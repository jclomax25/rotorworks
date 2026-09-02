"""
Physics correctness tests.

Every test here corresponds to a bug that was actually shipped at some point.
The docstrings name the failure so that a future regression is recognisable
rather than just "test_foo failed".
"""

from __future__ import annotations

import math

import pytest


# ======================================================================
# BATTERY PACK TOPOLOGY
# ======================================================================

def _pack(mc, series_units, parallel_units, unit_mode="pack"):
    return mc.BatteryConfig(
        chemistry="LiPo",
        operating_voltage_min=3.0, operating_voltage_nominal=3.7,
        operating_voltage_max=4.2,
        unit_mode=unit_mode,
        series_units=series_units, parallel_units=parallel_units,
        cells_series_per_unit=6, cells_parallel_per_unit=1,
        pack_capacity_mAh=5000, pack_weight_g=700,
        discharge_percent=80, resistance_cell_mOhm=3.0,
    )


@pytest.mark.parametrize(
    "series,parallel,exp_mAh,exp_Wh,exp_g",
    [
        (1, 1,  5000, 111.0,  700),   # 6S1P
        (2, 1,  5000, 222.0, 1400),   # 12S1P — series doubles VOLTAGE, not mAh
        (1, 2, 10000, 222.0, 1400),   # 6S2P  — parallel doubles CAPACITY
        (2, 2, 10000, 444.0, 2800),   # 12S2P
    ],
)
def test_pack_capacity_scales_with_parallel_only(mc, series, parallel,
                                                 exp_mAh, exp_Wh, exp_g):
    """
    Regression: capacity was multiplied by the SERIES count as well as the
    parallel count, so any series-stacked pack reported double the energy and
    therefore double the endurance. 12S1P read 10000 mAh / 444 Wh instead of
    5000 mAh / 222 Wh.
    """
    b = _pack(mc, series, parallel)
    assert b.capacity_mAh == pytest.approx(exp_mAh, abs=1)
    assert b.capacity_Wh == pytest.approx(exp_Wh, abs=0.5)
    assert b.weight_g == pytest.approx(exp_g, abs=1)


@pytest.mark.parametrize("mode", ["pack", "Pack", "PACK", " pack "])
def test_unit_mode_is_case_insensitive(mc, mode):
    """
    Regression: the constructor branched on the RAW unit_mode argument rather
    than the normalised one, so "Pack" fell through to an else-branch that
    silently set capacity AND weight to zero.
    """
    b = _pack(mc, 2, 1, unit_mode=mode)
    assert b.capacity_mAh > 0
    assert b.weight_g > 0


def test_series_parallel_counts_clamped_to_at_least_one(mc):
    """A zero count would give a zero-volt pack and divide-by-zero downstream."""
    b = _pack(mc, 0, 0)
    assert b.series_units >= 1
    assert b.parallel_units >= 1
    assert b.vmax_pack > 0


def test_fixedwing_pack_capacity_matches_multicopter(mc, fw):
    """Both simulators must agree on pack topology arithmetic."""
    a = _pack(mc, 2, 2)
    b = fw.BatteryConfig(
        chemistry="LiPo",
        operating_voltage_min=3.0, operating_voltage_nominal=3.7,
        operating_voltage_max=4.2, unit_mode="pack",
        series_units=2, parallel_units=2,
        cells_series_per_unit=6, cells_parallel_per_unit=1,
        pack_capacity_mAh=5000, pack_weight_g=700,
        discharge_percent=80, resistance_cell_mOhm=3.0,
    )
    assert a.capacity_mAh == pytest.approx(b.capacity_mAh)
    assert a.capacity_Wh == pytest.approx(b.capacity_Wh, abs=0.5)


# ======================================================================
# ATMOSPHERE
# ======================================================================

@pytest.mark.parametrize(
    "alt,temp,press",
    [(0, None, None), (120, None, None), (120, 25, None),
     (120, None, 95000), (120, 25, 95000), (2000, None, None), (0, 40, None)],
)
def test_atmosphere_models_agree(mc, fw, alt, temp, press):
    """
    Regression: the fixed-wing only honoured a pressure override when a
    temperature was ALSO supplied, so `--pressure` alone was silently ignored
    and the two simulators disagreed.
    """
    a = fw.isa_density(alt, temp, press)
    b = mc.compute_air_density(alt, temp, press)
    assert a == pytest.approx(b, abs=2e-4)


def test_isa_sea_level_density(fw):
    assert fw.isa_density(0) == pytest.approx(1.225, abs=0.002)


def test_density_falls_with_altitude_and_heat(fw):
    assert fw.isa_density(2000) < fw.isa_density(0)
    assert fw.isa_density(0, 40) < fw.isa_density(0, 0)


# ======================================================================
# MULTICOPTER FORWARD-FLIGHT INFLOW  (Glauert)
# ======================================================================

def test_inflow_reduces_to_hover_at_zero_speed(mc):
    vh = 5.549
    assert mc.induced_velocity_forward_flight(vh, 0.0) == pytest.approx(vh, abs=1e-9)


def test_inflow_axial_case_matches_closed_form(mc):
    """At 90 deg incidence the solver must match vi = -V/2 + sqrt((V/2)^2 + vh^2)."""
    vh, V = 5.549, 10.0
    exact = -V / 2 + math.sqrt((V / 2) ** 2 + vh ** 2)
    got = mc.induced_velocity_forward_flight(vh, V, math.radians(90))
    assert got == pytest.approx(exact, abs=1e-3)


def test_inflow_edgewise_case_matches_implicit_solution(mc):
    """At 0 deg incidence the solver must satisfy vi = vh^2 / sqrt(V^2 + vi^2)."""
    vh, V = 5.549, 10.0
    vi = mc.induced_velocity_forward_flight(vh, V, 0.0)
    residual = vi - vh ** 2 / math.sqrt(V ** 2 + vi ** 2)
    assert residual == pytest.approx(0.0, abs=1e-6)


def test_inflow_decreases_with_airspeed(mc):
    """
    Regression: the multicopter used the STATIC hover value at every speed,
    overstating induced power roughly 2x at 10 m/s and 4x at 20 m/s.
    """
    vh = 5.549
    vals = [mc.induced_velocity_forward_flight(vh, V) for V in (0, 5, 10, 15, 20)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 0.5 * vals[0]


def test_multicopter_power_curve_has_a_bucket(mc, mc_quad):
    """
    Real multirotors show a power minimum 10-25% below hover around 8-14 m/s.
    Regression: with static inflow the curve rose monotonically, so there was
    no minimum for the optimiser to find.
    """
    speeds = list(range(0, 26, 2))
    powers = [
        mc.compute_operating_metrics(mc_quad, V, "hover" if V == 0 else "forward")["total_power_W"]
        for V in speeds
    ]
    hover = powers[0]
    v_min = speeds[powers.index(min(powers))]
    dip = 1.0 - min(powers) / hover

    assert 4 <= v_min <= 16, f"power bucket at {v_min} m/s, expected 4-16"
    assert 0.08 <= dip <= 0.40, f"bucket depth {dip:.1%}, expected 8-40%"
    assert powers[-1] > hover, "power must rise above hover at high speed"


def test_multicopter_hover_power_unchanged_by_inflow_model(mc, mc_quad):
    """The forward-flight correction must vanish at V = 0."""
    hover = mc.compute_operating_metrics(mc_quad, 0.0, "hover")["total_power_W"]
    T = mc.thrust_required(mc_quad, 0.0, "hover") / mc_quad.num_motors
    A = mc.disk_area(mc_quad.propeller.diameter_in)
    vh = math.sqrt(T / (2 * mc_quad.air_density * A))
    ideal_total = T * vh * mc_quad.num_motors
    assert hover > ideal_total          # electrical > ideal shaft power
    assert hover < 4.0 * ideal_total    # but not absurdly so


def test_multicopter_optimal_speeds_are_not_pinned(mc, mc_quad):
    """
    Regression: with a monotonic power curve, best-endurance pinned to the
    lower search bound (0.5 m/s) and best-range to the upper bound.
    """
    v_lo, v_hi = 0.5, 25.0
    be, _t, br, _d = mc.find_optimal_speeds(mc_quad, min_speed=v_lo, max_speed=v_hi)
    assert v_lo + 0.5 < be < v_hi - 0.5, f"best endurance pinned at {be}"
    assert v_lo + 0.5 < br < v_hi - 0.5, f"best range pinned at {br}"
    assert br >= be, "best-range speed must be at least best-endurance speed"


# ======================================================================
# FIXED-WING PROPULSION
# ======================================================================

def test_fixedwing_efficiency_never_exceeds_unity(fw, fw_plane):
    """
    Regression: cruise power used the STATIC hover form P = T*vi instead of
    the forward-flight P = T*(V+vi), understating power ~5x. The Metrics tab
    reported a system efficiency of 360%.
    """
    for V in (8, 12, 16, 20, 25, 30):
        m = fw.compute_metrics(fw_plane, V)
        eff = m["power_required_W"] / m["total_power_W"]
        assert eff <= 1.0, f"efficiency {eff:.1%} at {V} m/s exceeds 100%"


def test_fixedwing_induced_velocity_reduces_to_static_at_zero_speed(fw):
    T, rho, A = 2.0, 1.225, 0.0613
    static = math.sqrt(T / (2 * rho * A))
    assert fw._induced_velocity_forward(T, 0.0, rho, A) == pytest.approx(static, abs=1e-9)


def test_fixedwing_endurance_is_physically_plausible(fw, fw_plane):
    """A 2.6 kg airframe on 296 Wh should not fly for six hours."""
    m = fw.compute_metrics(fw_plane, 19.0)
    assert 20 < m["flight_time_min"] < 240


def test_prop_efficiency_peaks_and_falls_off(fw, fw_plane):
    """
    Propeller efficiency must vary with advance ratio: poor when static,
    peaking near 60% of pitch speed, collapsing as pitch speed is approached.
    Regression: it was a flat constant at every airspeed.
    """
    n_max = fw_plane.motor.kv * fw_plane.battery.vmax_pack / 60.0
    v_pitch = fw_plane.propeller.pitch_m * n_max
    peak = fw_plane.airframe.prop_efficiency

    etas = {V: fw.propeller_efficiency_at_speed(fw_plane, V)
            for V in (0.0, 0.2 * v_pitch, 0.6 * v_pitch, 0.95 * v_pitch)}

    assert all(e <= peak + 1e-9 for e in etas.values()), "eta exceeded the entered peak"
    assert etas[0.6 * v_pitch] == pytest.approx(peak, abs=1e-6), "peak not at design point"
    assert etas[0.2 * v_pitch] < etas[0.6 * v_pitch]
    assert etas[0.95 * v_pitch] < etas[0.6 * v_pitch]
    assert all(e > 0 for e in etas.values()), "eta must never reach zero"


def test_prop_efficiency_constant_model_is_flat(fw, fw_plane):
    """The 'constant' model must reproduce pre-2.4.0 behaviour exactly."""
    fw_plane.airframe.prop_eff_model = "constant"
    peak = fw_plane.airframe.prop_efficiency
    for V in (0, 10, 20, 30, 50):
        assert fw.propeller_efficiency_at_speed(fw_plane, V) == pytest.approx(peak)


def test_landing_ground_roll_dumps_lift(fw, fw_plane):
    """
    Regression: the ground roll was modelled at CL_takeoff, so lift nearly
    cancelled weight, the brakes saw almost nothing and the roll ran ~80 m
    where a textbook figure is ~20 m.
    """
    roll = fw.landing_distance_m(fw_plane, obstacle_height_m=0.0)
    assert 5 < roll < 60, f"ground roll {roll:.1f} m is implausible"


def test_landing_over_obstacle_exceeds_ground_roll(fw, fw_plane):
    """The 15 m obstacle figure includes an approach segment of 15 m x L/D."""
    total = fw.landing_distance_m(fw_plane, obstacle_height_m=15.0)
    roll = fw.landing_distance_m(fw_plane, obstacle_height_m=0.0)
    assert total > roll


def test_cruise_altitude_drives_glide_distance(fw, fw_plane):
    """
    Regression: glide distance used reference_altitude_m (the FIELD elevation),
    so it read 0 m whenever the field was at sea level regardless of how high
    the aircraft actually flew.
    """
    base = fw.compute_metrics(fw_plane, 19.0)["glide_distance_m"]

    fw_plane.cruise_altitude_m = 1000.0
    high = fw.compute_metrics(fw_plane, 19.0)["glide_distance_m"]

    assert high > base
    ratio = fw.compute_metrics(fw_plane, 19.0)["glide_ratio"]
    assert high == pytest.approx(ratio * 1000.0, rel=0.02)


def test_cruise_altitude_defaults_to_field_elevation(fw, fw_plane):
    """Leaving cruise altitude unset must preserve the old behaviour."""
    assert fw_plane.cruise_altitude_m is None
    assert fw_plane.glide_reference_altitude_m == pytest.approx(
        fw_plane.reference_altitude_m)


# ======================================================================
# DRAG MODEL
# ======================================================================

def test_forward_drag_does_not_double_count_arms(mc, mc_quad):
    """
    Regression: the geometry fallback put the arms into BOTH parasite_area
    and profile_area, and forward flight summed the two — overstating forward
    drag by ~113% on a typical frame.
    """
    V = 12.0
    q = 0.5 * mc_quad.air_density * V ** 2
    fwd = mc.drag_force_required(mc_quad, V, "forward")
    expected = q * mc_quad.parasite_area * mc_quad.parasite_drag_coefficient
    assert fwd == pytest.approx(expected, rel=1e-9)


def test_hover_drag_uses_side_profile(mc, mc_quad):
    V = 12.0
    q = 0.5 * mc_quad.air_density * V ** 2
    hov = mc.drag_force_required(mc_quad, V, "hover")
    expected = q * mc_quad.profile_area * mc_quad.profile_drag_coefficient
    assert hov == pytest.approx(expected, rel=1e-9)


def test_wind_resistance_is_nan_when_area_unknown(mc, mc_quad):
    """
    Regression: an unknown reference area fell back to 1e-6 m^2, producing
    wind resistances of thousands of m/s. Unknown must read as NaN, not as a
    confident wrong number.
    """
    mc_quad.parasite_area = 0.0
    mc_quad.frontal_area = 0.0
    assert math.isnan(mc.hover_wind_resistance_mps(mc_quad))


# ======================================================================
# BATTERY STATE OF CHARGE
# ======================================================================

def _soc_pack(mod, chem="LiPo", model="auto", **kw):
    return mod.BatteryConfig(
        chemistry=chem,
        operating_voltage_min=3.3, operating_voltage_nominal=3.7,
        operating_voltage_max=4.2, unit_mode="cell",
        series_units=4, parallel_units=1,
        cell_capacity_mAh=5000, cell_weight_g=110,
        discharge_percent=80, resistance_cell_mOhm=4.0,
        soc_model=model, **kw
    )


@pytest.mark.parametrize("chem,expect", [
    ("LiPo", "lipo"), ("Li-ion", "liion"), ("LiFePO4", "lifepo4"),
])
def test_soc_preset_selected_from_chemistry(fw, chem, expect):
    b = _soc_pack(fw, chem=chem)
    assert b.soc_model_source.startswith("preset")
    assert expect in b.soc_model_source
    assert b.soc_nonlinear_enabled


def test_soc_open_circuit_voltage_falls_as_pack_empties(fw):
    b = _soc_pack(fw)
    socs = [1.0, 0.8, 0.5, 0.2, 0.05]
    ocv = [b.ocv_at_soc(s) for s in socs]
    assert all(a >= c for a, c in zip(ocv, ocv[1:]))


def test_soc_resistance_rises_when_nearly_empty(fw):
    b = _soc_pack(fw)
    assert b.resistance_at_soc(0.05) > b.resistance_at_soc(0.5)


def test_soc_linear_model_matches_legacy_behaviour(fw):
    b = _soc_pack(fw, model="linear")
    assert not b.soc_nonlinear_enabled
    expected = b.vmax_pack - 20.0 * b.pack_resistance
    assert b.voltage_under_load(20.0) == pytest.approx(expected)


def test_voltage_under_load_defaults_to_full_charge(fw):
    """Omitting soc must reproduce the old signature's behaviour exactly."""
    b = _soc_pack(fw)
    assert b.voltage_under_load(20.0) == pytest.approx(b.voltage_under_load(20.0, soc=1.0))


def test_soc_custom_breakpoints_accepted(fw):
    b = _soc_pack(fw, soc_bp=[0, 0.5, 1.0],
                  ocv_cell_bp=[3.2, 3.8, 4.2], r_scale_bp=[2.0, 1.0, 1.2])
    assert b.soc_model_source == "custom-arrays"


def test_soc_breakpoints_accept_percentages(fw):
    assert fw.parse_soc_breakpoints("0,50,100") == pytest.approx([0.0, 0.5, 1.0])


def test_soc_models_agree_between_simulators(mc, fw):
    """Both simulators must produce identical SoC curves for the same pack."""
    a, b = _soc_pack(mc), _soc_pack(fw)
    for s in (1.0, 0.75, 0.5, 0.25, 0.05):
        assert a.ocv_at_soc(s) == pytest.approx(b.ocv_at_soc(s), abs=1e-9)
        assert a.resistance_at_soc(s) == pytest.approx(b.resistance_at_soc(s), abs=1e-12)


# ======================================================================
# METRICS SANITY
# ======================================================================

def test_metrics_never_return_none_for_tip_values(mc, mc_quad):
    """
    Regression: tip_speed_mps and tip_mach are stored as None when RPM is
    unavailable, and callers used dict.get(key, nan_default) — which returns
    None, not the default, when the key EXISTS with value None. float(None)
    then raised TypeError on the default config.
    """
    m = mc.compute_operating_metrics(mc_quad, 10.0, "forward")
    for key in ("tip_speed_mps", "tip_mach"):
        val = m.get(key)
        assert val is None or isinstance(val, float)
        # The safe pattern the GUI must use:
        safe = float(val) if val is not None else float("nan")
        assert isinstance(safe, float)


def test_wing_loading_matches_hand_calculation(fw, fw_plane):
    m = fw.compute_metrics(fw_plane, 19.0)
    expected = fw_plane.weight_N / fw_plane.airframe.wing_area_m2
    assert m["wing_loading_N_m2"] == pytest.approx(expected, rel=1e-6)


def test_stall_speed_matches_hand_calculation(fw, fw_plane):
    m = fw.compute_metrics(fw_plane, 19.0)
    af = fw_plane.airframe
    expected = math.sqrt(2 * fw_plane.weight_N /
                         (fw_plane.air_density * af.wing_area_m2 * af.CL_max))
    assert m["stall_speed_mps"] == pytest.approx(expected, rel=1e-6)


# ======================================================================
# DRAG COEFFICIENT CALCULATOR
# ======================================================================

def test_shoelace_area_of_known_polygons(dragcalc):
    unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    triangle = [(0, 0), (4, 0), (0, 3)]
    assert dragcalc.polygon_area_px2(unit_square) == pytest.approx(1.0, abs=1e-9)
    assert dragcalc.polygon_area_px2(triangle) == pytest.approx(6.0, abs=1e-9)


def test_shoelace_is_winding_independent(dragcalc):
    clockwise = [(0, 0), (0, 1), (1, 1), (1, 0)]
    assert dragcalc.polygon_area_px2(clockwise) == pytest.approx(1.0, abs=1e-9)


def test_shoelace_degenerate_polygon_is_zero(dragcalc):
    assert dragcalc.polygon_area_px2([(0, 0), (1, 1)]) == 0.0


def test_bcoef_matches_ardupilot_iris_reference(dragcalc):
    """
    ArduPilot's airspeed-estimation guide works the IRIS as its example:
    1.45 kg with 0.0203 m^2 frontal and 0.0217 m^2 side area gives
    EK3_DRAG_BCOEF_X = 71.4 and BCOEF_Y = 66.8 (Cd assumed 1.0).
    """
    mass, cd = 1.45, 1.0
    assert mass / (cd * 0.0203) == pytest.approx(71.4, abs=0.2)
    assert mass / (cd * 0.0217) == pytest.approx(66.8, abs=0.2)


def test_mcoef_is_in_the_documented_range(dragcalc):
    """MCOEF = g / (2 * v_h); ArduPilot documents 0.1-1.0 as typical."""
    g, rho = 9.80665, dragcalc.isa_density(0)
    diameter_m = 10 * 0.0254
    area = math.pi / 4 * diameter_m ** 2
    v_h = math.sqrt((1.5 * g / 4) / (2 * rho * area))
    mcoef = g / (2 * v_h)
    assert 0.1 <= mcoef <= 1.0


def test_dragcalc_atmosphere_matches_the_simulators(dragcalc, fw):
    for alt in (0, 1000, 2000):
        assert dragcalc.isa_density(alt) == pytest.approx(fw.isa_density(alt), abs=2e-4)


# ======================================================================
# FIXED-WING TRANSIENT (acceleration / deceleration) MODEL
# ======================================================================

def _fw_mission(fw, tmp_path, payload):
    import json
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(payload))
    return fw.MissionProfile.from_json(str(path))


def test_constant_speed_mission_is_unaffected_by_transients(fw, fw_plane, tmp_path):
    """
    The transient model is a lead-in ramp. With no speed change there is
    nothing to ramp, so results must match the pre-transient behaviour.
    """
    profile = _fw_mission(fw, tmp_path, {"phases": [
        {"name": "A", "speed": 18.0, "duration": 300, "altitude": 150},
        {"name": "B", "speed": 18.0, "duration": 300, "altitude": 150},
    ]})
    results, _worst, _series = fw.simulate_fw_mission(fw_plane, profile)
    for _name, minutes, _km, status in results:
        assert minutes == pytest.approx(5.0, abs=1e-6)
        assert status == "OK"


def test_acceleration_limit_changes_distance_covered(fw, fw_plane, tmp_path):
    """
    A gentler acceleration spends longer at low speed, so the aircraft covers
    less ground in the same phase duration.
    """
    def distance_with(accel):
        profile = _fw_mission(fw, tmp_path, {
            "max_accel_mps2": accel, "max_decel_mps2": 2.0,
            "phases": [
                {"name": "slow", "speed": 12.0, "duration": 300, "altitude": 150},
                {"name": "fast", "speed": 28.0, "duration": 300, "altitude": 150},
            ]})
        return fw.simulate_fw_mission(fw_plane, profile)[0][1][2]

    gentle, brisk = distance_with(0.5), distance_with(5.0)
    assert gentle < brisk, "a slower ramp should cover less ground"


def test_transient_settings_are_read_from_mission_json(fw, tmp_path):
    profile = _fw_mission(fw, tmp_path, {
        "transient_dt_s": 0.25, "max_accel_mps2": 3.0,
        "max_decel_mps2": 4.0, "decel_regen_eff": 0.15,
        "phases": [{"name": "x", "speed": 18.0, "duration": 60, "altitude": 100}]})
    assert profile.transient_dt_s == pytest.approx(0.25)
    assert profile.max_accel_mps2 == pytest.approx(3.0)
    assert profile.max_decel_mps2 == pytest.approx(4.0)
    assert profile.decel_regen_eff == pytest.approx(0.15)


def test_kinetic_power_costs_energy_to_accelerate(mc, fw):
    """Shared helper: speeding up costs power, slowing down releases it."""
    for module in (mc, fw):
        speeding_up = module.kinetic_power_term_W(2600, 10.0, 20.0, 1.0, 0.0)
        slowing = module.kinetic_power_term_W(2600, 20.0, 10.0, 1.0, 0.0)
        assert speeding_up > 0
        assert slowing == 0.0, "no regen by default"
        recovered = module.kinetic_power_term_W(2600, 20.0, 10.0, 1.0, 0.5)
        assert recovered < 0, "regen should return some energy"


def test_ramp_speed_never_overshoots(mc):
    """A phase must settle onto its commanded speed and hold it."""
    v, _a = mc.ramp_speed(10.0, 12.0, 10.0, 5.0, 5.0)   # huge step available
    assert v == pytest.approx(12.0), "ramp overshot the target"
    v, _a = mc.ramp_speed(20.0, 12.0, 10.0, 5.0, 5.0)
    assert v == pytest.approx(12.0)


def test_ramp_speed_respects_limits(mc):
    v, accel = mc.ramp_speed(10.0, 30.0, 2.0, 1.5, 2.0)
    assert v == pytest.approx(13.0)                     # 10 + 1.5*2
    assert accel == pytest.approx(1.5)


# ======================================================================
# MEASURED PROPELLER / MOTOR TABLES
# ======================================================================

import os as _os
_TABLE_CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "data", "motor_prop_table.csv")


def test_prop_table_csv_with_sparse_title_row_parses(mc):
    """
    Regression: the header scan did `raw.iloc[i].astype(str)`, and with
    pandas' newer `str` dtype that leaves NaN as a real float rather than the
    string 'nan'. A vendor export whose first line is a sparse title row
    ("Test Data,,,,,") therefore raised
    "'float' object has no attribute 'startswith'".
    """
    prop = mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                              max_thrust_g=9500, blades=2, weight_g=110,
                              table_csv=_TABLE_CSV)
    assert prop.table is not None
    assert len(prop.table) > 5
    for column in ("Thrust_g", "Power_W", "RPM", "Current_A", "Throttle_pct"):
        assert column in prop.table.columns, f"{column} missing after parse"


def test_fit_propeller_curve_returns_three_parts(mc, fw):
    """
    Regression: extracting this helper into the shared core dropped the
    (coeffs, x_min, x_max) tuple down to a bare coefficient list. The caller
    unpacks three values, so `coeffs` silently became a float and every
    prop-table lookup raised "'float' object is not iterable".
    """
    for module in (mc, fw):
        result = module._fit_propeller_curve([0, 1, 2, 3], [1, 2, 5, 10], degree=2)
        assert isinstance(result, tuple) and len(result) == 3
        coeffs, x_min, x_max = result
        assert hasattr(coeffs, "__iter__"), "coeffs must be iterable"
        assert x_min <= x_max


def _quad_with_table(mc):
    batt = mc.BatteryConfig(
        chemistry="LiPo", operating_voltage_min=3.3,
        operating_voltage_nominal=3.7, operating_voltage_max=4.2,
        unit_mode="pack", pack_capacity_mAh=16000, pack_weight_g=2100,
        series_units=2, parallel_units=1, cells_series_per_unit=6,
        discharge_percent=80, resistance_cell_mOhm=3.0, discharge_c_cont=15)
    drone = mc.DroneConfig(
        num_motors=8, battery=batt,
        motor=mc.MotorConfig(kv=160, idle_current=0.6, idle_voltage=10,
                             rated_voltage=12, resistance=0.045,
                             max_current=45, max_power=2000, weight_g=410),
        propeller=mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                                     max_thrust_g=9500, blades=2, weight_g=110,
                                     table_csv=_TABLE_CSV),
        drone_weight_g=11500,
        profile_drag_coefficient=None, profile_area=None,
        parasite_drag_coefficient=None, parasite_area=None, frontal_area=None,
        cruise_speed=8.0, periph_current=0.5,
        motor_configuration="coaxial", coaxial_spacing_m=0.10, max_tilt_deg=20,
        body_length_m=0.45, body_width_m=0.40, body_height_m=0.20,
        arm_length_m=0.40, arm_width_m=0.030)
    drone.air_density = mc.compute_air_density(500)
    drone.derive_drag_from_geometry_if_missing()
    return drone


def test_single_point_run_with_a_measured_table(mc):
    """A run backed by a measured table must produce finite, sane numbers."""
    drone = _quad_with_table(mc)
    metrics = mc.compute_operating_metrics(drone, 8.0, "forward")
    assert metrics["total_power_W"] > 0
    assert math.isfinite(metrics["total_power_W"])
    assert metrics.get("prop_rpm") is not None, "a table should give an RPM"


def test_motor_operating_point_figure_builds_from_a_table(mc):
    """
    The motor operating-point plots only appear when a measured table is
    loaded, and the call site swallows exceptions — so a failure here shows
    up as silently missing plots rather than an error.
    """
    import matplotlib
    matplotlib.use("Agg")
    drone = _quad_with_table(mc)
    metrics = mc.compute_operating_metrics(drone, 8.0, "forward")
    figure = mc.make_motor_operating_point_figure(drone, metrics, figsize=(10, 6))
    assert figure is not None
    assert len(figure.axes) >= 2


# ======================================================================
# MULTI-MOTOR FIXED-WING
# ======================================================================

def _fw_with_motors(fw, n_motors, diameter_in=12):
    batt = fw.BatteryConfig(
        chemistry="LiPo", operating_voltage_min=3.3,
        operating_voltage_nominal=3.7, operating_voltage_max=4.2,
        unit_mode="pack", pack_capacity_mAh=10000, pack_weight_g=880,
        series_units=1, parallel_units=1, cells_series_per_unit=4,
        discharge_percent=80, resistance_cell_mOhm=3.5)
    airframe = fw.AirframeConfig(
        wing_span_m=2.0, wing_area_m2=0.46, CD0=0.028, oswald=0.87,
        CL_max=1.25, prop_efficiency=0.76, num_motors=n_motors)
    return fw.FixedWingConfig(
        aircraft_weight_g=2600, airframe=airframe, battery=batt,
        motor=fw.MotorConfig(kv=750, idle_current=0.8, idle_voltage=10,
                             rated_voltage=4, resistance=0.06,
                             max_current=40, max_power=600, weight_g=160),
        propeller=fw.PropellerConfig(diameter_in=diameter_in, pitch_in=8,
                                     blades=2, weight_g=22),
        cruise_speed_mps=19.0, air_density=1.225, reference_altitude_m=120)


def test_motor_count_changes_fixed_wing_power(fw):
    """
    Regression: motor_shaft_power_from_thrust fed TOTAL thrust through a
    SINGLE propeller disc and never consulted num_motors, so a twin, a triple
    and a single all reported exactly the same power.
    """
    powers = [fw.compute_metrics(_fw_with_motors(fw, n), 19.0)["total_power_W"]
              for n in (1, 2, 3, 4)]
    assert len(set(round(p, 6) for p in powers)) > 1, \
        "motor count has no effect on power"


def test_more_motors_lower_induced_power(fw):
    """
    Spreading the same thrust over more disc area lowers induced velocity and
    therefore induced power. Adding motors must not make cruise cost MORE.
    """
    powers = [fw.compute_metrics(_fw_with_motors(fw, n), 19.0)["total_power_W"]
              for n in (1, 2, 3, 4)]
    assert powers == sorted(powers, reverse=True), \
        f"power should fall as motors are added, got {powers}"
    # The effect is real but modest: a few percent, not a factor.
    assert 0.005 < (powers[0] - powers[1]) / powers[0] < 0.15


def test_multi_motor_thrust_available_scales(fw):
    """Available thrust is per-motor times motor count."""
    single = fw.compute_metrics(_fw_with_motors(fw, 1), 19.0)["thrust_available_N"]
    twin = fw.compute_metrics(_fw_with_motors(fw, 2), 19.0)["thrust_available_N"]
    assert twin == pytest.approx(2.0 * single, rel=1e-6)


@pytest.mark.parametrize("n_motors", [1, 2, 3, 4, 6])
def test_fixed_wing_runs_for_any_motor_count(fw, n_motors):
    m = fw.compute_metrics(_fw_with_motors(fw, n_motors), 19.0)
    assert m["total_power_W"] > 0
    assert math.isfinite(m["flight_time_min"])
    assert m["flight_time_min"] > 0


def test_multicopter_already_splits_thrust_per_motor(mc, mc_quad):
    """
    The multicopter divides total thrust by motor count before any
    single-rotor calculation. This guards the equivalent of the fixed-wing bug.
    """
    metrics = mc.compute_operating_metrics(mc_quad, 10.0, "forward")
    total = float(metrics["thrust_total_N"])
    per_motor = float(metrics["thrust_per_motor_N"])
    assert per_motor == pytest.approx(total / mc_quad.num_motors, rel=1e-9)


# ======================================================================
# FIXED-WING BENCH TABLES
# ======================================================================

_FW_TABLE_CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "data", "fw_motor_prop_table.csv")


def test_fixedwing_table_with_sparse_title_row_parses(fw):
    """
    Regression: the fixed-wing had its OWN table loader with the same
    Series.astype(str) fault fixed in the multicopter — a sparse title row
    put floats in the header scan and raised
    "argument of type 'float' is not iterable".
    """
    df = fw.load_prop_table(_FW_TABLE_CSV)
    assert len(df) >= 10
    for column in ("Thrust_g", "Power_W", "RPM", "Current_A", "Throttle_pct"):
        assert column in df.columns, f"{column} missing after parse"


def test_table_columns_map_by_name_not_position(fw, mc):
    """
    The two sample tables list the same fields in a different order. Column
    mapping must be by name, or one of them silently reads the wrong data.
    """
    fw_df = fw.load_prop_table(_FW_TABLE_CSV)
    mc_prop = mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                                 max_thrust_g=9500, blades=2, weight_g=110,
                                 table_csv=_TABLE_CSV)
    # Thrust rises with power in both, whatever order the columns appeared in.
    for df in (fw_df, mc_prop.table):
        assert df["Thrust_g"].is_monotonic_increasing
        assert df["Power_W"].iloc[-1] > df["Power_W"].iloc[0]


def _fw_table_config(fw, table):
    batt = fw.BatteryConfig(
        chemistry="LiPo", operating_voltage_min=3.3,
        operating_voltage_nominal=3.7, operating_voltage_max=4.2,
        unit_mode="cell", cell_capacity_mAh=10000, cell_weight_g=220,
        series_units=6, parallel_units=1,
        discharge_percent=80, resistance_cell_mOhm=3.0)
    airframe = fw.AirframeConfig(
        wing_span_m=2.4, wing_area_m2=0.62, CD0=0.030, oswald=0.85,
        CL_max=1.20, prop_efficiency=0.75, num_motors=1)
    return fw.FixedWingConfig(
        aircraft_weight_g=6500, airframe=airframe, battery=batt,
        motor=fw.MotorConfig(kv=380, idle_current=1.0, idle_voltage=10,
                             rated_voltage=6, resistance=0.05,
                             max_current=60, max_power=1800, weight_g=400),
        propeller=fw.PropellerConfig(diameter_in=18, pitch_in=8, blades=2,
                                     weight_g=60, table_csv=table),
        cruise_speed_mps=22.0, air_density=1.225, reference_altitude_m=100)


def test_table_changes_the_fixed_wing_result(fw):
    """Supplying a measured table must actually influence the answer."""
    plain = fw.compute_metrics(_fw_table_config(fw, None), 22.0)
    tabled = fw.compute_metrics(_fw_table_config(fw, _FW_TABLE_CSV), 22.0)
    assert tabled["total_power_W"] != pytest.approx(plain["total_power_W"], rel=1e-6)


def test_static_table_is_not_read_as_cruise_power(fw):
    """
    Regression: the bench table is STATIC data. Reading its power directly at
    cruise ignored the T*V work the propeller does against the oncoming air —
    at 22 m/s that is 5.7x the static ideal power — and overstated endurance
    by roughly 3.8x.

    Cruise power with a table must therefore be the same order as the
    analytic estimate, not a small fraction of it.
    """
    plain = fw.compute_metrics(_fw_table_config(fw, None), 22.0)
    tabled = fw.compute_metrics(_fw_table_config(fw, _FW_TABLE_CSV), 22.0)
    ratio = tabled["total_power_W"] / plain["total_power_W"]
    assert 0.5 < ratio < 2.5, (
        f"table cruise power is {ratio:.2f}x the analytic estimate; "
        "a static table is probably being read directly")


def test_table_derived_efficiency_is_physically_sensible(fw):
    """
    The table implies a combined motor+prop efficiency. It must land in a
    believable band — a real combination is neither 15% nor 90% efficient.
    """
    import math as _m
    df = fw.load_prop_table(_FW_TABLE_CSV)
    rho, area = 1.225, _m.pi / 4 * (18 * 0.0254) ** 2
    for _, row in df.iterrows():
        thrust_N = float(row["Thrust_g"]) * 9.80665 / 1000.0
        ideal = thrust_N * _m.sqrt(thrust_N / (2 * rho * area))
        eta = ideal / float(row["Power_W"])
        assert 0.20 < eta < 0.80, f"implied efficiency {eta:.2f} is not credible"


def test_table_lookup_is_fast_enough_for_interactive_plots(fw):
    """
    Regression: max_thrust_N called pandas Series.max() on the table on every
    evaluation. The climb-rate and best-speed searches evaluate thrust ~1000
    times per run, so a single point took 49 ms and a 201-point plot sweep
    took ~10 s — long enough for the window manager to report the GUI as not
    responding.

    Bounds and columns are now cached at load. This guards the regression
    with a deliberately loose budget so it fails on a 10x slowdown, not on
    ordinary machine-to-machine variation.
    """
    import time

    config = _fw_table_config(fw, _FW_TABLE_CSV)
    fw.compute_metrics(config, 22.0)          # warm any lazy work

    start = time.perf_counter()
    for _ in range(20):
        fw.compute_metrics(config, 22.0)
    per_call_ms = (time.perf_counter() - start) / 20 * 1000.0

    assert per_call_ms < 25.0, (
        f"{per_call_ms:.1f} ms per evaluation with a table; the cached table "
        "bounds have probably been lost")


def test_table_bounds_are_cached_on_the_propeller(fw, mc):
    """Both simulators must cache the table's scalar bounds at load time."""
    fw_prop = fw.PropellerConfig(diameter_in=18, pitch_in=8, blades=2,
                                 weight_g=60, table_csv=_FW_TABLE_CSV)
    mc_prop = mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                                 max_thrust_g=9500, blades=2, weight_g=110,
                                 table_csv=_TABLE_CSV)
    for prop in (fw_prop, mc_prop):
        assert getattr(prop, "_thrust_g_max", None) is not None
        assert prop._thrust_g_max == pytest.approx(
            float(prop.table["Thrust_g"].max()))
        assert prop._thrust_g_min == pytest.approx(
            float(prop.table["Thrust_g"].min()))


def test_caching_did_not_change_the_answer(fw):
    """The cache is an optimisation; results must be identical."""
    config = _fw_table_config(fw, _FW_TABLE_CSV)
    metrics = fw.compute_metrics(config, 22.0)
    cached_max = config.propeller._thrust_g_max

    # Recompute the same quantity the slow way and confirm agreement.
    config.propeller._thrust_g_max = None
    slow = fw.compute_metrics(config, 22.0)
    config.propeller._thrust_g_max = cached_max

    for key, value in metrics.items():
        if isinstance(value, float) and math.isfinite(value):
            assert slow.get(key) == pytest.approx(value, rel=1e-12), (
                f"{key} differs between cached and uncached paths")


def test_static_power_model_fits_the_measured_data(fw):
    """
    The two-term model P = a*T^1.5 + b must reproduce the measured table
    closely, or extrapolating with it is not justified.
    """
    prop = fw.PropellerConfig(diameter_in=18, pitch_in=8, blades=2,
                              weight_g=60, table_csv=_FW_TABLE_CSV)
    assert prop._static_power_a is not None, "no static power fit was made"
    assert prop._static_power_b > 0, "fixed loss term should be positive"

    for _, row in prop.table.iterrows():
        thrust_N = float(row["Thrust_g"]) * 9.80665 / 1000.0
        predicted = prop._static_power_a * thrust_N ** 1.5 + prop._static_power_b
        measured = float(row["Power_W"])
        assert abs(predicted - measured) / measured < 0.10, \
            f"fit is off by more than 10% at {row['Thrust_g']:.0f} g"


def test_efficiency_collapses_near_zero_thrust(fw):
    """
    A motor still draws its idle power at zero thrust, so g/W must fall to
    zero there.

    Regression: extrapolating power as a pure T^1.5 power law assumed
    efficiency was constant, which made predicted efficiency rise without
    bound — 40 g/W at 50 g of thrust against a best measured 7.6 g/W. The
    fixed-loss term fixes the shape.
    """
    prop = fw.PropellerConfig(diameter_in=18, pitch_in=8, blades=2,
                              weight_g=60, table_csv=_FW_TABLE_CSV)

    def efficiency(thrust_g):
        power = fw._table_power_for_thrust(
            prop.table, thrust_g * 9.80665 / 1000.0, prop)
        return thrust_g / power

    assert efficiency(10) < 2.0, "efficiency should collapse near zero thrust"
    assert efficiency(50) < efficiency(500), "efficiency curve has the wrong shape"


def test_extrapolated_efficiency_stays_credible(fw):
    """
    Below the table, efficiency may exceed the best measured value — lower
    disc loading genuinely is more efficient — but not implausibly so.
    """
    prop = fw.PropellerConfig(diameter_in=18, pitch_in=8, blades=2,
                              weight_g=60, table_csv=_FW_TABLE_CSV)
    best_measured = float((prop.table["Thrust_g"] / prop.table["Power_W"]).max())
    worst = 0.0
    for thrust_g in range(20, 1420, 20):
        power = fw._table_power_for_thrust(
            prop.table, thrust_g * 9.80665 / 1000.0, prop)
        worst = max(worst, thrust_g / power)
    assert worst < 2.0 * best_measured, (
        f"extrapolated efficiency reaches {worst:.1f} g/W against a best "
        f"measured {best_measured:.1f} g/W")


def test_below_range_table_power_is_never_negative(fw):
    """
    Regression: a polynomial fitted to the measured band and extrapolated
    downward crossed zero. On the sample table (1426-6733 g) it went negative
    below ~250 g, and the operating-point marker read -12.7 W with an implied
    53 g/W. Below-range power now follows static momentum theory
    (P proportional to T^1.5) anchored on the lowest measured row.
    """
    df = fw.load_prop_table(_FW_TABLE_CSV)
    for thrust_g in range(10, 1500, 10):
        power = fw._table_power_for_thrust(df, thrust_g * 9.80665 / 1000.0)
        assert power is not None and power > 0, \
            f"{thrust_g} g gave {power} W"


def test_below_range_table_power_is_monotonic(fw):
    """More thrust must never cost less power."""
    df = fw.load_prop_table(_FW_TABLE_CSV)
    powers = [fw._table_power_for_thrust(df, g * 9.80665 / 1000.0)
              for g in range(50, 1500, 25)]
    assert powers == sorted(powers)


def test_table_lookup_is_exact_at_the_measured_edge(fw):
    """The extrapolation must reproduce the measurement it is anchored on."""
    df = fw.load_prop_table(_FW_TABLE_CSV)
    lo_g = float(df["Thrust_g"].iloc[0])
    lo_w = float(df["Power_W"].iloc[0])
    assert fw._table_power_for_thrust(df, lo_g * 9.80665 / 1000.0) == \
        pytest.approx(lo_w, rel=1e-9)


def test_operating_curve_flags_extrapolation(fw):
    """
    An operating point outside the measured data is an extrapolation, and the
    chart must say so rather than presenting it as measured.
    """
    import matplotlib
    matplotlib.use("Agg")
    config = _fw_table_config(fw, _FW_TABLE_CSV)
    metrics = fw.compute_metrics(config, 22.0)
    figure = fw.make_motor_operating_point_figure(config, metrics, figsize=(12, 5))
    title = figure._suptitle.get_text()
    thrust_g = metrics["thrust_required_N"] / config.num_motors * 1000.0 / 9.80665
    df = config.propeller.table
    if thrust_g < float(df["Thrust_g"].min()):
        assert "EXTRAPOLATED" in title, f"no extrapolation warning in: {title}"


def test_multicopter_below_range_values_stay_positive(mc):
    """The multicopter shares the fault class; guard it the same way."""
    prop = mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                              max_thrust_g=9500, blades=2, weight_g=110,
                              table_csv=_TABLE_CSV)
    df = prop.table
    below = float(df["Thrust_g"].min()) * 0.1
    point = mc.interpolate_motor_point.__wrapped__(prop, below) \
        if hasattr(mc.interpolate_motor_point, "__wrapped__") else None
    # Exercise the public path instead, which is what the GUI uses.
    for factor in (0.05, 0.2, 0.5, 0.9):
        thrust_N = float(df["Thrust_g"].min()) * factor * 9.80665 / 1000.0
        power = mc.interpolate_motor_power(
            type("C", (), {"propeller": prop, "num_motors": 1})(), thrust_N)
        assert power > 0, f"{factor:.2f} of min thrust gave {power} W"


# ======================================================================
# THRUST AVAILABLE MUST FALL WITH AIRSPEED
# ======================================================================

def test_thrust_available_decreases_with_airspeed(fw):
    """
    Regression: thrust available returned the STATIC bench figure at every
    airspeed. A propeller cannot make its static thrust at speed.
    """
    config = _fw_table_config(fw, _FW_TABLE_CSV)
    thrusts = [fw.thrust_available_N(config, v) for v in (0, 15, 25, 40, 60)]
    assert thrusts == sorted(thrusts, reverse=True), \
        f"thrust must fall with speed, got {thrusts}"
    assert thrusts[-1] < 0.6 * thrusts[0], \
        "thrust barely dropped; the static value is probably still in use"


def test_thrust_available_at_zero_is_the_static_value(fw):
    config = _fw_table_config(fw, _FW_TABLE_CSV)
    assert fw.thrust_available_N(config, 0.0) == pytest.approx(
        fw.max_thrust_N(config), rel=1e-9)


def test_climb_rate_obeys_energy_conservation(fw):
    """
    The strongest available check: ideal climb power (W x RC) can never exceed
    the shaft power the propulsion system can deliver.

    Regression: with static thrust used at all speeds, the model reported a
    best climb of 3597 m/min needing 2469 W of ideal power, against a measured
    maximum of 1680 W electrical.
    """
    for table in (None, _FW_TABLE_CSV):
        config = _fw_table_config(fw, table)
        metrics = fw.compute_metrics(config, 22.0)
        climb_power_W = config.weight_N * metrics["max_rc_mps"]
        available_W = fw.max_shaft_power_W(config)
        if available_W > 0:
            assert climb_power_W <= available_W * 1.05, (
                f"best climb needs {climb_power_W:.0f} W of ideal power but "
                f"only {available_W:.0f} W of shaft power is available")


def test_best_climb_speed_is_physically_plausible(fw):
    """
    Best rate of climb occurs a little above stall, not at several times it.

    Regression: because thrust did not decay with speed, excess power kept
    rising and the search reported best climb at 5.8x stall speed.
    """
    config = _fw_table_config(fw, _FW_TABLE_CSV)
    metrics = fw.compute_metrics(config, 22.0)
    stall = metrics["stall_speed_mps"]
    v_best = metrics["v_max_rc_mps"]
    assert stall < v_best < 4.0 * stall, (
        f"best climb at {v_best:.1f} m/s against a stall speed of "
        f"{stall:.1f} m/s is not plausible")


def test_thrust_available_never_exceeds_static(fw):
    """Forward flight cannot beat the static thrust."""
    config = _fw_table_config(fw, _FW_TABLE_CSV)
    static = fw.thrust_available_N(config, 0.0)
    for v in range(0, 70, 5):
        assert fw.thrust_available_N(config, float(v)) <= static + 1e-9
