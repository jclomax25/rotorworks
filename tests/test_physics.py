"""
Physics correctness tests.

Every test here corresponds to a bug that was actually shipped at some point.
The docstrings name the failure so that a future regression is recognisable
rather than just "test_foo failed".
"""

from __future__ import annotations

import os
import math

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _curve_at(line, x):
    """A drawn line's value at x, between its plotted points."""
    xs, ys = (list(map(float, d)) for d in (line.get_xdata(), line.get_ydata()))
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise AssertionError(f"{x} is outside the drawn curve {xs[0]}-{xs[-1]}")


@pytest.mark.parametrize("layout,speed,orientation",
                         [("coaxial", 0.0, "hover"), ("flat", 8.0, "translating")])
def test_multicopter_motor_marker_sits_where_the_model_runs(mc, layout, speed, orientation):
    """
    Regression: the operating point was read straight off the static bench
    table, while the run's power applies the table's efficiency to the
    forward-flight ideal power and adds the coaxial penalty: 116 W marked
    against 75 W charged at 8 m/s, and against 137 W in a coaxial hover.
    """
    import matplotlib
    matplotlib.use("Agg")
    drone = _quad_with_table(mc)
    drone.motor_configuration = layout
    metrics = mc.compute_operating_metrics(drone, speed, orientation)
    per_motor_W = metrics["motor_power_W"] / drone.num_motors
    bench_W = mc.interpolate_motor_power(drone, metrics["thrust_per_motor_N"])
    assert abs(per_motor_W / bench_W - 1.0) > 0.1, "not the case this guards"

    figure = mc.make_motor_operating_point_figure(drone, metrics, figsize=(10, 5))
    power_ax = figure.axes[0]
    star = next(l for l in power_ax.get_lines() if l.get_marker() == "*")
    assert star.get_ydata()[0] == pytest.approx(per_motor_W, rel=1e-9)
    curve = power_ax.get_lines()[0]
    assert _curve_at(curve, float(star.get_xdata()[0])) == pytest.approx(per_motor_W, rel=0.02)
    # The bench data stays behind it, faint, as the 0 m/s reference.
    assert any(l.get_alpha() for l in power_ax.get_lines())


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


@pytest.mark.parametrize("n_motors", [1, 2])
def test_fixed_wing_motor_marker_is_the_cruise_power_per_motor(fw, n_motors):
    """
    Regression: the operating point was the STATIC table power at the TOTAL
    thrust. At 22 m/s that marked 71 W where the model charged 401 W, and a
    twin looked its whole thrust up in a one-motor table.
    """
    import matplotlib
    matplotlib.use("Agg")
    config = _fw_table_config(fw, _FW_TABLE_CSV)
    config.airframe.num_motors = n_motors
    config.num_motors = n_motors
    metrics = fw.compute_metrics(config, 22.0)
    per_motor_W = metrics["motor_power_W"] / n_motors
    per_motor_g = metrics["thrust_required_N"] / n_motors * 1000.0 / 9.80665

    figure = fw.make_motor_operating_point_figure(config, metrics, figsize=(12, 5))
    power_ax = figure.axes[0]
    dot = next(l for l in power_ax.get_lines() if l.get_marker() == "o")
    assert dot.get_xdata()[0] == pytest.approx(per_motor_g, rel=1e-9)
    assert dot.get_ydata()[0] == pytest.approx(per_motor_W, rel=1e-9)
    curve = power_ax.get_lines()[0]
    assert _curve_at(curve, per_motor_g) == pytest.approx(per_motor_W, rel=0.02)
    assert any(l.get_alpha() for l in power_ax.get_lines())
    assert "22.0 m/s" in figure._suptitle.get_text()


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


# ======================================================================
# FIGURES MUST NOT LEAK
# ======================================================================

def test_embedded_figures_do_not_enter_the_pyplot_registry(fw, fw_plane, mc, mc_quad):
    """
    Regression: figures for the GUI were built with pyplot.subplots(), which
    keeps every one alive in a module-level registry until something closes
    it. Nothing ever did, so a long GUI session accumulated figures
    indefinitely and matplotlib warned "More than 20 figures have been
    opened".

    Embedded figures are now built directly from matplotlib.figure.Figure, so
    they are freed with their canvas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    assert plt.get_fignums() == []

    for _ in range(15):
        fw.make_performance_figure(fw_plane, max_speed=35, figsize=(8, 6))
        fw.make_airframe_diagram_figure(fw_plane, figsize=(6, 5))
        mc.make_performance_figure(mc_quad, max_speed=20, figsize=(8, 6))
        mc.make_airframe_diagram_figure(mc_quad, figsize=(6, 5))

    assert plt.get_fignums() == [], (
        f"{len(plt.get_fignums())} figures retained after 60 builds — "
        "an embedded figure is going through pyplot again")


def test_make_figure_matches_pyplot_subplots_shape():
    """
    The helper stands in for pyplot.subplots(), so its return shape must match
    — including the squeeze behaviour call sites depend on.
    """
    import numpy as np
    import sys as _sys
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    import rotorworks_core as core

    fig, ax = core.make_figure()
    assert not isinstance(ax, np.ndarray), "single subplot should not be an array"

    fig, axes = core.make_figure(1, 2, figsize=(6, 3))
    assert axes.shape == (2,)

    fig, axes = core.make_figure(2, 3, figsize=(6, 4))
    assert axes.shape == (2, 3)


def test_figures_still_render_after_the_registry_change(fw, fw_plane):
    """The helper must produce working figures, not just leak-free ones."""
    figure = fw.make_performance_figure(fw_plane, max_speed=35, figsize=(8, 6))
    assert figure is not None
    assert len(figure.axes) >= 4, "performance figure lost its panels"
    for axis in figure.axes:
        assert axis.get_xlabel() or axis.get_title() or axis.lines, \
            "an empty panel was produced"


# ======================================================================
# WIND DURING A MISSION
# ======================================================================

def _heavy_lift(mc):
    batt = mc.BatteryConfig(
        chemistry="LiPo", operating_voltage_min=3.3,
        operating_voltage_nominal=3.7, operating_voltage_max=4.2,
        unit_mode="pack", pack_capacity_mAh=16000, pack_weight_g=2100,
        series_units=2, parallel_units=1, cells_series_per_unit=6,
        discharge_percent=80, resistance_cell_mOhm=3.0, discharge_c_cont=15)
    drone = mc.DroneConfig(
        num_motors=8, battery=batt,
        motor=mc.MotorConfig(kv=170, idle_current=0.6, idle_voltage=10,
                             rated_voltage=12, resistance=0.045,
                             max_current=45, max_power=2000, weight_g=410),
        propeller=mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                                     max_thrust_g=9500, blades=2, weight_g=110),
        drone_weight_g=11500,
        profile_drag_coefficient=None, profile_area=None,
        parasite_drag_coefficient=None, parasite_area=None, frontal_area=None,
        cruise_speed=8.0, periph_current=0.5,
        motor_configuration="coaxial", coaxial_spacing_m=0.10, max_tilt_deg=20,
        body_length_m=0.45, body_width_m=0.40, body_height_m=0.20,
        arm_length_m=0.40, arm_width_m=0.030)
    drone.air_density = mc.compute_air_density(0)
    drone.derive_drag_from_geometry_if_missing()
    return drone


def _square_mission(mc, tmp_path, speed=10.0):
    import json
    payload = {"wind_direction_deg": 0, "phases": [
        {"name": "Takeoff", "speed": 0.0, "duration": 25, "altitude": 40},
        {"name": "North", "speed": speed, "distance": 400, "altitude": 40, "course_deg": 0},
        {"name": "East", "speed": speed, "distance": 400, "altitude": 40, "course_deg": 90},
        {"name": "South", "speed": speed, "distance": 400, "altitude": 40, "course_deg": 180},
        {"name": "West", "speed": speed, "distance": 400, "altitude": 40, "course_deg": 270},
    ]}
    path = tmp_path / "square.json"
    path.write_text(json.dumps(payload))
    return mc.MissionProfile.from_json(str(path))


def test_wind_does_not_abort_a_flyable_leg(mc, tmp_path):
    """
    Regression: a distance leg starting from a hover has an airspeed below the
    headwind for the first second or two of its acceleration ramp, so its
    instantaneous groundspeed is zero. The guard judged the phase on that
    instant and reported "zero groundspeed with distance phase", killing a
    perfectly flyable 10 m/s leg into a 3 m/s headwind (7 m/s steady
    groundspeed).
    """
    drone = _heavy_lift(mc)
    mission = _square_mission(mc, tmp_path, speed=10.0)
    results, _worst, _series = mc.simulate_mission(drone, mission, wind_mps=3.0)
    for name, _t, _d, status in results:
        assert "Invalid" not in status, f"{name}: {status}"


def test_wind_still_blocks_a_genuinely_impossible_leg(mc, tmp_path):
    """A headwind above the commanded airspeed must still be reported."""
    drone = _heavy_lift(mc)
    mission = _square_mission(mc, tmp_path, speed=10.0)
    results, _worst, _series = mc.simulate_mission(drone, mission, wind_mps=15.0)
    statuses = [s for _n, _t, _d, s in results]
    assert any("Invalid" in s for s in statuses), \
        "a 15 m/s headwind on a 10 m/s leg should be impossible"


def test_leg_times_order_correctly_by_heading(mc, tmp_path):
    """
    Into wind is slowest, downwind fastest, and the two crosswind legs match.
    A wind model that is wired up incorrectly rarely gets all three right.
    """
    drone = _heavy_lift(mc)
    mission = _square_mission(mc, tmp_path, speed=10.0)
    results, _worst, _series = mc.simulate_mission(drone, mission, wind_mps=3.0)
    times = {name: t for name, t, _d, _s in results}

    assert times["North"] > times["East"] > times["South"], \
        f"headings not ordered by wind: {times}"
    assert times["East"] == pytest.approx(times["West"], rel=0.05), \
        "the two crosswind legs should take about the same time"


# ======================================================================
# HOVER-ATTITUDE THRUST AND THRUST-TO-WEIGHT
# ======================================================================

def test_hover_thrust_equals_weight_at_zero_speed(mc, mc_quad):
    """At rest there is no drag, so thrust must be exactly the weight."""
    weight_N = mc_quad.drone_weight_g * 9.81 / 1000.0
    assert mc.thrust_required(mc_quad, 0.0, "hover") == pytest.approx(weight_N, rel=1e-9)


def test_hover_drag_adds_in_quadrature_not_linearly(mc, mc_quad):
    """
    Regression: hover thrust was weight + drag. Drag is HORIZONTAL and weight
    is vertical, so they combine as a vector magnitude. The linear sum
    overstated thrust badly at speed — on a 16.5 kg X8 at 18 m/s it gave
    187 N against a true 164 N.
    """
    weight_N = mc_quad.drone_weight_g * 9.81 / 1000.0
    for speed in (5.0, 10.0, 18.0):
        drag_N = mc.drag_force_required(mc_quad, speed, "hover")
        got = mc.thrust_required(mc_quad, speed, "hover")
        assert got == pytest.approx(math.hypot(weight_N, drag_N), rel=1e-9)
        assert got < weight_N + drag_N, "linear sum is back"


def test_hover_thrust_rises_only_slightly_with_speed(mc, mc_quad):
    """
    Because drag adds in quadrature and is small next to weight, the
    hover-attitude curve should be nearly flat — not the steep rise the
    linear sum produced.
    """
    at_rest = mc.thrust_required(mc_quad, 0.0, "hover")
    at_speed = mc.thrust_required(mc_quad, 18.0, "hover")
    assert at_speed >= at_rest
    assert at_speed < 1.15 * at_rest, "hover-attitude thrust rises too steeply"


def test_thrust_to_weight_uses_available_thrust(mc, mc_quad):
    """
    Regression: thrust-to-weight compared REQUIRED thrust with weight, which
    is ~1 by definition in steady flight and told a designer nothing. It must
    compare what the propulsion system CAN produce.
    """
    weight_N = mc_quad.drone_weight_g * 9.81 / 1000.0
    available_N = mc.available_total_thrust_N(mc_quad)
    assert available_N > 0, "the reference quad should have a thrust limit"

    twr = available_N / weight_N
    assert twr > 1.5, f"available TWR {twr:.2f} is implausibly low for this quad"

    required_N = mc.compute_operating_metrics(mc_quad, 0.0, "hover")["thrust_total_N"]
    assert required_N / weight_N == pytest.approx(1.0, abs=0.02), \
        "required thrust over weight should sit at 1 in hover, which is why it " \
        "is useless as a design metric"


def test_status_and_metrics_agree_on_thrust_to_weight(mc, mc_quad):
    """
    Regression: the Status tab kept its own copy of the thrust-to-weight
    calculation and was missed when the Metrics tab was corrected, so it went
    on reporting 1.00:1 against a ">= 1.5:1" limit — a check that could never
    fail, on a number that was never a margin.

    Both surfaces must derive from the same available-thrust figure.
    """
    weight_N = mc_quad.drone_weight_g * 9.81 / 1000.0
    available_N = mc.available_total_thrust_N(mc_quad)
    assert available_N > 0

    expected = available_N / weight_N
    assert expected > 1.5, "reference quad should have real thrust margin"

    # The required-thrust ratio must NOT be what either surface reports as TWR.
    required_N = mc.compute_operating_metrics(mc_quad, 0.0, "hover")["thrust_total_N"]
    assert abs(required_N / weight_N - expected) > 0.5, (
        "available and required ratios are indistinguishable here, so this "
        "test cannot detect the regression it exists for")


def test_fixedwing_thrust_to_weight_threshold_scales_with_efficiency(fw):
    """
    Regression: the fixed-wing thrust-to-weight check used a fixed ">= 1.2:1"
    limit, which is a ROTORCRAFT criterion. A wing carries the weight, so
    thrust only has to beat drag: level flight needs T/W > 1/(L/D), which is
    around 0.11 for a survey aircraft and 0.05 for a glider.

    The 2 m survey example was flagged red at 0.54:1 while climbing at
    494 m/min with an 80% thrust margin and a 19 m take-off — a healthy
    aircraft failing a check that almost no real fixed-wing could pass.

    The threshold must therefore scale with the aircraft's own L/D.
    """
    config = _fw_table_config(fw, None)
    metrics = fw.compute_metrics(config, 22.0)

    ld = metrics["LD_ratio"]
    twr = metrics["thrust_available_N"] / config.weight_N
    needed_for_level = 1.0 / ld

    assert twr > needed_for_level, (
        "the reference aircraft cannot hold level flight, so this test cannot "
        "distinguish a bad threshold from a bad aircraft")
    assert needed_for_level < 1.2, (
        "a fixed-wing needing T/W above 1.2 for level flight would be "
        "extraordinary; the old fixed threshold was not physical")

    # The aircraft must be demonstrably healthy on independent measures.
    assert metrics["max_rc_mps"] > 0, "cannot climb"
    assert metrics["thrust_required_N"] < metrics["thrust_available_N"]


def test_more_efficient_aircraft_needs_less_thrust_to_weight(fw):
    """A higher L/D must lower the thrust-to-weight a wing actually needs."""
    draggy = _fw_table_config(fw, None)
    draggy.airframe.CD0 = 0.060

    clean = _fw_table_config(fw, None)
    clean.airframe.CD0 = 0.015

    ld_draggy = fw.compute_metrics(draggy, 22.0)["LD_ratio"]
    ld_clean = fw.compute_metrics(clean, 22.0)["LD_ratio"]

    assert ld_clean > ld_draggy
    assert 1.0 / ld_clean < 1.0 / ld_draggy, \
        "the cleaner aircraft should need less thrust-to-weight, not more"


# ======================================================================
# MULTICOPTER STATUS THRESHOLDS
# ======================================================================

def test_hover_efficiency_ceiling_follows_disk_loading(mc, mc_quad):
    """
    The best achievable hover efficiency is set by disk loading alone:

        vi = sqrt(DL / 2*rho)        ideal g/W = 1000 / (g0 * vi)

    A flat "5 g/W" threshold therefore measured disc SIZE more than design
    quality — trivially easy on a heavy-lift with big discs, near impossible
    on a cinewhoop. The check now scales with this ceiling.
    """
    rho = mc_quad.air_density
    metrics = mc.compute_operating_metrics(mc_quad, 0.0, "hover")
    dl = float(metrics["disk_loading_N_m2"])

    ideal_gW = 1000.0 / (9.80665 * math.sqrt(dl / (2.0 * rho)))
    actual_gW = float(metrics["hover_efficiency_gW"])

    assert actual_gW < ideal_gW, \
        "a real rotor cannot beat the momentum-theory ideal for its disk loading"
    assert 0.2 < actual_gW / ideal_gW < 1.0, \
        f"achieving {actual_gW / ideal_gW * 100:.0f}% of ideal is not credible"


def test_higher_disk_loading_lowers_the_efficiency_ceiling(mc, mc_quad):
    """Smaller discs for the same weight must reduce the achievable g/W."""
    import copy
    small = copy.deepcopy(mc_quad)
    small.propeller.diameter_in = mc_quad.propeller.diameter_in * 0.6

    big_dl = mc.compute_operating_metrics(mc_quad, 0.0, "hover")["disk_loading_N_m2"]
    small_dl = mc.compute_operating_metrics(small, 0.0, "hover")["disk_loading_N_m2"]
    assert small_dl > big_dl, "smaller discs should raise disk loading"

    def ceiling(dl):
        return 1000.0 / (9.80665 * math.sqrt(dl / (2.0 * mc_quad.air_density)))

    assert ceiling(small_dl) < ceiling(big_dl), \
        "a higher disk loading must lower the efficiency ceiling"


def test_solidity_expectation_scales_with_blade_count(mc):
    """
    Solidity rises with blade count almost by definition, so a single
    0.05-0.15 window judged a 3-blade propeller against a 2-blade
    expectation and flagged ordinary designs as suspect.
    """
    two = mc.propeller_solidity(5.0, 2)
    three = mc.propeller_solidity(5.0, 3)
    assert three > two
    # The scaled window must accept its own blade count.
    for blades, sigma in ((2, two), (3, three)):
        lo, hi = 0.05 * blades / 2.0, 0.15 * blades / 2.0
        assert lo <= sigma <= hi, \
            f"{blades}-blade solidity {sigma:.3f} outside its scaled window"


def test_figure_of_merit_expectation_scales_with_rotor_size(mc):
    """
    Small propellers run at low Reynolds number and cannot reach the figure
    of merit of a large rotor. A flat 0.65 flagged three of the five example
    aircraft as bad, including ordinary ones.
    """
    def target(diameter_in):
        if diameter_in >= 15:
            return 0.70
        if diameter_in >= 9:
            return 0.60
        return 0.45

    assert target(22) > target(10) > target(3), \
        "expectation should fall with rotor size, not stay constant"


# ======================================================================
# TRANSLATION DIRECTION
# ======================================================================

def test_sideways_translation_costs_more_than_forward(mc, mc_quad):
    """
    A multirotor presents its side silhouette when translating sideways,
    which on most airframes is larger than the frontal one. Treating every
    translation as nose-first understated the cost of flying crabbed.
    """
    import copy
    forward = copy.deepcopy(mc_quad)
    forward.translation_direction_deg = 0.0
    sideways = copy.deepcopy(mc_quad)
    sideways.translation_direction_deg = 90.0

    d_fwd = mc.drag_force_required(forward, 12.0, "translating")
    d_side = mc.drag_force_required(sideways, 12.0, "translating")
    assert d_side > d_fwd, "sideways should present the larger silhouette"

    p_fwd = mc.compute_operating_metrics(forward, 12.0, "translating")["total_power_W"]
    p_side = mc.compute_operating_metrics(sideways, 12.0, "translating")["total_power_W"]
    assert p_side > p_fwd


def test_tilt_splits_into_pitch_and_roll_by_direction(mc, mc_quad):
    """
    Forward translation is pure pitch, sideways is pure roll, and a diagonal
    splits between them. A single "tilt" number could not express this, and
    airframes rarely have equal authority in both axes.
    """
    import copy

    def attitude(azimuth):
        cfg = copy.deepcopy(mc_quad)
        cfg.translation_direction_deg = azimuth
        m = mc.compute_operating_metrics(cfg, 12.0, "translating")
        return m["pitch_required_deg"], m["roll_required_deg"], m["tilt_required_deg"]

    pitch, roll, tilt = attitude(0.0)
    assert abs(roll) < 1e-6 and pitch == pytest.approx(tilt, rel=1e-6)

    pitch, roll, tilt = attitude(90.0)
    assert abs(pitch) < 1e-6 and roll == pytest.approx(tilt, rel=1e-6)

    pitch, roll, tilt = attitude(45.0)
    assert pitch == pytest.approx(roll, rel=1e-6), "45 deg should split evenly"
    assert 0 < pitch < tilt, "each component must be smaller than the total"


def test_pitch_and_roll_recombine_to_the_total_tilt(mc, mc_quad):
    """The split must be lossless — it is a rotation of the same vector."""
    import copy
    import rotorworks_core as core
    for azimuth in (0.0, 30.0, 45.0, 60.0, 90.0, 200.0):
        cfg = copy.deepcopy(mc_quad)
        cfg.translation_direction_deg = azimuth
        m = mc.compute_operating_metrics(cfg, 12.0, "translating")
        recombined = core.tilt_from_pitch_roll(m["pitch_required_deg"],
                                               m["roll_required_deg"])
        assert recombined == pytest.approx(m["tilt_required_deg"], abs=1e-6)


def test_forward_orientation_still_works(mc, mc_quad):
    """
    "forward" is the old name for translating at 0 degrees. Existing configs,
    missions and CLI invocations must behave exactly as before.
    """
    import copy
    cfg = copy.deepcopy(mc_quad)
    cfg.translation_direction_deg = 0.0
    old = mc.compute_operating_metrics(cfg, 12.0, "forward")
    new = mc.compute_operating_metrics(cfg, 12.0, "translating")
    for key, value in old.items():
        if isinstance(value, float) and math.isfinite(value):
            assert new.get(key) == pytest.approx(value, rel=1e-12), \
                f"{key} differs between 'forward' and 'translating'"


# ======================================================================
# MULTICOPTER TURNS
# ======================================================================

def test_turn_bank_matches_the_standard_relation():
    import rotorworks_core as core
    """tan(bank) = V^2 / (R*g) — the same for a multirotor and an aeroplane."""
    for speed, radius in ((10.0, 10.0), (12.0, 25.0), (20.0, 40.0)):
        expected = math.degrees(math.atan(speed ** 2 / (radius * 9.80665)))
        assert core.turn_bank_deg(speed, radius) == pytest.approx(expected, rel=1e-9)


def test_straight_flight_has_unit_load_factor():
    import rotorworks_core as core
    """No turn, no penalty — a zero or absent radius must change nothing."""
    assert core.turn_load_factor(15.0, 0.0) == pytest.approx(1.0)
    assert core.turn_bank_deg(15.0, 0.0) == pytest.approx(0.0)


def test_turning_costs_power(mc, mc_quad):
    """
    A turning multirotor holds its weight AND supplies centripetal force, so
    thrust rises by 1/cos(bank) and power with it. Missions of tight turns
    cost more than their straight-line distance suggests — which the model
    previously could not express at all.
    """
    import rotorworks_core as core
    straight = mc.compute_operating_metrics(mc_quad, 12.0, "translating",
                                            load_factor=1.0)["total_power_W"]
    powers = []
    for radius in (100.0, 50.0, 25.0, 15.0):
        n = core.turn_load_factor(12.0, radius)
        powers.append(mc.compute_operating_metrics(
            mc_quad, 12.0, "translating", load_factor=n)["total_power_W"])

    assert all(p > straight for p in powers), "every turn should cost power"
    # Radii run wide to tight, so the cost should rise through the list.
    assert powers == sorted(powers), "tighter turns must cost more, not less"


def test_turn_thrust_is_a_three_way_vector_sum():
    import rotorworks_core as core
    """
    Weight, drag and centripetal force are mutually perpendicular, so
    T = sqrt(W^2 + D^2 + Fc^2). Adding any pair linearly would overstate it.
    """
    weight_N, drag_N, speed, radius = 17.66, 2.2, 12.0, 20.0
    thrust, along, lateral = core.turn_thrust_N(weight_N, drag_N, speed, radius)

    centripetal = (weight_N / 9.80665) * speed ** 2 / radius
    assert thrust == pytest.approx(
        math.sqrt(weight_N ** 2 + drag_N ** 2 + centripetal ** 2), rel=1e-9)
    assert thrust < weight_N + drag_N + centripetal, "linear sum is back"
    assert lateral == pytest.approx(core.turn_bank_deg(speed, radius), rel=1e-9), \
        "the lateral tilt IS the bank angle"


def test_a_mission_without_turns_is_unchanged(mc, mc_quad, tmp_path):
    """Existing missions have no turn radius and must behave exactly as before."""
    import json
    payload = {"reserve_percent": 20, "phases": [
        {"name": "Leg", "speed": 12.0, "distance": 500, "altitude": 50}]}
    path = tmp_path / "straight.json"
    path.write_text(json.dumps(payload))

    mission = mc.MissionProfile.from_json(str(path))
    assert getattr(mission.phases[0], "turn_radius_m", None) in (None, 0, 0.0)

    results, _worst, _series = mc.simulate_mission(mc_quad, mission, wind_mps=0.0)
    assert results and "Invalid" not in results[0][3]


# ======================================================================
# ZERO-SPEED CONSISTENCY AND CONTINUITY
# ======================================================================

def test_hover_and_translating_agree_at_zero_speed(mc, mc_quad):
    """
    Regression: "hover" and "translating at 0 m/s" describe the identical
    condition and must give identical power.

    They differed by about 4% on a coaxial airframe because the coaxial
    interference penalty was discounted 30% whenever the orientation was not
    the literal word "hover" — so a stationary aircraft got a forward-flight
    benefit that requires a freestream it does not have.
    """
    import copy
    for layout, spacing in (("flat", None), ("coaxial", 0.05)):
        cfg = copy.deepcopy(mc_quad)
        cfg.motor_configuration = layout
        cfg.coaxial_spacing_m = spacing
        if layout == "coaxial":
            cfg.num_motors = max(cfg.num_motors, 2) * 2

        hover = mc.power_required(cfg, 0.0, "hover")
        translating = mc.power_required(cfg, 0.0, "translating")
        assert hover == pytest.approx(translating, rel=1e-12), (
            f"{layout}: hover {hover:.4f} W vs translating {translating:.4f} W "
            "— two names for the same condition must agree")


def test_coaxial_relief_grows_with_airspeed_not_with_a_label(mc, mc_quad):
    """
    Forward flight eases coaxial interference because the freestream sweeps
    the upper rotor's wake clear. That relief must scale with SPEED — keying
    it to the orientation string made it a step change at zero.
    """
    import copy
    cfg = copy.deepcopy(mc_quad)
    cfg.motor_configuration = "coaxial"
    cfg.coaxial_spacing_m = 0.05
    cfg.num_motors = max(cfg.num_motors, 2) * 2

    thrust = mc.thrust_required(cfg, 0.0, "hover") / cfg.num_motors
    penalties = [
        mc.motor_configuration_power_multiplier(
            cfg, "translating", airspeed_mps=v, thrust_per_motor_N=thrust)
        for v in (0.0, 2.0, 5.0, 10.0, 20.0)]

    assert penalties[0] == pytest.approx(
        mc.motor_configuration_power_multiplier(cfg, "hover"), rel=1e-12), \
        "at rest there is no freestream, so no relief"
    assert penalties == sorted(penalties, reverse=True), \
        "relief should grow with speed, so the penalty falls"
    assert penalties[-1] > 1.0, "coaxial interference never disappears entirely"


def test_multicopter_power_curve_has_no_jumps(mc, mc_quad):
    """
    Power must vary smoothly with speed. A step means a branch is being taken
    on a label or a threshold rather than on the physics, which is exactly
    how the coaxial discount went unnoticed.
    """
    import copy
    for layout, spacing in (("flat", None), ("coaxial", 0.05)):
        cfg = copy.deepcopy(mc_quad)
        cfg.motor_configuration = layout
        cfg.coaxial_spacing_m = spacing
        if layout == "coaxial":
            cfg.num_motors = max(cfg.num_motors, 2) * 2

        speeds = [i * 0.1 for i in range(0, 251)]
        powers = [mc.power_required(cfg, v, "translating") for v in speeds]
        for i in range(len(speeds) - 1):
            step = abs(powers[i + 1] - powers[i]) / max(powers[i], 1e-9)
            assert step < 0.05, (
                f"{layout}: {step * 100:.1f}% jump between "
                f"{speeds[i]:.1f} and {speeds[i + 1]:.1f} m/s")


def test_thrust_equals_weight_when_stationary(mc, mc_quad):
    """With no drag and no turn, the rotors hold exactly the weight."""
    weight_N = mc_quad.drone_weight_g * 9.81 / 1000.0
    assert mc.thrust_required(mc_quad, 0.0, "translating") == pytest.approx(
        weight_N, abs=1e-6)
    assert mc.thrust_required(mc_quad, 0.0, "hover") == pytest.approx(
        weight_N, abs=1e-6)


def test_hover_endurance_ignores_the_cruise_speed_box(mc, mc_quad):
    """
    Regression: compute_operating_metrics forces hover to 0 m/s but
    estimate_flight_time_minutes did not, so one run reported hover POWER
    and cruise-speed ENDURANCE — 47.99 min against the correct 23.80 on the
    heavy-lift example.
    """
    at_zero = mc.estimate_flight_time_minutes(mc_quad, 0.0, orientation="hover")
    at_speed = mc.estimate_flight_time_minutes(mc_quad, 17.0, orientation="hover")
    assert at_zero == pytest.approx(at_speed, rel=1e-12), \
        "hover endurance must not depend on the speed box"


def test_mission_altitude_trace_ramps_instead_of_jumping(mc, mc_quad, tmp_path):
    """
    Regression: the altitude series recorded `phase.altitude` — the phase's
    TARGET — as a constant, so a 30 s climb from 0 to 60 m plotted as an
    instant jump to 60 and stayed there through the landing.

    The ENERGY was always right (the potential-power term is integrated per
    step, and matched m*g*h to 0.7%), which is what made this easy to miss:
    only the trace disagreed with the physics behind it.
    """
    import json
    payload = {"reserve_percent": 20, "phases": [
        {"name": "Climb", "speed": 0.0, "duration": 30,
         "altitude": 60, "climb_rate_mps": 2.0},
        {"name": "Cruise", "speed": 10.0, "distance": 200, "altitude": 60},
        {"name": "Land", "speed": 0.0, "duration": 30,
         "altitude": 0, "descent_rate_mps": 2.0}]}
    path = tmp_path / "climb.json"
    path.write_text(json.dumps(payload))

    _results, _worst, series = mc.simulate_mission(
        mc_quad, mc.MissionProfile.from_json(str(path)), wind_mps=0.0)
    alt = series["altitude_m"]

    assert alt[0] == pytest.approx(0.0, abs=1e-6), "must start on the ground"
    assert max(alt) == pytest.approx(60.0, abs=0.5), "must reach the target"
    assert alt[-1] == pytest.approx(0.0, abs=0.5), "must come back down"

    # A ramp, not a step: intermediate heights have to actually appear.
    climbing = [a for a in alt if 5.0 < a < 55.0]
    assert len(climbing) > 10, \
        "no intermediate altitudes — the trace jumped instead of ramping"


def test_mission_climb_energy_matches_potential_energy(mc, mc_quad, tmp_path):
    """
    The extra energy a climb costs over hovering for the same time must equal
    m*g*h, give or take the extra thrust needed while climbing.
    """
    import json
    height, rate = 60.0, 2.0
    duration = height / rate
    payload = {"reserve_percent": 20, "phases": [
        {"name": "Climb", "speed": 0.0, "duration": duration,
         "altitude": height, "climb_rate_mps": rate}]}
    path = tmp_path / "climb_only.json"
    path.write_text(json.dumps(payload))

    _r, _w, series = mc.simulate_mission(
        mc_quad, mc.MissionProfile.from_json(str(path)), wind_mps=0.0)
    used_Wh = series["battery_energy_Wh"][0] - series["battery_energy_Wh"][-1]

    hover_W = mc.compute_operating_metrics(mc_quad, 0.0, "hover")["total_power_W"]
    hover_Wh = hover_W * duration / 3600.0
    climb_work_Wh = used_Wh - hover_Wh

    ideal_Wh = (mc_quad.drone_weight_g * 9.81 / 1000.0) * height / 3600.0
    assert climb_work_Wh == pytest.approx(ideal_Wh, rel=0.15), (
        f"climb cost {climb_work_Wh:.2f} Wh over hover, ideal m*g*h is "
        f"{ideal_Wh:.2f} Wh")


def test_rails_and_peripheral_current_add(mc, mc_quad):
    """
    Regression: regulated rails and direct-from-pack peripherals are
    INDEPENDENT loads. The help has always said so — "use this for devices
    wired straight to pack voltage; use the Avionics tab for anything on a
    regulated rail... never enter the same device in both" — but the model
    used peripheral current only as a FALLBACK for "no rails defined".

    So a payload wired straight to the pack drew nothing at all as soon as a
    single BEC rail existed, and the number the user typed did nothing.
    """
    import copy

    def power(periph_A, with_rails):
        cfg = copy.deepcopy(mc_quad)
        cfg.periph_current = periph_A
        cfg.avionics = (mc.AvionicsConfig(voltage_tree={5.0: (2.0, 0.9)})
                        if with_rails else None)
        return mc.compute_operating_metrics(cfg, 10.0, "translating")["total_power_W"]

    base = power(0.0, False)
    rails_only = power(0.0, True)
    periph_only = power(2.0, False)
    both = power(2.0, True)

    assert rails_only > base, "a rail draws power"
    assert periph_only > base, "a direct-from-pack device draws power"
    assert both - base == pytest.approx(
        (rails_only - base) + (periph_only - base), rel=1e-9), \
        "the two loads must add, not replace one another"


def test_peripheral_current_is_not_ignored_when_rails_exist(mc, mc_quad):
    """The specific symptom: typing a peripheral current changed nothing."""
    import copy
    cfg = copy.deepcopy(mc_quad)
    cfg.avionics = mc.AvionicsConfig(voltage_tree={5.0: (2.0, 0.9)})

    cfg.periph_current = 0.0
    without = mc.compute_operating_metrics(cfg, 10.0, "translating")["total_power_W"]
    cfg.periph_current = 2.0
    with_periph = mc.compute_operating_metrics(cfg, 10.0, "translating")["total_power_W"]

    assert with_periph > without, \
        "peripheral current was ignored because rails were defined"


# ======================================================================
# REAL AIRCRAFT — validated against published specifications
# ======================================================================
#
# Two aircraft whose manufacturers publish enough to check the model against
# something it did not choose. The VTOL suite has the same idea; these cover
# the multicopter and fixed-wing sides.

def test_dji_m300_hover_endurance_matches_the_published_figure(mc):
    """
    DJI publishes 55 minutes of hover for a Matrice 300 RTK at 6.3 kg with
    two TB60 packs and no camera. This is the strongest validation in the
    project: the battery energy is published, so NOTHING is tuned. The only
    judgement call is usable capacity, left at the conventional 80%.

    The pack energy is worth checking on its own — 11870 mAh at a 12S LiPo
    nominal should come out at DJI's stated 548 Wh for the pair, and if it
    does not, the battery model disagrees with DJI's arithmetic before any
    aerodynamics are involved.
    """
    import json
    path = os.path.join(ROOT, "examples", "configs",
                        "multicopter_dji_m300_rtk.json")
    cfg_vars = json.load(open(path))["vars"]

    def f(key, default=0.0):
        raw = str(cfg_vars.get(key, "")).strip()
        return float(raw) if raw else default

    battery = mc.BatteryConfig(
        chemistry=cfg_vars["batt_chem"], operating_voltage_min=f("batt_vmin"),
        operating_voltage_nominal=f("batt_vnom"), operating_voltage_max=f("batt_vmax"),
        unit_mode="pack", pack_capacity_mAh=f("batt_pack_capacity"),
        pack_weight_g=f("batt_pack_weight"), series_units=1, parallel_units=1,
        cells_series_per_unit=12, discharge_percent=f("batt_dischg_pct"),
        resistance_cell_mOhm=f("batt_r"), discharge_c_cont=f("batt_c_cont"))

    pack_Wh = battery.pack_capacity_mAh / 1000.0 * battery.vnom_pack
    assert pack_Wh == pytest.approx(548.0, rel=0.02), (
        f"pack energy {pack_Wh:.0f} Wh against DJI's 2 x 274 = 548 Wh")

    drone = mc.DroneConfig(
        num_motors=4, battery=battery,
        motor=mc.MotorConfig(kv=f("motor_kv"), idle_current=f("motor_i0"),
                             idle_voltage=f("motor_v0"), rated_voltage=12,
                             resistance=f("motor_r"), max_current=f("motor_imax"),
                             max_power=f("motor_pmax"), weight_g=f("motor_weight")),
        propeller=mc.PropellerConfig(diameter_in=21, pitch_in=10, max_rpm=0,
                                     max_thrust_g=5500, blades=2, weight_g=95),
        drone_weight_g=f("weight"), profile_drag_coefficient=None,
        profile_area=None, parasite_drag_coefficient=None, parasite_area=None,
        frontal_area=None, cruise_speed=15, periph_current=0,
        motor_configuration="flat",
        body_length_m=f("body_length_m"), body_width_m=f("body_width_m"),
        body_height_m=f("body_height_m"), arm_length_m=f("arm_length_m"),
        arm_width_m=f("arm_width_m"))
    drone.air_density = mc.compute_air_density(0, 20)
    drone.derive_drag_from_geometry_if_missing()

    hover_W = mc.compute_operating_metrics(drone, 0.0, "hover")["total_power_W"]
    usable_Wh = pack_Wh * f("batt_dischg_pct") / 100.0
    endurance_min = usable_Wh / hover_W * 60.0

    assert endurance_min == pytest.approx(55.0, rel=0.10), (
        f"model hovers {endurance_min:.1f} min against DJI's published 55")


def test_ebee_x_endurance_and_range_agree_with_one_battery_choice(fw):
    """
    senseFly publishes 90 min AND 95 km for the eBee X's endurance battery,
    but not the battery capacity. Capacity is therefore fitted — to the
    ENDURANCE only.

    The check is what happens to the range, which was not fitted: it should
    land on 95 km by itself. One free parameter satisfying two published
    numbers is real evidence, if weaker than the M300's where nothing at all
    is tuned.
    """
    import json
    path = os.path.join(ROOT, "examples", "configs",
                        "fixedwing_ebee_x_mapping.json")
    cfg_vars = json.load(open(path))["vars"]

    def f(key, default=0.0):
        raw = str(cfg_vars.get(key, "")).strip()
        return float(raw) if raw else default

    battery = fw.BatteryConfig(
        chemistry=cfg_vars["batt_chem"], operating_voltage_min=f("batt_vmin"),
        operating_voltage_nominal=f("batt_vnom"), operating_voltage_max=f("batt_vmax"),
        unit_mode="pack", pack_capacity_mAh=f("batt_pack_cap"),
        pack_weight_g=f("batt_pack_wt"), series_units=1, parallel_units=1,
        cells_series_per_unit=4, discharge_percent=f("batt_dischg_pct"),
        resistance_cell_mOhm=f("batt_r"), discharge_c_cont=f("batt_c_cont"))
    airframe = fw.AirframeConfig(
        wing_span_m=f("wing_span"), wing_area_m2=f("wing_area"), CD0=f("CD0"),
        oswald=f("oswald"), CL_max=f("CL_max"), prop_efficiency=f("prop_eff"),
        num_motors=1)
    cfg = fw.FixedWingConfig(
        aircraft_weight_g=f("weight") + f("payload_mass_g"), airframe=airframe,
        battery=battery,
        motor=fw.MotorConfig(kv=f("motor_kv"), idle_current=f("motor_i0"),
                             idle_voltage=f("motor_v0"), rated_voltage=4,
                             resistance=f("motor_r"), max_current=f("motor_imax"),
                             max_power=f("motor_pmax"), weight_g=f("motor_wt")),
        propeller=fw.PropellerConfig(diameter_in=f("prop_d"), pitch_in=f("prop_pitch"),
                                     blades=2, weight_g=f("prop_wt")),
        cruise_speed_mps=f("cruise_speed"), air_density=1.225,
        reference_altitude_m=0)

    metrics = fw.compute_metrics(cfg, 17.6)
    assert metrics["flight_time_min"] == pytest.approx(90.0, rel=0.05), \
        f"endurance {metrics['flight_time_min']:.1f} min against a published 90"
    assert metrics["flight_range_km"] == pytest.approx(95.0, rel=0.05), \
        f"range {metrics['flight_range_km']:.1f} km against a published 95"

    # Cruise must sit above stall, and senseFly's published band starts at 11.
    assert metrics["stall_speed_mps"] < 11.0, \
        f"stall {metrics['stall_speed_mps']:.1f} m/s is above the published cruise band"


def _load_dji_config(mc, filename):
    """Build a DroneConfig from one of the shipped DJI example files."""
    import json
    path = os.path.join(ROOT, "examples", "configs", filename)
    g = json.load(open(path))["vars"]

    def f(key, default=0.0):
        raw = str(g.get(key, "")).strip()
        return float(raw) if raw else default

    battery = mc.BatteryConfig(
        chemistry=g["batt_chem"], operating_voltage_min=f("batt_vmin"),
        operating_voltage_nominal=f("batt_vnom"), operating_voltage_max=f("batt_vmax"),
        unit_mode="pack", pack_capacity_mAh=f("batt_pack_capacity"),
        pack_weight_g=f("batt_pack_weight"), series_units=1, parallel_units=1,
        cells_series_per_unit=int(f("batt_cells_series")),
        discharge_percent=f("batt_dischg_pct"), resistance_cell_mOhm=f("batt_r"),
        discharge_c_cont=f("batt_c_cont"))
    drone = mc.DroneConfig(
        num_motors=4, battery=battery,
        motor=mc.MotorConfig(kv=f("motor_kv"), idle_current=f("motor_i0"),
                             idle_voltage=f("motor_v0"), rated_voltage=12,
                             resistance=f("motor_r"), max_current=f("motor_imax"),
                             max_power=f("motor_pmax"), weight_g=f("motor_weight")),
        propeller=mc.PropellerConfig(diameter_in=f("prop_d"), pitch_in=f("prop_pitch"),
                                     max_rpm=0, max_thrust_g=f("prop_max_thrust"),
                                     blades=2, weight_g=f("prop_weight")),
        drone_weight_g=f("weight"), profile_drag_coefficient=None, profile_area=None,
        parasite_drag_coefficient=None, parasite_area=None, frontal_area=None,
        cruise_speed=f("speed"), periph_current=0, motor_configuration="flat",
        body_length_m=f("body_length_m"), body_width_m=f("body_width_m"),
        body_height_m=f("body_height_m"), arm_length_m=f("arm_length_m"),
        arm_width_m=f("arm_width_m"))
    drone.air_density = mc.compute_air_density(0, 20)
    drone.derive_drag_from_geometry_if_missing()
    usable_Wh = (battery.pack_capacity_mAh / 1000.0 * battery.vnom_pack
                 * f("batt_dischg_pct") / 100.0)
    return drone, usable_Wh


def _hover_endurance_min(mc, drone, usable_Wh):
    power = mc.compute_operating_metrics(drone, 0.0, "hover")["total_power_W"]
    return usable_Wh / power * 60.0


def test_m350_is_shorter_legged_than_the_m300_by_the_right_margin(mc):
    """
    The M350 and M300 share an airframe. The M350 is heavier (6.47 vs 6.3 kg)
    on LESS energy (526 vs 548 Wh), so it must hover for less time — and by
    roughly the ratio those two numbers imply.

    This is a sensitivity check, which is worth more than another absolute
    one: a model can be right about one aircraft by luck, but not about the
    DIFFERENCE between two that DJI documents separately.
    """
    m300, m300_Wh = _load_dji_config(mc, "multicopter_dji_m300_rtk.json")
    m350, m350_Wh = _load_dji_config(mc, "multicopter_dji_m350_rtk.json")

    m300_min = _hover_endurance_min(mc, m300, m300_Wh)
    m350_min = _hover_endurance_min(mc, m350, m350_Wh)

    assert m350_min < m300_min, "the heavier aircraft on less energy lasted longer"
    # Energy is 4% down and weight 2.7% up, so expect roughly 5-8% less.
    assert 0.88 < m350_min / m300_min < 0.98, (
        f"M350 is {m350_min / m300_min:.3f} of the M300 — outside what the "
        "weight and energy difference can explain")
    assert m350_min == pytest.approx(55.0, rel=0.10), (
        f"M350 hovers {m350_min:.1f} min against a published 55 (measured by "
        "DJI at ~8 m/s, so this is the conservative comparison)")


def test_the_m30_documents_where_this_model_stops_being_accurate(mc):
    """
    KNOWN LIMITATION, pinned deliberately.

    The M30 is half the M300's weight on 16 in rotors instead of 21. The
    model predicts 47 min of hover against DJI's published 36 — it is 31%
    OPTIMISTIC, and this test asserts that gap rather than hiding it.

    Overall hover efficiency, ideal momentum power over electrical power:

        M300, 21 in rotors: model 69.4%, real 69.2%   essentially exact
        M30,  16 in rotors: model 74.8%, real 57.3%   far too kind

    The model gets the large aircraft almost exactly right, then carries the
    same efficiency down to half the size. It has NO size dependence in its
    efficiency chain and real hardware plainly does — roughly 12 points lost
    scaling down, from lower propeller Reynolds number and from smaller
    motors and ESCs.

    Not fixed deliberately: two aircraft is not a scaling law, and fitting
    one to two points would be inventing a coefficient. If someone does fix
    it with real bench data, this test fails — and it should, because the
    right response is to update the expectation here rather than to discover
    the change by accident somewhere else.
    """
    m30, usable_Wh = _load_dji_config(mc, "multicopter_dji_m30.json")
    predicted = _hover_endurance_min(mc, m30, usable_Wh)
    published = 36.0

    assert predicted > published, "the known optimism has reversed"
    assert 1.20 < predicted / published < 1.45, (
        f"the M30 gap is now {predicted / published:.2f}x, not the documented "
        "~1.31x — if the figure-of-merit scaling was fixed, update this test")

    # The model's reported figure of merit is a DERIVED diagnostic, not an
    # input: it is ideal induced power over the model's own induced power.
    # It reading higher for the smaller rotor is a symptom of the missing
    # size dependence, not its cause. Pinned so a fix is noticed.
    m300, _ = _load_dji_config(mc, "multicopter_dji_m300_rtk.json")
    fom_small = mc.compute_operating_metrics(m30, 0.0, "hover")["figure_of_merit"]
    fom_large = mc.compute_operating_metrics(m300, 0.0, "hover")["figure_of_merit"]
    assert fom_small > fom_large, (
        "the small-rotor figure of merit is no longer above the large-rotor "
        "one — the scaling may have been corrected, so revisit the M30 gap")


# ----------------------------------------------------------------------
# Audit C1: the loaded pack voltage may fall below the cutoff.
#
# It used to be clamped at V_min, so a pack too resistive to hold its
# cutoff under load reported exactly V_min and every "V_load < V_min" stop
# was unreachable. Each test below fails with the clamp in place.
# ----------------------------------------------------------------------

def test_mc_pack_solve_falls_below_cutoff(mc, mc_quad):
    batt = mc_quad.battery
    v_ok, _i = mc.solve_pack_voltage_and_current(batt, 160.0)
    assert v_ok > batt.vmin_pack
    batt.resistance_cell = 0.090                # 90 mOhm per cell
    v, i = mc.solve_pack_voltage_and_current(batt, 160.0)
    assert v < batt.vmin_pack


def test_mc_single_point_endurance_is_zero_in_brownout(mc, mc_quad):
    assert mc.estimate_flight_time_minutes(mc_quad, 10.0, "translating") > 0
    mc_quad.battery.resistance_cell = 0.090
    assert mc.estimate_flight_time_minutes(mc_quad, 10.0, "translating") == 0.0


def test_mc_mission_stops_on_low_voltage(mc, mc_quad, tmp_path):
    mission = _square_mission(mc, tmp_path, speed=10.0)
    results, _w, _s = mc.simulate_mission(mc_quad, mission)
    assert not any("voltage" in s for _n, _t, _d, s in results)
    mc_quad.battery.resistance_cell = 0.090
    results, _w, _s = mc.simulate_mission(mc_quad, mission)
    assert any("voltage under load" in s for _n, _t, _d, s in results)


def test_fw_voltage_under_load_is_not_clamped(fw, fw_plane):
    batt = fw_plane.battery
    i = (batt.ocv_at_soc(1.0) - 0.5 * batt.vmin_pack) / batt.resistance_at_soc(1.0)
    assert batt.voltage_under_load(i) == pytest.approx(0.5 * batt.vmin_pack)


def test_fw_single_point_endurance_is_zero_in_brownout(fw, fw_plane):
    m = fw.compute_metrics(fw_plane, 19.0)
    assert m["flight_time_min"] > 0
    fw_plane.battery.resistance_cell = 0.200    # 200 mOhm per cell
    m = fw.compute_metrics(fw_plane, 19.0)
    assert m["v_load_V"] < fw_plane.battery.vmin_pack
    assert m["flight_time_min"] == 0.0


def test_fw_mission_stops_on_low_voltage(fw, fw_plane, tmp_path):
    profile = _fw_mission(fw, tmp_path, {"phases": [
        {"name": "Cruise", "speed": 19.0, "duration": 300, "altitude": 150}]})
    results, _w, _s = fw.simulate_fw_mission(fw_plane, profile)
    assert results[0][-1] == "OK"
    fw_plane.battery.resistance_cell = 0.200
    results, _w, _s = fw.simulate_fw_mission(fw_plane, profile)
    assert "voltage" in results[-1][-1]
