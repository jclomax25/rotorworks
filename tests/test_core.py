"""
Tests for the shared core module.

These test `rotorworks_core` directly, and — just as importantly — assert that
both simulators are genuinely delegating to it rather than carrying their own
copies again. A future contributor who pastes a local implementation back into
one simulator should see these fail.
"""

from __future__ import annotations

import math

import pytest


@pytest.fixture(scope="session")
def core():
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import rotorworks_core
    return rotorworks_core


# ======================================================================
# Both simulators must be USING the core, not shadowing it
# ======================================================================

@pytest.mark.parametrize("attr", [
    "parse_float_list", "parse_soc_breakpoints", "wind_components_mps",
    "groundspeed_along_track_mps", "thermal_step",
])
def test_simulators_share_the_same_function_objects(mc, fw, core, attr):
    """
    Not just "same behaviour" — literally the same object. If someone pastes a
    local copy back into one simulator, this catches it immediately.
    """
    assert getattr(mc, attr) is getattr(fw, attr), f"{attr} has diverged again"
    expected = getattr(core, attr)
    assert getattr(mc, attr) is expected, f"{attr} is not the core implementation"


def test_tooltip_class_is_shared(mc, fw, core):
    assert mc._Tooltip is fw._Tooltip is core.Tooltip


def test_soc_presets_are_shared(mc, fw, core):
    assert mc.SOC_PRESETS is fw.SOC_PRESETS is core.SOC_PRESETS


def test_atmosphere_delegates_to_core(mc, fw, core):
    """
    Both simulators keep their historical function names but must produce the
    core's values. They previously had separate implementations that disagreed
    about pressure overrides.
    """
    for alt, temp, press in [(0, None, None), (500, 35, None),
                             (500, None, 95000), (2000, 15, 90000)]:
        want = core.air_density(alt, temp, press)
        assert mc.compute_air_density(alt, temp, press) == pytest.approx(want, abs=1e-12)
        assert fw.isa_density(alt, temp, press) == pytest.approx(want, abs=1e-12)


def test_core_does_not_import_tkinter_at_module_scope(core):
    """
    Headless CLI runs must work on machines with no tk. The tooltip imports it
    lazily inside _show; nothing else may import it at module level.
    """
    import inspect
    source = inspect.getsource(core)
    module_level = [
        line for line in source.split("\n")
        if line.startswith(("import ", "from ")) and "tkinter" in line
    ]
    assert not module_level, f"core imports tkinter at module scope: {module_level}"


# ======================================================================
# Atmosphere
# ======================================================================

def test_sea_level_matches_isa(core):
    assert core.air_density(0) == pytest.approx(1.225, abs=0.002)


def test_temperature_and_pressure_are_independent(core):
    """
    Regression: an earlier fixed-wing implementation only honoured a pressure
    override when a temperature was ALSO supplied, so pressure alone was
    silently ignored.
    """
    baseline = core.air_density(500)
    temp_only = core.air_density(500, temperature_C=35)
    press_only = core.air_density(500, pressure_Pa=95000)
    both = core.air_density(500, temperature_C=35, pressure_Pa=95000)

    assert temp_only != pytest.approx(baseline, abs=1e-6)
    assert press_only != pytest.approx(baseline, abs=1e-6)
    assert both != pytest.approx(temp_only, abs=1e-6)
    assert both != pytest.approx(press_only, abs=1e-6)


def test_density_obeys_the_gas_law(core):
    """rho = P / (R*T) exactly, when both are pinned."""
    rho = core.air_density(0, temperature_C=15, pressure_Pa=101325)
    expected = 101325 / (core.R_AIR * (15 + 273.15))
    assert rho == pytest.approx(expected, rel=1e-12)


# ======================================================================
# Interpolation
# ======================================================================

def test_interpolation_clamps_instead_of_extrapolating(core):
    xp, fp = [0.0, 0.5, 1.0], [10.0, 20.0, 30.0]
    assert core.interp_linear_clamped(-5.0, xp, fp) == 10.0
    assert core.interp_linear_clamped(99.0, xp, fp) == 30.0
    assert core.interp_linear_clamped(0.25, xp, fp) == pytest.approx(15.0)


def test_interpolation_rejects_mismatched_vectors(core):
    with pytest.raises(ValueError):
        core.interp_linear_clamped(0.5, [0, 1], [1, 2, 3])


def test_eval_poly_matches_manual_expansion(core):
    # 2x^2 + 3x + 4 at x = 5  ->  50 + 15 + 4
    assert core.eval_poly([2, 3, 4], 5) == pytest.approx(69.0)
    assert core.eval_poly(None, 5) is None
    assert core.eval_poly([], 5) is None


# ======================================================================
# Parsing
# ======================================================================

def test_parse_float_list_handles_blanks_and_whitespace(core):
    assert core.parse_float_list("1, 2 ,3") == [1.0, 2.0, 3.0]
    assert core.parse_float_list("") is None
    assert core.parse_float_list(None) is None
    assert core.parse_float_list([1, 2]) == [1.0, 2.0]


def test_soc_breakpoints_accept_fractions_or_percentages(core):
    assert core.parse_soc_breakpoints("0,0.5,1.0") == pytest.approx([0.0, 0.5, 1.0])
    assert core.parse_soc_breakpoints("0,50,100") == pytest.approx([0.0, 0.5, 1.0])


def test_soc_breakpoints_are_clamped(core):
    assert core.parse_soc_breakpoints("-10,150") == pytest.approx([0.0, 1.0])


# ======================================================================
# State of charge
# ======================================================================

@pytest.mark.parametrize("label,expected", [
    ("LiPo", "lipo"), ("li-po", "lipo"), ("Li-Ion", "liion"),
    ("NMC", "liion"), ("LFP", "lifepo4"), ("LiFePO4", "lifepo4"),
    ("unobtainium", None), (None, None),
])
def test_chemistry_labels_map_to_presets(core, label, expected):
    assert core.battery_preset_key(label) == expected


def test_soc_curves_are_sorted_and_deduplicated(core):
    s, v, r = core.normalize_soc_curves(
        [1.0, 0.0, 0.5, 0.5], [4.2, 3.0, 3.8, 3.9], [1.5, 2.6, 1.0, 1.1])
    assert s == sorted(s)
    assert len(s) == 3                       # the duplicate 0.5 collapsed
    assert v[s.index(0.5)] == 3.9            # last value wins


def test_soc_curve_requires_two_points(core):
    with pytest.raises(ValueError):
        core.normalize_soc_curves([0.5], [3.8], [1.0])


def test_preset_curves_are_monotonic_in_voltage(core):
    """Open-circuit voltage must rise with state of charge for every preset."""
    for name, table in core.SOC_PRESETS.items():
        ocv = table["ocv_cell_bp"]
        assert all(a <= b for a, b in zip(ocv, ocv[1:])), f"{name} OCV not monotonic"


def test_preset_resistance_rises_at_both_extremes(core):
    """Cells get more resistive when nearly empty and when fully charged."""
    for name, table in core.SOC_PRESETS.items():
        r = table["r_scale_bp"]
        assert r[0] > min(r), f"{name}: resistance should be high at 0% SoC"
        assert r[-1] > min(r), f"{name}: resistance should be high at 100% SoC"


class _FakePack:
    """Minimal duck-typed stand-in, to test the SoC helpers in isolation."""
    def __init__(self, chemistry="LiPo"):
        self.chemistry = chemistry
        self.series_cells = 4
        self.vmax_pack = 16.8
        self.vmin_pack = 13.2
        self.pack_resistance = 0.016
        self.usable_Wh = 59.2
        self.soc_nonlinear_enabled = False
        self.soc_model_source = "linear-fallback"
        self.soc_bp = []
        self.ocv_cell_bp = []
        self.r_scale_bp = []


def test_configure_picks_preset_from_chemistry(core):
    pack = _FakePack("LiPo")
    core.configure_battery_soc_model(pack, "auto", None, None, None, None)
    assert pack.soc_model_source == "preset:lipo"
    assert pack.soc_nonlinear_enabled


def test_configure_linear_disables_the_curve(core):
    pack = _FakePack()
    core.configure_battery_soc_model(pack, "linear", None, None, None, None)
    assert not pack.soc_nonlinear_enabled
    assert pack.soc_model_source == "linear-selected"


def test_configure_falls_back_when_chemistry_is_unknown(core):
    pack = _FakePack("unobtainium")
    core.configure_battery_soc_model(pack, "auto", None, None, None, None)
    assert pack.soc_model_source == "linear-fallback"
    assert not pack.soc_nonlinear_enabled


def test_configure_prefers_explicit_arrays_over_preset(core):
    pack = _FakePack("LiPo")
    core.configure_battery_soc_model(
        pack, "auto", None, [0, 0.5, 1.0], [3.2, 3.8, 4.2], [2.0, 1.0, 1.2])
    assert pack.soc_model_source == "custom-arrays"


def test_configure_survives_a_bad_csv_path(core):
    """A missing curve file must fall back, not raise."""
    pack = _FakePack("LiPo")
    core.configure_battery_soc_model(pack, "auto", "/no/such/file.csv", None, None, None)
    assert pack.soc_model_source == "preset:lipo"


def test_voltage_under_load_defaults_to_full_charge(core):
    pack = _FakePack()
    core.configure_battery_soc_model(pack, "auto", None, None, None, None)
    assert core.pack_voltage_under_load(pack, 20.0) == pytest.approx(
        core.pack_voltage_under_load(pack, 20.0, soc=1.0))


def test_voltage_can_fall_below_the_cutoff(core):
    """Audit C1: the loaded voltage used to be clamped at V_min, so no
    "V_load < V_min" check could ever fire. It may now sag below the cutoff,
    and is floored only at 0 V."""
    pack = _FakePack()
    core.configure_battery_soc_model(pack, "auto", None, None, None, None)
    ocv = core.pack_ocv_from_soc(pack, 1.0)
    r = core.pack_resistance_from_soc(pack, 1.0)
    i = (ocv - 0.5 * pack.vmin_pack) / r          # sags to half the cutoff
    assert core.pack_voltage_under_load(pack, i) == pytest.approx(0.5 * pack.vmin_pack)
    assert core.pack_voltage_under_load(pack, i) < pack.vmin_pack
    assert core.pack_voltage_under_load(pack, 10_000.0) == 0.0


def test_constant_power_solve_is_exact(core):
    pack = _FakePack()
    core.configure_battery_soc_model(pack, "auto", None, None, None, None)
    v, i, ok = core.solve_pack_for_power(pack, 300.0, soc=0.6)
    assert ok
    assert v * i == pytest.approx(300.0)
    assert v == pytest.approx(core.pack_voltage_under_load(pack, i, soc=0.6))
    # The physical (high-voltage) root, not the collapsed one.
    assert v > 0.5 * core.pack_ocv_from_soc(pack, 0.6)


def test_constant_power_solve_reports_an_overload(core):
    """Beyond OCV^2 / 4R the pack cannot deliver the power. The solve
    returns the maximum-power point, well below cutoff, instead of
    diverging."""
    pack = _FakePack()
    core.configure_battery_soc_model(pack, "auto", None, None, None, None)
    ocv = core.pack_ocv_from_soc(pack, 1.0)
    r = core.pack_resistance_from_soc(pack, 1.0)
    v, i, ok = core.solve_pack_for_power(pack, 1.01 * ocv * ocv / (4.0 * r))
    assert not ok
    assert v == pytest.approx(0.5 * ocv)
    assert i == pytest.approx(0.5 * ocv / r)
    assert v < pack.vmin_pack
    assert core.solve_pack_for_power(pack, 0.0) == (pytest.approx(ocv), 0.0, True)


def test_cli_explicit_dests_sees_typed_flags_only(core):
    """Audit B1: a typed flag counts even when it equals the default."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve_percent", type=float, default=20.0)
    parser.add_argument("--wind_direction_deg", type=float, default=0.0)
    parser.add_argument("--gui", action="store_true")
    assert core.cli_explicit_dests(parser, ["--reserve_percent", "20"]) == {"reserve_percent"}
    assert core.cli_explicit_dests(parser, []) == set()
    assert core.cli_explicit_dests(parser, ["--gui"]) == {"gui"}
    # The parser's own defaults are untouched afterwards.
    assert parser.parse_args([]).reserve_percent == 20.0


def test_usable_soc_maps_onto_the_cell_curve(core):
    """Audit C2: missions track usable-energy SoC, the curves span the full
    cell capacity. With 80% usable, usable 0.0 is the cell's 0.2."""
    pack = _FakePack()
    core.configure_battery_soc_model(pack, "auto", None, None, None, None)
    pack.usable_fraction = 0.8
    assert core.cell_soc_from_usable(pack, 1.0) == pytest.approx(1.0)
    assert core.cell_soc_from_usable(pack, 0.5) == pytest.approx(0.6)
    assert core.cell_soc_from_usable(pack, 0.0) == pytest.approx(0.2)
    assert core.cell_soc_from_usable(pack, None) == pytest.approx(1.0)
    assert core.pack_voltage_under_load(pack, 0.0, soc=0.0) == pytest.approx(
        core.pack_ocv_from_soc(pack, 0.2))
    v, i, _ok = core.solve_pack_for_power(pack, 150.0, soc=0.0)
    assert v == pytest.approx(core.pack_voltage_under_load(pack, i, soc=0.0))
    # Without a usable fraction the two scales coincide.
    del pack.usable_fraction
    assert core.cell_soc_from_usable(pack, 0.3) == pytest.approx(0.3)


def test_soc_after_energy_draw_stays_in_range(core):
    pack = _FakePack()
    assert core.soc_after_energy_draw(pack, 1.0, 0.0) == pytest.approx(1.0)
    assert core.soc_after_energy_draw(pack, 1.0, 29.6) == pytest.approx(0.5, abs=0.01)
    assert core.soc_after_energy_draw(pack, 1.0, 1e6) == 0.0


@pytest.mark.parametrize("source,expected", [
    ("preset:lipo", "preset-lipo"),
    ("csv:/tmp/curve.csv", "csv"),
    ("custom-arrays", "custom-arrays"),
    ("", "linear-fallback"),
    (None, "linear-fallback"),
])
def test_model_source_labels(core, source, expected):
    assert core.soc_model_short_label(source) == expected


# ======================================================================
# Wind
# ======================================================================

def test_headwind_when_flying_into_the_wind(core):
    head, cross = core.wind_components_mps(10.0, 0.0, 0.0)
    assert head == pytest.approx(10.0)
    assert cross == pytest.approx(0.0, abs=1e-9)


def test_tailwind_is_negative_headwind(core):
    head, _ = core.wind_components_mps(10.0, 180.0, 0.0)
    assert head == pytest.approx(-10.0)


def test_pure_crosswind(core):
    head, cross = core.wind_components_mps(10.0, 90.0, 0.0)
    assert head == pytest.approx(0.0, abs=1e-9)
    assert abs(cross) == pytest.approx(10.0)


def test_groundspeed_reduces_to_simple_form_without_crosswind(core):
    assert core.groundspeed_along_track_mps(20.0, 5.0) == pytest.approx(15.0)
    assert core.groundspeed_along_track_mps(20.0, 5.0, 0.0) == pytest.approx(15.0)


def test_crosswind_costs_along_track_speed(core):
    """Part of the airspeed vector is spent crabbing."""
    plain = core.groundspeed_along_track_mps(20.0, 0.0, 0.0)
    crabbed = core.groundspeed_along_track_mps(20.0, 0.0, 10.0)
    assert crabbed < plain
    assert crabbed == pytest.approx(math.sqrt(400 - 100))


def test_groundspeed_is_zero_when_crosswind_exceeds_airspeed(core):
    assert core.groundspeed_along_track_mps(5.0, 0.0, 10.0) == 0.0


def test_groundspeed_never_negative(core):
    assert core.groundspeed_along_track_mps(5.0, 50.0) == 0.0


# ======================================================================
# Thermal
# ======================================================================

def test_component_heats_when_dissipating_power(core):
    warmer = core.thermal_step(25.0, 25.0, power_loss_W=10.0,
                               thermal_resistance_C_per_W=2.0,
                               thermal_mass_J_per_C=50.0, dt_s=1.0)
    assert warmer > 25.0


def test_component_cools_toward_ambient_with_no_load(core):
    cooler = core.thermal_step(80.0, 25.0, power_loss_W=0.0,
                               thermal_resistance_C_per_W=2.0,
                               thermal_mass_J_per_C=50.0, dt_s=1.0)
    assert 25.0 < cooler < 80.0


def test_thermal_equilibrium_is_stable(core):
    """At T = ambient + P*R the temperature must not move."""
    ambient, power, r_th = 25.0, 10.0, 2.0
    equilibrium = ambient + power * r_th
    stepped = core.thermal_step(equilibrium, ambient, power, r_th, 50.0, 1.0)
    assert stepped == pytest.approx(equilibrium, abs=1e-9)


def test_winding_current_satisfies_the_motor_equation(core):
    """Audit M3, F9: at part throttle the winding carries more current than
    the ESC draws, and P_in = (V_emf + I*R) * I."""
    p, rpm, kv, r, v_supply = 120.0, 6000.0, 920.0, 0.115, 16.0
    i_w = core.motor_winding_current_A(p, rpm, kv, r, v_supply)
    assert (rpm / kv + i_w * r) * i_w == pytest.approx(p, rel=1e-9)
    assert i_w > p / v_supply


def test_winding_current_floors_at_the_supply_current(core):
    # Unknown RPM or Kv: the supply-side current is the only answer.
    assert core.motor_winding_current_A(120.0, None, 920.0, 0.1, 16.0) == pytest.approx(7.5)
    assert core.motor_winding_current_A(120.0, 6000.0, None, 0.1, 16.0) == pytest.approx(7.5)
    # A back-EMF above the supply cannot be reached: floored.
    assert core.motor_winding_current_A(120.0, 20000.0, 920.0, 0.1, 16.0) == pytest.approx(7.5)
    assert core.motor_winding_current_A(0.0, 6000.0, 920.0, 0.1, 16.0) == 0.0


def test_forward_rpm_reduces_to_static_and_reproduces_thrust(core):
    d, pitch, rho, ct = 0.3, 0.2, 1.225, 0.1
    assert core.rpm_from_thrust_forward(5.0, 0.0, d, pitch, rho, ct) == pytest.approx(
        core.rpm_from_thrust(5.0, d, rho, ct))
    v = 20.0
    rpm = core.rpm_from_thrust_forward(5.0, v, d, pitch, rho, ct)
    n = rpm / 60.0
    j, j0 = v / (n * d), pitch / d
    assert ct * (1.0 - j / j0) * rho * n * n * d ** 4 == pytest.approx(5.0, rel=1e-9)
    assert n > v / pitch            # faster than pitch speed, or no thrust


def test_long_thermal_step_settles_without_overshoot(core):
    """Audit F1: explicit Euler overshot, then diverged, once dt > 2*tau.
    tau here is R*C = 100 s; T_ss = 25 + 10*2 = 45 C."""
    t_ss = 45.0
    for dt in (150.0, 250.0, 900.0, 10_000.0):
        t = core.thermal_step(25.0, 25.0, 10.0, 2.0, 50.0, dt)
        assert 25.0 < t <= t_ss + 1e-9, dt
    assert core.thermal_step(25.0, 25.0, 10.0, 2.0, 50.0, 10_000.0) == pytest.approx(t_ss)
    # Cooling from above never undershoots ambient either.
    assert 25.0 <= core.thermal_step(300.0, 25.0, 0.0, 2.0, 50.0, 900.0) < 300.0


def test_thermal_step_is_the_exact_first_order_solution(core):
    import math
    t = core.thermal_step(25.0, 25.0, 10.0, 2.0, 50.0, 100.0)   # one tau
    assert t == pytest.approx(45.0 - 20.0 * math.exp(-1.0))


def test_short_thermal_step_matches_euler(core):
    """At the 0.5 s step the multicopter and VTOL use, the exact solution
    and the old Euler update agree to well under a millidegree."""
    euler = 25.0 + (10.0 - 0.0) / 50.0 * 0.5
    assert core.thermal_step(25.0, 25.0, 10.0, 2.0, 50.0, 0.5) == pytest.approx(euler, abs=1e-3)


# ======================================================================
# Propeller curve fitting
# ======================================================================

def test_fit_recovers_a_known_quadratic(core):
    xs = [0, 1, 2, 3, 4]
    ys = [2 * x * x + 3 * x + 4 for x in xs]
    coeffs, x_min, x_max = core.fit_propeller_curve(xs, ys, degree=2)
    assert coeffs == pytest.approx([2.0, 3.0, 4.0], abs=1e-6)
    assert (x_min, x_max) == (0.0, 4.0)


def test_fit_returns_the_three_part_contract(core):
    """
    Callers unpack ``coeffs, x_min, x_max``. Returning a bare list instead
    bound `coeffs` to a float and broke every prop-table lookup.
    """
    result = core.fit_propeller_curve([0, 1, 2, 3], [1, 2, 5, 10], degree=2)
    assert isinstance(result, tuple) and len(result) == 3
    coeffs, x_min, x_max = result
    assert isinstance(coeffs, list) and len(coeffs) == 3
    assert isinstance(x_min, float) and isinstance(x_max, float)


def test_fit_returns_none_with_too_few_points(core):
    assert core.fit_propeller_curve([1.0, 2.0], [1.0, 2.0], degree=2) is None


def test_fit_ignores_non_finite_samples(core):
    xs = [0, 1, 2, 3, float("nan")]
    ys = [4, 9, 18, 31, 5.0]
    assert core.fit_propeller_curve(xs, ys, degree=2) is not None


# ======================================================================
# AIRFRAME DIAGRAM GEOMETRY
# ======================================================================

def test_polygon_has_one_vertex_per_motor(core):
    for n in (3, 4, 6, 8):
        verts = core.regular_polygon_vertices(n, 0.15)
        assert len(verts) == n
        radii = [math.hypot(x, y) for x, y in verts]
        assert all(r == pytest.approx(0.15, abs=1e-9) for r in radii), \
            "polygon must be equilateral"


def test_rotor_ring_detects_overlap(core):
    """Big props on short arms must be flagged as overlapping."""
    tight = core.rotor_ring_layout(num_positions=6, body_circumradius_m=0.11,
                                   arm_length_m=0.10,
                                   prop_diameter_m=15 * 0.0254)
    assert tight["overlaps"]
    assert tight["adjacent_gap_m"] < 0


def test_rotor_ring_reports_clearance_when_it_fits(core):
    roomy = core.rotor_ring_layout(num_positions=4, body_circumradius_m=0.11,
                                   arm_length_m=0.225,
                                   prop_diameter_m=10 * 0.0254)
    assert not roomy["overlaps"]
    assert roomy["adjacent_gap_m"] > 0


def test_rotor_spacing_matches_the_ring_formula(core):
    """Neighbour spacing on a ring of N points at radius R is 2R*sin(pi/N)."""
    n, body, arm = 6, 0.12, 0.20
    layout = core.rotor_ring_layout(n, body, arm, 0.25)
    r = body + arm
    assert layout["motor_spacing_m"] == pytest.approx(2 * r * math.sin(math.pi / n))
    assert layout["rotor_radius_m"] == pytest.approx(r)


def test_more_rotors_on_the_same_arm_means_less_clearance(core):
    gaps = [core.rotor_ring_layout(n, 0.12, 0.22, 10 * 0.0254)["adjacent_gap_m"]
            for n in (4, 6, 8)]
    assert gaps[0] > gaps[1] > gaps[2]


def test_single_wing_motor_sits_on_the_centreline(core):
    layout = core.wing_rotor_positions(1, 2.0, 12 * 0.0254)
    assert layout["positions_y_m"] == [0.0]
    assert not layout["overlaps"]


def test_wing_motors_are_symmetric_about_the_centreline(core):
    for n in (2, 3, 4, 5):
        pos = core.wing_rotor_positions(n, 2.0, 10 * 0.0254)["positions_y_m"]
        assert len(pos) == n
        assert sum(pos) == pytest.approx(0.0, abs=1e-9), "layout is not symmetric"


def test_odd_wing_motor_count_includes_a_centreline_prop(core):
    pos = core.wing_rotor_positions(3, 2.0, 10 * 0.0254)["positions_y_m"]
    assert any(abs(p) < 1e-9 for p in pos)


def test_wing_props_stay_inboard_of_the_tips(core):
    layout = core.wing_rotor_positions(4, 2.0, 10 * 0.0254)
    assert layout["tip_overhang_m"] <= 0, "a disc reaches past the wing tip"


def test_oversized_wing_props_are_flagged_as_overlapping(core):
    layout = core.wing_rotor_positions(4, 2.0, 20 * 0.0254)
    assert layout["overlaps"]
    assert layout["adjacent_gap_m"] < 0


# ======================================================================
# SENSITIVITY AND COMPARISON ENGINE
# ======================================================================

class _ToyConfig:
    """Minimal stand-in: flight time rises with capacity, falls with weight."""
    def __init__(self):
        self.weight_g = 1000.0
        self.capacity_Ah = 5.0
        self.unused = 1.0


def _toy_evaluate_factory(core, base):
    def evaluate(cfg):
        cfg = base if cfg is None else cfg
        return 60.0 * cfg.capacity_Ah / (cfg.weight_g / 1000.0)
    evaluate.base_config = base
    return evaluate


def test_sensitivity_ranks_by_influence(core):
    base = _ToyConfig()
    rows = core.sensitivity_sweep(
        [("Weight", lambda c, f: setattr(c, "weight_g", c.weight_g * f)),
         ("Capacity", lambda c, f: setattr(c, "capacity_Ah", c.capacity_Ah * f))],
        _toy_evaluate_factory(core, base))
    assert [r["name"] for r in rows][0] == "Weight", "widest swing should sort first"
    assert rows[0]["span"] >= rows[1]["span"]


def test_sensitivity_keeps_an_input_that_does_nothing(core):
    """
    A lever with zero span is a real finding — "this does not matter" — so it
    must appear in the table rather than being quietly dropped.
    """
    base = _ToyConfig()
    rows = core.sensitivity_sweep(
        [("Unused", lambda c, f: setattr(c, "unused", c.unused * f))],
        _toy_evaluate_factory(core, base))
    assert len(rows) == 1
    assert rows[0]["span"] == pytest.approx(0.0)


def test_sensitivity_returns_nothing_when_baseline_fails(core):
    base = _ToyConfig()

    def evaluate(cfg):
        return None
    evaluate.base_config = base
    assert core.sensitivity_sweep([("x", lambda c, f: None)], evaluate) == []


def test_sensitivity_survives_a_lever_that_raises(core):
    base = _ToyConfig()

    def boom(cfg, factor):
        raise RuntimeError("bad lever")

    rows = core.sensitivity_sweep(
        [("Weight", lambda c, f: setattr(c, "weight_g", c.weight_g * f)),
         ("Broken", boom)],
        _toy_evaluate_factory(core, base))
    assert [r["name"] for r in rows] == ["Weight"], "a raising lever must be dropped"


def test_sensitivity_does_not_mutate_the_base_config(core):
    base = _ToyConfig()
    before = base.weight_g
    core.sensitivity_sweep(
        [("Weight", lambda c, f: setattr(c, "weight_g", c.weight_g * f))],
        _toy_evaluate_factory(core, base))
    assert base.weight_g == before, "the sweep modified the caller's config"


def test_comparison_reports_signed_change_and_percentage(core):
    rows = core.compare_metric_sets(
        {"t": 30.0, "p": 100.0}, {"t": 33.0, "p": 90.0},
        [("t", "Flight time", 2), ("p", "Power", 1)])
    by_key = {r["key"]: r for r in rows}
    assert by_key["t"]["delta"] == pytest.approx(3.0)
    assert by_key["t"]["delta_pct"] == pytest.approx(10.0)
    assert by_key["t"]["direction"] == 1
    assert by_key["p"]["delta"] == pytest.approx(-10.0)
    assert by_key["p"]["direction"] == -1


def test_comparison_flags_a_metric_that_is_missing(core):
    rows = core.compare_metric_sets({"t": 30.0}, {}, [("t", "Flight time", 2)])
    assert rows[0]["comparable"] is False
    assert rows[0]["delta"] is None


def test_comparison_against_itself_is_all_zero(core):
    data = {"t": 30.0, "p": 100.0}
    rows = core.compare_metric_sets(data, dict(data),
                                    [("t", "T", 2), ("p", "P", 2)])
    assert all(r["delta"] == pytest.approx(0.0) for r in rows)


def test_format_delta_is_signed_and_handles_none(core):
    assert core.format_delta(3.5, 2) == "+3.50"
    assert core.format_delta(-3.5, 1) == "-3.5"
    assert core.format_delta(None) == "—"
    assert core.format_delta(float("nan")) == "—"



# ======================================================================
# PROPELLER COEFFICIENTS
# ======================================================================

def _table_path(name):
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)


def test_coefficients_derived_from_a_table_are_self_consistent(core, mc):
    """
    C_T should be nearly constant across a static table — the coefficient
    exists precisely to remove the RPM dependence. A large spread would mean
    the fit is not describing a single regime and should not be trusted.
    """
    prop = mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                              max_thrust_g=9500, blades=2, weight_g=110,
                              table_csv=_table_path("motor_prop_table.csv"))
    derived = core.derive_prop_coefficients_from_table(prop.table, 22.0)
    assert derived is not None
    assert derived["points"] > 10
    assert derived["c_t_spread"] < 1.20, \
        f"C_T varies by {derived['c_t_spread']:.2f}x across the table"
    assert 0.02 < derived["c_t"] < 0.20, "C_T outside any plausible range"
    assert 0.005 < derived["c_p"] < 0.15, "C_P outside any plausible range"


def test_geometry_estimate_matches_measured_propellers(core, mc, fw):
    """
    Regression: the original C_T fit was invented rather than measured and
    came out roughly TWICE too high against both sample propellers. Since
    RPM goes as 1/sqrt(C_T) that understated propeller speed by about 40%,
    and it passed an earlier eyeball check only because the resulting tip
    speeds landed inside a plausible band.

    The fit is now calibrated against those measurements, so it must stay
    close to them.
    """
    mc_prop = mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                                 max_thrust_g=9500, blades=2, weight_g=110,
                                 table_csv=_table_path("motor_prop_table.csv"))
    fw_prop = fw.PropellerConfig(diameter_in=18, pitch_in=8, blades=2,
                                 weight_g=60,
                                 table_csv=_table_path("fw_motor_prop_table.csv"))

    for prop, diameter, pitch in ((mc_prop, 22.0, 7.2), (fw_prop, 18.0, 8.0)):
        measured = core.derive_prop_coefficients_from_table(prop.table, diameter)
        estimated = core.estimate_prop_thrust_coefficient(diameter, pitch, 2)
        error = abs(estimated / measured["c_t"] - 1.0)
        assert error < 0.25, (
            f"{diameter}x{pitch}: estimate {estimated:.4f} vs measured "
            f"{measured['c_t']:.4f} — {error * 100:.0f}% off")


def test_derivation_needs_an_rpm_column(core):
    """Without a rotational speed there is no way to non-dimensionalise."""
    import pandas as pd
    df = pd.DataFrame({"Thrust_g": [1000, 2000], "Power_W": [100, 250]})
    assert core.derive_prop_coefficients_from_table(df, 18.0) is None


def test_power_coefficient_follows_from_thrust_coefficient(core):
    """
    C_P is derived from C_T through momentum theory rather than fitted
    separately, so the two cannot drift into disagreement.
    """
    for c_t in (0.05, 0.10, 0.20):
        c_p = core.estimate_prop_power_coefficient(c_t)
        assert c_p > 0
        assert core.estimate_prop_power_coefficient(c_t * 2) > c_p, \
            "a higher thrust coefficient must cost more power"


# ======================================================================
# POWER BUDGET
# ======================================================================

def test_power_budget_balances(core):
    """
    Delivered plus lost must equal the total exactly. If it does not, the
    table is claiming to account for energy it has not accounted for.
    """
    rows = core.build_power_budget(
        total_in_W=1000.0, motor_shaft_W=780.0, motor_copper_W=60.0,
        battery_i2r_W=25.0, esc_loss_W=35.0, peripheral_W=20.0,
        rails=[{"name": "5V", "voltage_V": 5, "current_A": 2.0, "efficiency": 0.9}])

    delivered = sum(r["watts"] for r in rows if r["kind"] == "delivered")
    lost = sum(r["watts"] for r in rows if r["kind"] == "lost")
    total = next(r["watts"] for r in rows if r["kind"] == "total")

    assert delivered + lost == pytest.approx(total, rel=1e-9)
    assert sum(r["pct"] for r in rows
               if r["kind"] in ("delivered", "lost")) == pytest.approx(100.0, abs=1e-6)


def test_battery_loss_is_not_double_counted(core):
    """
    Regression: terminal power is measured at the SAGGED voltage, so the
    pack's own I2R loss is already expressed as reduced voltage rather than
    extra current. Listing it as a loss inside the terminal power charged it
    twice and produced a negative "Unaccounted" row exactly equal to the I2R.

    The total is therefore taken at the cells: P_cells = P_terminals + I2R.
    """
    # These must add up at the TERMINALS: 925 + 50 + 25 = 1000. The pack's
    # own I2R then sits on top to give the cell-side total.
    i2r = 25.0
    rows = core.build_power_budget(
        total_in_W=1000.0, motor_shaft_W=925.0, motor_copper_W=50.0,
        battery_i2r_W=i2r, esc_loss_W=25.0)

    total = next(r["watts"] for r in rows if r["kind"] == "total")
    assert total == pytest.approx(1000.0 + i2r), \
        "the total should be cell-side, i.e. terminal power plus pack I2R"
    assert not any("Unaccounted" in r["name"] for r in rows), \
        "a residual row means the accounting does not close"


def test_power_budget_flags_an_incomplete_breakdown(core):
    """
    If the itemised rows genuinely do not add up, that must be visible rather
    than silently absorbed into the percentages.
    """
    rows = core.build_power_budget(
        total_in_W=1000.0, motor_shaft_W=100.0, motor_copper_W=10.0,
        battery_i2r_W=0.0, esc_loss_W=10.0)
    assert any("Unaccounted" in r["name"] for r in rows), \
        "880 W missing should be reported, not hidden"


def test_rail_losses_follow_efficiency(core):
    """A less efficient regulator must waste more for the same delivered power."""
    def loss_for(efficiency):
        rows = core.build_power_budget(
            total_in_W=500.0, motor_shaft_W=400.0, motor_copper_W=20.0,
            battery_i2r_W=10.0, esc_loss_W=20.0,
            rails=[{"name": "5V", "voltage_V": 5, "current_A": 2.0,
                    "efficiency": efficiency}])
        return next(r["watts"] for r in rows if "regulator loss" in r["name"])

    assert loss_for(0.80) > loss_for(0.95)


# ======================================================================
# PER-ROTOR LOAD SHARING
# ======================================================================

def _x_quad(core, arm=0.237):
    """Four rotors in an X, +x forward and +y right."""
    import math as _m
    return [(arm * _m.cos(a), arm * _m.sin(a))
            for a in (_m.pi / 4, 3 * _m.pi / 4, 5 * _m.pi / 4, 7 * _m.pi / 4)]


def test_drag_through_the_cg_loads_every_rotor_equally(core):
    """With no moment arm there is no moment, so nothing is unbalanced."""
    thrusts = core.rotor_thrust_distribution(
        _x_quad(core), 17.66, drag_N=2.2, translation_azimuth_deg=0.0,
        drag_height_above_cg_m=0.0)
    assert all(t == pytest.approx(17.66 / 4) for t in thrusts)


def test_rotor_thrusts_always_sum_to_the_total(core):
    """
    The split redistributes thrust; it must not create or destroy any. If the
    sum drifted, the aircraft would be silently heavier or lighter than the
    weight it is holding up.
    """
    for azimuth in (0.0, 45.0, 90.0, 180.0, 270.0):
        thrusts = core.rotor_thrust_distribution(
            _x_quad(core), 17.66, drag_N=2.2,
            translation_azimuth_deg=azimuth, drag_height_above_cg_m=0.06)
        assert sum(thrusts) == pytest.approx(17.66, rel=1e-9)


def test_trailing_rotors_work_harder_than_leading_ones(core):
    """
    Drag above the CG pitches the nose down, so the rotors BEHIND the CG must
    push harder to hold attitude. Which rotor saturates first is decided by
    this, not by the average.
    """
    positions = _x_quad(core)          # 0,3 are forward; 1,2 are aft
    thrusts = core.rotor_thrust_distribution(
        positions, 17.66, drag_N=2.2, translation_azimuth_deg=0.0,
        drag_height_above_cg_m=0.06)
    forward_pair = (thrusts[0] + thrusts[3]) / 2
    aft_pair = (thrusts[1] + thrusts[2]) / 2
    assert aft_pair > forward_pair


def test_travel_direction_selects_which_rotors_load(core):
    """
    Flying forwards loads the aft pair; flying right loads the left-hand pair.
    The imbalance follows the direction of travel, which is the whole point of
    having a translation direction.
    """
    positions = _x_quad(core)
    fwd = core.rotor_thrust_distribution(
        positions, 17.66, drag_N=2.2, translation_azimuth_deg=0.0,
        drag_height_above_cg_m=0.06)
    right = core.rotor_thrust_distribution(
        positions, 17.66, drag_N=2.2, translation_azimuth_deg=90.0,
        drag_height_above_cg_m=0.06)
    assert fwd != right, "direction of travel must change which rotors load"
    # Same magnitude of imbalance, different rotors carrying it.
    assert core.rotor_load_spread(fwd)["spread"] == pytest.approx(
        core.rotor_load_spread(right)["spread"], rel=1e-9)


def test_load_spread_reports_a_meaningful_imbalance(core):
    spread = core.rotor_load_spread([4.0, 4.0, 5.0, 5.0])
    assert spread["max_N"] == 5.0 and spread["min_N"] == 4.0
    assert spread["spread"] == pytest.approx(1.25)
    assert spread["imbalance_pct"] == pytest.approx(11.111, abs=0.01)

    even = core.rotor_load_spread([4.4] * 4)
    assert even["spread"] == pytest.approx(1.0)
    assert even["imbalance_pct"] == pytest.approx(0.0)


def test_no_rotor_is_asked_to_pull_downward(core):
    """A rotor cannot produce negative thrust, so the split must floor at 0."""
    thrusts = core.rotor_thrust_distribution(
        _x_quad(core), 5.0, drag_N=50.0, translation_azimuth_deg=0.0,
        drag_height_above_cg_m=0.5)
    assert all(t >= 0.0 for t in thrusts)


def test_power_coefficient_is_the_shaft_coefficient(core, mc):
    """Audit C4: C_P was fitted to the table's Power_W, electrical power at
    the ESC input. With a torque column it is the shaft coefficient."""
    import math
    prop = mc.PropellerConfig(diameter_in=22, pitch_in=7.2, max_rpm=0,
                              max_thrust_g=9500, blades=2, weight_g=110,
                              table_csv=_table_path("motor_prop_table.csv"))
    fit = core.derive_prop_coefficients_from_table(prop.table, 22.0)
    assert fit["c_p_basis"] == "shaft"
    d = 22.0 * 0.0254
    row = prop.table.iloc[len(prop.table) // 2]
    n = float(row["RPM"]) / 60.0
    shaft_cp = float(row["Torque_Nm"]) * 2 * math.pi * n / (1.225 * n ** 3 * d ** 5)
    elec_cp = float(row["Power_W"]) / (1.225 * n ** 3 * d ** 5)
    assert shaft_cp < elec_cp
    assert fit["c_p"] == pytest.approx(shaft_cp, rel=0.25)
    no_torque = prop.table.drop(columns=["Torque_Nm"])
    fallback = core.derive_prop_coefficients_from_table(no_torque, 22.0)
    assert fallback["c_p_basis"] == "electrical"
    assert fallback["c_p"] > fit["c_p"]


def test_power_coefficient_estimate_uses_the_propeller_convention(core):
    """Audit C3: ideal C_P = C_T^1.5 * sqrt(2/pi) in T = C_T rho n^2 D^4."""
    import math
    c_t, fm = 0.08, 0.65
    assert core.estimate_prop_power_coefficient(c_t, fm) == pytest.approx(
        c_t ** 1.5 * math.sqrt(2.0 / math.pi) / fm)
    # Check it against momentum theory directly: P = T^1.5 / sqrt(2 rho A).
    rho, d, n = 1.225, 0.3, 100.0
    t = c_t * rho * n ** 2 * d ** 4
    p_ideal = t ** 1.5 / math.sqrt(2 * rho * math.pi * d * d / 4)
    # (The figure of merit is capped at 0.95, so compare at that.)
    assert core.estimate_prop_power_coefficient(c_t, 0.95) == pytest.approx(
        p_ideal / 0.95 / (rho * n ** 3 * d ** 5), rel=1e-9)
