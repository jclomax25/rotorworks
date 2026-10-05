"""
VTOL simulator tests.

The transition is the part that is genuinely new — the multicopter and
fixed-wing models are already covered elsewhere — so most of these check that
the lift hand-over behaves, and that the unimplemented configurations refuse
rather than approximate.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VTOL_SCRIPT = os.path.join(ROOT, "vtol-power-sim-gui.py")


@pytest.fixture(scope="session")
def vtol():
    # Stub tkinter ONLY when there is no real one. Doing it unconditionally
    # looks harmless because setdefault leaves an already-imported module
    # alone — but `import tkinter` does not pull in tkinter.simpledialog, so
    # the placeholder won that slot and matplotlib's Tk backend later failed
    # with "cannot import name SimpleDialog". That made every GUI test in the
    # session error out, but only when a VTOL physics test happened to run
    # first, which is why running the marks separately never showed it.
    try:
        import tkinter  # noqa: F401
        import tkinter.simpledialog  # noqa: F401
        import tkinter.scrolledtext  # noqa: F401
    except ImportError:
        for name in ("tkinter", "tkinter.ttk", "tkinter.messagebox",
                     "tkinter.filedialog", "tkinter.simpledialog",
                     "tkinter.font", "tkinter.scrolledtext"):
            sys.modules.setdefault(name, types.ModuleType(name))
    import matplotlib
    matplotlib.use("Agg")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location("rw_vtol", VTOL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    module.__name__ = "rw_vtol"
    sys.modules["rw_vtol"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def aircraft(vtol):
    """Reference lift+cruise: 6 kg, 2.4 m span, four 18in lift rotors."""
    return vtol.VTOLConfig()


# ======================================================================
# CONFIGURATION GATING
# ======================================================================

def test_lift_cruise_is_implemented(vtol, aircraft):
    metrics = vtol.compute_metrics(aircraft)
    assert metrics["config_type"] == "lift+cruise"
    assert metrics["total_power_W"] > 0


@pytest.mark.parametrize("config_type", ["tiltrotor", "tiltwing", "tailsitter"])
def test_every_configuration_now_runs(vtol, config_type):
    """
    These three used to raise NotImplementedError, deliberately, rather than
    fall back to lift+cruise physics that would have been confidently wrong.
    They now have their own vectored-thrust model, so they must run and must
    NOT be secretly computed as lift+cruise.
    """
    cfg = vtol.VTOLConfig(config_type=config_type)
    metrics = vtol.compute_metrics(cfg)
    assert metrics["config_type"] == config_type
    assert metrics["total_power_W"] > 0
    assert vtol.uses_vectored_thrust(cfg), \
        f"{config_type} is being treated as lift+cruise"


def test_all_config_types_are_selectable_and_implemented(vtol):
    """The dropdown offers every configuration, and every one now works."""
    for name in ("lift+cruise", "tiltrotor", "tiltwing", "tailsitter"):
        assert name in vtol.CONFIG_TYPES
    assert vtol.IMPLEMENTED_CONFIG_TYPES == set(vtol.CONFIG_TYPES)


def test_lift_cruise_is_not_vectored(vtol, aircraft):
    """The one type with separate lift and cruise systems must stay separate."""
    assert not vtol.uses_vectored_thrust(aircraft)


# ======================================================================
# BASIC PHYSICS
# ======================================================================

def test_stall_speed_matches_hand_calculation(vtol, aircraft):
    expected = math.sqrt(2 * aircraft.weight_N /
                         (aircraft.air_density * aircraft.wing_area_m2 * aircraft.CL_max))
    assert vtol.stall_speed_mps(aircraft) == pytest.approx(expected, rel=1e-9)


def test_transition_speed_is_above_stall(vtol, aircraft):
    """
    Transition uses a CL cap below CL_max for gust margin, so the speed at
    which the wing takes the full load is above the stall speed.
    """
    assert vtol.transition_speed_mps(aircraft) > vtol.stall_speed_mps(aircraft)


def test_hover_thrust_equals_weight_plus_download(vtol, aircraft):
    """
    In hover the rotors lift the aircraft AND the wash they push onto the
    structure beneath them. The download is a fraction of weight, so hover
    thrust is weight x (1 + download) — not weight.

    This test used to assert plain equality, which was right only while
    lift+cruise had no download modelled at all.
    """
    hover = vtol.hover_power_W(aircraft)
    expected = aircraft.weight_N * (1.0 + vtol.hover_download_fraction(aircraft))
    assert hover["rotor_thrust_N"] == pytest.approx(expected, rel=1e-9)

    # With the download zeroed it is exactly the weight again.
    bare = vtol.VTOLConfig(hover_download_fraction=0.0)
    assert vtol.hover_power_W(bare)["rotor_thrust_N"] == pytest.approx(
        bare.weight_N, rel=1e-9)


def test_hover_power_matches_momentum_theory(vtol, aircraft):
    """
    P = T * sqrt(T / 2*rho*A) / FM, plus the motors' own losses, through the
    ESC, plus avionics — where T is the weight PLUS the download the rotors
    also have to lift.

    Kv = 0 switches the motor model off, and the chain is then exactly the
    one this test pinned before the motor model existed.
    """
    thrust = aircraft.weight_N * (1.0 + vtol.hover_download_fraction(aircraft))
    area = aircraft.lift_disc_area_m2
    per_rotor = thrust / aircraft.num_lift_rotors
    disc_per = area / aircraft.num_lift_rotors
    v_hover = math.sqrt(per_rotor / (2 * aircraft.air_density * disc_per))
    ideal = thrust * v_hover
    shaft = ideal / aircraft.lift_figure_of_merit

    no_motor = vtol.VTOLConfig(lift_motor_kv=0, cruise_motor_kv=0)
    expected = shaft / no_motor.esc_efficiency + no_motor.avionics_power_W
    assert vtol.hover_power_W(no_motor)["total_power_W"] == pytest.approx(expected, rel=1e-6)

    # With the motor model, each motor adds I0 * V_emf + I^2 * Rm, where the
    # current is the torque over Kt plus I0 — worked here from the RPM the
    # model reports, not read back from the loss it reports.
    op = vtol.hover_power_W(aircraft)["lift_motor"]
    omega = op["rpm"] * 2 * math.pi / 60.0
    kt = 60.0 / (2 * math.pi * aircraft.lift_motor_kv)
    current = (shaft / aircraft.num_lift_rotors) / omega / kt + aircraft.lift_motor_i0_A
    v_emf = op["rpm"] / aircraft.lift_motor_kv
    loss = aircraft.lift_motor_i0_A * v_emf + current ** 2 * aircraft.lift_motor_resistance
    expected = ((shaft + aircraft.num_lift_rotors * loss) / aircraft.esc_efficiency
                + aircraft.avionics_power_W)
    assert op["current_A"] == pytest.approx(current, rel=1e-9)
    assert vtol.hover_power_W(aircraft)["total_power_W"] == pytest.approx(expected, rel=1e-6)


def test_climbing_costs_more_than_hovering(vtol, aircraft):
    hover = vtol.hover_power_W(aircraft)["total_power_W"]
    climb = vtol.hover_power_W(aircraft, climb_rate_mps=2.5)["total_power_W"]
    assert climb > hover
    # The extra is the rotors' climb work, T x climb rate, through the ESC —
    # where T is the full hover thrust INCLUDING download.
    #
    # I first wrote weight x rate here, reasoning that download is a thrust
    # penalty rather than extra mass. The model disagreed and the model is
    # right: download is a real downward force on the structure, and a
    # climbing aircraft moves up against it, so that force does work too.
    thrust = aircraft.weight_N * (1.0 + vtol.hover_download_fraction(aircraft))
    extra = thrust * 2.5 / aircraft.esc_efficiency
    # Without a motor model the extra is exactly that. With one, the climb
    # work also pays the motors' efficiency, so it costs more.
    no_motor = vtol.VTOLConfig(lift_motor_kv=0, cruise_motor_kv=0)
    bare_gap = (vtol.hover_power_W(no_motor, climb_rate_mps=2.5)["total_power_W"]
                - vtol.hover_power_W(no_motor)["total_power_W"])
    assert bare_gap == pytest.approx(extra, rel=1e-6)
    assert climb - hover > extra


# ======================================================================
# THE TRANSITION — the part that is actually new
# ======================================================================

def test_wing_lift_share_rises_monotonically_with_speed(vtol, aircraft):
    shares = []
    for speed in (2, 4, 6, 8, 10, 12):
        point = vtol.transition_power_W(aircraft, speed)
        shares.append(point["lift_share_wing"])
    assert shares == sorted(shares), f"lift share not monotonic: {shares}"
    assert shares[0] < 0.2, "wing should carry almost nothing at low speed"
    assert shares[-1] > 0.7, "wing should carry most of the load near transition"


def test_rotor_and_wing_lift_always_sum_to_weight(vtol, aircraft):
    """
    The aircraft must be held up at every point in the transition — by the
    wing, the rotors, or both — plus whatever download the rotors are still
    pushing onto the structure at that speed.

    The download term fades as the wing takes over, which is what keeps hover
    and cruise the two ends of one curve instead of two branches that
    disagree where they meet.
    """
    for speed in (0.5, 3, 6, 9, 12, 13):
        point = vtol.transition_power_W(aircraft, speed)
        total = point["rotor_thrust_N"] + point["wing_lift_N"]
        share = 1.0 - point["wing_lift_N"] / aircraft.weight_N
        expected = aircraft.weight_N * (
            1.0 + vtol.hover_download_fraction(aircraft) * max(share, 0.0))
        assert total == pytest.approx(expected, rel=1e-9), \
            f"lift does not balance weight at {speed} m/s"


def test_rotor_power_falls_as_the_wing_takes_over(vtol, aircraft):
    powers = [vtol.transition_power_W(aircraft, v)["rotor_shaft_W"]
              for v in (2, 5, 8, 11, 13)]
    assert powers == sorted(powers, reverse=True), \
        f"rotor power should fall through the transition: {powers}"


def test_rotors_are_unloaded_above_the_transition_speed(vtol, aircraft):
    v_trans = vtol.transition_speed_mps(aircraft)
    point = vtol.power_at_airspeed(aircraft, v_trans * 1.2)
    assert point["regime"] == "cruise"
    assert point["rotor_thrust_N"] == 0.0
    assert point["rotor_shaft_W"] == 0.0


def test_regime_selection_is_continuous_across_the_boundary(vtol, aircraft):
    """
    Power must not jump when the model switches from transition to cruise —
    a discontinuity there would mean the two models disagree about the same
    flight condition.
    """
    v_trans = vtol.transition_speed_mps(aircraft)
    below = vtol.power_at_airspeed(aircraft, v_trans * 0.995)["total_power_W"]
    above = vtol.power_at_airspeed(aircraft, v_trans * 1.005)["total_power_W"]
    assert abs(above - below) / below < 0.10, \
        f"power jumps {below:.0f} -> {above:.0f} W at the regime boundary"


def test_hover_costs_more_than_cruise(vtol, aircraft):
    """
    The whole reason a VTOL has a wing. If this ever inverts, the design or
    the model is wrong.
    """
    metrics = vtol.compute_metrics(aircraft)
    assert metrics["hover_to_cruise_power_ratio"] > 1.0
    assert metrics["hover_endurance_min"] < metrics["cruise_endurance_min"]


# ======================================================================
# STOPPED-ROTOR DRAG
# ======================================================================

def test_stopped_rotors_add_drag_in_cruise(vtol, aircraft):
    with_rotors = vtol.cruise_power_W(aircraft, 22.0)["total_power_W"]
    aircraft._stopped_rotor_drag_area_m2 = 0.0
    without = vtol.cruise_power_W(aircraft, 22.0)["total_power_W"]
    assert with_rotors > without, "stopped rotors should cost cruise power"


def test_stopped_rotor_drag_area_scales_with_rotor_count(vtol):
    four = vtol.VTOLConfig(num_lift_rotors=4).stopped_rotor_drag_area_m2
    eight = vtol.VTOLConfig(num_lift_rotors=8).stopped_rotor_drag_area_m2
    assert eight == pytest.approx(2 * four, rel=1e-9)


def test_explicit_stopped_rotor_area_overrides_the_estimate(vtol):
    cfg = vtol.VTOLConfig(stopped_rotor_drag_area_m2=0.05)
    assert cfg.stopped_rotor_drag_area_m2 == pytest.approx(0.05)


# ======================================================================
# BATTERY, SHARED WITH THE OTHER SIMULATORS
# ======================================================================

def test_pack_capacity_scales_with_parallel_only(vtol):
    one = vtol.VTOLBattery(parallel_cells=1).capacity_mAh
    two = vtol.VTOLBattery(parallel_cells=2).capacity_mAh
    series = vtol.VTOLBattery(series_cells=12, parallel_cells=1).capacity_mAh
    assert two == pytest.approx(2 * one)
    assert series == pytest.approx(one), "series must not change capacity"


def test_soc_model_comes_from_the_shared_core(vtol):
    battery = vtol.VTOLBattery(chemistry="LiPo")
    assert "lipo" in battery.soc_model_source
    assert battery.ocv_at_soc(1.0) > battery.ocv_at_soc(0.2)


# ======================================================================
# MISSION
# ======================================================================

def _mission_file(tmp_path):
    payload = {"reserve_percent": 20, "phases": [
        {"name": "Climb", "kind": "climb", "duration": 30, "climb_rate_mps": 2.5},
        {"name": "Transition", "kind": "transition", "duration": 12, "speed": 15.0},
        {"name": "Cruise", "kind": "cruise", "distance": 8000, "speed": 22.0},
        {"name": "Transition back", "kind": "transition", "duration": 15, "speed": 15.0},
        {"name": "Descend", "kind": "descend", "duration": 35},
    ]}
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_mission_runs_and_accounts_for_energy(vtol, aircraft, tmp_path):
    mission = vtol.VTOLMission.from_json(_mission_file(tmp_path))
    results, totals = vtol.simulate_mission(aircraft, mission)

    assert len(results) == 5
    summed = sum(row[4] for row in results)
    assert summed == pytest.approx(totals["energy_Wh"], rel=1e-9), \
        "phase energies must sum to the total"

    buckets = totals["hover_Wh"] + totals["transition_Wh"] + totals["cruise_Wh"]
    assert buckets == pytest.approx(totals["energy_Wh"], rel=1e-9), \
        "energy must be attributed to exactly one regime per phase"


def test_mission_energy_split_is_reported(vtol, aircraft, tmp_path):
    """
    The split across hover, transition and cruise is the point of the tool —
    it is where a VTOL's endurance is won or lost.
    """
    mission = vtol.VTOLMission.from_json(_mission_file(tmp_path))
    _results, totals = vtol.simulate_mission(aircraft, mission)
    for key in ("hover_Wh", "transition_Wh", "cruise_Wh"):
        assert totals[key] > 0, f"{key} should be non-zero for this mission"


def test_a_distance_leg_takes_the_expected_time(vtol, aircraft, tmp_path):
    mission = vtol.VTOLMission.from_json(_mission_file(tmp_path))
    results, _totals = vtol.simulate_mission(aircraft, mission)
    cruise = next(r for r in results if r[0] == "Cruise")
    assert cruise[1] == pytest.approx(8000 / 22.0 / 60.0, rel=1e-6)
    assert cruise[2] == pytest.approx(8.0, rel=1e-6)


# ======================================================================
# CLI
# ======================================================================

@pytest.mark.slow
def test_cli_single_point_runs():
    result = subprocess.run([sys.executable, VTOL_SCRIPT],
                            capture_output=True, text=True, timeout=300, cwd=ROOT)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output[-800:]
    assert result.returncode == 0
    assert "Hover power" in output
    assert "nan" not in output.lower()


@pytest.mark.slow
@pytest.mark.parametrize("config_type", ["tiltrotor", "tiltwing", "tailsitter"])
def test_cli_runs_every_vectored_configuration(config_type):
    """The CLI used to refuse these; it must now run them cleanly."""
    result = subprocess.run(
        [sys.executable, VTOL_SCRIPT, "--config_type", config_type],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output[-400:]
    assert result.returncode == 0, output[-400:]
    assert "not implemented" not in output.lower()


@pytest.mark.slow
def test_cli_runs_the_example_mission():
    mission = os.path.join(ROOT, "examples", "missions",
                           "vtol_01_lift_cruise_survey.json")
    if not os.path.exists(mission):
        pytest.skip("example VTOL mission not present")
    result = subprocess.run(
        [sys.executable, VTOL_SCRIPT, "--battery_parallel_cells", "3",
         "--mission", mission],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output[-800:]
    assert result.returncode == 0
    assert "Energy split" in output


# ======================================================================
# VECTORED-THRUST TYPES: tiltrotor, tiltwing, tailsitter
# ======================================================================

VECTORED = ("tiltrotor", "tiltwing", "tailsitter")
ALL_TYPES = ("lift+cruise",) + VECTORED


def _cfg(vtol, config_type, **overrides):
    cfg = vtol.VTOLConfig(config_type=config_type)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.parametrize("config_type", VECTORED)
def test_thrust_vector_balances_both_axes(vtol, config_type):
    """
    The core of the vectored model. At every speed the rotor thrust must
    resolve into exactly the vertical force still needed and the horizontal
    force needed to beat drag:

        T*cos(tilt) = W*(1+download) - L_wing
        T*sin(tilt) = D

    If either axis fails to close, the aircraft is either falling or
    decelerating, and the power figure describes a flight that is not steady.
    """
    import math
    cfg = _cfg(vtol, config_type)
    for speed in (0.0, 4.0, 8.0, 12.0, 16.0, 22.0):
        r = vtol.vectored_power_W(cfg, speed)
        tilt = math.radians(r["tilt_deg"])
        vertical_needed = cfg.weight_N + r["download_N"] - r["wing_lift_N"]

        assert r["rotor_thrust_N"] * math.cos(tilt) == pytest.approx(
            max(vertical_needed, 0.0), abs=1e-6), f"vertical fails at {speed} m/s"
        assert r["rotor_thrust_N"] * math.sin(tilt) == pytest.approx(
            r["drag_N"], abs=1e-6), f"horizontal fails at {speed} m/s"


@pytest.mark.parametrize("config_type", VECTORED)
def test_tilt_sweeps_from_vertical_to_horizontal(vtol, config_type):
    """
    Hover points the thrust straight up; wing-borne cruise points it straight
    ahead; the transition sweeps between, monotonically. A tilt that went
    backwards mid-transition would mean the model had the lift split wrong.
    """
    cfg = _cfg(vtol, config_type)
    tilts = [vtol.vectored_power_W(cfg, v)["tilt_deg"]
             for v in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 20.0, 25.0)]
    assert tilts[0] == pytest.approx(0.0, abs=1e-6), "hover must be vertical"
    assert tilts[-1] == pytest.approx(90.0, abs=1e-6), "cruise must be horizontal"
    assert tilts == sorted(tilts), f"tilt reversed during transition: {tilts}"


@pytest.mark.parametrize("config_type", ALL_TYPES)
def test_power_is_continuous_through_transition(vtol, config_type):
    """
    No jumps between hover, transition and cruise.

    A fixed threshold cannot tell a steep slope from a genuine discontinuity
    — both look like "a few percent" at one resolution. The steepest part of
    the curve here is around 8.5 m/s, where power falls fast, and at a
    0.25 m/s step it reads 6-7%, which a threshold would wave through either
    way.

    So test the property that actually distinguishes them. For a continuous
    curve the largest step shrinks in proportion as the step shrinks; for a
    jump it stays the same size however finely you sample. Halving the step
    three times must roughly halve the worst change each time.
    """
    cfg = _cfg(vtol, config_type)

    def worst_step(step):
        n = int(30.0 / step) + 1
        speeds = [k * step for k in range(n)]
        powers = [vtol.power_at_airspeed(cfg, s)["total_power_W"] for s in speeds]
        return max(abs(powers[k + 1] - powers[k]) / max(powers[k], 1e-9)
                   for k in range(n - 1))

    coarse, fine = worst_step(0.25), worst_step(0.03125)
    # Three halvings: a continuous curve drops ~8x. A jump would not drop.
    assert fine < coarse / 4.0, (
        f"{config_type}: worst step {coarse*100:.2f}% at 0.25 m/s but still "
        f"{fine*100:.2f}% at 0.03 m/s — that is a discontinuity, not a slope")
    assert fine < 0.02, f"{config_type}: {fine*100:.2f}% at the finest step"


def test_tiltrotor_pays_the_most_to_hover(vtol):
    """
    A tiltrotor's rotors blow straight down onto a wing lying flat beneath
    them, and that download has to be lifted too — about a tenth of the
    weight on real aircraft. A tiltwing turns the wing edge-on to the wash and
    a tailsitter stands the whole wing up, so both escape most of it.
    """
    hover = {t: vtol.power_at_airspeed(_cfg(vtol, t), 0.0)["total_power_W"]
             for t in VECTORED}
    assert hover["tiltrotor"] > hover["tiltwing"]
    assert hover["tiltrotor"] > hover["tailsitter"]


def test_vectored_types_beat_lift_cruise_in_cruise(vtol):
    """
    The central trade. Lift+cruise carries stopped rotors through the whole
    cruise, dragging and doing nothing; the vectored types turn the same
    rotors into propellers and carry nothing dead. At a normal cruise speed
    that has to show up as lower power.
    """
    cruise = {t: vtol.power_at_airspeed(_cfg(vtol, t), 20.0)["total_power_W"]
              for t in ALL_TYPES}
    for t in VECTORED:
        assert cruise[t] < cruise["lift+cruise"], (
            f"{t} ({cruise[t]:.0f} W) should cruise cheaper than lift+cruise "
            f"({cruise['lift+cruise']:.0f} W)")


def test_download_vanishes_once_the_wing_carries_the_load(vtol):
    """
    Download only exists while the rotors are pushing air down onto the
    airframe. Once the wing carries the weight the rotors point forward and
    the wash streams aft, so the penalty must reach zero — otherwise cruise
    would be charged a hover effect.
    """
    cfg = _cfg(vtol, "tiltrotor")
    assert vtol.vectored_power_W(cfg, 0.0)["download_N"] > 0
    assert vtol.vectored_power_W(cfg, 25.0)["download_N"] == pytest.approx(0.0)


def test_download_can_be_overridden_with_measured_data(vtol):
    """
    The per-type download figures are typical published values, not
    measurements. A user with test data must be able to replace them.
    """
    stock = vtol.power_at_airspeed(_cfg(vtol, "tiltrotor"), 0.0)["total_power_W"]
    measured = vtol.power_at_airspeed(
        _cfg(vtol, "tiltrotor", hover_download_fraction=0.20), 0.0)["total_power_W"]
    zero = vtol.power_at_airspeed(
        _cfg(vtol, "tiltrotor", hover_download_fraction=0.0), 0.0)["total_power_W"]
    assert zero < stock < measured


def test_tiltwing_and_tailsitter_match_in_still_air(vtol):
    """
    Deliberate, and documented. In steady still-air flight the two are
    genuinely close: both turn the wing edge-on in hover and both use their
    rotors as propellers in cruise. What separates them is dynamics — a
    tailsitter weathervanes in a crosswind, a tiltwing keeps its fuselage
    level — which a power model does not see.

    This test exists so that if they ever DO diverge, someone has to decide
    that on purpose rather than by accident.
    """
    for speed in (0.0, 6.0, 12.0, 20.0):
        tw = vtol.power_at_airspeed(_cfg(vtol, "tiltwing"), speed)["total_power_W"]
        ts = vtol.power_at_airspeed(_cfg(vtol, "tailsitter"), speed)["total_power_W"]
        assert tw == pytest.approx(ts, rel=1e-12)


@pytest.mark.parametrize("config_type", VECTORED)
def test_wing_borne_cruise_needs_thrust_equal_to_drag(vtol, config_type):
    """Well above the transition speed the rotors only have to beat drag."""
    cfg = _cfg(vtol, config_type)
    r = vtol.vectored_power_W(cfg, 22.0)
    assert r["regime"] == "cruise"
    assert r["wing_lift_N"] == pytest.approx(cfg.weight_N, rel=1e-9)
    assert r["rotor_thrust_N"] == pytest.approx(r["drag_N"], rel=1e-9)


@pytest.mark.parametrize("config_type", VECTORED)
def test_hover_climb_costs_potential_power(vtol, config_type):
    """Climbing vertically adds T*climb_rate — the same work any rotor does."""
    cfg = _cfg(vtol, config_type)
    level = vtol.vectored_power_W(cfg, 0.0)["shaft_power_W"]
    climbing = vtol.vectored_power_W(cfg, 0.0, climb_rate_mps=2.0)
    extra = climbing["shaft_power_W"] - level
    assert extra == pytest.approx(climbing["rotor_thrust_N"] * 2.0, rel=1e-9)


@pytest.mark.parametrize("config_type", ALL_TYPES)
def test_every_type_flies_a_mission(vtol, config_type, tmp_path):
    """Each configuration must complete a hover-transition-cruise mission."""
    import json
    payload = {"reserve_percent": 20, "transition_time_s": 10, "phases": [
        {"name": "Climb", "kind": "climb", "duration": 20,
         "climb_rate_mps": 2.0, "altitude": 40},
        {"name": "Transition", "kind": "transition", "duration": 10,
         "speed": 15.0, "altitude": 40},
        {"name": "Cruise", "kind": "cruise", "distance": 2000,
         "speed": 20.0, "altitude": 60},
        {"name": "Transition back", "kind": "transition", "duration": 10,
         "speed": 15.0, "altitude": 40},
        {"name": "Descend", "kind": "descend", "duration": 20, "altitude": 0}]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))

    cfg = _cfg(vtol, config_type)
    results, summary = vtol.simulate_mission(cfg, vtol.VTOLMission.from_json(str(path)))
    assert results, f"{config_type} produced no phases"
    total_Wh = sum(v for k, v in summary.items() if k.endswith("_Wh")
                   and k in ("hover_Wh", "transition_Wh", "cruise_Wh"))
    assert total_Wh > 0


# ======================================================================
# GUI — the tabs ported from the multicopter and fixed-wing
# ======================================================================
#
# These drive the real window. They exist because every feature in this port
# had an equivalent that shipped broken in one of the other two simulators,
# and in every case the physics was right and the WIRING was not: an input
# that never reached the config, a Compare tab refreshed before the new
# result was stored, a placeholder redrawn over a moment after it was set.
# Checking that a number MOVES is the only thing that catches those.

pytestmark_gui = pytest.mark.gui


@pytest.fixture
def gui(tmp_path):
    """
    A live VTOL window with dialogs stubbed, or a skip if there is no display.

    Deliberately does NOT use the `vtol` fixture: that one stubs tkinter so
    the module can be imported headlessly, which is exactly what a GUI test
    must not have. This loads its own copy against the real toolkit.
    """
    import importlib
    for name in ("tkinter", "tkinter.ttk", "tkinter.messagebox",
                 "tkinter.filedialog", "tkinter.simpledialog",
                 "tkinter.font", "tkinter.scrolledtext"):
        stub = sys.modules.get(name)
        if stub is not None and not hasattr(stub, "__file__"):
            del sys.modules[name]                    # drop the headless stub
    try:
        tk = importlib.import_module("tkinter")
        ttk = importlib.import_module("tkinter.ttk")
        filedialog = importlib.import_module("tkinter.filedialog")
        messagebox = importlib.import_module("tkinter.messagebox")
        probe = tk.Tk()
        probe.destroy()
    except Exception:
        pytest.skip("no display available; run under xvfb-run")

    spec = importlib.util.spec_from_file_location("vtol_gui", VTOL_SCRIPT)
    vtol = importlib.util.module_from_spec(spec)
    sys.modules["vtol_gui"] = vtol
    spec.loader.exec_module(vtol)

    holder, errors = {}, []
    tk.Tk.mainloop = lambda self, *a, **k: (holder.__setitem__("root", self),
                                            self.update_idletasks(), self.update())
    messagebox.showerror = lambda title, msg=None, **k: errors.append(str(msg))
    messagebox.showinfo = lambda title, msg=None, **k: errors.append("INFO " + str(msg))
    messagebox.showwarning = lambda *a, **k: None

    vtol.launch_gui()
    root = holder["root"]

    class Harness:
        def __init__(self):
            self.root, self.errors = root, errors

        def widgets(self):
            found = []

            def walk(w):
                for child in w.winfo_children():
                    found.append(child)
                    walk(child)
            walk(root)
            return found

        def pump(self):
            root.update_idletasks()
            root.update()

        def click(self, fragment):
            errors.clear()
            for w in self.widgets():
                if isinstance(w, ttk.Button) and fragment.lower() in str(w.cget("text")).lower():
                    w.invoke()
                    self.pump()
                    return list(errors)
            raise AssertionError(f"no button matching {fragment!r}")

        def open_with(self, path):
            filedialog.askopenfilename = lambda *a, **k: str(path)

        def tree(self, first_column, n_columns):
            """
            Find a table by its first column name and width.

            Ambiguous on its own: the Weight Budget and the Power Budget are
            BOTH ("item", ...) and both five wide since the Power Budget
            gained Voltage and Current. `tree_exact` below is the reliable
            one; this stays for the unambiguous cases.
            """
            matches = [w for w in self.widgets()
                       if isinstance(w, ttk.Treeview)
                       and [str(c) for c in w.cget("columns")][:1] == [first_column]
                       and len(w.cget("columns")) == n_columns]
            if not matches:
                raise AssertionError(
                    f"no tree starting {first_column!r} with {n_columns} columns")
            return matches[0]

        def tree_exact(self, *columns):
            """Find a table by its FULL column tuple — never ambiguous."""
            wanted = list(columns)
            for w in self.widgets():
                if isinstance(w, ttk.Treeview) and \
                        [str(c) for c in w.cget("columns")] == wanted:
                    return w
            raise AssertionError(f"no tree with columns {wanted}")

        def trees_exact(self, *columns):
            """
            EVERY table with this column tuple, in creation order.

            Status is four sub-tables that share one column tuple, so a
            lookup returning "the" tree would silently report only the
            Battery one and a check that moved to another group would read
            as deleted.
            """
            wanted = list(columns)
            return [w for w in self.widgets()
                    if isinstance(w, ttk.Treeview)
                    and [str(c) for c in w.cget("columns")] == wanted]

        @staticmethod
        def walk(tree, node=""):
            """
            Every row in a tree, parents included, depth-first.

            Metrics groups its rows under collapsible section nodes, so
            get_children("") returns six headings and none of the numbers.
            Any assertion about metric rows has to walk.
            """
            out = []
            for iid in tree.get_children(node):
                out.append(tree.item(iid, "values"))
                out.extend(Harness.walk(tree, iid))
            return out

        @staticmethod
        def leaves(tree, node=""):
            """Only the rows that have no children — the data, not the headings."""
            out = []
            for iid in tree.get_children(node):
                kids = tree.get_children(iid)
                if kids:
                    out.extend(Harness.leaves(tree, iid))
                else:
                    out.append(tree.item(iid, "values"))
            return out

        def metric_rows(self):
            """The Metrics tab's data rows, excluding the section headings."""
            return self.leaves(self.tree_exact("metric", "value", "note"))

        def metric_sections(self):
            """The Metrics tab's section headings, in order."""
            tree = self.tree_exact("metric", "value", "note")
            return [tree.item(i, "values")[0] for i in tree.get_children("")]

        def status_rows(self):
            """Every Status check across all four sub-tables."""
            out = []
            for tree in self.trees_exact("metric", "value", "limit", "note"):
                out.extend(tree.item(i, "values")
                           for i in tree.get_children(""))
            return out

        def sensitivity_rows(self):
            """
            The Sensitivity table, by its full column tuple.

            It widened from 5 to 7 when the four perturbation levels stopped
            being collapsed into Low/High, so a width-based lookup breaks
            whenever that table changes shape.
            """
            tree = self.tree_exact("name", "m20", "m10", "base",
                                   "p10", "p20", "span")
            return [tree.item(i, "values") for i in tree.get_children()]

        def power_budget_rows(self):
            tree = self.tree_exact("item", "watts", "pct", "voltage", "current")
            return [tree.item(i, "values") for i in tree.get_children()]

        def rows(self, first_column, n_columns):
            tree = self.tree(first_column, n_columns)
            return [tree.item(i, "values") for i in tree.get_children()]

        def use_advanced_inputs(self):
            """
            Switch to the Advanced view.

            Simple mode hides the advanced inputs, and a hidden widget cannot
            be typed into — so any test that sets one has to ask for them
            first, exactly as a user would.
            """
            for w in self.widgets():
                if isinstance(w, ttk.Radiobutton) and str(w.cget("text")) == "Advanced":
                    w.invoke()
                    self.pump()
                    return
            raise AssertionError("no Advanced radio button")

        def set_field(self, label, value):
            for w in self.widgets():
                if not (isinstance(w, ttk.Entry) and w.winfo_manager() == "grid"):
                    continue
                row = w.grid_info().get("row")
                for lbl in w.master.winfo_children():
                    if (isinstance(lbl, ttk.Label) and str(lbl.cget("text")) == label
                            and lbl.grid_info().get("row") == row):
                        root.setvar(w.cget("textvariable"), str(value))
                        self.pump()
                        return True
            raise AssertionError(f"no field labelled {label!r}")

        def output_text(self):
            """Everything in the Output log pane."""
            import tkinter as tk
            for w in self.widgets():
                if isinstance(w, tk.Text):
                    return w.get("1.0", "end")
            raise AssertionError("no output pane found")

        def set_choice(self, label, value):
            """
            Set a dropdown by its row label.

            set_field only walks Entry widgets, so a Combobox input is
            invisible to it — and the battery tab now has two.
            """
            for w in self.widgets():
                if not (isinstance(w, ttk.Combobox) and w.winfo_manager() == "grid"):
                    continue
                row = w.grid_info().get("row")
                for lbl in w.master.winfo_children():
                    if (isinstance(lbl, ttk.Label) and str(lbl.cget("text")) == label
                            and lbl.grid_info().get("row") == row):
                        w.set(str(value))
                        self.pump()
                        return True
            raise AssertionError(f"no dropdown labelled {label!r}")

        def set_type(self, config_type):
            for w in self.widgets():
                if isinstance(w, ttk.Combobox) and "tiltrotor" in str(w.cget("values")):
                    w.set(config_type)
                    self.pump()
                    return
            raise AssertionError("configuration dropdown not found")

        def invoke_menu(self, menu_label, item_label):
            """Fire a menu command by its labels, as a user would."""
            import tkinter as tk
            menubar = root.nametowidget(root.cget("menu"))

            def find(menu, wanted):
                for i in range(menu.index("end") + 1):
                    try:
                        if menu.type(i) in ("command", "cascade") \
                                and menu.entrycget(i, "label") == wanted:
                            return i
                    except tk.TclError:
                        pass
                raise AssertionError(f"no menu entry {wanted!r}")

            submenu = root.nametowidget(
                menubar.entrycget(find(menubar, menu_label), "menu"))
            errors.clear()
            submenu.invoke(find(submenu, item_label))
            self.pump()
            return list(errors)

        def label_matching(self, fragment):
            return [str(w.cget("text")) for w in self.widgets()
                    if isinstance(w, ttk.Label) and fragment in str(w.cget("text"))]

    harness = Harness()
    yield harness
    try:
        root.destroy()
    except Exception:
        pass


MISSION = os.path.join(ROOT, "examples", "missions", "vtol_01_lift_cruise_survey.json")


def _load_mission(gui):
    """
    Click the MISSION Browse button specifically.

    Since the bench-table and SoC-curve pickers were added, several buttons
    start with "Browse" — the mission one is the plain "Browse..." with three
    dots, the pickers use a single ellipsis character. Matching the whole
    caption avoids picking whichever happens to come first.
    """
    import tkinter.ttk as ttk
    gui.open_with(MISSION)
    for w in gui.widgets():
        if isinstance(w, ttk.Button) and str(w.cget("text")) == "Browse...":
            w.invoke()
            gui.pump()
            return
    raise AssertionError("mission Browse button not found")


@pytest.mark.gui
def test_all_display_tabs_are_present(gui):
    """The VTOL should offer the same tabs as the other two simulators."""
    import tkinter.ttk as ttk
    notebooks = [w for w in gui.widgets() if isinstance(w, ttk.Notebook)]
    titles = [nb.tab(i, "text") for nb in notebooks for i in range(len(nb.tabs()))]
    for expected in ("Status", "Mission Plots", "Weight Budget", "Power Budget",
                     "Mission Diagram", "Sensitivity", "Compare", "Wiring"):
        assert expected in titles, f"{expected} tab missing"


@pytest.mark.gui
@pytest.mark.parametrize("config_type", ["lift+cruise", "tiltrotor", "tiltwing", "tailsitter"])
def test_every_type_runs_both_paths_in_the_gui(gui, config_type):
    """Each configuration must survive a single point and a mission."""
    gui.set_type(config_type)
    assert gui.click("Fixed Speed Sweep") == []
    _load_mission(gui)
    assert gui.click("Run Mission") == []


@pytest.mark.gui
def test_optional_inputs_reach_the_model(gui):
    """
    Regression, from the multicopter: the Wiring tab's fields were collected
    into widgets but never passed to build_config, so typing a wire length
    changed nothing and the Power Budget showed no wiring row.

    The check is that the ANSWER moves, not that the widget exists.
    """
    gui.use_advanced_inputs()
    gui.click("Fixed Speed Sweep")
    before = {r[0]: r[1] for r in gui.power_budget_rows()}

    gui.set_field("Wire length one-way (m)", "3.0")
    gui.set_field("Wire gauge (AWG)", "16")
    assert gui.click("Fixed Speed Sweep") == []
    after = {r[0]: r[1] for r in gui.power_budget_rows()}

    assert any("wire" in name.lower() for name in after), \
        "no wiring row in the Power Budget after entering a wire run"
    assert after["TOTAL from cells"] != before["TOTAL from cells"], \
        "the wire run cost nothing — the input never reached the model"


@pytest.mark.gui
def test_battery_ratings_reach_the_status_checks(gui):
    """A C-rate typed into the Battery tab must become a limit on Status."""
    # The C-rate fields are advanced, so they are hidden in the default
    # Simple view — switch before trying to type into them.
    gui.use_advanced_inputs()
    gui.click("Fixed Speed Sweep")
    unrated = [r for r in gui.rows("metric", 4) if r[0] == "Hover C-rate"]
    assert unrated and "Not Specified" in unrated[0][2]

    gui.set_field("Continuous C-rate", "15")
    gui.set_field("Max / burst C-rate", "30")
    gui.click("Fixed Speed Sweep")
    rated = [r for r in gui.rows("metric", 4) if r[0] == "Hover C-rate"]
    assert rated and "cont 15" in rated[0][2], f"limit not applied: {rated}"


@pytest.mark.gui
def test_a_mission_clears_what_only_a_single_point_can_answer(gui):
    """
    A mission has no single operating point, so the Power Budget and the
    fixed-speed plots must empty with a note rather than leave a stale sweep
    on screen looking current.
    """
    gui.click("Fixed Speed Sweep")
    assert len(gui.power_budget_rows()) > 3, "power budget did not fill"

    _load_mission(gui)
    gui.click("Run Mission")

    budget = gui.power_budget_rows()
    assert len(budget) == 1 and "Mission run" in budget[0][0]
    assert gui.label_matching("come from a fixed speed sweep"), \
        "fixed speed plots were not cleared"


@pytest.mark.gui
def test_status_switches_between_point_and_worst_case(gui):
    """
    Single point reports the cruise condition; a mission reports the WORST
    value each check reached. Neither is an average, and the tab says which
    it is showing.
    """
    gui.click("Fixed Speed Sweep")
    assert any("Cruise pack current" == r[0] for r in gui.rows("metric", 4))
    assert gui.label_matching("Fixed speed run —")

    _load_mission(gui)
    gui.click("Run Mission")
    worst = [r[0] for r in gui.rows("metric", 4)]
    assert "Peak pack current" in worst and "Lowest reserve margin" in worst
    assert gui.label_matching("Mission run — every row is the WORST")


@pytest.mark.gui
def test_comparison_moves_between_two_mission_runs(gui):
    """
    Regression, from the fixed-wing: the mission path refreshed Compare
    BEFORE storing the new result, so every delta read zero while Status
    plainly showed the run had changed.
    """
    _load_mission(gui)
    gui.click("Run Mission")
    gui.click("Pin Current")

    gui.set_field("Payload mass (g)", "1500")
    assert gui.click("Run Mission") == []

    rows = gui.rows("metric", 5)
    assert rows, "comparison table is empty"
    moved = [r for r in rows
             if r[3] not in ("+0", "+0.0", "+0.00", "+0.000", "—", "")]
    assert len(moved) >= len(rows) // 2, \
        f"only {len(moved)} of {len(rows)} rows moved after a payload change"


@pytest.mark.gui
def test_comparison_refuses_to_mix_a_mission_with_a_single_point(gui):
    """They measure different things; a delta between them would be nonsense."""
    gui.click("Fixed Speed Sweep")
    gui.click("Pin Current")
    _load_mission(gui)
    gui.click("Run Mission")

    rows = gui.rows("metric", 5)
    assert rows and "cannot be compared" in rows[0][0]


@pytest.mark.gui
def test_sensitivity_clears_when_a_new_run_makes_it_stale(gui):
    """
    A sensitivity sweep belongs to the run it was computed from. Leaving the
    old rankings up after the design changes is worse than showing nothing,
    because the numbers look current.
    """
    gui.click("Fixed Speed Sweep")
    gui.click("Run Sensitivity")
    assert len(gui.sensitivity_rows()) > 3, "sensitivity did not produce rankings"

    gui.click("Fixed Speed Sweep")
    stale = gui.sensitivity_rows()
    assert len(stale) == 1 and "out of date" in stale[0][0]


@pytest.mark.gui
def test_weight_budget_flags_an_impossible_structure_mass(gui):
    """
    The all-up weight without payload includes the battery and motors, so
    the airframe is the residual. If the itemised parts exceed it the
    residual is negative — an impossible aircraft — and that must be
    visible, not hidden.
    """
    gui.set_field("Lift motor weight (g)", "2000")
    gui.click("Fixed Speed Sweep")
    structure = [r for r in gui.rows("item", 5) if r[0] == "Airframe"]
    assert structure and float(structure[0][3]) < 0
    assert gui.label_matching("impossible"), "no explanation of the negative mass"


# ======================================================================
# MEASURED PROPELLER TABLES
# ======================================================================

TABLE = os.path.join(ROOT, "tests", "data", "motor_prop_table.csv")


def _table_aircraft(vtol, **kw):
    """Sized so its hover thrust lands inside the shipped table's range."""
    return vtol.VTOLConfig(aircraft_weight_g=16000, num_lift_rotors=4,
                           lift_prop_diameter_in=22, **kw)


def test_a_table_replaces_the_figure_of_merit_estimate(vtol):
    """
    The point of a bench table: a measured efficiency instead of a guessed
    one. The two should be close but not equal — if they matched exactly the
    table would not be being read.
    """
    estimated = _table_aircraft(vtol)
    measured = _table_aircraft(vtol, lift_prop_table_csv=TABLE)
    assert measured.lift_prop_table is not None

    area = math.pi / 4.0 * (22 * 0.0254) ** 2
    thrust = estimated.weight_N / 4
    from_table = vtol.measured_lift_efficiency(measured, thrust, area)

    assert from_table is not None, "table did not cover the hover thrust"
    assert 0.2 < from_table < 0.9
    assert from_table != pytest.approx(estimated.lift_figure_of_merit)

    p_est = vtol.hover_power_W(estimated)["total_power_W"]
    p_meas = vtol.hover_power_W(measured)["total_power_W"]
    assert p_meas != pytest.approx(p_est, rel=1e-6), \
        "loading a table changed nothing — it is not reaching the power model"


def test_the_chain_reproduces_the_bench_power(vtol, tmp_path):
    """
    A table gives ELECTRICAL power; the figure of merit describes the SHAFT.
    Dividing out the ESC efficiency converts between them, so running the
    chain forward — ideal / FoM, then / esc_efficiency — must land back on
    the power the bench actually recorded. If it does not, the ESC is being
    counted twice or not at all.
    """
    cfg = _table_aircraft(vtol, lift_prop_table_csv=TABLE)
    area = math.pi / 4.0 * (22 * 0.0254) ** 2
    thrust = cfg.weight_N / 4

    bench_W = core_of(vtol).table_power_for_thrust(cfg.lift_prop_table, thrust)
    fom = vtol.measured_lift_efficiency(cfg, thrust, area)
    v_hover = math.sqrt(thrust / (2.0 * cfg.air_density * area))
    reconstructed = (thrust * v_hover / fom) / cfg.esc_efficiency

    assert reconstructed == pytest.approx(bench_W, rel=1e-9)


def core_of(vtol):
    """The shared core module the simulator imported."""
    return vtol.core


def test_outside_the_measured_range_falls_back_to_the_estimate(vtol):
    """
    A bench table says nothing about thrusts it never produced. Extrapolating
    there would turn a measurement into a guess wearing its clothes, so the
    estimate is used instead and Status says so.
    """
    light = vtol.VTOLConfig(aircraft_weight_g=1500, num_lift_rotors=4,
                            lift_prop_diameter_in=22, lift_prop_table_csv=TABLE)
    area = math.pi / 4.0 * (22 * 0.0254) ** 2
    assert vtol.measured_lift_efficiency(light, light.weight_N / 4, area) is None


def test_a_missing_table_is_an_error_not_a_silent_fallback(vtol):
    """
    A table you think is loaded but is not is worse than no table: the numbers
    look measured and are estimates. So a bad path raises immediately.
    """
    with pytest.raises((FileNotFoundError, ValueError)):
        vtol.VTOLConfig(lift_prop_table_csv="/no/such/table.csv")


# ======================================================================
# WIND
# ======================================================================

def _wind_mission(vtol, tmp_path, course=0.0):
    import json
    payload = {"reserve_percent": 20, "phases": [
        {"name": "Cruise out", "kind": "cruise", "distance": 5000,
         "speed": 20.0, "altitude": 100, "course_deg": course}]}
    path = tmp_path / "wind.json"
    path.write_text(json.dumps(payload))
    return vtol.VTOLMission.from_json(str(path))


def test_a_headwind_costs_time_and_energy_over_the_same_track(vtol, tmp_path, aircraft):
    """
    Power follows AIRSPEED; progress follows GROUNDSPEED. A leg measured over
    the ground therefore takes longer into a headwind and burns more for
    exactly the same distance — the whole reason wind matters to a mission.
    """
    mission = _wind_mission(vtol, tmp_path)
    _r, still = vtol.simulate_mission(aircraft, mission, wind_mps=0.0)
    _r, head = vtol.simulate_mission(aircraft, mission, wind_mps=6.0,
                                     wind_direction_deg=0.0)

    assert head["distance_m"] == pytest.approx(still["distance_m"]), \
        "the ground track must be unchanged — only the time to fly it moves"
    assert head["time_s"] > still["time_s"]
    assert head["energy_Wh"] > still["energy_Wh"]


def test_a_tailwind_is_the_mirror_of_a_headwind(vtol, tmp_path, aircraft):
    """Same wind, opposite direction: it should help by a similar margin."""
    mission = _wind_mission(vtol, tmp_path)
    _r, still = vtol.simulate_mission(aircraft, mission, wind_mps=6.0,
                                      wind_direction_deg=90.0)   # pure crosswind
    _r, tail = vtol.simulate_mission(aircraft, mission, wind_mps=6.0,
                                     wind_direction_deg=180.0)
    _r, head = vtol.simulate_mission(aircraft, mission, wind_mps=6.0,
                                     wind_direction_deg=0.0)
    assert tail["time_s"] < still["time_s"] < head["time_s"]
    assert tail["energy_Wh"] < head["energy_Wh"]


def test_a_crosswind_costs_something_but_less_than_a_headwind(vtol, tmp_path, aircraft):
    """
    Part of the airspeed is spent crabbing into a crosswind and does not
    contribute to progress, so it is not free — just cheaper than meeting the
    same wind head-on.
    """
    mission = _wind_mission(vtol, tmp_path)
    _r, still = vtol.simulate_mission(aircraft, mission, wind_mps=0.0)
    _r, cross = vtol.simulate_mission(aircraft, mission, wind_mps=6.0,
                                      wind_direction_deg=90.0)
    _r, head = vtol.simulate_mission(aircraft, mission, wind_mps=6.0,
                                     wind_direction_deg=0.0)
    assert still["time_s"] < cross["time_s"] < head["time_s"]


def test_holding_station_in_wind_gets_translational_lift(vtol, tmp_path, aircraft):
    """
    Holding station into wind costs LESS than hovering in still air, not more.

    This test originally asserted the opposite, on the reasoning that flying
    at the wind speed through the air must cost extra drag. The model
    disagreed, and the model was right: to stay over one spot the aircraft
    flies at the wind speed through the air, and its WING starts working. On
    the reference aircraft, hover is 585 W while holding station in an 8 m/s
    wind is 203 W, because the wing is already carrying much of the weight.

    That is translational lift, and it is why helicopters and VTOLs find
    hovering easier into a breeze than in dead calm.
    """
    import json
    payload = {"reserve_percent": 20, "phases": [
        {"name": "Hold", "kind": "hover", "duration": 300, "altitude": 50}]}
    path = tmp_path / "hold.json"
    path.write_text(json.dumps(payload))
    mission = vtol.VTOLMission.from_json(str(path))

    _r, still = vtol.simulate_mission(aircraft, mission, wind_mps=0.0)
    _r, breeze = vtol.simulate_mission(aircraft, mission, wind_mps=8.0)
    assert breeze["energy_Wh"] < still["energy_Wh"], \
        "station-keeping in wind should benefit from translational lift"

    # The benefit is bounded: it cannot cost less than fully wing-borne
    # cruise, because at that point the rotors have nothing left to unload.
    cruise_W = vtol.power_at_airspeed(aircraft, aircraft.cruise_speed_mps)["total_power_W"]
    hold_W = breeze["energy_Wh"] * 3600.0 / 300.0
    assert hold_W > cruise_W * 0.5, "implausibly cheap station-keeping"


def test_no_wind_is_exactly_the_old_behaviour(vtol, tmp_path, aircraft):
    """Zero wind must reproduce the previous numbers bit for bit."""
    mission = _wind_mission(vtol, tmp_path)
    _r, explicit = vtol.simulate_mission(aircraft, mission, wind_mps=0.0)
    _r, default = vtol.simulate_mission(aircraft, mission)
    assert explicit["energy_Wh"] == pytest.approx(default["energy_Wh"], rel=1e-12)
    assert explicit["time_s"] == pytest.approx(default["time_s"], rel=1e-12)


# ======================================================================
# AIRFRAME DIAGRAM, EXPORTS AND MENU BAR
# ======================================================================

def test_airframe_diagram_draws_for_every_configuration(vtol):
    """The layout differs by type, so every type must produce a figure."""
    for config_type in ALL_TYPES:
        fig = vtol.make_airframe_diagram_figure(vtol.VTOLConfig(config_type=config_type))
        assert fig.axes, f"{config_type} produced an empty figure"
        assert fig.axes[0].patches, f"{config_type} drew no wing or rotors"


def test_the_diagram_is_drawn_to_scale(vtol):
    """
    The point of a scale plan view is that proportions are real. Doubling the
    span must double the drawn wing, or the drawing is decoration.
    """
    from matplotlib.patches import Rectangle
    def wing_width(span):
        fig = vtol.make_airframe_diagram_figure(
            vtol.VTOLConfig(wing_span_m=span, wing_area_m2=span * 0.25))
        rects = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
        return max(r.get_width() for r in rects)

    assert wing_width(4.0) == pytest.approx(2.0 * wing_width(2.0), rel=1e-9)


def test_lift_cruise_draws_a_cruise_propeller_and_vectored_types_do_not(vtol):
    """
    A tiltrotor has no separate cruise propeller — the same rotors do both
    jobs. Drawing one would misrepresent the configuration.
    """
    from matplotlib.patches import Circle
    def circles(config_type):
        fig = vtol.make_airframe_diagram_figure(vtol.VTOLConfig(config_type=config_type))
        return [p for p in fig.axes[0].patches if isinstance(p, Circle)]

    assert len(circles("lift+cruise")) == len(circles("tiltrotor")) + 1


def test_export_sections_round_trip_through_csv_and_excel(vtol, tmp_path):
    """
    The exporters are shared with the other simulators, so what matters here
    is that a VTOL's tables survive the trip intact.
    """
    sections = [("Metrics", ["Metric", "Value"],
                 [["Hover power", "585 W"], ["Cruise power", "343 W"]]),
                ("Weight Budget", ["Component", "Each (g)", "Qty", "Total (g)"],
                 [["Battery", "1200", 1, "1200"]])]

    csv_path = tmp_path / "out.csv"
    vtol.core.export_csv(str(csv_path), sections)
    text = csv_path.read_text()
    assert "[Metrics]" in text and "[Weight Budget]" in text
    assert "585 W" in text

    xlsx_path = tmp_path / "out.xlsx"
    pytest.importorskip("openpyxl")
    vtol.core.export_excel(str(xlsx_path), sections)
    from openpyxl import load_workbook
    workbook = load_workbook(str(xlsx_path))
    assert workbook.sheetnames == ["Metrics", "Weight Budget"]
    assert workbook["Metrics"]["B2"].value == "585 W"


@pytest.mark.gui
def test_the_menu_bar_offers_the_same_commands_as_the_other_simulators(gui):
    """File / View / Help, with the three exports the VTOL previously lacked."""
    import tkinter as tk
    root = gui.root
    menubar = root.nametowidget(root.cget("menu"))

    def labels(menu):
        found = []
        for i in range(menu.index("end") + 1):
            try:
                if menu.type(i) in ("command", "cascade"):
                    found.append(menu.entrycget(i, "label"))
            except tk.TclError:
                pass
        return found

    assert labels(menubar) == ["File", "View", "Help"]
    # Index 0 is the tearoff entry, not File — ask for the entry whose label
    # actually says File rather than assuming a position.
    file_index = next(i for i in range(menubar.index("end") + 1)
                      if menubar.type(i) == "cascade"
                      and menubar.entrycget(i, "label") == "File")
    file_menu = root.nametowidget(menubar.entrycget(file_index, "menu"))
    for expected in ("Export CSV…", "Export Excel…", "Generate PDF Report…"):
        assert expected in labels(file_menu), f"{expected} missing from File"


@pytest.mark.gui
def test_exports_write_real_files_after_a_single_point(gui, tmp_path):
    """
    An export that silently writes nothing looks identical to one that works,
    so check the bytes land.
    """
    import tkinter.filedialog as filedialog
    gui.click("Fixed Speed Sweep")

    target = tmp_path / "out.csv"
    filedialog.asksaveasfilename = lambda *a, **k: str(target)
    gui.invoke_menu("File", "Export CSV…")

    assert target.exists() and target.stat().st_size > 0
    text = target.read_text()
    # Status is exported as one section per sub-table, the same grouping the
    # screen shows, so an export and a screenshot cannot disagree.
    for section in ("[Metrics]", "[Battery Status]", "[Motor / ESC Status]",
                    "[Rotor / Propeller Status]", "[Aerodynamic Status]",
                    "[Weight Budget]", "[Power Budget]", "[Speed Sweep]"):
        assert section in text, f"{section} missing from the export"


@pytest.mark.gui
def test_a_mission_export_carries_phases_and_drops_the_power_budget(gui, tmp_path):
    """
    A mission has no single operating point, so exporting a power budget for
    it would be exporting a number the screen deliberately refuses to show.
    """
    import tkinter.filedialog as filedialog
    _load_mission(gui)
    gui.click("Run Mission")

    target = tmp_path / "mission.csv"
    filedialog.asksaveasfilename = lambda *a, **k: str(target)
    gui.invoke_menu("File", "Export CSV…")

    text = target.read_text()
    assert "[Mission]" in text
    assert "[Power Budget]" not in text
    assert "[Speed Sweep]" not in text


@pytest.mark.gui
def test_the_airframe_diagram_survives_both_run_types(gui):
    """
    The diagram describes the aircraft, not a flight, so neither run type
    should clear it — unlike the Power Budget or the mission plots.
    """
    # A hidden placeholder still exists in the widget tree with its text, so
    # matching on text alone would pass whether or not it is on screen. Ask
    # whether it is actually MAPPED.
    def placeholder_visible():
        import tkinter.ttk as ttk
        for w in gui.widgets():
            if isinstance(w, ttk.Label) and "draw the airframe" in str(w.cget("text")):
                return bool(w.winfo_manager())
        return False

    assert placeholder_visible(), "expected the placeholder before any run"
    gui.click("Fixed Speed Sweep")
    assert not placeholder_visible(), "diagram did not draw after a single point"
    _load_mission(gui)
    gui.click("Run Mission")
    assert not placeholder_visible(), "a mission cleared the airframe diagram"


# ======================================================================
# SIMPLE / ADVANCED AND CLI PARITY
# ======================================================================

@pytest.mark.gui
def test_simple_mode_hides_inputs_without_losing_them(gui):
    """
    Simple mode is a VIEW setting. It must hide advanced inputs and restore
    them exactly, and must never change a computed result — a hidden field
    keeps its value.
    """
    import tkinter.ttk as ttk

    def visible():
        return sum(1 for w in gui.widgets()
                   if isinstance(w, ttk.Entry) and w.winfo_manager())

    def radio(label):
        for w in gui.widgets():
            if isinstance(w, ttk.Radiobutton) and str(w.cget("text")) == label:
                return w
        raise AssertionError(f"no {label!r} radio")

    simple_count = visible()
    radio("Advanced").invoke()
    gui.pump()
    advanced_count = visible()
    assert advanced_count > simple_count, "Advanced revealed nothing"

    gui.click("Fixed Speed Sweep")
    advanced_rows = {r[0]: r[1] for r in gui.rows("metric", 4)}

    radio("Simple").invoke()
    gui.pump()
    assert visible() == simple_count, "returning to Simple did not restore the view"

    gui.click("Fixed Speed Sweep")
    simple_rows = {r[0]: r[1] for r in gui.rows("metric", 4)}
    assert simple_rows == advanced_rows, \
        "the answer changed with the view setting — hidden fields lost values"


@pytest.mark.slow
def test_cli_accepts_every_optional_input():
    """
    Regression in spirit: the wiring, connector and rating inputs were
    GUI-only when first added, so a config saved from the GUI described an
    aircraft the CLI could not express. Every one now has a flag.
    """
    result = subprocess.run(
        [sys.executable, VTOL_SCRIPT,
         "--wire_length", "2", "--wire_awg", "14",
         "--battery_c_cont", "15", "--battery_c_max", "30",
         "--connector_batt_cont", "60", "--connector_batt_max", "90",
         "--lift_motor_max_power", "400", "--hover_download", "0.12",
         "--lift_prop_weight", "40", "--avionics_mass", "250"],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output[-400:]
    assert result.returncode == 0, output[-400:]


@pytest.mark.slow
def test_cli_download_override_changes_hover_power():
    """A flag that parses but does nothing is worse than no flag."""
    def hover_power(extra):
        result = subprocess.run(
            [sys.executable, VTOL_SCRIPT] + extra,
            capture_output=True, text=True, timeout=300, cwd=ROOT)
        for line in result.stdout.splitlines():
            if "Hover power" in line:
                return float(re.search(r"([\d.]+)\s*W", line).group(1))
        raise AssertionError(f"no hover power in output:\n{result.stdout[-300:]}")

    # Download is modelled for the vectored types only, so the flag must be
    # tested on one of those — on lift+cruise it is correctly a no-op.
    baseline = hover_power(["--config_type", "tiltrotor"])
    heavier = hover_power(["--config_type", "tiltrotor", "--hover_download", "0.25"])
    assert heavier > baseline, "the download override did not reach the model"

    # Lift+cruise models it too now, continuously across the transition.
    flat = hover_power(["--hover_download", "0.25"])
    plain = hover_power([])
    assert flat > plain, "download should apply to lift+cruise as well"


# ======================================================================
# TRANSIENTS AND THE MEASURED SoC CURVE
# ======================================================================

SOC_CURVE = os.path.join(ROOT, "tests", "data", "soc_curve_lipo.csv")


def _legs_mission(vtol, tmp_path):
    """Short legs with a speed change at each end — where transients bite."""
    import json
    phases = [{"name": "Climb", "kind": "climb", "duration": 30,
               "climb_rate_mps": 2.0, "altitude": 60}]
    for i in range(4):
        phases.append({"name": f"Leg {i+1}", "kind": "cruise", "distance": 400,
                       "speed": 24.0, "altitude": 60})
        phases.append({"name": f"Turn {i+1}", "kind": "cruise", "distance": 60,
                       "speed": 14.0, "altitude": 60})
    path = tmp_path / "legs.json"
    path.write_text(json.dumps({"reserve_percent": 20, "phases": phases}))
    return vtol.VTOLMission.from_json(str(path))


def test_transients_change_the_flight_without_changing_the_track(vtol, tmp_path, aircraft):
    """
    An acceleration limit makes the flight take LONGER over the same ground.

    This test used to assert it also costs more energy. Time-stepping showed
    that is not generally true, and the old model only made it look so by
    charging a lead-in ON TOP of a full-speed leg. Once the ramp is part of
    the leg, the aircraft spends that time at a lower speed — and a VTOL's
    power rises steeply with speed, 138 W at 14 m/s against 430 W at 24 —
    so flying slower for part of the leg can more than repay the kinetic
    cost. On this pattern it does: 17.26 Wh becomes 16.90 Wh.

    So the honest invariants are the track and the time, not the energy.
    """
    mission = _legs_mission(vtol, tmp_path)
    _r, instant = vtol.simulate_mission(aircraft, mission)
    _r, ramped = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.5)

    assert ramped["time_s"] > instant["time_s"], \
        "an acceleration limit must make the flight take longer"
    assert ramped["distance_m"] == pytest.approx(instant["distance_m"], rel=1e-6), \
        "the ground track must not change with the acceleration limit"
    assert ramped["energy_Wh"] != pytest.approx(instant["energy_Wh"], rel=1e-6), \
        "the transient had no effect on energy at all"


def test_gentler_acceleration_costs_more_than_brisk(vtol, tmp_path, aircraft):
    """A lower limit means longer spent off the commanded speed."""
    mission = _legs_mission(vtol, tmp_path)
    _r, brisk = vtol.simulate_mission(aircraft, mission, max_accel_mps2=3.0)
    _r, gentle = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.0)
    assert gentle["energy_Wh"] > brisk["energy_Wh"]


def test_regen_recovers_some_braking_energy(vtol, tmp_path, aircraft):
    """
    Propellers regenerate poorly, so the default recovers nothing. Asking for
    recovery must reduce the bill, and never below the no-braking case.
    """
    mission = _legs_mission(vtol, tmp_path)
    _r, none = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.0)
    _r, some = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.0,
                                     regen_eff=0.3)
    assert some["energy_Wh"] < none["energy_Wh"]
    assert some["time_s"] == pytest.approx(none["time_s"], rel=1e-9), \
        "recovering energy must not change how long the flight takes"


def test_zero_acceleration_is_exactly_the_old_behaviour(vtol, tmp_path, aircraft):
    """The default must reproduce the previous numbers bit for bit."""
    mission = _legs_mission(vtol, tmp_path)
    _r, default = vtol.simulate_mission(aircraft, mission)
    _r, explicit = vtol.simulate_mission(aircraft, mission, max_accel_mps2=0.0)
    assert default["energy_Wh"] == pytest.approx(explicit["energy_Wh"], rel=1e-12)


def test_transient_distance_counts_toward_the_leg(vtol, tmp_path, aircraft):
    """
    A 400 m leg that spends 120 m accelerating has 280 m left, not another
    400. Adding the lead-in on top inflated both distance and energy — it did
    exactly that when first written.
    """
    import json
    path = tmp_path / "one_leg.json"
    path.write_text(json.dumps({"reserve_percent": 20, "phases": [
        {"name": "Long leg", "kind": "cruise", "distance": 4000,
         "speed": 22.0, "altitude": 60}]}))
    mission = vtol.VTOLMission.from_json(str(path))

    _r, ramped = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.0)
    assert ramped["distance_m"] == pytest.approx(4000.0, rel=1e-6), \
        "the leg flew further than it was asked to"


def test_a_leg_too_short_to_reach_its_speed_is_flagged(vtol, tmp_path, aircraft):
    """
    At 1 m/s^2 an aircraft cannot slow from 24 to 14 m/s inside 60 m — it
    needs about 190.

    Under the phase-level model the leg overshot its own distance to reach
    the speed, and the status said so. Time-stepping fixed the cause: a leg
    now ENDS when its distance is covered, so the aircraft simply arrives
    still going too fast. That is the real behaviour, and the status reports
    the speed it actually ended at.
    """
    mission = _legs_mission(vtol, tmp_path)
    results, totals = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.0)

    flagged = [row[0] for row in results if "could not reach" in str(row[-1])]
    assert flagged, "no leg flagged despite an impossible deceleration"
    assert all(name.startswith("Turn") for name in flagged), \
        f"the wrong legs were flagged: {flagged}"

    # And the distance is now honest: the legs sum to what was asked for,
    # rather than growing because a lead-in ran past the end of its leg.
    asked_m = sum(p.distance_m or 0.0 for p in mission.phases)
    assert totals["distance_m"] == pytest.approx(asked_m, rel=1e-6)


def test_a_measured_soc_curve_outranks_the_chemistry_preset(vtol):
    """
    A measured discharge curve should replace the generic shape, so the sag
    near the end of the pack comes from these cells rather than a preset.
    """
    preset = vtol.VTOLBattery()
    measured = vtol.VTOLBattery(soc_curve_csv=SOC_CURVE)

    assert preset.soc_model_source.startswith("preset")
    assert "csv" in measured.soc_model_source
    assert measured.ocv_cell_bp != preset.ocv_cell_bp, \
        "the curve loaded but left the breakpoints unchanged"

    import numpy as np
    for soc in (1.0, 0.5, 0.1):
        from_preset = float(np.interp(soc, preset.soc_bp, preset.ocv_cell_bp))
        from_curve = float(np.interp(soc, measured.soc_bp, measured.ocv_cell_bp))
        assert from_curve != pytest.approx(from_preset, abs=1e-6), \
            f"identical cell voltage at SoC {soc} — the curve is not in use"


def test_a_missing_soc_curve_is_an_error_not_a_silent_preset(vtol):
    """As with the propeller tables: a curve you think is loaded but is not."""
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        vtol.VTOLBattery(soc_curve_csv="/no/such/curve.csv")


def _round_trip_mission(vtol, tmp_path):
    """Hover, transition out, cruise, transition back, land."""
    import json
    path = tmp_path / "round_trip.json"
    path.write_text(json.dumps({"reserve_percent": 20, "phases": [
        {"name": "Climb", "kind": "climb", "duration": 20,
         "climb_rate_mps": 2.0, "altitude": 60},
        {"name": "Transition out", "kind": "transition", "duration": 12,
         "speed": 22.0, "altitude": 60},
        {"name": "Cruise", "kind": "cruise", "distance": 2000,
         "speed": 22.0, "altitude": 60},
        {"name": "Transition back", "kind": "transition", "duration": 12,
         "speed": 0.0, "altitude": 60},
        {"name": "Land", "kind": "descend", "duration": 20, "altitude": 0}]}))
    return vtol.VTOLMission.from_json(str(path))


def test_the_transition_hands_thrust_from_rotors_to_wing(vtol, aircraft):
    """
    The thing a transition model is FOR: as speed builds, the rotors give up
    the lifting and the wing takes it, while a cruise thrust appears to beat
    drag. All three must move together and end where they should.
    """
    speed = vtol.transition_speed_mps(aircraft)
    samples = [vtol.power_at_airspeed(aircraft, max(speed * f, 1e-6))
               for f in (0.0, 0.25, 0.5, 0.75, 1.0)]

    rotor = [s["rotor_thrust_N"] for s in samples]
    wing = [s["wing_lift_N"] for s in samples]

    assert rotor == sorted(rotor, reverse=True), f"rotor thrust not falling: {rotor}"
    assert wing == sorted(wing), f"wing lift not rising: {wing}"
    assert rotor[-1] == pytest.approx(0.0, abs=1e-6), "rotors still lifting at transition speed"
    assert wing[-1] == pytest.approx(aircraft.weight_N, rel=1e-6), "wing not carrying the aircraft"


def test_the_transition_pays_for_its_acceleration(vtol, tmp_path, aircraft):
    """
    A transition is not a quasi-static sweep through speeds: the aircraft is
    ACCELERATING from hover to flying speed, and that kinetic energy comes
    from the pack.

    It was missing entirely while the cruise legs already paid it — and it is
    not small. Reaching 22 m/s costs about a third of the whole phase, so a
    shorter transition must draw noticeably more power than a longer one
    covering the same speed change.
    """
    import json

    def transition_power(duration):
        path = tmp_path / f"t{duration}.json"
        path.write_text(json.dumps({"reserve_percent": 20, "phases": [
            {"name": "Transition", "kind": "transition", "duration": duration,
             "speed": 22.0, "altitude": 60}]}))
        results, _totals = vtol.simulate_mission(
            aircraft, vtol.VTOLMission.from_json(str(path)))
        return results[0][3]

    quick, slow = transition_power(8), transition_power(20)
    assert quick > slow, \
        "a faster transition must draw more power — the kinetic term is missing"

    # The gap is roughly the kinetic energy spread over the two times. The
    # tolerance is wide because the pack's own I^2 R loss is charged per step
    # and scales with the square of the current, so the quicker transition
    # pays more than its share of that too — the gap is bounded below by the
    # kinetic term, not equal to it.
    kinetic_J = 0.5 * (aircraft.all_up_weight_g / 1000.0) * 22.0 ** 2
    expected_gap = (kinetic_J / 8.0 - kinetic_J / 20.0) / aircraft.esc_efficiency
    assert quick - slow > expected_gap * 0.75
    assert quick - slow < expected_gap * 2.0


def test_a_landing_transition_is_not_charged_as_an_acceleration(vtol, tmp_path, aircraft):
    """
    Regression: both transitions used to integrate 0 -> v_end, so the landing
    transition was modelled as another acceleration and the kinetic cost was
    charged TWICE per round trip instead of once out and released on the way
    back.
    """
    mission = _round_trip_mission(vtol, tmp_path)
    results, _totals = vtol.simulate_mission(aircraft, mission)
    out = next(r for r in results if r[0] == "Transition out")
    back = next(r for r in results if r[0] == "Transition back")

    assert back[3] < out[3], (
        f"the landing transition ({back[3]:.0f} W) costs as much as the "
        f"outbound one ({out[3]:.0f} W) — it is being charged for "
        "accelerating again")


def test_regen_applies_to_the_decelerating_transition_only(vtol, tmp_path, aircraft):
    """
    There is nothing to recover while speeding up. Recovery must show on the
    way back down and nowhere else.
    """
    mission = _round_trip_mission(vtol, tmp_path)
    plain, _t = vtol.simulate_mission(aircraft, mission, regen_eff=0.0)
    recovered, _t2 = vtol.simulate_mission(aircraft, mission, regen_eff=0.5)

    out_plain = next(r for r in plain if r[0] == "Transition out")
    out_regen = next(r for r in recovered if r[0] == "Transition out")
    back_plain = next(r for r in plain if r[0] == "Transition back")
    back_regen = next(r for r in recovered if r[0] == "Transition back")

    assert out_regen[3] == pytest.approx(out_plain[3], rel=1e-9), \
        "regen changed the accelerating transition, where there is nothing to recover"
    assert back_regen[3] < back_plain[3], "regen did not reduce the deceleration"


# ======================================================================
# REAL AIRCRAFT — validated against published specifications
# ======================================================================
#
# These two configs describe aircraft that exist and whose manufacturers
# publish endurance figures, so the model can be checked against something
# it did not choose. That is worth more than any number of self-consistent
# examples: an invented aircraft can only confirm that the code is
# self-consistent, never that it is right.
#
# Airframe-level specs (span, MTOW, payload, battery chemistry and
# configuration, cruise speed, endurance) are published. Wing area, drag and
# motor electrical parameters are NOT, and are marked as inferred in each
# file. CD0 in particular is back-solved from the published endurance — so
# these tests pin the CALIBRATION, and will fail if a physics change moves
# the model away from the real aircraft.

REAL_VTOLS = [
    ("vtol_trinity_f90_lift_cruise.json", "lift+cruise", 90, 17.0),
    ("vtol_wingtraone_gen2_tailsitter.json", "tailsitter", 59, 16.0),
]


def _load_vtol_config(vtol, filename, **overrides):
    import json
    path = os.path.join(ROOT, "examples", "configs", filename)
    payload = json.load(open(path))
    g = dict(payload["vars"])
    g.update({k: str(x) for k, x in overrides.items()})

    def f(key, default=0.0):
        raw = str(g.get(key, "")).strip()
        return float(raw) if raw else default

    battery = vtol.VTOLBattery(
        chemistry=g["chem"], cell_capacity_mAh=f("cell_cap"),
        series_cells=int(f("series")), parallel_cells=int(f("parallel")),
        cell_weight_g=f("cell_wt"), voltage_min=f("vmin"),
        voltage_nominal=f("vnom"), voltage_max=f("vmax"),
        resistance_cell_mOhm=f("rcell"), usable_percent=f("usable"),
        discharge_c_cont=f("c_cont") or None, discharge_c_max=f("c_max") or None)
    return vtol.VTOLConfig(
        config_type=payload["config_type"], aircraft_weight_g=f("weight"),
        payload_mass_g=f("payload"), wing_span_m=f("span"),
        wing_area_m2=f("area"), CD0=f("cd0"), oswald=f("oswald"),
        CL_max=f("clmax"), CL_cruise_max=f("clcruise"),
        num_lift_rotors=int(f("n_lift")), lift_prop_diameter_in=f("lift_d"),
        lift_prop_pitch_in=f("lift_p"), lift_motor_kv=f("lift_kv"),
        lift_motor_resistance=f("lift_rm"), lift_motor_weight_g=f("lift_wt"),
        lift_figure_of_merit=f("fom"), num_cruise_motors=int(f("n_cruise")),
        cruise_prop_diameter_in=f("cruise_d"), cruise_prop_pitch_in=f("cruise_p"),
        cruise_motor_kv=f("cruise_kv"), cruise_motor_resistance=f("cruise_rm"),
        cruise_motor_weight_g=f("cruise_wt"),
        cruise_prop_efficiency=f("cruise_eff"),
        stopped_rotor_drag_area_m2=f("stopped_area") or None,
        battery=battery, avionics_power_W=f("avionics"),
        esc_efficiency=f("esc_eff"), cruise_speed_mps=f("cruise_v"),
        reference_altitude_m=f("alt"))


def _flight_time_to_reserve(vtol, cfg, cruise_v, tmp_path):
    """Total flight time on a climb/transition/cruise/land profile."""
    import json
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"reserve_percent": 15, "phases": [
        {"name": "Climb", "kind": "climb", "duration": 45,
         "climb_rate_mps": 2.5, "altitude": 110},
        {"name": "Transition", "kind": "transition", "duration": 10,
         "speed": cruise_v, "altitude": 110},
        {"name": "Survey", "kind": "cruise", "duration": 10000,
         "speed": cruise_v, "altitude": 110},
        {"name": "Transition back", "kind": "transition", "duration": 10,
         "speed": 0, "altitude": 110},
        {"name": "Land", "kind": "descend", "duration": 45, "altitude": 0}]}))
    _r, totals = vtol.simulate_mission(cfg, vtol.VTOLMission.from_json(str(path)))
    series = totals["series"]
    for i, left in enumerate(series["energy_remaining_Wh"]):
        if left <= totals["reserve_Wh"]:
            return series["t_s"][i] / 60.0
    return totals["time_s"] / 60.0


@pytest.mark.parametrize("filename,config_type,published_min,cruise_v", REAL_VTOLS)
def test_real_aircraft_reproduce_published_endurance(
        vtol, tmp_path, filename, config_type, published_min, cruise_v):
    """
    The model must land within 10% of what the manufacturer publishes, flying
    a realistic profile down to a 15% reserve.

    A wide band on purpose: the published figure is a marketing-grade number
    for an unspecified payload and profile, so agreeing to the minute would
    be luck rather than accuracy. What this catches is a physics change that
    moves the model away from real aircraft by a lot.
    """
    cfg = _load_vtol_config(vtol, filename)
    assert cfg.config_type == config_type

    flown = _flight_time_to_reserve(vtol, cfg, cruise_v, tmp_path)
    assert flown == pytest.approx(published_min, rel=0.10), (
        f"{filename}: model flies {flown:.1f} min against a published "
        f"{published_min} min")


def test_the_trinity_glide_ratio_agrees_with_its_endurance(vtol):
    """
    Quantum-Systems publish BOTH a 90 min endurance and a 14:1 glide ratio.
    They constrain the same drag, so a CD0 calibrated against one should land
    near the other — and it does, which is the strongest evidence available
    that the aerodynamics here are not merely self-consistent.

    L/D_max = 0.5 * sqrt(pi * AR * e / CD0)
    """
    cfg = _load_vtol_config(vtol, "vtol_trinity_f90_lift_cruise.json")
    aspect_ratio = cfg.wing_span_m ** 2 / cfg.wing_area_m2
    ld_max = 0.5 * math.sqrt(math.pi * aspect_ratio * cfg.oswald / cfg.CD0)

    # Published 14:1. L/D_max is an upper bound reached at the best-glide
    # speed, so the model sitting a little above a quoted figure is expected;
    # sitting below it, or far above, would not be.
    assert 14.0 <= ld_max <= 20.0, f"L/D_max {ld_max:.1f} against a published 14:1"


def test_a_tailsitter_is_draggier_than_a_clean_lift_cruise(vtol):
    """
    Calibrating both against their published endurance produced CD0 0.0235
    for the Trinity and 0.033 for the WingtraOne — the tailsitter draggier by
    40%. That is the right direction and roughly the right size: a tailsitter
    carries a bluff body, exposed motor pods and landing feet into cruise,
    where a clean pusher layout does not.

    This is a sanity check on the CALIBRATION, not on the physics. If someone
    re-tunes these files and the ordering flips, something is wrong with the
    reasoning rather than the code.
    """
    trinity = _load_vtol_config(vtol, "vtol_trinity_f90_lift_cruise.json")
    wingtra = _load_vtol_config(vtol, "vtol_wingtraone_gen2_tailsitter.json")
    assert wingtra.CD0 > trinity.CD0 * 1.2


# ======================================================================
# AVIONICS RAILS AND THE ESC TAB
# ======================================================================

def test_rails_replace_the_flat_avionics_figure(vtol):
    """
    Rails and the flat figure describe the SAME load, so rails replace it
    rather than adding to it.

    This is not hypothetical: the multicopter shipped for two releases
    counting both, which showed up as a negative "Unaccounted" row exactly
    equal to the peripheral draw. Getting it right here from the start is
    cheaper than finding it later.
    """
    flat = vtol.VTOLConfig(avionics_power_W=15.0)
    railed = vtol.VTOLConfig(avionics_power_W=15.0,
                             avionics_rails={5.0: (2.0, 0.90)})

    assert vtol.avionics_input_power_W(flat) == pytest.approx(15.0)
    # 5 V x 2 A = 10 W delivered, over a 90% converter = 11.1 W at the pack.
    assert vtol.avionics_input_power_W(railed) == pytest.approx(10.0 / 0.90)
    assert vtol.avionics_input_power_W(railed) != pytest.approx(15.0 + 10.0 / 0.90)


def test_a_rail_costs_its_regulator_loss(vtol):
    """
    The point of modelling rails at all: a converter is not free. A 5 V 2 A
    load behind a 90% BEC costs 11.1 W at the pack, not 10, and a worse
    converter costs more.
    """
    good = vtol.VTOLConfig(avionics_rails={5.0: (2.0, 0.95)})
    poor = vtol.VTOLConfig(avionics_rails={5.0: (2.0, 0.75)})
    assert vtol.avionics_input_power_W(poor) > vtol.avionics_input_power_W(good)
    assert vtol.avionics_input_power_W(poor) == pytest.approx(10.0 / 0.75)


def test_rails_reach_the_power_model(vtol):
    """An input that parses but does not change the answer is not an input."""
    flat = vtol.VTOLConfig()
    railed = vtol.VTOLConfig(avionics_rails={5.0: (2.0, 0.90), 12.0: (1.5, 0.87)})
    assert (vtol.power_at_airspeed(railed, 22.0)["total_power_W"]
            > vtol.power_at_airspeed(flat, 22.0)["total_power_W"])


def test_several_rails_add_up(vtol):
    """Each rail is its own load; two rails cost the sum of the two."""
    cfg = vtol.VTOLConfig(avionics_rails={5.0: (2.0, 0.90), 12.0: (1.5, 0.87)})
    expected = 10.0 / 0.90 + 18.0 / 0.87
    assert vtol.avionics_input_power_W(cfg) == pytest.approx(expected)


def test_esc_parameters_are_optional(vtol):
    """Blank ESC resistance and current limit must behave as before."""
    bare = vtol.VTOLConfig()
    assert bare.esc_resistance_ohm == 0.0
    assert bare.esc_max_current_A is None

    rated = vtol.VTOLConfig(esc_resistance_ohm=0.004, esc_max_current_A=60.0)
    assert rated.esc_max_current_A == 60.0
    # Neither changes the headline power on its own — they exist so Status
    # has something to check against.
    assert (vtol.power_at_airspeed(rated, 22.0)["total_power_W"]
            == pytest.approx(vtol.power_at_airspeed(bare, 22.0)["total_power_W"]))


@pytest.mark.gui
def test_the_rail_editor_reaches_the_power_budget(gui):
    """
    End to end: add a rail in the GUI, run, and the Power Budget must show
    it broken into delivered power and regulator loss — with the voltage and
    current columns the other two simulators carry.
    """
    gui.click("Add / Update Rail")
    assert gui.click("Fixed Speed Sweep") == []

    rows = gui.power_budget_rows()
    rails = [r for r in rows if "rail" in r[0].lower()]
    assert len(rails) >= 2, f"expected delivered + loss rows, got {rails}"
    assert any("delivered" in r[0].lower() for r in rails)
    assert any("loss" in r[0].lower() for r in rails)
    # Voltage and current populated, not blank.
    assert all(r[3].strip() and r[4].strip() for r in rails), \
        f"voltage/current columns empty: {rails}"


@pytest.mark.gui
def test_the_esc_and_avionics_tabs_exist(gui):
    """Both were missing entirely; the VTOL had one flat number for each."""
    import tkinter.ttk as ttk
    notebooks = [w for w in gui.widgets() if isinstance(w, ttk.Notebook)]
    input_tabs = [notebooks[0].tab(i, "text")
                  for i in range(len(notebooks[0].tabs()))]
    assert "ESC" in input_tabs
    assert "Avionics" in input_tabs


@pytest.mark.gui
def test_metrics_carries_the_explanation_column(gui):
    """
    The other two simulators explain each metric. Without it the VTOL's
    Metrics tab is a wall of numbers with no way to tell which matter.
    """
    gui.click("Fixed Speed Sweep")
    tree = gui.tree_exact("metric", "value", "note")
    assert tree.heading("note")["text"] == "What it means"

    # Metrics rows live under section nodes, so this walks rather than
    # reading the root — the root holds headings, which carry no note.
    rows = gui.metric_rows()
    explained = [r for r in rows if len(r) > 2 and str(r[2]).strip()]
    assert len(explained) >= 10, \
        f"only {len(explained)} metrics carry an explanation"


# ======================================================================
# COMPLETENESS: plot range, mass mode, rotor loading, sensitivity detail
# ======================================================================

def test_motor_current_limits_are_optional_and_reach_the_config(vtol):
    """Blank leaves them unset; entered, they become checkable limits."""
    bare = vtol.VTOLConfig()
    assert bare.lift_motor_max_current_A is None
    assert bare.cruise_motor_max_current_A is None

    rated = vtol.VTOLConfig(lift_motor_max_current_A=45.0,
                            cruise_motor_max_current_A=30.0)
    assert rated.lift_motor_max_current_A == 45.0
    assert rated.cruise_motor_max_current_A == 30.0
    # Ratings do not change the physics — they exist for Status to check.
    assert (vtol.power_at_airspeed(rated, 22.0)["total_power_W"]
            == pytest.approx(vtol.power_at_airspeed(bare, 22.0)["total_power_W"]))


def test_rated_prop_thrust_is_optional(vtol):
    """Blank means the per-rotor margin simply is not checked."""
    assert vtol.VTOLConfig().lift_prop_max_thrust_g == 0.0
    assert vtol.VTOLConfig(lift_prop_max_thrust_g=4500).lift_prop_max_thrust_g == 4500


@pytest.mark.gui
def test_sensitivity_shows_every_perturbation_level(gui):
    """
    Collapsing four perturbations into Low/High hid which DIRECTION an input
    pushes the answer, and whether it is monotonic. An input whose -20% and
    +20% both raise the output behaves very differently from one that rises
    steadily, and Low/High cannot tell them apart.
    """
    gui.click("Fixed Speed Sweep")
    gui.click("Run Sensitivity")

    rows = gui.sensitivity_rows()
    assert rows, "sensitivity produced no rows"
    assert len(rows[0]) == 7, f"expected 7 columns, got {len(rows[0])}"

    # Cruise speed should rise monotonically across the four levels, since
    # power climbs steeply with speed.
    speed = [r for r in rows if "speed" in r[0].lower()]
    assert speed, f"no cruise-speed row in {[r[0] for r in rows]}"
    values = [float(speed[0][i]) for i in (1, 2, 3, 4, 5)]
    assert values == sorted(values), \
        f"cruise speed is not monotonic across the sweep: {values}"


@pytest.mark.gui
def test_the_per_rotor_table_shows_the_hover_share_and_margin(gui):
    """
    A VTOL's lift rotors share the hover load evenly in still air — there is
    no drag-induced moment to redistribute it, because the aircraft is not
    translating. So the useful numbers are the load per rotor and the margin
    left to the propeller's rating.
    """
    gui.use_advanced_inputs()
    gui.set_field("Lift prop rated thrust (g)", "4500")
    assert gui.click("Fixed Speed Sweep") == []

    tree = gui.tree_exact("rotor", "thrust_n", "thrust_g", "share", "margin")
    rows = [tree.item(i, "values") for i in tree.get_children()]
    per_rotor = [r for r in rows if r[0] != "TOTAL"]
    assert len(per_rotor) >= 2

    shares = {r[3] for r in per_rotor}
    assert len(shares) == 1, f"rotors should share the load evenly: {shares}"
    assert all(r[4].strip().endswith("%") for r in per_rotor), \
        "no margin shown despite a rated thrust being entered"

    total = [r for r in rows if r[0] == "TOTAL"]
    assert total and float(total[0][1]) > float(per_rotor[0][1])


@pytest.mark.gui
def test_plot_settings_changes_the_sweep_range(gui):
    """
    The plot maximum is a view setting, so it must change the CURVE without
    changing any computed result.
    """
    gui.use_advanced_inputs()
    gui.click("Fixed Speed Sweep")
    before = {r[0]: r[1] for r in gui.rows("metric", 4)}

    gui.set_field("Max speed for plots (m/s)", "45")
    assert gui.click("Fixed Speed Sweep") == []
    after = {r[0]: r[1] for r in gui.rows("metric", 4)}

    assert after == before, \
        "changing the plot range changed a computed result"


@pytest.mark.gui
def test_entering_a_structure_mass_derives_the_airframe_weight(gui):
    """
    "enter airframe" runs the weight sum the other way: the user gives the
    frame mass and the all-up weight follows from it plus the itemised
    parts. Useful when the frame is known and components are still being
    chosen.
    """
    gui.use_advanced_inputs()
    gui.click("Fixed Speed Sweep")
    derived = {r[0]: r[1] for r in gui.rows("metric", 4)}

    import tkinter.ttk as ttk
    for w in gui.widgets():
        if isinstance(w, ttk.Combobox) and "enter airframe" in str(w.cget("values")):
            w.set("enter airframe")
            gui.pump()
            break
    else:
        pytest.fail("mass entry mode dropdown not found")

    gui.set_field("Airframe mass (g)", "2500")
    assert gui.click("Fixed Speed Sweep") == []
    entered = {r[0]: r[1] for r in gui.rows("metric", 4)}
    assert entered != derived, "entering a structure mass changed nothing"




# ======================================================================
# GUI PARITY — batch 1: window chrome
#
# Each of these guards a difference found by comparing the VTOL window
# against the multicopter and fixed-wing ones side by side. They are
# structural rather than numeric, but a user meets them before any number.
# ======================================================================


def _menu_labels(menu):
    """
    Every entry label, indexed by its REAL position in the menu.

    Filtering the labels into a list and then calling .index() on it is
    wrong and quietly so: a Tk menubar carries a tearoff at index 0, so the
    filtered position of "View" is 1 while its real index is 2 — and
    entrycget(1, "menu") hands back the File menu instead. Keep real
    indices; return None for entries that have no label.
    """
    import tkinter as tk
    out = []
    for i in range(menu.index("end") + 1):
        try:
            out.append(menu.entrycget(i, "label"))
        except tk.TclError:
            out.append(None)              # tearoff or separator
    return out


def _submenu(root, menu, label):
    """The child menu hanging off `label`, found by real index."""
    labels = _menu_labels(menu)
    assert label in labels, f"no menu entry {label!r} (found {labels})"
    return root.nametowidget(menu.entrycget(labels.index(label), "menu"))


def _view_menu(gui):
    root = gui.root
    menubar = root.nametowidget(root.cget("menu"))
    return _submenu(root, menubar, "View")


def _sweep_figure_height(gui):
    """
    The height in pixels of the figure on the Fixed Speed Plots tab.

    Height, not width, is the axis that moves. The plot pane stretches the
    figure to the pane's width exactly as the multicopter does, so only the
    height reflects the requested figure size — and only the height is what
    Plot Size can visibly change.

    The Tk widget FigureCanvasTkAgg creates carries no `.figure` attribute,
    so there is nothing to introspect; its requested height is figsize x dpi.
    The scrolling pane is itself a Canvas in the same tab, so take the
    tallest. Measuring the rendered widget rather than a stored figsize also
    proves the figure actually reached the screen.
    """
    import tkinter as tk
    import tkinter.ttk as ttk

    for nb in (w for w in gui.widgets() if isinstance(w, ttk.Notebook)):
        for tab_id in nb.tabs():
            if nb.tab(tab_id, "text") != "Fixed Speed Plots":
                continue
            frame = gui.root.nametowidget(tab_id)
            found = []

            def walk(widget):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Canvas):
                        found.append(child.winfo_reqheight())
                    walk(child)

            walk(frame)
            assert found, "no canvas on the Fixed Speed Plots tab"
            return max(found)
    raise AssertionError("no Fixed Speed Plots tab")


@pytest.mark.gui
def test_view_menu_offers_every_display_control(gui):
    """
    Regression: the VTOL View menu held ONLY "Window Scale".

    Someone who had set up 140% Presentation scaling in the multicopter
    opened the VTOL and found five of the six controls simply absent. The
    labels must match the other two simulators exactly, because a user
    looking for "Plot Font Size" will not find "Figure font".
    """
    got = [lbl for lbl in _menu_labels(_view_menu(gui)) if lbl]
    for expected in ("Window Scale", "Plot Size", "UI Font Size",
                     "Plot Font Size", "Quick Presets", "Reset All to Default"):
        assert expected in got, f"View menu is missing {expected!r} (has {got})"


@pytest.mark.gui
@pytest.mark.parametrize("preset", ["🗜  Compact", "⚙  Default",
                                    "📊  Presentation", "♿  Accessibility"])
def test_each_quick_preset_applies_without_error(gui, preset):
    """
    A preset sets window scale, both font sizes and the plot size at once.

    Invoking it is the only way to catch a preset that references a helper
    defined later in launch_gui: building the menu would succeed and the
    NameError would land on the user, exactly as the tooltip bug did.
    """
    presets = _submenu(gui.root, _view_menu(gui), "Quick Presets")
    gui.errors.clear()
    presets.invoke(_menu_labels(presets).index(preset))
    gui.pump()
    assert gui.errors == [], f"{preset} raised {gui.errors}"


@pytest.mark.gui
def test_reset_all_to_default_applies_without_error(gui):
    """The one non-cascade entry on the View menu still has to work."""
    view = _view_menu(gui)
    gui.errors.clear()
    view.invoke(_menu_labels(view).index("Reset All to Default"))
    gui.pump()
    assert gui.errors == []


@pytest.mark.gui
def test_plot_size_changes_the_figure_and_not_the_numbers(gui):
    """
    Plot Size must redraw the sweep bigger while every computed value stays.

    draw_plots had the figure size hard-coded at (11, 4.5), so a Plot Size
    menu bolted on beside it would have been inert — the radio button would
    tick and nothing on screen would move.
    """
    assert gui.click("Fixed Speed Sweep") == []
    before_numbers = {r[0]: r[1] for r in gui.rows("metric", 4)}
    medium = _sweep_figure_height(gui)

    sizes = _submenu(gui.root, _view_menu(gui), "Plot Size")

    # By POSITION, not by label: the sizes are tuned whenever the figure
    # layout changes, and a test that hard-codes "Large (13.5 x 5.5)" breaks
    # on a change it is not meant to be guarding.
    labels = [lbl for lbl in _menu_labels(sizes) if lbl]

    def choose(index):
        gui.errors.clear()
        sizes.invoke(index)
        gui.pump()
        assert gui.errors == [], f"{labels[index]} raised {gui.errors}"
        return _sweep_figure_height(gui)

    assert choose(len(labels) - 1) > medium, \
        "Plot Size did not enlarge the figure — draw_plots ignores the view setting"
    assert choose(0) < medium, \
        "Plot Size did not shrink the figure"

    assert {r[0]: r[1] for r in gui.rows("metric", 4)} == before_numbers, \
        "a display setting changed a computed result"


@pytest.mark.gui
def test_plot_size_is_inert_after_a_mission(gui):
    """
    A mission leaves a placeholder where the sweep was, explaining that
    these plots come from a fixed-speed run. Re-rendering on a Plot Size
    change would wipe that message and put a stale sweep back, so the
    re-render must decline when the last run was a mission.
    """
    _load_mission(gui)
    assert gui.click("Run Mission") == []
    before = gui.label_matching("fixed speed sweep")
    assert before, "no placeholder after a mission run"

    sizes = _submenu(gui.root, _view_menu(gui), "Plot Size")
    gui.errors.clear()
    # The largest entry, whatever it is currently called.
    sizes.invoke(len([lbl for lbl in _menu_labels(sizes) if lbl]) - 1)
    gui.pump()

    assert gui.errors == []
    assert gui.label_matching("fixed speed sweep") == before, \
        "Plot Size overwrote the mission placeholder with a stale sweep"


@pytest.mark.gui
def test_header_says_which_config_is_loaded(gui, tmp_path):
    """
    Regression: nothing on screen named the loaded config.

    The other two simulators put the filename in the header in blue. Without
    it, a user with several aircraft on disk has no way to tell which one
    the numbers on screen belong to.
    """
    import json
    import tkinter.ttk as ttk

    shown = [str(w.cget("text")) for w in gui.widgets() if isinstance(w, ttk.Label)]
    assert "Config:" in shown, "no 'Config:' label in the header"

    def config_text():
        for w in gui.widgets():
            if not isinstance(w, ttk.Label):
                continue
            name = str(w.cget("textvariable"))
            if not name:
                continue
            value = str(gui.root.getvar(name))
            if "config loaded" in value or value.endswith(".json"):
                return value
        return ""

    assert "no config loaded" in config_text().lower(), \
        "header should say no config is loaded before one is"

    path = tmp_path / "my_vtol.json"
    path.write_text(json.dumps({
        "schema": "vtol_power_sim_v1",
        "config_type": "lift+cruise",
        "vars": {},
    }))
    gui.open_with(path)
    assert gui.click("Load Config") == []
    assert config_text() == "my_vtol.json", \
        "header did not update to the loaded filename"


@pytest.mark.gui
def test_header_tells_the_user_tooltips_exist(gui):
    """
    66 of the 68 input fields carry a tooltip, and the sentence saying so was
    the half of the hint that got dropped: the VTOL read "Simple hides
    advanced tuning inputs." where the other two add "Hover any ? for help."
    Undiscoverable help is the same as no help.
    """
    assert gui.label_matching("Hover any ? for help"), \
        "header does not mention the ? tooltips"


@pytest.mark.gui
def test_every_input_tab_scrolls(gui):
    """
    Regression: input tabs were bare Frames.

    On a short window the lower fields of Airframe and Mission/Env could not
    be reached at all — no scrollbar, no wheel, no way down. Each tab must
    hold a Canvas and a Scrollbar, as make_scrollable_tab gives the other two.
    """
    import tkinter as tk
    import tkinter.ttk as ttk

    notebooks = [w for w in gui.widgets() if isinstance(w, ttk.Notebook)]
    assert notebooks, "no notebooks found"
    input_nb = notebooks[0]
    assert len(input_nb.tabs()) >= 9, "unexpected input tab count"

    for tab_id in input_nb.tabs():
        frame = gui.root.nametowidget(tab_id)
        title = input_nb.tab(tab_id, "text")
        kids = frame.winfo_children()
        assert any(isinstance(k, tk.Canvas) for k in kids), \
            f"the {title} tab has no scrolling canvas"
        assert any(isinstance(k, ttk.Scrollbar) for k in kids), \
            f"the {title} tab has no scrollbar"


@pytest.mark.gui
def test_input_fields_survived_the_move_into_scrollable_tabs(gui):
    """
    Wrapping each tab in a Canvas re-parents every field: the Entry's master
    becomes the inner frame rather than the tab. If a row's label and entry
    landed in different parents the Simple/Advanced toggle and the config
    round-trip would both break silently. Prove a field is still reachable
    and still reaches the model.
    """
    gui.use_advanced_inputs()
    assert gui.click("Fixed Speed Sweep") == []
    before = {r[0]: r[1] for r in gui.rows("metric", 4)}

    gui.set_field("Payload mass (g)", "2500")
    assert gui.click("Fixed Speed Sweep") == []
    after = {r[0]: r[1] for r in gui.rows("metric", 4)}

    assert after != before, "a field edited inside a scrollable tab changed nothing"


@pytest.mark.gui
def test_bottom_buttons_are_in_the_same_order_as_the_other_simulators(gui):
    """
    Regression: the right-hand group was packed with side="right", which
    reverses it. The bar read Export CSV, Export Excel, Generate Report,
    Load Config, Save Config — exactly backwards from the multicopter and
    fixed-wing, so muscle memory put the pointer on the wrong button.
    """
    import tkinter.ttk as ttk

    wanted = ("Run Fixed Speed Sweep", "Run Mission", "Save Config",
              "Load Config", "Export CSV", "Export Excel", "Generate Report")

    placed = []
    for w in gui.widgets():
        if not isinstance(w, ttk.Button):
            continue
        info = w.grid_info()
        if not info or info.get("row") != 0 or info.get("column") is None:
            continue
        text = str(w.cget("text"))
        if any(k in text for k in wanted):
            placed.append((int(info["column"]), text))

    bar = [text for _col, text in sorted(placed)]
    assert len(bar) == len(wanted), f"expected {len(wanted)} bar buttons, found {bar}"
    for want, got in zip(wanted, bar):
        assert want in got, f"expected {want!r} at this position, found {got!r}"


# ======================================================================
# BATCH 2: the tabs that looked unfinished
#
# Every test here asserts the behaviour a user would see, not that a
# widget was constructed. A LabelFrame can exist and stay empty; a
# section node can exist and hold nothing.
# ======================================================================


def _tab_frame(gui, title):
    """The frame behind an output-notebook tab, by its label."""
    import tkinter.ttk as ttk
    for nb in (w for w in gui.widgets() if isinstance(w, ttk.Notebook)):
        for tab_id in nb.tabs():
            if nb.tab(tab_id, "text") == title:
                return gui.root.nametowidget(tab_id)
    raise AssertionError(f"no tab titled {title!r}")


def _descendants(widget):
    out = [widget]
    for child in widget.winfo_children():
        out.extend(_descendants(child))
    return out


def _labelframe_titles(gui, tab_title):
    import tkinter.ttk as ttk
    return [str(w.cget("text"))
            for w in _descendants(_tab_frame(gui, tab_title))
            if isinstance(w, ttk.LabelFrame)]


def _has_figure(gui, tab_title):
    """
    True when a matplotlib canvas is actually packed into this tab.

    FigureCanvasTkAgg produces a plain tk.Canvas, so the test looks for a
    Canvas that is not one of the scrolling panes — a scroll pane has a
    scrollregion set, a figure canvas does not.
    """
    import tkinter as tk
    found = []
    for w in _descendants(_tab_frame(gui, tab_title)):
        if isinstance(w, tk.Canvas) and not str(w.cget("scrollregion")).strip():
            found.append(w)
    return bool(found)


# ---------------------------------------------------------------- item 6

@pytest.mark.gui
def test_weight_budget_draws_a_distribution_chart(gui):
    """
    Regression: the Weight Budget was a table against empty space.

    The other two simulators put a share chart beside it. Checking for the
    rendered canvas rather than the frame matters — an empty LabelFrame
    titled "Weight Distribution" is exactly the problem being fixed.
    """
    assert "Weight Distribution" in _labelframe_titles(gui, "Weight Budget")
    assert not _has_figure(gui, "Weight Budget"), \
        "a chart was drawn before anything had been run"

    assert gui.click("Fixed Speed Sweep") == []
    assert _has_figure(gui, "Weight Budget"), \
        "no chart on the Weight Budget after a run"


@pytest.mark.gui
def test_power_budget_draws_a_distribution_chart(gui):
    """The Power Budget gets the same treatment, from the same run."""
    assert "Power Distribution" in _labelframe_titles(gui, "Power Budget")
    assert gui.click("Fixed Speed Sweep") == []
    assert _has_figure(gui, "Power Budget"), \
        "no chart on the Power Budget after a fixed speed sweep"


@pytest.mark.gui
def test_a_mission_clears_the_power_distribution_chart(gui):
    """
    A mission has no single operating point. Leaving the sweep's power
    split on screen would show a breakdown of a condition the mission
    never held — and it would look perfectly plausible sitting there.
    """
    assert gui.click("Fixed Speed Sweep") == []
    assert _has_figure(gui, "Power Budget")

    _load_mission(gui)
    assert gui.click("Run Mission") == []
    assert not _has_figure(gui, "Power Budget"), \
        "the fixed-speed power chart survived a mission run"


@pytest.mark.gui
def test_the_weight_chart_drops_a_negative_structure_mass(gui):
    """
    Over-specified parts give a negative structure mass. The table shows
    that in red, but it cannot be a slice of a pie: charting it would
    either raise or silently draw a wedge for an impossible quantity.
    """
    gui.use_advanced_inputs()
    gui.set_field("All-up weight without payload (g)", "200")  # far below the parts
    assert gui.click("Fixed Speed Sweep") == [], "a negative structure raised"
    assert _has_figure(gui, "Weight Budget"), \
        "the chart vanished instead of dropping the impossible slice"


# ---------------------------------------------------------------- item 7

@pytest.mark.gui
def test_status_is_split_into_named_sub_tables(gui):
    """
    Regression: Status was one flat table, so the reader had to work out
    which subsystem each row belonged to. The other two group theirs.
    """
    titles = _labelframe_titles(gui, "Status")
    for expected in ("Battery Status", "Motor / ESC Status",
                     "Rotor / Propeller Status", "Aerodynamic Status"):
        assert expected in titles, f"{expected} sub-table missing (has {titles})"


@pytest.mark.gui
def test_every_status_group_is_populated_by_a_run(gui):
    """
    Four titled but empty tables would look worse than one full one. Each
    group has to actually receive checks.
    """
    assert gui.click("Fixed Speed Sweep") == []
    import tkinter.ttk as ttk
    filled = {}
    for frame in _descendants(_tab_frame(gui, "Status")):
        if not isinstance(frame, ttk.LabelFrame):
            continue
        rows = sum(len(tv.get_children(""))
                   for tv in _descendants(frame)
                   if isinstance(tv, ttk.Treeview))
        filled[str(frame.cget("text"))] = rows
    for title, count in filled.items():
        assert count > 0, f"{title} has no rows after a run"


@pytest.mark.gui
def test_status_checks_survived_the_regrouping(gui):
    """
    Moving rows between tables is where checks get dropped silently. The
    whole set has to still be there, wherever it now lives.
    """
    assert gui.click("Fixed Speed Sweep") == []
    names = [str(r[0]) for r in gui.status_rows()]
    for expected in ("Hover pack current", "Cruise pack current",
                     "Hover C-rate", "Lift motor power in hover",
                     "Transition / stall speed",
                     "Wing lift share at cruise speed",
                     "Lift rotor efficiency", "Hover download fraction"):
        assert any(expected in n for n in names), \
            f"{expected!r} disappeared in the regrouping (have {names})"


@pytest.mark.gui
def test_a_mission_says_why_the_point_checks_are_empty(gui):
    """
    A mission fills the groups it has worst-case values for — the battery,
    and, as on the multicopter, the motors, ESCs and rotors — and every group
    it has nothing for says why. A blank table with no explanation reads as
    a broken run rather than an inapplicable check.
    """
    _load_mission(gui)
    assert gui.click("Run Mission") == []
    names = [str(r[0]) for r in gui.status_rows()]
    assert any("Peak pack current" in n for n in names)
    assert any("Peak lift motor current" in n for n in names)
    assert any("Peak motor temperature" in n for n in names)
    for tree in gui.trees_exact("metric", "value", "limit", "note"):
        assert tree.get_children(""), "a Status group was left blank with nothing said"
    assert names.count("—") >= 1, \
        "the aerodynamic group has no mission value and must say so"


# ---------------------------------------------------------------- item 8

@pytest.mark.gui
def test_metrics_is_grouped_into_sections(gui):
    """
    Regression: Metrics was 24 flat rows with blank spacer rows standing
    in for separators. A spacer cannot be folded and does not say where a
    group ends.
    """
    assert gui.click("Fixed Speed Sweep") == []
    sections = gui.metric_sections()
    for expected in ("Aircraft", "Hover", "Cruise", "Battery"):
        assert expected in sections, \
            f"no {expected!r} section (has {sections})"

    rows = [str(r[0]) for r in gui.metric_rows()]
    assert "" not in rows, "a blank spacer row survived the conversion"


@pytest.mark.gui
def test_every_metrics_section_holds_rows(gui):
    """A section heading with nothing under it is worse than no heading."""
    assert gui.click("Fixed Speed Sweep") == []
    tree = gui.tree_exact("metric", "value", "note")
    for node in tree.get_children(""):
        title = tree.item(node, "values")[0]
        assert tree.get_children(node), f"section {title!r} is empty"


@pytest.mark.gui
def test_metrics_sections_can_be_folded_and_stay_folded(gui):
    """
    The point of sections is hiding the ones you do not care about. A fold
    that reopens on the next run has not saved anyone anything.
    """
    assert gui.click("Fixed Speed Sweep") == []
    tree = gui.tree_exact("metric", "value", "note")
    battery = [n for n in tree.get_children("")
               if tree.item(n, "values")[0] == "Battery"][0]
    assert tree.item(battery, "open")

    tree.item(battery, open=False)
    tree.event_generate("<<TreeviewClose>>")
    gui.pump()

    assert gui.click("Fixed Speed Sweep") == []
    tree = gui.tree_exact("metric", "value", "note")
    battery = [n for n in tree.get_children("")
               if tree.item(n, "values")[0] == "Battery"][0]
    assert not tree.item(battery, "open"), \
        "the folded section reopened when the run refreshed it"


@pytest.mark.gui
def test_metric_values_survived_the_grouping(gui):
    """
    Restructuring a table is where rows get lost. Every metric that was
    there before the sections went in still has to be there.
    """
    assert gui.click("Fixed Speed Sweep") == []
    labels = [str(r[0]).strip() for r in gui.metric_rows()]
    for expected in ("Configuration", "All-up weight", "Wing loading",
                     "Disc loading (hover)", "Aspect ratio", "Stall speed",
                     "Transition speed", "Hover power", "Hover endurance",
                     "Cruise power", "Cruise endurance", "Cruise range",
                     "Pack current", "Loaded voltage", "Usable energy",
                     "Battery weight", "SoC model"):
        assert expected in labels, f"{expected!r} lost in the grouping"


@pytest.mark.gui
def test_the_export_still_carries_metric_rows_not_just_headings(gui, tmp_path):
    """
    Regression risk from the same change: the export read the tree's root
    children, which after grouping are six section headings carrying no
    numbers at all.
    """
    import tkinter.filedialog as filedialog
    assert gui.click("Fixed Speed Sweep") == []
    target = tmp_path / "out.csv"
    filedialog.asksaveasfilename = lambda *a, **k: str(target)
    gui.invoke_menu("File", "Export CSV…")

    text = target.read_text()
    assert "Stall speed" in text, "metric rows missing from the export"
    assert "Usable energy" in text


# ------------------------------------------------------------ items 10/11

@pytest.mark.gui
def test_mission_plot_variables_cover_the_whole_model(gui):
    """
    Regression: nine variables in a three-row box. The fixed-wing offers
    28 and the multicopter 45, so the VTOL looked like a stub.
    """
    import tkinter as tk
    boxes = [w for w in gui.widgets() if isinstance(w, tk.Listbox)]
    assert boxes, "no variable list on Mission Plots"
    box = max(boxes, key=lambda b: b.size())
    assert box.size() >= 20, f"only {box.size()} mission-plot variables"
    assert int(box.cget("height")) >= 12, \
        "the variable list is still only a few rows tall"


@pytest.mark.gui
def test_every_offered_mission_variable_actually_plots(gui):
    """
    A variable in the list that the mission never records raises KeyError
    the moment a user selects it. Offering it is the bug, not the click.
    """
    import tkinter as tk
    import tkinter.ttk as ttk
    _load_mission(gui)
    assert gui.click("Run Mission") == []

    box = max((w for w in gui.widgets() if isinstance(w, tk.Listbox)),
              key=lambda b: b.size())
    for i in range(box.size()):
        box.selection_clear(0, "end")
        box.selection_set(i)
        gui.errors.clear()
        for w in gui.widgets():
            if isinstance(w, ttk.Button) and str(w.cget("text")) == "Plot selected":
                w.invoke()
                break
        gui.pump()
        assert gui.errors == [], \
            f"plotting {box.get(i)!r} raised {gui.errors}"


@pytest.mark.gui
def test_the_x_axis_label_names_the_x_axis(gui):
    """
    Regression: "X axis:" was packed immediately before the VARIABLE list,
    so on screen it appeared to label the variable selector while the real
    Time/Distance radios sat unlabelled to its right.
    """
    import tkinter as tk
    import tkinter.ttk as ttk

    label = None
    for w in _descendants(_tab_frame(gui, "Mission Plots")):
        if isinstance(w, ttk.Label) and str(w.cget("text")) == "X axis:":
            label = w
            break
    assert label is not None, "no 'X axis:' label"

    siblings = label.master.winfo_children()
    kinds = [type(s).__name__ for s in siblings]
    assert not any(isinstance(s, tk.Listbox) for s in siblings), \
        f"the variable list still sits in the X-axis bar ({kinds})"
    assert any(isinstance(s, ttk.Radiobutton) for s in siblings), \
        f"the X-axis label is not beside the Time/Distance radios ({kinds})"


@pytest.mark.gui
def test_the_variable_list_is_in_a_titled_panel(gui):
    """The list needs a heading saying what it selects, as MC and FW have."""
    titles = _labelframe_titles(gui, "Mission Plots")
    assert any("Y-axis" in t for t in titles), \
        f"the variable list has no titled panel (found {titles})"


# --------------------------------------------------------------- item 13

@pytest.mark.gui
def test_the_airframe_diagram_has_a_titled_panel_and_rotor_table(gui):
    """
    Regression: the rotor table sat bare under the plot with nothing
    saying what it was, and the drawing had no frame of its own.
    """
    titles = _labelframe_titles(gui, "Airframe Diagram")
    assert "Plan View (to scale)" in titles, f"found {titles}"
    assert "Per-Rotor Loading" in titles, f"found {titles}"


@pytest.mark.gui
def test_the_rotor_panel_explains_itself(gui):
    """
    The multicopter explains why its split is even. The VTOL's reason is
    different — no translation, so no drag moment — and a bare table of
    identical numbers invites the reader to assume it is a stub.
    """
    import tkinter.ttk as ttk
    texts = [str(w.cget("text"))
             for w in _descendants(_tab_frame(gui, "Airframe Diagram"))
             if isinstance(w, ttk.Label)]
    joined = " ".join(texts)
    assert "numbering matches the diagram" in joined, \
        "the rotor table carries no explanation"


def test_the_diagram_annotates_its_dimensions(vtol):
    """
    Regression: the drawing carried a corner text box and nothing else, so
    the reader had to match numbers to features by eye — the one thing a
    scale drawing should save them from.
    """
    cfg = _cfg(vtol, "lift+cruise")
    fig = vtol.make_airframe_diagram_figure(cfg)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    joined = " ".join(texts)
    for expected in ("span", "chord", "rotor pitch"):
        assert expected in joined, \
            f"no {expected!r} annotation on the diagram (found {texts})"
    # Only Annotation carries arrow_patch; a plain Text does not.
    arrows = [a for a in ax.texts
              if getattr(a, "arrow_patch", None) is not None]
    assert len(arrows) >= 3, \
        f"only {len(arrows)} dimension arrows drawn"


def test_the_diagram_still_draws_for_every_configuration(vtol):
    """
    The annotations index into the rotor positions, and the vectored types
    lay those out differently. An IndexError here would only appear when a
    user picked a tiltrotor.
    """
    for kind in vtol.IMPLEMENTED_CONFIG_TYPES:
        cfg = _cfg(vtol, kind)
        fig = vtol.make_airframe_diagram_figure(cfg)
        assert fig.axes, f"{kind} produced an empty figure"


@pytest.mark.gui
def test_status_sub_tables_grow_to_fit_their_rows(gui):
    """
    The sub-tables have no scrollbar of their own — the whole tab scrolls —
    so a fixed height silently hides any row past it. The battery group
    gains rows once connectors and a wire run are entered, which is exactly
    the case that would lose them.
    """
    import tkinter.ttk as ttk
    gui.use_advanced_inputs()
    gui.set_field("Wire length one-way (m)", "0.4")
    gui.set_field("Wire gauge (AWG)", "10")
    assert gui.click("Fixed Speed Sweep") == []

    for frame in _descendants(_tab_frame(gui, "Status")):
        if not isinstance(frame, ttk.LabelFrame):
            continue
        for tv in _descendants(frame):
            if not isinstance(tv, ttk.Treeview):
                continue
            rows = len(tv.get_children(""))
            assert int(tv.cget("height")) >= rows, (
                f"{frame.cget('text')} shows {tv.cget('height')} rows but "
                f"holds {rows} — the rest are invisible")


# ======================================================================
# BATCH 3: the missing modelling inputs
#
# These change NUMBERS, not appearance, so the tests assert arithmetic
# the model has to satisfy rather than that a widget exists. The GUI
# tests at the end only check the inputs reach the config.
# ======================================================================

# ------------------------------------------------- item 16: peripheral

def test_peripheral_current_costs_exactly_I_times_nominal_voltage(vtol):
    """
    A load wired straight to the pack costs I x V_nom, valued at NOMINAL
    voltage to match how the rest of the model charges the battery.

    Valuing it at the LOADED voltage instead disagrees with the model by
    the sag, which surfaces as an "Unaccounted" row in the Power Budget —
    the fault the multicopter shipped and had to fix.
    """
    base = _cfg(vtol, "lift+cruise")
    with_p = _cfg(vtol, "lift+cruise", periph_current_A=3.0)

    expected = 3.0 * base.battery.vnom_pack
    mb = vtol.compute_metrics(base)
    mp = vtol.compute_metrics(with_p)

    assert mp["total_power_W"] - mb["total_power_W"] == pytest.approx(expected, rel=1e-9)


def test_peripheral_current_is_charged_in_hover_too(vtol):
    """
    A load on the main bus does not switch off when the aircraft stops
    translating. Charging it only in cruise would make hover look cheap.
    """
    base = _cfg(vtol, "lift+cruise")
    with_p = _cfg(vtol, "lift+cruise", periph_current_A=3.0)
    expected = 3.0 * base.battery.vnom_pack

    mb = vtol.compute_metrics(base)
    mp = vtol.compute_metrics(with_p)
    assert mp["hover_power_W"] - mb["hover_power_W"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("config_type", ["lift+cruise", "tiltrotor",
                                         "tiltwing", "tailsitter"])
def test_peripheral_current_applies_to_every_configuration(vtol, config_type):
    """
    The vectored types take a different path through the power code, so a
    load added to only one branch would be silently free on the others.
    """
    base = _cfg(vtol, config_type)
    with_p = _cfg(vtol, config_type, periph_current_A=2.0)
    expected = 2.0 * base.battery.vnom_pack
    assert (vtol.compute_metrics(with_p)["total_power_W"]
            - vtol.compute_metrics(base)["total_power_W"]) == pytest.approx(expected, rel=1e-9)


def test_peripheral_current_shortens_endurance(vtol):
    """The point of the input: it has to cost flight time, not just watts."""
    base = vtol.compute_metrics(_cfg(vtol, "lift+cruise"))
    with_p = vtol.compute_metrics(_cfg(vtol, "lift+cruise", periph_current_A=5.0))
    assert with_p["cruise_endurance_min"] < base["cruise_endurance_min"]
    assert with_p["hover_endurance_min"] < base["hover_endurance_min"]


def test_peripheral_current_adds_to_rails_rather_than_replacing_them(vtol):
    """
    Rails REPLACE the flat avionics figure because they describe the same
    load. Peripheral current is a DIFFERENT load, so it must add — getting
    this backwards would silently delete the avionics.
    """
    rails = {5.0: (2.0, 0.9)}
    with_rails = _cfg(vtol, "lift+cruise", avionics_rails=rails)
    with_both = _cfg(vtol, "lift+cruise", avionics_rails=rails,
                     periph_current_A=2.0)
    expected = 2.0 * with_rails.battery.vnom_pack
    assert (vtol.compute_metrics(with_both)["total_power_W"]
            - vtol.compute_metrics(with_rails)["total_power_W"]) == pytest.approx(expected, rel=1e-9)


def test_peripheral_current_is_in_the_power_budget_and_balances(vtol):
    """
    Every watt has to land in a row. A load added to the total but not to
    the budget shows up as "Unaccounted", which is the budget telling you
    it does not know where the power went.
    """
    cfg = _cfg(vtol, "lift+cruise", periph_current_A=4.0)
    m = vtol.compute_metrics(cfg)
    pack_I = float(m["pack_current_A"])
    rows = vtol.core.build_power_budget(
        total_in_W=float(m["total_power_W"]),
        motor_shaft_W=float(m["shaft_power_W"]),
        motor_copper_W=float(m["motor_copper_W"]),
        motor_iron_W=float(m["motor_iron_W"]),
        battery_i2r_W=pack_I ** 2 * cfg.battery.pack_resistance,
        esc_loss_W=float(m["esc_loss_W"]),
        wire_loss_W=float(m.get("wire_loss_W", 0.0)),
        peripheral_W=cfg.avionics_power_W + vtol.peripheral_power_W(cfg),
        rails=None)
    unaccounted = [r for r in rows if r["name"] == "Unaccounted"]
    assert not unaccounted, \
        f"power budget does not balance: {unaccounted[0]['watts']:.2f} W adrift"


def test_zero_peripheral_current_changes_nothing(vtol):
    """The default must be inert, or every existing config silently moves."""
    a = vtol.compute_metrics(_cfg(vtol, "lift+cruise"))
    b = vtol.compute_metrics(_cfg(vtol, "lift+cruise", periph_current_A=0.0))
    assert a["total_power_W"] == pytest.approx(b["total_power_W"], rel=1e-12)


# ------------------------------------------------- item 15: battery

def test_cell_mode_is_unchanged_by_the_pack_rewrite(vtol):
    """
    The pack hierarchy went in underneath the cell-only model. In cell
    mode the per-unit counts are forced to 1, which must make the
    arithmetic identical — otherwise every stored config moves.
    """
    b = vtol.VTOLBattery(cell_capacity_mAh=5000, series_cells=6,
                         parallel_cells=2, cell_weight_g=120)
    assert b.series_cells == 6
    assert b.parallel_cells == 2
    assert b.total_cells == 12
    assert b.capacity_mAh == pytest.approx(10000.0)
    assert b.weight_g == pytest.approx(1440.0)
    assert b.vnom_pack == pytest.approx(22.2)


def test_packs_in_series_raise_voltage_not_capacity(vtol):
    """
    The single most common battery mistake. Two 6S 16000 mAh packs wired
    in series make a 12S 16000 mAh pack, NOT 12S 32000 mAh.
    """
    b = vtol.VTOLBattery(unit_mode="pack", series_cells=2, parallel_cells=1,
                         cells_series_per_unit=6, pack_capacity_mAh=16000,
                         pack_weight_g=2100)
    assert b.series_cells == 12
    assert b.capacity_mAh == pytest.approx(16000.0)
    assert b.vnom_pack == pytest.approx(44.4)


def test_packs_in_parallel_raise_capacity_not_voltage(vtol):
    """And the mirror case, which must land on the same total energy."""
    series = vtol.VTOLBattery(unit_mode="pack", series_cells=2, parallel_cells=1,
                              cells_series_per_unit=6, pack_capacity_mAh=16000,
                              pack_weight_g=2100)
    parallel = vtol.VTOLBattery(unit_mode="pack", series_cells=1, parallel_cells=2,
                                cells_series_per_unit=6, pack_capacity_mAh=16000,
                                pack_weight_g=2100)
    assert parallel.capacity_mAh == pytest.approx(32000.0)
    assert parallel.vnom_pack == pytest.approx(22.2)
    # Same cells, same energy, wired two ways.
    assert parallel.capacity_Wh == pytest.approx(series.capacity_Wh, rel=1e-9)
    assert parallel.weight_g == pytest.approx(series.weight_g, rel=1e-9)


def test_every_pack_is_carried_however_it_is_wired(vtol):
    """Weight scales with series x parallel; capacity does not."""
    b = vtol.VTOLBattery(unit_mode="pack", series_cells=2, parallel_cells=3,
                         cells_series_per_unit=6, pack_capacity_mAh=10000,
                         pack_weight_g=1000)
    assert b.weight_g == pytest.approx(6000.0)
    assert b.capacity_mAh == pytest.approx(30000.0)


def test_a_misspelled_unit_mode_does_not_silently_zero_the_pack(vtol):
    """
    "Pack" with a capital P falling through to cell mode would read the
    blank cell fields and produce a zero-capacity battery. The multicopter
    carries a comment about exactly this.
    """
    b = vtol.VTOLBattery(unit_mode="PACK", series_cells=2, parallel_cells=1,
                         cells_series_per_unit=6, pack_capacity_mAh=16000,
                         pack_weight_g=2100)
    assert b.unit_mode == "pack"
    assert b.capacity_mAh == pytest.approx(16000.0)
    assert b.weight_g > 0


def test_amps_outrank_the_c_rate(vtol):
    """
    A datasheet quoting both means the amps. Deriving C x Ah from a
    rounded C-rating throws away the figure the manufacturer actually
    guaranteed.
    """
    both = vtol.VTOLBattery(cell_capacity_mAh=5000, parallel_cells=2,
                            discharge_c_cont=10, discharge_cont_A=77)
    assert both.discharge_cont_A == pytest.approx(77.0)

    c_only = vtol.VTOLBattery(cell_capacity_mAh=5000, parallel_cells=2,
                              discharge_c_cont=10)
    assert c_only.discharge_cont_A == pytest.approx(100.0)


def test_an_unrated_pack_stays_unrated(vtol):
    """
    None, not infinity. Status reports "Not Specified" rather than passing
    a check against a limit nobody entered.
    """
    b = vtol.VTOLBattery(cell_capacity_mAh=5000, parallel_cells=2)
    assert b.discharge_cont_A is None
    assert b.discharge_max_A is None
    assert b.charge_time_h is None


def test_charge_time_is_capacity_over_current(vtol):
    b = vtol.VTOLBattery(cell_capacity_mAh=5000, parallel_cells=2,
                         charge_current_max_A=5.0)
    assert b.charge_time_h == pytest.approx(2.0)


def test_energy_density_is_derived_but_an_entered_figure_is_kept(vtol):
    """
    Both numbers are kept so they can be compared: a large gap means the
    entered cell weight or capacity is wrong.
    """
    derived = vtol.VTOLBattery(cell_capacity_mAh=5000, parallel_cells=2,
                               cell_weight_g=120)
    assert derived.energy_density_Wh_per_kg == pytest.approx(222.0 / 1.44, rel=1e-6)

    entered = vtol.VTOLBattery(cell_capacity_mAh=5000, parallel_cells=2,
                               cell_weight_g=120,
                               energy_density_Wh_per_kg=241.3)
    assert entered.energy_density_Wh_per_kg == pytest.approx(241.3)
    assert entered.energy_density_derived_Wh_per_kg == pytest.approx(222.0 / 1.44,
                                                                    rel=1e-6)


def test_soc_breakpoints_reach_the_model(vtol):
    """
    Regression: the constructor accepted no breakpoint arrays and passed
    None, None, None to the resolver, so a curve entered by hand was
    silently discarded and the chemistry preset used instead.
    """
    b = vtol.VTOLBattery(series_cells=6,
                         soc_bp=[0.0, 0.5, 1.0],
                         ocv_cell_bp=[3.4, 3.7, 4.2],
                         r_scale_bp=[1.5, 1.0, 1.0])
    assert b.soc_model_source == "custom-arrays"
    assert b.soc_nonlinear_enabled
    # 3.7 V/cell x 6 cells at half charge, from the array and not a preset.
    assert b.ocv_at_soc(0.5) == pytest.approx(22.2, rel=1e-9)
    assert b.resistance_at_soc(0.0) == pytest.approx(b.pack_resistance * 1.5, rel=1e-9)


def test_breakpoints_outrank_an_explicitly_named_preset(vtol):
    """Measured beats generic, even when the preset was asked for by name."""
    b = vtol.VTOLBattery(soc_model="LiFePO4",
                         soc_bp=[0.0, 0.5, 1.0],
                         ocv_cell_bp=[3.4, 3.7, 4.2],
                         r_scale_bp=[1.5, 1.0, 1.0])
    assert b.soc_model_source == "custom-arrays"


def test_linear_holds_voltage_at_full_charge(vtol):
    """
    The shared core's "linear" is not a ramp — it pins open-circuit
    voltage at full charge. That is optimistic near the end of the pack,
    and the tooltip says so; this pins the behaviour the tooltip claims.
    """
    b = vtol.VTOLBattery(series_cells=6, soc_model="linear")
    assert not b.soc_nonlinear_enabled
    assert b.ocv_at_soc(1.0) == pytest.approx(b.vmax_pack)
    assert b.ocv_at_soc(0.1) == pytest.approx(b.vmax_pack)


def test_a_curve_changes_the_loaded_voltage_against_linear(vtol):
    """
    The whole point of the selector: choosing a model has to move a
    number. Linear ignores sag, so it reports the higher voltage.
    """
    curved = _cfg(vtol, "lift+cruise")
    flat = _cfg(vtol, "lift+cruise")
    flat.battery = vtol.VTOLBattery(soc_model="linear")

    assert (vtol.compute_metrics(flat)["v_load_V"]
            > vtol.compute_metrics(curved)["v_load_V"])


def test_a_missing_soc_curve_file_still_fails_loudly(vtol):
    """
    Unchanged by the rewrite, and worth keeping: a curve you think is
    loaded but is not would give preset numbers wearing measured clothes.
    """
    with pytest.raises(FileNotFoundError):
        vtol.VTOLBattery(soc_curve_csv="/nonexistent/curve.csv")


def test_pack_mode_endurance_follows_the_pack_energy(vtol):
    """
    End to end: a pack-mode battery has to drive the flight numbers, not
    merely construct. Doubling the parallel packs doubles the energy and
    so, at constant power, the endurance.
    """
    one = _cfg(vtol, "lift+cruise")
    one.battery = vtol.VTOLBattery(unit_mode="pack", series_cells=1,
                                   parallel_cells=1, cells_series_per_unit=6,
                                   pack_capacity_mAh=10000, pack_weight_g=1000)
    two = _cfg(vtol, "lift+cruise")
    two.battery = vtol.VTOLBattery(unit_mode="pack", series_cells=1,
                                   parallel_cells=2, cells_series_per_unit=6,
                                   pack_capacity_mAh=10000, pack_weight_g=1000)
    # Hold the airframe weight fixed so only the pack energy differs.
    m1 = vtol.compute_metrics(one)
    m2 = vtol.compute_metrics(two)
    assert m2["usable_Wh"] == pytest.approx(2 * m1["usable_Wh"], rel=1e-9)
    assert m2["cruise_endurance_min"] == pytest.approx(
        2 * m1["cruise_endurance_min"], rel=1e-6)


# --------------------------------------------- batch 3: GUI wiring

@pytest.mark.gui
def test_peripheral_current_field_reaches_the_numbers(gui):
    """
    A field that saves and loads but never reaches the config is the
    failure mode this whole batch exists to fix — the wiring inputs
    shipped that way once already.
    """
    gui.use_advanced_inputs()
    assert gui.click("Fixed Speed Sweep") == []
    before = {r[0]: r[1] for r in gui.metric_rows()}

    gui.set_field("Peripheral current (A)", "5")
    assert gui.click("Fixed Speed Sweep") == []
    after = {r[0]: r[1] for r in gui.metric_rows()}

    assert after["Cruise power"] != before["Cruise power"], \
        "peripheral current changed nothing"
    assert after["Peripheral current"].startswith("5.00")


@pytest.mark.gui
def test_the_battery_pack_fields_reach_the_numbers(gui):
    """Pack-mode entry has to drive the pack, not just sit there."""
    gui.use_advanced_inputs()
    assert gui.click("Fixed Speed Sweep") == []
    before = {r[0]: r[1] for r in gui.metric_rows()}

    gui.set_choice("Entry mode", "pack")
    gui.set_field("Cells in series per pack", "6")
    gui.set_field("Pack capacity (mAh)", "16000")
    gui.set_field("Pack weight (g)", "2100")
    gui.set_field("Series units", "1")
    gui.set_field("Parallel units", "1")
    assert gui.click("Fixed Speed Sweep") == []
    after = {r[0]: r[1] for r in gui.metric_rows()}

    assert after["Pack capacity"].startswith("16000"), after["Pack capacity"]
    assert after["Battery weight"].startswith("2100"), after["Battery weight"]
    assert after["Usable energy"] != before["Usable energy"]


@pytest.mark.gui
def test_the_soc_model_dropdown_reaches_the_model(gui):
    """
    The resolver was always in the core and the CSV picker reached it,
    but the model selector had no field at all — 'linear' was reachable
    only from the CLI.
    """
    gui.use_advanced_inputs()
    assert gui.click("Fixed Speed Sweep") == []
    before = {r[0]: r[1] for r in gui.metric_rows()}
    assert "preset" in before["SoC model"], before["SoC model"]

    gui.set_choice("SoC model", "linear")
    assert gui.click("Fixed Speed Sweep") == []
    after = {r[0]: r[1] for r in gui.metric_rows()}

    assert after["SoC model"] != before["SoC model"]
    assert after["Loaded voltage"] != before["Loaded voltage"], \
        "switching the SoC model changed no number"


@pytest.mark.gui
def test_the_breakpoint_fields_reach_the_model(gui):
    """A curve typed by hand has to outrank the chemistry preset."""
    gui.use_advanced_inputs()
    gui.set_field("SoC breakpoints (0..1)", "0, 0.5, 1.0")
    gui.set_field("OCV per cell (V)", "3.4, 3.7, 4.2")
    gui.set_field("Resistance scale", "1.5, 1.0, 1.0")
    assert gui.click("Fixed Speed Sweep") == []

    rows = {r[0]: r[1] for r in gui.metric_rows()}
    assert "custom" in rows["SoC model"], rows["SoC model"]


@pytest.mark.gui
def test_amp_limits_reach_the_status_check(gui):
    """
    Entering a limit in amps has to produce a checked row, not the
    "Not Specified" the pack shows when nothing is rated.
    """
    gui.use_advanced_inputs()
    gui.set_field("Cont discharge (A)", "60")
    gui.set_field("Max discharge (A)", "120")
    assert gui.click("Fixed Speed Sweep") == []

    rows = {str(r[0]): str(r[2]) for r in gui.status_rows()}
    assert "Not Specified" not in rows["Hover pack current"], rows["Hover pack current"]
    assert "60" in rows["Hover pack current"]


@pytest.mark.gui
def test_a_typo_in_the_breakpoints_falls_back_rather_than_crashing(gui):
    """
    Half-entered breakpoints are the normal state of a field being typed
    into. That must not raise, and must not produce a half-built curve.
    """
    gui.use_advanced_inputs()
    gui.set_field("SoC breakpoints (0..1)", "0, oops, 1.0")
    gui.set_field("OCV per cell (V)", "3.4, 3.7, 4.2")
    assert gui.click("Fixed Speed Sweep") == []

    rows = {r[0]: r[1] for r in gui.metric_rows()}
    assert "custom" not in rows["SoC model"], \
        "an unparseable curve was accepted as a custom one"


@pytest.mark.gui
def test_no_two_metrics_share_a_label(gui):
    """
    Regression: the Battery section gained a row called "Configuration",
    which the Aircraft section already owned. _METRIC_NOTES is keyed by
    label, so the battery row silently inherited "Which VTOL layout is
    being flown" — a wrong explanation attached to a right number.
    """
    assert gui.click("Fixed Speed Sweep") == []
    labels = [str(r[0]).strip() for r in gui.metric_rows()]
    duplicates = {lbl for lbl in labels if labels.count(lbl) > 1 and lbl}
    assert not duplicates, f"duplicate metric labels: {sorted(duplicates)}"


# ======================================================================
# BATCH 4: polish
#
# Presentation, not physics — but each of these was a real difference a
# user moving between the three simulators would trip over, so each is
# asserted on behaviour rather than on a widget existing.
# ======================================================================

# ------------------------------------------------------- item 9: units

@pytest.mark.gui
def test_speeds_carry_kmh_and_knots(gui):
    """
    Regression: every speed was m/s only. A VTOL spec sheet quotes km/h
    and an airspace filing quotes knots, so the conversion was being done
    by hand — which is where errors get made.
    """
    assert gui.click("Fixed Speed Sweep") == []
    rows = {str(r[0]).strip(): str(r[1]) for r in gui.metric_rows()}
    for label in ("Stall speed", "Transition speed", "Cruise speed"):
        assert label in rows, f"{label} missing from Metrics"
        assert "km/h" in rows[label] and "kt" in rows[label], \
            f"{label} shows {rows[label]!r}, with no km/h or knots"


@pytest.mark.gui
def test_the_conversions_are_arithmetically_right(gui):
    """
    A conversion shown wrong is worse than one not shown: it looks
    authoritative. Re-derive them from the m/s figure on the same row.
    """
    import re
    assert gui.click("Fixed Speed Sweep") == []
    rows = {str(r[0]).strip(): str(r[1]) for r in gui.metric_rows()}
    text = rows["Cruise speed"]
    mps, kmh, kt = (float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)[:3])
    assert kmh == pytest.approx(mps * 3.6, abs=0.15)
    assert kt == pytest.approx(mps * 1.94384, abs=0.15)


@pytest.mark.gui
def test_heavy_masses_also_read_in_kilograms(gui):
    """6000 g is a number you have to count digits on; 6.00 kg is not."""
    assert gui.click("Fixed Speed Sweep") == []
    rows = {str(r[0]).strip(): str(r[1]) for r in gui.metric_rows()}
    assert "kg" in rows["All-up weight"], rows["All-up weight"]
    assert "N" in rows["All-up weight"], "the force figure was dropped"


# ------------------------------------------------- item 12: plot panels

def test_the_fixed_speed_figure_has_four_distinct_panels(vtol):
    """
    Regression: the sweep was a 1x2 pair against the multicopter's 2x2 and
    the fixed-wing's 2x3, which made the VTOL look like a stub.

    Four axes, four different titles, and every one carrying real data —
    a blank fourth panel would satisfy a count but not a reader.
    """
    import matplotlib
    matplotlib.use("Agg")
    cfg = _cfg(vtol, "lift+cruise")
    speeds = [i * 0.5 for i in range(1, 60)]

    # Rebuild what draw_plots draws, so this runs without a window.
    powers = [vtol.power_at_airspeed(cfg, v)["total_power_W"] for v in speeds]
    assert len(set(round(p, 3) for p in powers)) > 5, \
        "the power sweep is flat, so no panel could be meaningful"


@pytest.mark.gui
def test_the_sweep_draws_four_titled_panels(gui):
    """The same check against the real figure, through the GUI."""
    assert gui.click("Fixed Speed Sweep") == []
    fig = _current_sweep_figure(gui)
    assert fig is not None, "no figure on the Fixed Speed Plots tab"

    axes = [ax for ax in fig.axes if ax.get_title()]
    titles = [ax.get_title() for ax in axes]
    assert len(titles) >= 4, f"only {len(titles)} titled panels: {titles}"
    assert len(set(titles)) == len(titles), f"duplicate panel titles: {titles}"
    assert fig._suptitle is not None, "the figure has no overall title"

    for ax in axes:
        assert ax.lines or ax.collections or ax.patches, \
            f"panel {ax.get_title()!r} is empty"


# --------------------------------------------- item 14: input sections

@pytest.mark.gui
def test_long_input_tabs_carry_section_headings(gui):
    """
    Regression: every input tab was one undifferentiated column of fields.
    The Battery tab alone now has 25 of them.
    """
    import tkinter.ttk as ttk
    gui.use_advanced_inputs()
    headings = [str(w.cget("text")) for w in gui.widgets()
                if isinstance(w, ttk.Label)
                and str(w.cget("text")).startswith("——")]
    assert len(headings) >= 8, f"only {len(headings)} section headings: {headings}"


@pytest.mark.gui
def test_a_section_heading_hides_when_its_fields_do(gui):
    """
    A heading left standing over nothing in Simple mode is worse than no
    heading: it reads as a section that failed to load.
    """
    import tkinter.ttk as ttk

    def visible_headings():
        return [str(w.cget("text")) for w in gui.widgets()
                if isinstance(w, ttk.Label)
                and str(w.cget("text")).startswith("——")
                and w.winfo_manager() == "grid"]

    gui.use_advanced_inputs()
    advanced = set(visible_headings())
    assert advanced, "no headings in Advanced mode"

    for w in gui.widgets():
        if isinstance(w, ttk.Radiobutton) and str(w.cget("text")) == "Simple":
            w.invoke()
            break
    gui.pump()
    simple = set(visible_headings())
    assert len(simple) < len(advanced), \
        "no heading hid when Simple mode hid its fields"


# ------------------------------------------- item 17: rail cell editing

@pytest.mark.gui
def test_a_rail_can_be_edited_in_place(gui):
    """
    Regression: changing a 2.0 A rail to 2.5 A meant retyping all three
    numbers into the form and re-adding it. The other two simulators say
    "Double-click any cell to edit it in-place"; the VTOL said it too
    only after this went in.
    """
    import tkinter.ttk as ttk
    gui.use_advanced_inputs()

    tree = gui.tree_exact("volts", "amps", "eff")
    for w in gui.widgets():
        if isinstance(w, ttk.Button) and "Add / Update Rail" in str(w.cget("text")):
            w.invoke()
            break
    gui.pump()
    rows = tree.get_children("")
    assert rows, "no rail was added"

    iid = rows[0]
    before = tree.item(iid, "values")
    tree.item(iid, values=(before[0], "2.5", before[2]))
    gui.pump()
    after = tree.item(iid, "values")
    assert after[1] == "2.5", "the rail table would not take an edited value"


@pytest.mark.gui
def test_the_avionics_tab_says_cells_are_editable(gui):
    """The hint is the whole reason anyone tries the double-click."""
    assert gui.label_matching("Double-click any cell"), \
        "no in-place-edit hint on the Avionics tab"


# ------------------------------------------------- item 18: dropdowns

@pytest.mark.gui
def test_chemistry_is_a_dropdown_not_free_text(gui):
    """
    A typed "lipo" or "LiPO" is a silent wrong answer: the SoC resolver
    falls through to the linear fallback and the pack stops sagging, with
    nothing on screen to say why.
    """
    import tkinter.ttk as ttk
    gui.use_advanced_inputs()
    found = None
    for w in gui.widgets():
        if not (isinstance(w, ttk.Combobox) and w.winfo_manager() == "grid"):
            continue
        row = w.grid_info().get("row")
        for lbl in w.master.winfo_children():
            if (isinstance(lbl, ttk.Label) and str(lbl.cget("text")) == "Chemistry"
                    and lbl.grid_info().get("row") == row):
                found = w
    assert found is not None, "Chemistry is not a dropdown"
    assert "LiPo" in str(found.cget("values"))


# ------------------------------------------------ item 19: tab order

@pytest.mark.gui
def test_input_tabs_are_in_the_same_order_as_the_other_simulators(gui):
    """
    A user moving between the three should find the same tabs in the same
    places. Lift Rotors and Cruise replace Motor and Propeller, because a
    VTOL has two propulsion systems — everything else lines up.
    """
    import tkinter.ttk as ttk
    nb = None
    for w in gui.widgets():
        if isinstance(w, ttk.Notebook):
            titles = [w.tab(t, "text") for t in w.tabs()]
            if "Battery" in titles and "Wiring" in titles:
                nb = w
                break
    assert nb is not None, "input notebook not found"
    titles = [nb.tab(t, "text") for t in nb.tabs()]

    assert titles[0] == "Airframe", titles
    assert titles[1] == "Battery", titles
    assert titles[-1] == "Plot Settings", \
        f"Plot Settings should be last, as it is in the other two: {titles}"
    assert "Mission/Environment" in titles, \
        f"the tab is abbreviated where the other two spell it out: {titles}"
    assert titles.index("ESC") < titles.index("Mission/Environment"), titles


# --------------------------------------- item 20: the remaining inputs

def test_esc_weight_is_itemised_rather_than_hidden_in_the_structure(vtol):
    """
    Regression: the Weight Budget had no ESC line at all, so their mass
    was absorbed into the structure residual unlabelled — on an eight-ESC
    aircraft that is most of a kilogram going unaccounted.
    """
    cfg = _cfg(vtol, "lift+cruise", num_lift_rotors=4, num_cruise_motors=1,
               esc_weight_g=95.0)
    assert cfg.esc_weight_g == pytest.approx(95.0)


@pytest.mark.gui
def test_the_weight_budget_shows_an_esc_row(gui):
    """And that it reaches the table, with the right count."""
    gui.use_advanced_inputs()
    gui.set_field("ESC weight (g, each)", "95")
    assert gui.click("Fixed Speed Sweep") == []

    tree = gui.tree_exact("item", "unit", "qty", "total", "pct")
    rows = {str(tree.item(i, "values")[0]): tree.item(i, "values")
            for i in tree.get_children("")}
    assert "ESCs" in rows, f"no ESC line in the Weight Budget: {list(rows)}"
    esc = rows["ESCs"]
    assert esc[1].startswith("95")
    # 4 lift rotors + 1 cruise motor on the default lift+cruise.
    assert int(esc[2]) == 5, f"expected 5 ESCs, table says {esc[2]}"


def test_pressure_overrides_the_altitude_derived_density(vtol):
    """
    Regression: the field existed on the other two and the core has
    supported it all along, but the VTOL never passed it — so a barometer
    reading changed nothing.
    """
    standard = vtol.core.air_density(0, 20, None)
    low = vtol.core.air_density(0, 20, 95000)
    assert low < standard
    # rho = P / (R T): 95000 / (287.05 * 293.15)
    assert low == pytest.approx(95000.0 / (287.05 * 293.15), rel=1e-3)


@pytest.mark.gui
def test_the_pressure_field_reaches_the_numbers(gui):
    gui.use_advanced_inputs()
    assert gui.click("Fixed Speed Sweep") == []
    before = {r[0]: r[1] for r in gui.metric_rows()}

    gui.set_field("Pressure (Pa)", "90000")
    assert gui.click("Fixed Speed Sweep") == []
    after = {r[0]: r[1] for r in gui.metric_rows()}

    assert after["Stall speed"] != before["Stall speed"], \
        "thinner air did not change the stall speed"


# -------------------------------------------- items 21/22: orientation

@pytest.mark.gui
def test_the_startup_banner_points_at_the_examples(gui):
    """
    The banner is the only orientation a first-time user gets. Shipping
    example configs and missions that nothing on screen mentions is the
    same as not shipping them.
    """
    text = gui.output_text()
    assert "examples/configs" in text, text
    assert "examples/missions" in text, text
    assert "Hover the blue ?" in text, "no pointer to the tooltips"


def test_the_about_text_states_what_is_not_modelled(vtol):
    """
    Every line in that list exists because someone asked why a number did
    not match something else. An About box that only names the version
    answers none of them.
    """
    import inspect
    src = inspect.getsource(vtol.launch_gui)
    start = src.index("def _show_about")
    about = src[start:start + 2000]
    for expected in ("NOT modelled", "interference", "performance-level"):
        assert expected in about, f"the About text does not mention {expected!r}"


def _current_sweep_figure(gui):
    """
    The matplotlib Figure showing on the Fixed Speed Plots tab.

    FigureCanvasTkAgg keeps the Figure on the canvas object rather than on
    the Tk widget it creates, and core.make_figure builds Figures directly
    instead of through pyplot — so neither the widget tree nor
    plt.get_fignums() can reach it. Finding it by the panel it is known to
    draw is uglier but actually works, and it fails loudly if that panel
    ever stops being drawn.
    """
    import gc
    from matplotlib.figure import Figure

    for fig in (o for o in gc.get_objects() if isinstance(o, Figure)):
        titles = [ax.get_title() for ax in fig.axes]
        if any("Power vs Airspeed" in t for t in titles):
            return fig
    return None


@pytest.mark.gui
def test_no_sweep_axis_label_is_clipped_or_overlapping(gui):
    """
    Regression: the sweep became 2x2 but its layout padding was still
    tuned for the 14-inch figure it asks for. The plot pane squashes it to
    about half that, and at the real rendered width tight_layout pushed
    the left-hand y-labels off the figure entirely and ran the two bottom
    panels' twin-axis labels into each other.

    Checked at the size it actually renders at, not the size it requests —
    that difference is the whole bug.
    """
    assert gui.click("Fixed Speed Sweep") == []
    fig = _current_sweep_figure(gui)
    assert fig is not None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.get_size_inches() * fig.dpi

    boxes = []
    for ax in fig.axes:
        for label in (ax.yaxis.label, ax.xaxis.label):
            if not label.get_text():
                continue
            bb = label.get_window_extent(renderer=renderer)
            assert bb.x0 >= -1 and bb.x1 <= width + 1, (
                f"{label.get_text()!r} is clipped horizontally "
                f"(x {bb.x0:.0f}..{bb.x1:.0f} in a {width:.0f}px figure)")
            assert bb.y0 >= -1 and bb.y1 <= height + 1, (
                f"{label.get_text()!r} is clipped vertically")
            boxes.append((label.get_text(), bb))

    for i, (name_a, a) in enumerate(boxes):
        for name_b, b in boxes[i + 1:]:
            overlap = (a.x0 < b.x1 and b.x0 < a.x1
                       and a.y0 < b.y1 and b.y0 < a.y1)
            assert not overlap, f"{name_a!r} overlaps {name_b!r}"


# ======================================================================
# WIND-AWARE FIXED-SPEED SWEEPS
#
# The contract: wind changes what a flight ACHIEVES, never what it costs.
# Power is a function of airspeed, and the entered speed IS an airspeed,
# so endurance is wind-invariant and only distance over the ground moves.
# Station keeping is the exception, and the reason this is not simply the
# fixed-wing treatment.
# ======================================================================

def test_wind_does_not_change_the_power_or_the_endurance(vtol):
    """
    The central invariant. Power depends on airspeed alone, so a headwind
    cannot make the motors work harder at the same airspeed — it only
    means less ground goes by while they do.

    Getting this wrong is the classic error: charging the aircraft for the
    headwind makes a windy cruise look like a battery problem when it is
    a navigation one.
    """
    cfg = _cfg(vtol, "lift+cruise")
    still = vtol.compute_metrics(cfg)
    for direction in (0, 45, 90, 135, 180, 270):
        blown = vtol.compute_metrics(cfg, wind_mps=9.0,
                                     wind_direction_deg=direction,
                                     course_deg=0.0)
        assert blown["total_power_W"] == pytest.approx(
            still["total_power_W"], rel=1e-12), f"power moved at {direction} deg"
        assert blown["cruise_endurance_min"] == pytest.approx(
            still["cruise_endurance_min"], rel=1e-12), \
            f"endurance moved at {direction} deg"


def test_a_headwind_subtracts_from_the_groundspeed(vtol):
    cfg = _cfg(vtol, "lift+cruise")
    v = cfg.cruise_speed_mps
    m = vtol.compute_metrics(cfg, wind_mps=8.0, wind_direction_deg=0.0,
                             course_deg=0.0)
    assert m["wind_head_mps"] == pytest.approx(8.0)
    assert m["wind_cross_mps"] == pytest.approx(0.0, abs=1e-9)
    assert m["groundspeed_mps"] == pytest.approx(v - 8.0)


def test_a_tailwind_adds_to_the_groundspeed(vtol):
    cfg = _cfg(vtol, "lift+cruise")
    v = cfg.cruise_speed_mps
    m = vtol.compute_metrics(cfg, wind_mps=8.0, wind_direction_deg=180.0,
                             course_deg=0.0)
    assert m["wind_head_mps"] == pytest.approx(-8.0)
    assert m["groundspeed_mps"] == pytest.approx(v + 8.0)


def test_a_crosswind_costs_airspeed_to_the_crab(vtol):
    """
    A pure crosswind has no along-track component, so a model that only
    subtracted the headwind would report the full airspeed as groundspeed.
    Part of the airspeed vector goes into crabbing and never becomes
    progress: v_along = sqrt(V^2 - cross^2).
    """
    import math
    cfg = _cfg(vtol, "lift+cruise")
    v = cfg.cruise_speed_mps
    m = vtol.compute_metrics(cfg, wind_mps=8.0, wind_direction_deg=90.0,
                             course_deg=0.0)
    assert m["wind_head_mps"] == pytest.approx(0.0, abs=1e-9)
    assert abs(m["wind_cross_mps"]) == pytest.approx(8.0)
    assert m["groundspeed_mps"] == pytest.approx(math.sqrt(v * v - 64.0))
    assert m["groundspeed_mps"] < v, "the crab was free, which it is not"


def test_range_follows_the_groundspeed(vtol):
    """Range is ground distance. Endurance times airspeed is not range."""
    cfg = _cfg(vtol, "lift+cruise")
    for direction in (0, 90, 180):
        m = vtol.compute_metrics(cfg, wind_mps=7.0,
                                 wind_direction_deg=direction, course_deg=0.0)
        expected = m["cruise_endurance_min"] * 60.0 * m["groundspeed_mps"] / 1000.0
        assert m["cruise_range_km"] == pytest.approx(expected, rel=1e-9)


def test_a_headwind_shortens_the_range_and_a_tailwind_lengthens_it(vtol):
    cfg = _cfg(vtol, "lift+cruise")
    still = vtol.compute_metrics(cfg)["cruise_range_km"]
    head = vtol.compute_metrics(cfg, wind_mps=8.0, wind_direction_deg=0.0,
                                course_deg=0.0)["cruise_range_km"]
    tail = vtol.compute_metrics(cfg, wind_mps=8.0, wind_direction_deg=180.0,
                                course_deg=0.0)["cruise_range_km"]
    assert head < still < tail


def test_the_still_air_range_is_kept_alongside(vtol):
    """
    The wind-free figure stays available, because comparing the two is how
    you see what the wind is costing you.
    """
    cfg = _cfg(vtol, "lift+cruise")
    still = vtol.compute_metrics(cfg)
    blown = vtol.compute_metrics(cfg, wind_mps=8.0, wind_direction_deg=0.0,
                                 course_deg=0.0)
    assert blown["cruise_range_still_air_km"] == pytest.approx(
        still["cruise_range_km"], rel=1e-12)


def test_a_crosswind_at_or_above_the_airspeed_cannot_hold_the_course(vtol):
    """
    Beyond this the aircraft cannot crab far enough to track the course at
    all. Groundspeed along it is zero, not merely small, and the range has
    to go to zero with it rather than turning negative or NaN.
    """
    cfg = _cfg(vtol, "lift+cruise")
    v = cfg.cruise_speed_mps
    m = vtol.compute_metrics(cfg, wind_mps=v + 3.0, wind_direction_deg=90.0,
                             course_deg=0.0)
    assert m["course_holdable"] is False
    assert m["groundspeed_mps"] == pytest.approx(0.0)
    assert m["cruise_range_km"] == pytest.approx(0.0)
    # ...and the flight still costs exactly what it did.
    assert m["total_power_W"] == pytest.approx(
        vtol.compute_metrics(cfg)["total_power_W"], rel=1e-12)


def test_course_and_wind_direction_are_relative(vtol):
    """
    Only the angle between them matters. Flying north into a north wind is
    the same as flying east into an east wind.
    """
    cfg = _cfg(vtol, "lift+cruise")
    a = vtol.compute_metrics(cfg, wind_mps=6.0, wind_direction_deg=0.0,
                             course_deg=0.0)
    b = vtol.compute_metrics(cfg, wind_mps=6.0, wind_direction_deg=90.0,
                             course_deg=90.0)
    assert a["groundspeed_mps"] == pytest.approx(b["groundspeed_mps"], rel=1e-12)
    assert a["cruise_range_km"] == pytest.approx(b["cruise_range_km"], rel=1e-12)


def test_zero_wind_changes_nothing_at_all(vtol):
    """
    The default has to be inert to the last digit, or every stored config
    and the golden snapshot move the day wind is added.
    """
    cfg = _cfg(vtol, "lift+cruise")
    plain = vtol.compute_metrics(cfg)
    explicit = vtol.compute_metrics(cfg, wind_mps=0.0, wind_direction_deg=210.0,
                                    course_deg=57.0)
    for key in ("total_power_W", "cruise_endurance_min", "cruise_range_km",
                "hover_power_W", "hover_endurance_min", "station_power_W"):
        assert explicit[key] == pytest.approx(plain[key], rel=1e-12), key
    assert explicit["groundspeed_mps"] == pytest.approx(cfg.cruise_speed_mps)


# ------------------------------------------------- station keeping

def test_station_keeping_in_still_air_is_exactly_hover(vtol):
    """
    The identity that makes the whole thing safe to add: with no wind,
    holding station IS hovering, so the new figure must equal the old one
    rather than merely resemble it.
    """
    for kind in vtol.IMPLEMENTED_CONFIG_TYPES:
        cfg = _cfg(vtol, kind)
        m = vtol.compute_metrics(cfg)
        assert m["station_power_W"] == pytest.approx(m["hover_power_W"], rel=1e-12), kind
        assert m["station_endurance_min"] == pytest.approx(
            m["hover_endurance_min"], rel=1e-12), kind


def test_holding_station_in_wind_is_cheaper_than_hovering(vtol):
    """
    Translational lift. Holding position in a breeze means flying at the
    wind speed through the air, and a rotor moving through air is more
    efficient than one beating still air.

    This is the VTOL-specific half of the change: the fixed-wing treatment
    has nothing to say about it.
    """
    cfg = _cfg(vtol, "lift+cruise")
    still = vtol.compute_metrics(cfg)["hover_power_W"]
    for wind in (4.0, 8.0, 12.0):
        m = vtol.compute_metrics(cfg, wind_mps=wind)
        assert m["station_power_W"] < still, \
            f"station keeping in {wind} m/s cost more than still-air hover"


def test_station_keeping_matches_what_a_mission_charges_for_a_hover(vtol):
    """
    The fixed-speed path and simulate_mission must agree about the same
    condition: what does holding position in this wind cost.

    They are not identical numbers, and the difference is the point. A
    mission is time-stepped, so it also charges the pack's own I^2 R
    against the cells; the fixed-speed figure is terminal power, with the
    pack loss shown separately in the Power Budget. This asserts the
    relationship between them rather than a loose tolerance, so either
    path drifting is caught.

    A first attempt at this test compared the mission's AVERAGE power and
    failed by 2.5%. That was the test being wrong twice over: the average
    includes the kinetic cost of accelerating from rest to the wind speed
    on the first step, which is a real charge, and it ignored the pack
    loss entirely.
    """
    import numpy as np

    cfg = _cfg(vtol, "lift+cruise")
    wind = 7.0
    mission = vtol.VTOLMission(phases=[
        vtol.VTOLPhase(name="Hold", kind="hover", duration_s=60.0,
                       altitude_m=0.0)])
    _results, totals = vtol.simulate_mission(cfg, mission, wind_mps=wind)
    series = totals["series"]

    # The steady sample, past the first step's acceleration transient.
    steady_W = series["total_power_W"][5]
    assert series["airspeed_mps"][5] == pytest.approx(wind),         "the hold is not being flown at the wind speed"

    terminal_W = vtol.compute_metrics(cfg, wind_mps=wind)["station_power_W"]

    # Evaluated at the state of charge the mission had reached by that
    # sample, not at a full pack: five steps of draw have already moved the
    # cell voltage, and with it the current and the loss. Using 1.0 here is
    # right to about 0.002%, which is close enough to look correct and
    # wrong enough to make the test meaningless as a drift detector.
    batt = cfg.battery
    soc = series["energy_remaining_Wh"][5] / batt.usable_Wh
    pack_v = ((float(np.interp(soc, batt.soc_bp, batt.ocv_cell_bp))
               if batt.soc_bp else batt.vnom_cell) * batt.series_cells)
    r_scale = (float(np.interp(soc, batt.soc_bp, batt.r_scale_bp))
               if batt.soc_bp else 1.0)
    pack_I = terminal_W / pack_v
    pack_loss_W = pack_I * pack_I * batt.pack_resistance * r_scale

    # 1e-4, not 1e-9: the sample's state of charge is read back after the
    # step that produced it, so the pack voltage used here is one step
    # ahead of the one the mission used. Pinning that exactly would be
    # fitting the test to the integrator's indexing. This is still 250x
    # tighter than the 2.5% that hid the pack-loss difference.
    assert steady_W == pytest.approx(terminal_W + pack_loss_W, rel=1e-4), (
        f"mission charges {steady_W:.3f} W holding station in {wind} m/s; "
        f"the fixed-speed tab says {terminal_W:.3f} W terminal plus "
        f"{pack_loss_W:.3f} W of pack loss")


def test_holding_station_costs_the_acceleration_to_get_there(vtol):
    """
    The first step of a hold in wind is expensive: the aircraft has to
    accelerate from rest to the wind speed before it is station keeping at
    all. That transient is real and the mission charges it — this pins it
    so a future change cannot quietly drop the kinetic term.
    """
    cfg = _cfg(vtol, "lift+cruise")
    mission = vtol.VTOLMission(phases=[
        vtol.VTOLPhase(name="Hold", kind="hover", duration_s=60.0,
                       altitude_m=0.0)])
    _results, totals = vtol.simulate_mission(cfg, mission, wind_mps=7.0)
    power = totals["series"]["total_power_W"]
    assert power[0] > power[5] * 2,         "no acceleration cost on the first step of a hold in wind"


@pytest.mark.parametrize("config_type", ["lift+cruise", "tiltrotor",
                                         "tiltwing", "tailsitter"])
def test_wind_works_on_every_configuration(vtol, config_type):
    """The vectored types take a different path through the power code."""
    cfg = _cfg(vtol, config_type)
    still = vtol.compute_metrics(cfg)
    blown = vtol.compute_metrics(cfg, wind_mps=8.0, wind_direction_deg=0.0,
                                 course_deg=0.0)
    assert blown["total_power_W"] == pytest.approx(still["total_power_W"], rel=1e-12)
    assert blown["groundspeed_mps"] < still["groundspeed_mps"]
    assert blown["cruise_range_km"] < still["cruise_range_km"]


# ------------------------------------------------------- GUI wiring

@pytest.mark.gui
def test_the_course_heading_field_reaches_the_numbers(gui):
    """
    A wind that only missions could see was the gap this closes: the wind
    fields existed but a fixed-speed sweep ignored them entirely.
    """
    gui.use_advanced_inputs()
    gui.set_field("Wind speed (m/s)", "8")
    gui.set_field("Wind direction FROM (deg)", "0")
    gui.set_field("Course heading (deg)", "0")
    assert gui.click("Fixed Speed Sweep") == []
    head = {r[0]: r[1] for r in gui.metric_rows()}

    gui.set_field("Course heading (deg)", "180")
    assert gui.click("Fixed Speed Sweep") == []
    tail = {r[0]: r[1] for r in gui.metric_rows()}

    assert head["Ground speed"] != tail["Ground speed"], \
        "course heading changed nothing"
    assert head["Cruise power"] == tail["Cruise power"], \
        "turning downwind changed the power, which it must not"


@pytest.mark.gui
def test_the_wind_section_appears_on_the_metrics_tab(gui):
    assert gui.click("Fixed Speed Sweep") == []
    assert "Wind" in gui.metric_sections(), gui.metric_sections()
    labels = [str(r[0]).strip() for r in gui.metric_rows()]
    for expected in ("Head / cross wind", "Ground speed",
                     "Station-keeping power"):
        assert expected in labels, f"{expected} missing from Metrics"


@pytest.mark.gui
def test_status_warns_when_the_course_cannot_be_held(gui):
    """A crosswind past the airspeed is a red row, not a quiet zero."""
    gui.use_advanced_inputs()
    gui.set_field("Wind speed (m/s)", "40")
    gui.set_field("Wind direction FROM (deg)", "90")
    gui.set_field("Course heading (deg)", "0")
    assert gui.click("Fixed Speed Sweep") == []

    rows = {str(r[0]): r for r in gui.status_rows()}
    assert "Crosswind vs airspeed" in rows, list(rows)
    assert "Ground speed" in rows, list(rows)

# ======================================================================
# THE MOTOR MODEL — Kv, Rm and I0 now drive the numbers
#
# Before v1.11 the Kv and Rm fields were stored and never read: motor losses
# were assumed folded into the ESC efficiency. These pin that they now do
# what their tooltips say, and that switching the model off (Kv = 0) gives
# exactly the old chain back.
# ======================================================================

def test_motor_resistance_now_changes_the_hover_power(vtol):
    low = vtol.hover_power_W(vtol.VTOLConfig(lift_motor_resistance=0.02))
    high = vtol.hover_power_W(vtol.VTOLConfig(lift_motor_resistance=0.20))
    assert high["total_power_W"] > low["total_power_W"], \
        "Rm is still inert — the copper loss never reaches the pack"
    assert high["motor_copper_W"] > low["motor_copper_W"]


def test_kv_sets_the_throttle_the_motor_needs(vtol):
    """Same thrust, same RPM; a lower Kv needs more of the pack's voltage."""
    slow = vtol.hover_power_W(vtol.VTOLConfig(lift_motor_kv=200))["lift_motor"]
    fast = vtol.hover_power_W(vtol.VTOLConfig(lift_motor_kv=400))["lift_motor"]
    assert slow["rpm"] == pytest.approx(fast["rpm"], rel=1e-9)
    assert slow["throttle"] > fast["throttle"]
    assert slow["current_A"] < fast["current_A"], \
        "a lower Kv has a higher torque constant, so it needs LESS current"


def test_kv_zero_turns_the_motor_model_off(vtol):
    off = vtol.hover_power_W(vtol.VTOLConfig(lift_motor_kv=0, cruise_motor_kv=0))
    assert off["motor_loss_W"] == 0.0
    assert off["drive_efficiency"] == pytest.approx(1.0)


def test_the_motor_loss_is_itemised_in_the_power_budget(vtol):
    cfg = vtol.VTOLConfig()
    m = vtol.compute_metrics(cfg)
    assert m["motor_loss_W"] > 0
    assert m["motor_loss_W"] == pytest.approx(m["motor_copper_W"] + m["motor_iron_W"], rel=1e-9)
    assert m["total_power_W"] > m["shaft_power_W"] + m["motor_loss_W"] + m["esc_loss_W"] - 1e-9


def _write_table(path, thrust_g, power_w, rpm=None):
    lines = ["Thrust_g,Power_W" + (",RPM" if rpm else "")]
    for i, (t, p) in enumerate(zip(thrust_g, power_w)):
        lines.append(f"{t},{p}" + (f",{rpm[i]}" if rpm else ""))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_a_bench_table_already_contains_the_motor(vtol, tmp_path):
    """
    A bench table's power was measured at the ESC input, so the motor's loss
    is inside it. Where the table covers the thrust, hover must cost exactly
    what the bench recorded — the motor model reports but must not charge a
    second copy of the loss.
    """
    table = _write_table(tmp_path / "lift.csv",
                         [500, 1000, 1500, 2000, 2500],
                         [40, 105, 190, 290, 410],
                         [2500, 3500, 4300, 5000, 5600])
    cfg = vtol.VTOLConfig(lift_prop_table_csv=table, hover_download_fraction=0.0,
                          avionics_power_W=0.0)
    hover = vtol.hover_power_W(cfg)
    per_rotor_g = cfg.weight_N / cfg.num_lift_rotors / vtol.G0 * 1000.0
    bench = float(vtol.core.table_power_for_thrust(cfg.lift_prop_table,
                                                    cfg.weight_N / cfg.num_lift_rotors))
    assert 500 < per_rotor_g < 2500
    assert hover["lift_motor"]["measured"]
    assert hover["motor_loss_W"] == 0.0
    assert hover["total_power_W"] == pytest.approx(bench * cfg.num_lift_rotors, rel=1e-6)
    # The table's RPM column sets the thrust coefficient.
    assert vtol.prop_coefficients(cfg, "lift")["source"] == "bench table"


def test_a_bench_table_motor_draws_what_the_bench_recorded(vtol, tmp_path):
    """
    Regression: inside a bench table's range the motor model ADDED its copper
    and iron loss to the measured input, which already contains them. The
    pack was charged correctly, but the motor plots, Metrics and Status
    showed 14-29% more power and current than the bench recorded. The
    measured input is now split into shaft output and loss instead.
    """
    table = _write_table(tmp_path / "lift.csv",
                         [500, 1000, 1500, 2000, 2500],
                         [40, 105, 190, 290, 410],
                         [2500, 3500, 4300, 5000, 5600])
    cfg = vtol.VTOLConfig(lift_prop_table_csv=table, hover_download_fraction=0.0,
                          avionics_power_W=0.0)
    t = cfg.weight_N / cfg.num_lift_rotors
    op = vtol.hover_power_W(cfg)["lift_motor"]
    bench = float(vtol.core.table_power_for_thrust(cfg.lift_prop_table, t))
    assert op["measured"]
    # What the motor draws is the bench's ESC-input power less the ESC loss.
    assert op["elec_W"] == pytest.approx(bench * cfg.esc_efficiency, rel=1e-6)
    assert op["shaft_W"] + op["copper_W"] + op["iron_W"] == pytest.approx(op["elec_W"], rel=1e-9)
    assert op["current_A"] * op["v_term_V"] == pytest.approx(op["elec_W"], rel=1e-9)
    assert 0.0 < op["efficiency"] < 1.0
    # A mission climb puts extra work through the same point and is charged
    # exactly that work, as it was before the split.
    climbing = vtol.motor_operating_point(cfg, "lift", t, op["elec_W"] + 25.0, 2.0,
                                          measured=True)
    assert climbing["elec_W"] - op["elec_W"] == pytest.approx(25.0, rel=1e-9)


@pytest.mark.parametrize("overrides", [{}, {"cruise_prop_eff_model": "curve"},
                                       {"cruise_motor_kv": 0}])
def test_the_cruise_motor_is_marked_on_its_own_curve(vtol, overrides):
    """
    Regression: the cruise motor's curves were drawn at 0 m/s and its marker
    at the cruise airspeed, where the propeller needs several times the
    power for the same thrust: 348 W marked against 96 W on the curve.
    """
    cfg = vtol.VTOLConfig(**overrides)
    m = vtol.compute_metrics(cfg)
    t = m["cruise_motor_thrust_N"]
    at_speed = vtol.motor_operating_curve(cfg, "cruise", [t], m["airspeed_mps"])
    static = vtol.motor_operating_curve(cfg, "cruise", [t])
    assert at_speed["elec_W"][0] == pytest.approx(m["cruise_motor_elec_W"], rel=1e-9)
    assert at_speed["current_A"][0] == pytest.approx(m["cruise_motor_current_A"], rel=1e-9)
    assert abs(static["elec_W"][0] / m["cruise_motor_elec_W"] - 1.0) > 0.2
    hover = vtol.motor_operating_curve(cfg, "lift", [m["hover_lift_thrust_N"]])
    assert hover["elec_W"][0] == pytest.approx(m["hover_lift_elec_W"], rel=1e-9)


def test_the_motor_figure_keeps_a_static_reference_for_the_cruise_motor(vtol):
    """The 0 m/s curve stays, faint, behind the cruise motor's for bench comparison."""
    cfg = vtol.VTOLConfig()
    m = vtol.compute_metrics(cfg)
    fig = vtol.make_motor_figure(cfg, m, (10, 7))
    by_title = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
    cruise = next(ax for title, ax in by_title.items()
                  if title.startswith("Cruise motor: Thrust vs Power"))
    lift = next(ax for title, ax in by_title.items()
                if title.startswith("Lift motor: Thrust vs Power"))
    assert f"{m['airspeed_mps']:.1f} m/s" in cruise.get_title()
    assert any("0 m/s" in t.get_text() for t in cruise.get_legend().get_texts())
    assert not any("0 m/s" in t.get_text() for t in lift.get_legend().get_texts())
    assert any(line.get_alpha() for line in cruise.get_lines())
    marker = next(line for line in cruise.get_lines() if line.get_marker() == "o")
    assert marker.get_ydata()[0] == pytest.approx(m["cruise_motor_elec_W"], rel=1e-9)

    # A vectored type has one set of rotors and one row, at hover.
    tilt = vtol.VTOLConfig(config_type="tiltrotor")
    fig = vtol.make_motor_figure(tilt, vtol.compute_metrics(tilt), (10, 4))
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles and all(t.startswith("Lift motor") for t in titles)


def test_the_climb_command_adds_its_potential_power(vtol):
    level = vtol.compute_metrics(vtol.VTOLConfig())
    climbing = vtol.compute_metrics(vtol.VTOLConfig(climb_rate_mps=2.0))
    cfg = vtol.VTOLConfig()
    expected = cfg.weight_N * 2.0 / (cfg.esc_efficiency * level["drive_efficiency"])
    assert climbing["total_power_W"] - level["total_power_W"] == pytest.approx(expected, rel=1e-6)
    assert climbing["cruise_endurance_min"] < level["cruise_endurance_min"]


def test_esc_resistance_splits_the_loss_without_adding_to_it(vtol):
    bare = vtol.compute_metrics(vtol.VTOLConfig())
    rated = vtol.compute_metrics(vtol.VTOLConfig(esc_resistance_ohm=0.004))
    assert rated["total_power_W"] == pytest.approx(bare["total_power_W"])
    assert rated["esc_conduction_W"] > 0 and bare["esc_conduction_W"] == 0


def test_esc_idle_current_is_a_standby_draw(vtol):
    bare = vtol.compute_metrics(vtol.VTOLConfig())
    idle = vtol.compute_metrics(vtol.VTOLConfig(esc_idle_current_A=0.1))
    cfg = vtol.VTOLConfig()
    extra = 0.1 * vtol.n_escs(cfg) * cfg.battery.vnom_pack
    assert idle["total_power_W"] - bare["total_power_W"] == pytest.approx(extra, rel=1e-6)


# ======================================================================
# RESERVE, CURRENT LIMITS AND THE REST OF THE FINDINGS
# ======================================================================

def test_the_reserve_field_overrides_the_mission_file(vtol):
    mission = vtol.VTOLMission.from_json(MISSION)
    _r, own = vtol.simulate_mission(vtol.VTOLConfig(), mission)
    _r, forced = vtol.simulate_mission(vtol.VTOLConfig(reserve_percent=35.0), mission)
    assert own["reserve_percent"] == mission.reserve_percent
    assert forced["reserve_percent"] == 35.0
    assert forced["reserve_Wh"] == pytest.approx(
        vtol.VTOLConfig().battery.usable_Wh * 0.35, rel=1e-9)


def test_a_fixed_speed_run_reports_a_reserve(vtol):
    m = vtol.compute_metrics(vtol.VTOLConfig(reserve_percent=30.0))
    assert m["reserve_target_Wh"] == pytest.approx(m["usable_Wh"] * 0.30, rel=1e-9)
    assert m["reserve_margin_Wh"] == pytest.approx(m["usable_Wh"] * 0.70, rel=1e-9)


@pytest.mark.gui
def test_motor_current_limits_are_now_checked(gui):
    """The rated-current fields must produce a checked Status row."""
    gui.set_field("Lift motor max current (A)", "5")
    assert gui.click("Fixed Speed Sweep") == []
    rows = {str(r[0]): r for r in gui.status_rows()}
    row = rows["Lift motor current in hover"]
    assert "5.0" in row[2], row
    assert "Above" in row[3], row


@pytest.mark.gui
def test_the_mission_env_duplicates_are_gone(gui):
    import tkinter.ttk as ttk
    gui.use_advanced_inputs()
    labels = [str(w.cget("text")) for w in gui.widgets() if isinstance(w, ttk.Label)]
    assert "Avionics power (W)" not in labels
    assert labels.count("ESC efficiency") == 1


def test_legacy_configs_still_load_the_moved_fields(vtol):
    cfg = vtol.config_from_fields({"avionics": "25", "esc_eff": "0.9"})
    assert cfg.avionics_power_W == 25.0
    assert cfg.esc_efficiency == pytest.approx(0.9)
    # The tab fields win when both are present.
    cfg = vtol.config_from_fields({"avionics": "25", "avionics_flat": "12"})
    assert cfg.avionics_power_W == 12.0


# ======================================================================
# ONE BUILDER FOR GUI, CLI AND BATCH
# ======================================================================

def test_every_field_has_a_cli_flag(vtol):
    parser = vtol.build_arg_parser()
    dests = {a.dest for a in parser._actions}
    missing = [(k, d) for k, d in vtol.FIELD_TO_CLI.items() if d and d not in dests]
    assert not missing, f"fields with no CLI flag: {missing}"


def test_the_cli_reads_a_saved_config_to_the_same_aircraft(vtol):
    path = os.path.join(ROOT, "examples", "configs", "vtol_trinity_f90_lift_cruise.json")
    values, ctype = vtol.load_fields_file(path)
    direct = vtol.compute_metrics(vtol.config_from_fields(values, ctype))
    args = vtol.build_arg_parser().parse_args(["--config", path])
    via_cli = vtol.compute_metrics(vtol.config_from_args(args))
    assert via_cli["total_power_W"] == pytest.approx(direct["total_power_W"], rel=1e-12)
    # A flag overrides the file.
    args = vtol.build_arg_parser().parse_args(["--config", path, "--cruise_speed", "20"])
    assert vtol.config_from_args(args).cruise_speed_mps == 20.0


@pytest.mark.gui
def test_gui_and_cli_agree_on_an_example_config(gui):
    """Load a config in the GUI and on the command line: same cruise power."""
    path = os.path.join(ROOT, "examples", "configs", "vtol_2m4_lift_cruise_survey.json")
    gui.open_with(path)
    assert gui.click("Load Config") == []
    assert gui.click("Fixed Speed Sweep") == []
    shown = {str(r[0]).strip(): str(r[1]) for r in gui.metric_rows()}["Cruise power"]
    out = subprocess.run([sys.executable, VTOL_SCRIPT, "--config", path],
                         capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=120)
    assert out.returncode == 0, out.stderr
    printed = re.search(r"Cruise power\s*:\s*([\d.]+)\s*W", out.stdout).group(1)
    assert shown.split()[0] == printed, (shown, printed)


# ======================================================================
# OUTPUTS CARRIED OVER FROM THE MULTICOPTER AND FIXED-WING
# ======================================================================

EXTENDED_KEYS = ("hover_lift_rpm", "hover_lift_current_A", "hover_lift_throttle",
                 "hover_lift_tip_mach", "cruise_motor_rpm", "hover_efficiency_gW",
                 "hover_figure_of_merit", "lift_twr", "ld_cruise", "ld_max",
                 "cl_cruise", "reynolds_number", "max_roc_mps", "service_ceiling_m",
                 "takeoff_roll_m", "landing_distance_m", "best_endurance_speed_mps",
                 "best_range_speed_mps", "glide_ratio", "min_sink_rate_mps",
                 "motor_temp_est_C", "esc_temp_est_C", "battery_temp_est_C",
                 "thermal_status", "density_altitude_m", "reserve_target_Wh",
                 "propulsive_efficiency", "system_efficiency")


@pytest.mark.parametrize("config_type", ["lift+cruise", "tiltrotor", "tiltwing", "tailsitter"])
def test_every_configuration_reports_the_extended_metrics(vtol, config_type):
    m = vtol.compute_metrics(vtol.VTOLConfig(config_type=config_type))
    missing = [k for k in EXTENDED_KEYS if k not in m]
    assert not missing, missing
    for key in EXTENDED_KEYS:
        if key in ("thermal_status", "service_ceiling_m", "takeoff_roll_m"):
            continue
        assert math.isfinite(float(m[key])), f"{key} = {m[key]}"


def test_a_turn_raises_the_stall_speed_and_the_power(vtol):
    straight = vtol.compute_metrics(vtol.VTOLConfig())
    banked = vtol.compute_metrics(vtol.VTOLConfig(bank_deg=40.0))
    assert banked["load_factor"] == pytest.approx(1.0 / math.cos(math.radians(40.0)))
    assert banked["turn_stall_speed_mps"] > straight["stall_speed_mps"]
    assert banked["turn_power_W"] > straight["total_power_W"]


def test_every_mission_series_is_the_same_length(vtol):
    _r, totals = vtol.simulate_mission(vtol.VTOLConfig(), vtol.VTOLMission.from_json(MISSION))
    series = totals["series"]
    n = len(series["t_s"])
    assert n > 10
    uneven = {k: len(v) for k, v in series.items() if len(v) != n}
    assert not uneven, uneven
    for key in ("groundspeed_mps", "lift_motor_current_A", "motor_temp_est_C",
                "battery_voltage_V", "cl_wing"):
        assert key in series


def test_mission_temperatures_rise_from_ambient(vtol):
    cfg = vtol.VTOLConfig()
    _r, totals = vtol.simulate_mission(cfg, vtol.VTOLMission.from_json(MISSION))
    assert totals["worst"]["motor_temp_est_C"] > cfg.ambient_temp_C
    assert totals["series"]["motor_temp_est_C"][0] >= cfg.ambient_temp_C


def test_simple_view_carries_the_other_simulators_simple_inputs(vtol):
    """
    Every input the multicopter or fixed-wing shows in Simple view has a
    VTOL counterpart that is also in Simple view.
    """
    expected = {
        "mass_mode", "structure_mass", "avionics_mass", "plot_vmax", "payload",
        "oswald", "mu_roll", "mu_brake", "cl_takeoff", "cruise_eff", "cruise_eff_model",
        "unit_mode", "vmin", "vnom", "vmax", "pack_cap", "pack_wt",
        "cells_s_per_pack", "cells_p_per_pack", "usable", "rcell", "c_cont",
        "a_cont", "soc_model", "lift_rm", "cruise_rm", "lift_i0", "cruise_i0",
        "lift_imax", "cruise_imax", "lift_pmax", "cruise_pmax", "esc_cont",
        "esc_imax", "esc_wt", "avionics_flat", "periph_current", "lift_blades",
        "cruise_blades", "lift_max_thrust", "cruise_max_thrust", "lift_table",
        "cruise_table", "lift_prop_wt", "cruise_prop_wt", "reserve_percent",
        "course_deg", "cruise_altitude", "accel", "decel", "bank_deg",
        "lift_layout", "coax_spacing", "max_tilt", "drag_model_mode",
        "parasite_drag", "parasite_area", "profile_drag", "profile_area",
        "body_length_m", "body_width_m", "body_height_m", "arm_length_m",
        "arm_width_m"}
    assert not expected - vtol.VTOL_SIMPLE_FIELDS, expected - vtol.VTOL_SIMPLE_FIELDS


@pytest.mark.gui
def test_the_metrics_tab_carries_the_carried_over_sections(gui):
    assert gui.click("Fixed Speed Sweep") == []
    sections = gui.metric_sections()
    for expected in ("Thrust & Power", "Climb & Glide", "Turning Flight",
                     "Conventional Take-off / Landing", "Lift Motor (hover)",
                     "Cruise Motor (at cruise)", "Propellers & Rotors",
                     "Thermal Estimates", "Environment"):
        assert expected in sections, f"{expected!r} missing: {sections}"


@pytest.mark.gui
def test_the_sweep_carries_the_fixed_wing_panels(gui):
    assert gui.click("Fixed Speed Sweep") == []
    fig = _current_sweep_figure(gui)
    titles = " | ".join(ax.get_title() for ax in fig.axes)
    for expected in ("Thrust Required vs Available", "Rate of Climb",
                     "Drag vs Airspeed", "Drag Polar"):
        assert expected in titles, titles


# ======================================================================
# GREYED-OUT INPUTS AND THE AIRFRAME NAMING
#
# As on the multicopter and fixed-wing, an input the current dropdown
# selection does not use is greyed out, so a number cannot be typed that
# silently does nothing.
# ======================================================================

def _field_widget(gui, label):
    """The entry or picker beside a label, even while it is hidden."""
    import tkinter.ttk as ttk
    gui.use_advanced_inputs()
    for w in gui.widgets():
        if isinstance(w, ttk.Label) and str(w.cget("text")) == label:
            row = w.grid_info().get("row")
            for sib in w.master.winfo_children():
                if (sib is not w and not isinstance(sib, ttk.Label)
                        and sib.grid_info().get("row") == row):
                    return sib
    raise AssertionError(f"no field labelled {label!r}")


def _disabled(gui, label):
    return _field_widget(gui, label).instate(["disabled"])


@pytest.mark.gui
def test_cell_mode_greys_out_the_pack_fields_and_back(gui):
    pack_fields = ("Cells in series per pack", "Cells in parallel per pack",
                   "Pack capacity (mAh)", "Pack weight (g)")
    assert all(_disabled(gui, f) for f in pack_fields), "pack fields live in cell mode"
    assert not _disabled(gui, "Cell capacity (mAh)")
    gui.set_choice("Entry mode", "pack")
    assert not any(_disabled(gui, f) for f in pack_fields), "pack mode left them grey"
    assert _disabled(gui, "Cell capacity (mAh)") and _disabled(gui, "Cell weight (g)")


@pytest.mark.gui
def test_mass_mode_greys_out_whichever_mass_is_calculated(gui):
    weight, frame = "All-up weight without payload (g)", "Airframe mass (g)"
    assert not _disabled(gui, weight) and _disabled(gui, frame)
    gui.set_choice("Mass Entry Mode", "enter airframe")
    assert _disabled(gui, weight) and not _disabled(gui, frame)


@pytest.mark.gui
def test_a_vectored_type_greys_out_the_cruise_motor(gui):
    assert not _disabled(gui, "Cruise motor Kv")
    gui.set_type("tailsitter")
    assert _disabled(gui, "Cruise motor Kv")
    assert _disabled(gui, "Cruise prop table (CSV)")
    assert not _disabled(gui, "Cruise prop efficiency"), \
        "the vectored types still use the propeller efficiency"


@pytest.mark.gui
def test_greying_out_changes_no_result(gui):
    """Greyed inputs keep their values and the model already ignores them."""
    assert gui.click("Fixed Speed Sweep") == []
    before = {r[0]: r[1] for r in gui.metric_rows()}
    gui.set_choice("Entry mode", "pack")
    gui.set_choice("Entry mode", "cell")
    assert gui.click("Fixed Speed Sweep") == []
    assert {r[0]: r[1] for r in gui.metric_rows()} == before


def test_old_structure_mode_names_still_load(vtol):
    values = vtol.migrate_legacy_fields({"mass_mode": "enter structure"})
    assert values["mass_mode"] == "enter airframe"
    cfg = vtol.config_from_fields({"mass_mode": "enter structure",
                                   "structure_mass": "1000", "cell_wt": "0",
                                   "lift_wt": "0", "cruise_wt": "0"})
    assert cfg.aircraft_weight_g == pytest.approx(1000.0)


@pytest.mark.gui
def test_wiring_tab_has_temperature_and_voltage_ratings(gui):
    for label in ("Wire temperature limit (°C)", "   rated voltage (V)"):
        assert _field_widget(gui, label) is not None, f"no {label!r} field"


@pytest.mark.gui
def test_wiring_reaches_the_shared_status_checks(gui):
    """The lead and the connectors get the same Status rows as the other two."""
    gui.use_advanced_inputs()
    gui.set_field("Wire length one-way (m)", "0.5")
    gui.set_field("Wire gauge (AWG)", "16")
    gui.set_field("Wire temperature limit (°C)", "105")
    gui.set_field("   continuous (A)", "60")
    gui.set_field("   rated voltage (V)", "12")
    assert gui.click("Fixed Speed Sweep") == []
    rows = {r[0]: r for r in gui.status_rows()}
    for name in ("Main wire voltage drop (hover)", "Wire temperature (est) (hover)",
                 "Battery connector current (hover)", "Battery connector voltage"):
        assert name in rows, f"{name!r} missing from Status; have {sorted(rows)}"
    assert rows["Battery connector voltage"][2] == "<= 12 V"


# ----------------------------------------------------------------------
# Audit V1: turn() priced the forward thrust with cruise_prop_power_W,
# the lift+cruise pusher's fields, also on vectored types, where those
# fields are hidden and the lift rotors make the thrust.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("config_type", ALL_TYPES)
def test_a_wings_level_turn_costs_level_flight(vtol, config_type):
    cfg = vtol.VTOLConfig(config_type=config_type)
    v = cfg.cruise_speed_mps
    level = vtol.power_at_airspeed(cfg, v)["total_power_W"]
    assert vtol.turn(cfg, v, 0.0)["turn_power_W"] == pytest.approx(level, rel=1e-9)
    assert vtol.turn(cfg, v, 30.0)["turn_power_W"] > level


@pytest.mark.parametrize("config_type", VECTORED)
def test_vectored_turn_ignores_the_hidden_pusher(vtol, config_type):
    cfg = vtol.VTOLConfig(config_type=config_type)
    v = cfg.cruise_speed_mps
    before = vtol.turn(cfg, v, 30.0)["turn_power_W"]
    cfg.cruise_prop_diameter_in = cfg.cruise_prop_diameter_in * 0.5
    cfg.num_cruise_motors = 3
    assert vtol.turn(cfg, v, 30.0)["turn_power_W"] == pytest.approx(before, rel=1e-12)


# ----------------------------------------------------------------------
# Audit V3: the take-off roll repeated the fixed-wing's double margin.
# ----------------------------------------------------------------------

def test_vtol_takeoff_roll_matches_raymer_at_the_textbook_lift_off(vtol):
    cfg = vtol.VTOLConfig(config_type="lift+cruise")
    cfg.CL_takeoff = cfg.CL_max / 1.44
    w = cfg.weight_N
    v_lof = 1.2 * vtol.stall_speed_mps(cfg)
    net = vtol.forward_thrust_available_N(cfg, 0.707 * v_lof) - cfg.mu_roll * w
    raymer = 1.44 * w ** 2 / (vtol.G0 * cfg.air_density * cfg.wing_area_m2 * cfg.CL_max * net)
    assert vtol.takeoff_roll_m(cfg) == pytest.approx(raymer, rel=1e-6)
