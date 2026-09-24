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
    P = T * sqrt(T / 2*rho*A) / FM, plus ESC and avionics, where T is the
    weight PLUS the download the rotors also have to lift.
    """
    thrust = aircraft.weight_N * (1.0 + vtol.hover_download_fraction(aircraft))
    area = aircraft.lift_disc_area_m2
    per_rotor = thrust / aircraft.num_lift_rotors
    disc_per = area / aircraft.num_lift_rotors
    v_hover = math.sqrt(per_rotor / (2 * aircraft.air_density * disc_per))
    ideal = thrust * v_hover
    shaft = ideal / aircraft.lift_figure_of_merit
    expected = shaft / aircraft.esc_efficiency + aircraft.avionics_power_W
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
    assert climb - hover == pytest.approx(extra, rel=1e-6)


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
            for w in self.widgets():
                if not isinstance(w, ttk.Treeview):
                    continue
                cols = [str(c) for c in w.cget("columns")]
                if cols and cols[0] == first_column and len(cols) == n_columns:
                    return w
            raise AssertionError(f"no tree starting {first_column!r} with {n_columns} columns")

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
    gui.open_with(MISSION)
    for w in gui.widgets():
        if str(type(w).__name__) == "Button" and str(w.cget("text")).startswith("Browse"):
            w.invoke()
            gui.pump()
            return
    # ttk buttons
    gui.click("Browse")


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
    assert gui.click("Single-Point") == []
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
    gui.click("Single-Point")
    before = {r[0]: r[1] for r in gui.rows("item", 3)}

    gui.set_field("Wire length one-way (m)", "3.0")
    gui.set_field("Wire gauge (AWG)", "16")
    assert gui.click("Single-Point") == []
    after = {r[0]: r[1] for r in gui.rows("item", 3)}

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
    gui.click("Single-Point")
    unrated = [r for r in gui.rows("metric", 4) if r[0] == "Hover C-rate"]
    assert unrated and "Not Specified" in unrated[0][2]

    gui.set_field("Continuous C-rate", "15")
    gui.set_field("Max / burst C-rate", "30")
    gui.click("Single-Point")
    rated = [r for r in gui.rows("metric", 4) if r[0] == "Hover C-rate"]
    assert rated and "cont 15" in rated[0][2], f"limit not applied: {rated}"


@pytest.mark.gui
def test_a_mission_clears_what_only_a_single_point_can_answer(gui):
    """
    A mission has no single operating point, so the Power Budget and the
    fixed-speed plots must empty with a note rather than leave a stale sweep
    on screen looking current.
    """
    gui.click("Single-Point")
    assert len(gui.rows("item", 3)) > 3, "power budget did not fill"

    _load_mission(gui)
    gui.click("Run Mission")

    budget = gui.rows("item", 3)
    assert len(budget) == 1 and "Mission run" in budget[0][0]
    assert gui.label_matching("come from a single-point run"), \
        "fixed speed plots were not cleared"


@pytest.mark.gui
def test_status_switches_between_point_and_worst_case(gui):
    """
    Single point reports the cruise condition; a mission reports the WORST
    value each check reached. Neither is an average, and the tab says which
    it is showing.
    """
    gui.click("Single-Point")
    assert any("Cruise pack current" == r[0] for r in gui.rows("metric", 4))
    assert gui.label_matching("Single-point run —")

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
    gui.click("Single-Point")
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
    gui.click("Single-Point")
    gui.click("Run Sensitivity")
    assert len(gui.rows("name", 5)) > 3, "sensitivity did not produce rankings"

    gui.click("Single-Point")
    stale = gui.rows("name", 5)
    assert len(stale) == 1 and "out of date" in stale[0][0]


@pytest.mark.gui
def test_weight_budget_flags_an_impossible_structure_mass(gui):
    """
    Airframe weight includes the battery and motors, so structure is the
    residual. If the itemised parts exceed it the residual is negative — an
    impossible aircraft — and that must be visible, not hidden.
    """
    gui.set_field("Lift motor weight (g)", "2000")
    gui.click("Single-Point")
    structure = [r for r in gui.rows("item", 5) if "structure" in r[0]]
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
    gui.click("Single-Point")

    target = tmp_path / "out.csv"
    filedialog.asksaveasfilename = lambda *a, **k: str(target)
    gui.invoke_menu("File", "Export CSV…")

    assert target.exists() and target.stat().st_size > 0
    text = target.read_text()
    for section in ("[Metrics]", "[Status]", "[Weight Budget]",
                    "[Power Budget]", "[Speed Sweep]"):
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
    gui.click("Single-Point")
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

    gui.click("Single-Point")
    advanced_rows = {r[0]: r[1] for r in gui.rows("metric", 4)}

    radio("Simple").invoke()
    gui.pump()
    assert visible() == simple_count, "returning to Simple did not restore the view"

    gui.click("Single-Point")
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


def test_transients_cost_time_and_energy(vtol, tmp_path, aircraft):
    """
    Accelerating costs power on top of steady drag, and a mission of short
    legs pays it at every speed change. Ignoring it makes a survey look
    cheaper than it flies.
    """
    mission = _legs_mission(vtol, tmp_path)
    _r, instant = vtol.simulate_mission(aircraft, mission)
    _r, ramped = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.5)

    assert ramped["time_s"] > instant["time_s"]
    assert ramped["energy_Wh"] > instant["energy_Wh"]


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
    needs about 190. The model reports the overshoot instead of quietly
    clamping it, because an unflyable pattern is worth knowing about.
    """
    mission = _legs_mission(vtol, tmp_path)
    results, _totals = vtol.simulate_mission(aircraft, mission, max_accel_mps2=1.0)
    flagged = [row[0] for row in results if "overshot" in str(row[-1])]
    assert flagged, "no leg flagged despite an impossible deceleration"
    assert all(name.startswith("Turn") for name in flagged), \
        f"the wrong legs were flagged: {flagged}"


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

    # The gap should be roughly the kinetic energy spread over the two times.
    kinetic_J = 0.5 * (aircraft.all_up_weight_g / 1000.0) * 22.0 ** 2
    expected_gap = (kinetic_J / 8.0 - kinetic_J / 20.0) / aircraft.esc_efficiency
    assert quick - slow == pytest.approx(expected_gap, rel=0.25)


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
