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
def test_unimplemented_configurations_refuse(vtol, config_type):
    """
    These must raise, not silently fall back to lift+cruise. Their transition
    physics differs enough that an approximation would be confidently wrong.
    """
    cfg = vtol.VTOLConfig(config_type=config_type)
    with pytest.raises(NotImplementedError) as excinfo:
        vtol.compute_metrics(cfg)
    assert config_type in str(excinfo.value)
    assert "lift+cruise" in str(excinfo.value), "message should say what IS available"


def test_all_config_types_are_selectable(vtol):
    """The dropdown must offer every planned configuration."""
    for name in ("lift+cruise", "tiltrotor", "tiltwing", "tailsitter"):
        assert name in vtol.CONFIG_TYPES
    assert vtol.IMPLEMENTED_CONFIG_TYPES == {"lift+cruise"}


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


def test_hover_thrust_equals_weight(vtol, aircraft):
    point = vtol.hover_power_W(aircraft)
    assert point["rotor_thrust_N"] == pytest.approx(aircraft.weight_N, rel=1e-9)
    assert point["wing_lift_N"] == 0.0


def test_hover_power_matches_momentum_theory(vtol, aircraft):
    """P = T * sqrt(T / 2*rho*A) / FM, plus ESC and avionics."""
    thrust = aircraft.weight_N
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
    # The extra is the potential power, weight x climb rate, through the ESC.
    extra = aircraft.weight_N * 2.5 / aircraft.esc_efficiency
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
    """The aircraft must be held up at every point in the transition."""
    for speed in (0.5, 3, 6, 9, 12, 13):
        point = vtol.transition_power_W(aircraft, speed)
        total = point["rotor_thrust_N"] + point["wing_lift_N"]
        assert total == pytest.approx(aircraft.weight_N, rel=1e-9), \
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
def test_cli_rejects_an_unimplemented_configuration():
    result = subprocess.run(
        [sys.executable, VTOL_SCRIPT, "--config_type", "tailsitter"],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, "should refuse cleanly, not crash"
    assert "not implemented" in output.lower()


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
