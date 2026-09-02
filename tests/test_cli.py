"""
Command-line interface tests.

These run the simulators as real subprocesses, the way a user or a CI job
would. They catch a whole class of bug that importing the module cannot:
missing argparse entries, arguments that are read but never defined, and
GUI-only features that were never wired into the headless path.

Marked `slow` because each case spawns a Python process.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

TIMEOUT = 300


# ----------------------------------------------------------------------
# Minimal fully-valid argument sets
# ----------------------------------------------------------------------

MC_BASE = [
    "--num_motors", "4", "--weight", "1450",
    "--battery_unit_mode", "pack",
    "--battery_pack_capacity", "5200", "--battery_pack_weight_g", "520",
    "--battery_series_units", "1", "--battery_parallel_units", "1",
    "--battery_cells_series_per_unit", "4",
    "--battery_operating_voltage_min", "3.3",
    "--battery_operating_voltage_nominal", "3.7",
    "--battery_operating_voltage_max", "4.2",
    "--battery_resistance_cell", "4.0",
    "--battery_discharge_percent", "80",
    "--battery_discharge_c_cont", "25",
    "--motor_kv", "920", "--motor_resistance", "0.115",
    "--prop_diameter", "10", "--prop_pitch", "4.5",
]

FW_BASE = [
    "--weight", "2600", "--wing_span", "2.0", "--wing_area", "0.46",
    "--CD0", "0.028", "--CL_max", "1.25", "--oswald", "0.87",
    "--battery_unit_mode", "pack",
    "--battery_pack_capacity", "10000", "--battery_pack_weight_g", "880",
    "--battery_series_units", "1", "--battery_cells_series_per_unit", "4",
    "--battery_operating_voltage_min", "3.3",
    "--battery_operating_voltage_nominal", "3.7",
    "--battery_operating_voltage_max", "4.2",
    "--battery_resistance_cell", "3.5",
    "--battery_discharge_percent", "80",
    "--battery_discharge_c_cont", "15",
    "--motor_kv", "750", "--motor_resistance", "0.06",
    "--prop_diameter", "12", "--prop_pitch", "8",
]


def run_sim(script: str, args: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script] + args,
        capture_output=True, text=True, timeout=TIMEOUT,
    )


def assert_clean(result: subprocess.CompletedProcess, context: str = ""):
    """A successful run: exit 0, no traceback anywhere in the output."""
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, f"{context}\n{combined[-1500:]}"
    assert result.returncode == 0, f"{context} exit={result.returncode}\n{combined[-1500:]}"


def assert_no_crash(result: subprocess.CompletedProcess, context: str = ""):
    """A handled failure: may exit non-zero, but must not dump a traceback."""
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, f"{context}\n{combined[-1500:]}"


# ----------------------------------------------------------------------
# Multicopter
# ----------------------------------------------------------------------

@pytest.mark.parametrize("extra,label", [
    (["--speed", "10"], "cruise"),
    (["--speed", "0", "--orientation", "hover"], "hover"),
    (["--speed", "25"], "high speed"),
    (["--speed", "10", "--altitude", "3000"], "altitude"),
    (["--speed", "10", "--temperature", "45"], "hot day"),
    (["--speed", "10", "--temperature", "-20"], "cold day"),
    (["--speed", "10", "--pressure", "95000"], "pressure override"),
    (["--speed", "10", "--payload_mass_g", "800"], "payload"),
    (["--speed", "10", "--wind", "8", "--wind_direction_deg", "270",
      "--course_deg", "90"], "wind"),
    (["--speed", "10", "--num_motors", "8", "--motor_configuration", "coaxial",
      "--coaxial_spacing_m", "0.09"], "coaxial X8"),
    (["--speed", "10", "--avionics_voltage_tree", "5.0:(2,0.9), 12.0:(1.5,0.85)"],
     "avionics rails"),
    (["--speed", "10", "--esc_voltage_rating", "4", "--esc_cont_current", "30",
      "--esc_max_current", "40", "--esc_resistance", "0.002"], "esc"),
    (["--speed", "10", "--prop_max_rpm", "9000", "--prop_max_thrust", "1100"],
     "prop limits"),
    (["--speed", "10", "--battery_soc_model", "linear"], "linear soc"),
    (["--speed", "10", "--climb_rate_mps", "2.0"], "climb"),
])
def test_multicopter_cli(paths, extra, label):
    assert_clean(run_sim(paths["multicopter"], MC_BASE + extra), label)


@pytest.mark.parametrize("mission", sorted(
    glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples", "missions", "mc_*.json"))))
def test_multicopter_missions(paths, mission):
    assert_clean(run_sim(paths["multicopter"], MC_BASE + ["--mission", mission]),
                 os.path.basename(mission))


# ----------------------------------------------------------------------
# Fixed-wing
# ----------------------------------------------------------------------

@pytest.mark.parametrize("extra,label", [
    (["--cruise_speed", "18"], "cruise"),
    (["--cruise_speed", "12"], "slow"),
    (["--cruise_speed", "30"], "fast"),
    (["--cruise_speed", "18", "--altitude", "2500"], "altitude"),
    (["--cruise_speed", "18", "--pressure", "95000"], "pressure override"),
    (["--cruise_speed", "18", "--cruise_altitude", "500"], "cruise altitude"),
    (["--cruise_speed", "18", "--bank_deg", "30"], "banked turn"),
    (["--cruise_speed", "18", "--bank_deg", "55"], "steep bank"),
    (["--cruise_speed", "18", "--payload_mass_g", "600"], "payload"),
    (["--cruise_speed", "18", "--wind", "10", "--wind_direction_deg", "180",
      "--course_deg", "0"], "wind"),
    (["--cruise_speed", "18", "--avionics_voltage_tree",
      "5.0:(1.4,0.9), 12.0:(0.8,0.88)"], "avionics rails"),
    (["--cruise_speed", "18", "--esc_voltage_rating", "4",
      "--esc_cont_current", "60", "--esc_max_current", "80",
      "--esc_idle_current", "0.05", "--esc_resistance", "0.003",
      "--esc_weight", "60"], "esc"),
    (["--cruise_speed", "18", "--prop_eff_model", "constant"], "constant prop eff"),
    (["--cruise_speed", "18", "--prop_eff_model", "curve"], "curve prop eff"),
    (["--cruise_speed", "18", "--battery_soc_model", "linear"], "linear soc"),
    (["--cruise_speed", "18", "--battery_soc_bp", "0,50,100",
      "--battery_ocv_cell_bp", "3.2,3.8,4.2",
      "--battery_r_scale_bp", "2.0,1.0,1.2"], "custom soc curve"),
])
def test_fixedwing_cli(paths, extra, label):
    assert_clean(run_sim(paths["fixedwing"], FW_BASE + extra), label)


@pytest.mark.parametrize("mission", sorted(
    glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples", "missions", "fw_*.json"))))
def test_fixedwing_missions(paths, mission):
    """
    Regression: the fixed-wing CLI had no --mission argument at all, so every
    one of these failed with 'unrecognized arguments' while the GUI ran them.
    """
    assert_clean(run_sim(paths["fixedwing"], FW_BASE + ["--mission", mission]),
                 os.path.basename(mission))


# ----------------------------------------------------------------------
# Both simulators expose --help without importing tkinter
# ----------------------------------------------------------------------

@pytest.mark.parametrize("key", ["multicopter", "fixedwing"])
def test_help_works(paths, key):
    assert_clean(run_sim(paths[key], ["--help"]), f"{key} --help")


# ----------------------------------------------------------------------
# Optional arguments really are optional
# ----------------------------------------------------------------------

def test_optional_ratings_can_be_omitted(paths):
    """
    Regression: reference-only values (motor max current/power, ESC specs,
    prop limits, charge current, energy density) were all mandatory, so a
    minimal but physically complete run was rejected.
    """
    assert_clean(run_sim(paths["multicopter"], MC_BASE + ["--speed", "10"]),
                 "minimal multicopter")
    assert_clean(run_sim(paths["fixedwing"], FW_BASE + ["--cruise_speed", "18"]),
                 "minimal fixed-wing")


def test_cell_and_pack_capacity_are_alternatives(paths):
    """Cell mode must not demand a pack capacity, or vice versa."""
    cell_args = [
        "--num_motors", "4", "--weight", "1450", "--speed", "10",
        "--battery_unit_mode", "cell",
        "--battery_cell_capacity", "5000", "--battery_cell_weight_g", "120",
        "--battery_series_units", "6", "--battery_parallel_units", "2",
        "--battery_operating_voltage_min", "3.3",
        "--battery_operating_voltage_nominal", "3.7",
        "--battery_operating_voltage_max", "4.2",
        "--battery_resistance_cell", "3.0",
        "--battery_discharge_c_cont", "25",
        "--motor_kv", "400", "--motor_resistance", "0.05",
        "--prop_diameter", "15", "--prop_pitch", "5",
    ]
    assert_clean(run_sim(paths["multicopter"], cell_args), "cell mode")


# ----------------------------------------------------------------------
# Bad input must be handled, not dumped as a traceback
# ----------------------------------------------------------------------

def test_missing_mission_file_is_reported_cleanly(paths, tmp_path):
    missing = str(tmp_path / "does_not_exist.json")
    for key, base, extra in (("multicopter", MC_BASE, []),
                             ("fixedwing", FW_BASE, ["--cruise_speed", "18"])):
        r = run_sim(paths[key], base + extra + ["--mission", missing])
        assert_no_crash(r, f"{key} missing mission")
        assert "not found" in (r.stdout + r.stderr).lower()


def test_malformed_mission_file_is_reported_cleanly(paths, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"phases": [{"name": "x"}]}')
    r = run_sim(paths["multicopter"], MC_BASE + ["--mission", str(bad)])
    assert_no_crash(r, "malformed mission")


def test_invalid_json_is_reported_cleanly(paths, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    r = run_sim(paths["multicopter"], MC_BASE + ["--mission", str(broken)])
    assert_no_crash(r, "invalid json")
    assert "json" in (r.stdout + r.stderr).lower()


@pytest.mark.parametrize("extra,label", [
    (["--speed", "5", "--weight", "25000"], "overweight"),
    (["--speed", "5", "--payload_mass_g", "20000"], "huge payload"),
    (["--speed", "5", "--weight", "150"], "tiny"),
    (["--speed", "10", "--altitude", "8000"], "very high"),
    (["--speed", "5", "--wind", "15"], "wind exceeds speed"),
    (["--speed", "15", "--max_tilt_deg", "5"], "tight tilt limit"),
    (["--speed", "5", "--num_motors", "1"], "single motor"),
    (["--speed", "5", "--num_motors", "16"], "sixteen motors"),
])
def test_multicopter_edge_cases(paths, extra, label):
    assert_no_crash(run_sim(paths["multicopter"], MC_BASE + extra), label)


@pytest.mark.parametrize("extra,label", [
    (["--cruise_speed", "5"], "below stall"),
    (["--cruise_speed", "45"], "very fast"),
    (["--cruise_speed", "20", "--weight", "15000"], "overweight"),
    (["--cruise_speed", "18", "--altitude", "6000"], "very high"),
    (["--cruise_speed", "18", "--bank_deg", "70"], "extreme bank"),
    (["--cruise_speed", "18", "--CD0", "0.10"], "very draggy"),
    (["--cruise_speed", "18", "--CL_max", "0.5"], "low CL_max"),
    (["--cruise_speed", "18", "--wind", "20"], "strong headwind"),
])
def test_fixedwing_edge_cases(paths, extra, label):
    assert_no_crash(run_sim(paths["fixedwing"], FW_BASE + extra), label)
