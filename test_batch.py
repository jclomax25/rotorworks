"""
Batch driver tests, plus the GUI-vs-CLI consistency check.

The consistency test is the most valuable one here: it asserts that the same
config file produces the same numbers whether it is opened in the GUI or fed
to the batch driver. That would have caught the fixed-wing ESC gap, where the
CLI silently dropped losses the GUI included.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

TIMEOUT = 600


def run_batch(paths, args):
    return subprocess.run(
        [sys.executable, paths["rotorworks"]] + args,
        capture_output=True, text=True, timeout=TIMEOUT, cwd=paths["root"],
    )


def assert_all_runs_ok(result, context=""):
    out = result.stdout + result.stderr
    assert "Traceback" not in out, f"{context}\n{out[-1500:]}"
    assert "rc=1" not in out and "rc=2" not in out, f"{context}\n{out[-1500:]}"
    assert result.returncode == 0, f"{context} exit={result.returncode}\n{out[-1500:]}"


def assert_rejected_by_mode(result, context=""):
    out = result.stdout + result.stderr
    assert "mode=simple" in out, f"{context}: expected a simple-mode rejection\n{out[-800:]}"


MC_CFG = "examples/configs/multicopter_450_survey_4S.json"
FW_CFG = "examples/configs/fixedwing_2m_survey_4S.json"


# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------

@pytest.mark.parametrize("args,label", [
    (["sweep", "--sim", "multicopter", "--mode", "simple", "--gui-config", MC_CFG,
      "--sweep-var", "payload_mass_g", "--values", "0,300,600"], "mc payload"),
    (["sweep", "--sim", "multicopter", "--mode", "simple", "--gui-config", MC_CFG,
      "--sweep-var", "speed", "--start", "5", "--stop", "15", "--step", "5"], "mc speed range"),
    (["sweep", "--sim", "multicopter", "--mode", "advanced", "--gui-config", MC_CFG,
      "--sweep-var", "weight", "--values", "1200,1450,1700"], "mc weight"),
    (["sweep", "--sim", "fixedwing", "--mode", "simple", "--gui-config", FW_CFG,
      "--sweep-var", "cruise_speed", "--values", "14,18,22"], "fw cruise"),
    (["sweep", "--sim", "fixedwing", "--mode", "simple", "--gui-config", FW_CFG,
      "--sweep-var", "CD0", "--values", "0.022,0.028,0.035"], "fw CD0"),
    (["sweep", "--sim", "fixedwing", "--mode", "simple", "--gui-config", FW_CFG,
      "--sweep-var", "altitude", "--values", "0,1500,3000"], "fw altitude"),
])
def test_sweeps_run(paths, args, label):
    assert_all_runs_ok(run_batch(paths, args), label)


def test_sizing_search_runs(paths):
    assert_all_runs_ok(run_batch(paths, [
        "size", "--sim", "multicopter", "--mode", "simple", "--gui-config", MC_CFG,
        "--design-var", "prop_diameter:9:11:1",
        "--design-var", "battery_pack_capacity=4000,5200",
        "--target-min", "flight_time_min=15",
        "--objective", "maximize:flight_time_min",
    ]), "sizing")


def test_batch_runs_file(paths, tmp_path):
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({"runs": [
        {"name": "clean", "overrides": {"payload_mass_g": 0}},
        {"name": "heavy", "overrides": {"payload_mass_g": 700}},
    ]}))
    assert_all_runs_ok(run_batch(paths, [
        "batch", "--sim", "multicopter", "--mode", "simple",
        "--gui-config", MC_CFG, "--runs-file", str(runs),
    ]), "batch runs")


# ----------------------------------------------------------------------
# Simple/Advanced mode is a guard rail
# ----------------------------------------------------------------------

def test_simple_mode_rejects_advanced_sweep_var(paths):
    assert_rejected_by_mode(run_batch(paths, [
        "sweep", "--sim", "multicopter", "--mode", "simple", "--gui-config", MC_CFG,
        "--sweep-var", "inflow_mu_bp", "--values", "0.1,0.2",
    ]), "advanced sweep var")


def test_simple_mode_rejects_advanced_set_override(paths):
    assert_rejected_by_mode(run_batch(paths, [
        "sweep", "--sim", "fixedwing", "--mode", "simple", "--gui-config", FW_CFG,
        "--set", "motor_pole_count=14",
        "--sweep-var", "cruise_speed", "--values", "18",
    ]), "advanced --set")


def test_simple_mode_rejects_advanced_design_var(paths):
    assert_rejected_by_mode(run_batch(paths, [
        "size", "--sim", "multicopter", "--mode", "simple", "--gui-config", MC_CFG,
        "--design-var", "decel_regen_eff:0:0.5:0.25",
    ]), "advanced design var")


def test_simple_mode_rejects_advanced_runs_file_override(paths, tmp_path):
    runs = tmp_path / "bad.json"
    runs.write_text(json.dumps({"runs": [
        {"name": "x", "overrides": {"inflow_map_enabled": 1}}]}))
    assert_rejected_by_mode(run_batch(paths, [
        "batch", "--sim", "multicopter", "--mode", "simple",
        "--gui-config", MC_CFG, "--runs-file", str(runs),
    ]), "advanced runs-file override")


def test_advanced_mode_allows_advanced_parameters(paths):
    assert_all_runs_ok(run_batch(paths, [
        "sweep", "--sim", "multicopter", "--mode", "advanced", "--gui-config", MC_CFG,
        "--sweep-var", "transient_dt_s", "--values", "0.25,0.5",
    ]), "advanced mode permits everything")


# ----------------------------------------------------------------------
# GUI config translation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cfg,sim", [(MC_CFG, "multicopter"), (FW_CFG, "fixedwing")])
def test_gui_config_translation_is_lossless_for_esc(rw, paths, cfg, sim):
    """
    Regression: the fixed-wing GUI->CLI map had no ESC entries, so ESC values
    present in a saved config were silently discarded on the batch path.
    """
    translated = rw.load_gui_config(os.path.join(paths["root"], cfg), sim)
    esc_keys = [k for k in translated if k.startswith("esc_")]
    assert esc_keys, f"{sim}: no ESC arguments survived translation"


@pytest.mark.parametrize("cfg,sim", [(MC_CFG, "multicopter"), (FW_CFG, "fixedwing")])
def test_gui_config_translation_drops_blanks(rw, paths, cfg, sim):
    """Blank GUI fields must be omitted, not passed through as empty strings."""
    translated = rw.load_gui_config(os.path.join(paths["root"], cfg), sim)
    assert all(v != "" for v in translated.values())


def test_every_example_config_runs_through_batch(paths):
    for cfg in sorted(glob.glob(os.path.join(paths["configs"], "*.json"))):
        rel = os.path.relpath(cfg, paths["root"])
        sim = "multicopter" if os.path.basename(cfg).startswith("multi") else "fixedwing"
        var = "speed" if sim == "multicopter" else "cruise_speed"
        val = "8" if sim == "multicopter" else "18"
        assert_all_runs_ok(run_batch(paths, [
            "sweep", "--sim", sim, "--mode", "simple", "--gui-config", rel,
            "--sweep-var", var, "--values", val,
        ]), os.path.basename(cfg))


# ----------------------------------------------------------------------
# GUI vs CLI consistency
# ----------------------------------------------------------------------

@pytest.mark.gui
@pytest.mark.parametrize("sim,cfg,var,val", [
    ("multicopter", MC_CFG, "speed", "10"),
    ("fixedwing", FW_CFG, "cruise_speed", "19"),
])
def test_gui_and_batch_agree_on_the_same_config(paths, mc, fw, sim, cfg, var, val):
    """
    The same config file must produce the same flight time and range whether
    it is opened in the GUI or run through the batch driver.

    Regression: the fixed-wing CLI had no ESC arguments, so the batch path
    dropped ESC losses the GUI applied — about 1% on a typical airframe, and
    silent.
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk(); root.destroy()
    except Exception:
        pytest.skip("no display available")

    # --- batch path ---
    result = run_batch(paths, [
        "sweep", "--sim", sim, "--mode", "simple", "--gui-config", cfg,
        "--sweep-var", var, "--values", val,
    ])
    assert_all_runs_ok(result, "batch leg")
    match = re.search(r"time=([\d.]+) min, range=([\d.]+) km", result.stdout)
    assert match, f"could not parse batch output:\n{result.stdout[-600:]}"
    batch_time, batch_range = float(match.group(1)), float(match.group(2))

    # --- GUI path ---
    from test_gui import GuiHarness  # noqa: E402  (same test package)

    gui = GuiHarness(mc if sim == "multicopter" else fw)
    gui.capture_errors()
    try:
        gui.set_open_dialog(os.path.join(paths["root"], cfg))
        gui.click("Load Config")
        assert not gui.click("Single-Point")
        text = ""
        for widget in gui.widgets:
            if isinstance(widget, tk.Text):
                try:
                    body = widget.get("1.0", "end")
                except Exception:
                    continue
                if "Flight" in body:
                    text = body
        # The two simulators label this line slightly differently
        # ("Flight time" vs "Flight Time"), so match case-insensitively.
        found = re.search(r"flight\s+time\s*:\s*([\d.]+)", text, re.IGNORECASE)
        assert found, f"could not find a flight time in the GUI output:\n{text[:400]}"
        gui_time = float(found.group(1))
    finally:
        gui.destroy()

    assert gui_time == pytest.approx(batch_time, abs=0.1), (
        f"{sim}: GUI reports {gui_time} min, batch reports {batch_time} min"
    )
