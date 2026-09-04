"""
Full coverage matrix: every example config and mission, through both the GUI
and the CLI, with and without a measured propeller table.

Why this file exists
--------------------
Two bugs reached the user because the suite had a hole in exactly this shape:

  * `name 'm' is not defined` — the Status tab's table-range check referenced
    a variable that exists in one simulator and not the other. It fired only
    when a table was loaded, and no GUI test loaded one, so 44 GUI tests
    passed while every table-backed run crashed.

  * The propeller-table loaders each carried their own NaN-handling fault.
    The multicopter's was found and fixed; the fixed-wing's survived another
    six releases because nothing exercised it.

Both were "works in one interface, or one table state, and not the other".
The fix is not another individual test — it is covering the product of the
dimensions that actually vary:

    {multicopter, fixed-wing} x {every config} x {no table, table}
                              x {every mission} x {GUI, CLI}

Marked `slow`; the CLI half spawns a subprocess per case.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

TIMEOUT = 300

HERE = os.path.dirname(os.path.abspath(__file__))
MC_TABLE = os.path.join(HERE, "data", "motor_prop_table.csv")
FW_TABLE = os.path.join(HERE, "data", "fw_motor_prop_table.csv")


def _configs(prefix):
    return sorted(glob.glob(os.path.join(
        os.path.dirname(HERE), "examples", "configs", f"{prefix}*.json")))


def _missions(prefix):
    return sorted(glob.glob(os.path.join(
        os.path.dirname(HERE), "examples", "missions", f"{prefix}_*.json")))


def _ids(paths):
    return [os.path.basename(p).replace(".json", "") for p in paths]


MC_CONFIGS, FW_CONFIGS = _configs("multicopter"), _configs("fixedwing")
MC_MISSIONS, FW_MISSIONS = _missions("mc"), _missions("fw")


# ======================================================================
# CLI HALF
# ======================================================================

MC_BASE = [
    "--num_motors", "4", "--weight", "1450",
    "--battery_unit_mode", "pack",
    "--battery_pack_capacity", "5200", "--battery_pack_weight_g", "520",
    "--battery_series_units", "1", "--battery_parallel_units", "1",
    "--battery_cells_series_per_unit", "4",
    "--battery_operating_voltage_min", "3.3",
    "--battery_operating_voltage_nominal", "3.7",
    "--battery_operating_voltage_max", "4.2",
    "--battery_resistance_cell", "4.0", "--battery_discharge_percent", "80",
    "--battery_discharge_c_cont", "25",
    "--motor_kv", "920", "--motor_resistance", "0.115",
    "--prop_diameter", "22", "--prop_pitch", "6.6",
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
    "--battery_resistance_cell", "3.5", "--battery_discharge_percent", "80",
    "--battery_discharge_c_cont", "15",
    "--motor_kv", "750", "--motor_resistance", "0.06",
    "--prop_diameter", "18", "--prop_pitch", "8",
]


def _run_cli(paths, sim, extra):
    script = paths["multicopter" if sim == "mc" else "fixedwing"]
    base = MC_BASE if sim == "mc" else FW_BASE
    return subprocess.run([sys.executable, script] + base + extra,
                          capture_output=True, text=True,
                          timeout=TIMEOUT, cwd=paths["root"])


def _assert_cli_clean(result, context):
    out = result.stdout + result.stderr
    assert "Traceback" not in out, f"{context}\n{out[-1200:]}"
    assert result.returncode == 0, f"{context} exit={result.returncode}\n{out[-1200:]}"
    lowered = out.lower()
    assert "nan" not in lowered.replace("n/a", ""), \
        f"{context}: NaN in output\n" + \
        "\n".join(l for l in out.split("\n") if "nan" in l.lower())[:400]


@pytest.mark.parametrize("with_table", [False, True], ids=["no-table", "table"])
@pytest.mark.parametrize("sim,speed_flag,speed", [
    ("mc", "--speed", "8"), ("fw", "--cruise_speed", "19")])
def test_cli_single_point(paths, sim, speed_flag, speed, with_table):
    """Every simulator runs a single point, with and without a table."""
    extra = [speed_flag, speed]
    if with_table:
        extra += ["--prop_table", MC_TABLE if sim == "mc" else FW_TABLE]
    _assert_cli_clean(_run_cli(paths, sim, extra),
                      f"{sim} single-point table={with_table}")


@pytest.mark.parametrize("with_table", [False, True], ids=["no-table", "table"])
@pytest.mark.parametrize("mission", MC_MISSIONS, ids=_ids(MC_MISSIONS))
def test_cli_multicopter_missions(paths, mission, with_table):
    extra = ["--mission", mission]
    if with_table:
        extra += ["--prop_table", MC_TABLE]
    _assert_cli_clean(_run_cli(paths, "mc", extra),
                      f"mc {os.path.basename(mission)} table={with_table}")


@pytest.mark.parametrize("with_table", [False, True], ids=["no-table", "table"])
@pytest.mark.parametrize("mission", FW_MISSIONS, ids=_ids(FW_MISSIONS))
def test_cli_fixedwing_missions(paths, mission, with_table):
    extra = ["--mission", mission]
    if with_table:
        extra += ["--prop_table", FW_TABLE]
    _assert_cli_clean(_run_cli(paths, "fw", extra),
                      f"fw {os.path.basename(mission)} table={with_table}")


def test_cli_table_actually_changes_the_answer(paths):
    """
    A table that is loaded but ignored would pass every "does it run" check.
    Confirm the numbers actually move.
    """
    def flight_time(sim, speed_flag, speed, table):
        extra = [speed_flag, speed]
        if table:
            extra += ["--prop_table", table]
        out = _run_cli(paths, sim, extra).stdout
        found = re.search(r"flight\s+time[^:\n]*:\s*([\d.]+)", out, re.IGNORECASE)
        assert found, f"no flight time in {sim} output"
        return float(found.group(1))

    for sim, flag, speed, table in (("mc", "--speed", "8", MC_TABLE),
                                    ("fw", "--cruise_speed", "19", FW_TABLE)):
        plain = flight_time(sim, flag, speed, None)
        tabled = flight_time(sim, flag, speed, table)
        assert plain != pytest.approx(tabled, rel=1e-6), \
            f"{sim}: the propeller table had no effect on flight time"


# ======================================================================
# GUI HALF
# ======================================================================
# One window per simulator for the whole module: building a Tk window costs
# seconds, and every case reloads a config anyway.

tk = pytest.importorskip("tkinter", reason="tkinter not installed")
from tkinter import ttk  # noqa: E402


def _display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


# NOTE: the skip is deliberately NOT module-level. The CLI half needs no
# display, and skipping the whole file on a headless box would silently drop
# half the coverage this file exists to provide.
def _require_display():
    if not _display_available():
        pytest.skip("no display; run under xvfb-run")


@pytest.fixture(scope="module")
def gui_mc(mc):
    _require_display()
    from test_gui import GuiHarness
    harness = GuiHarness(mc)
    harness.capture_errors()
    yield harness
    harness.destroy()


@pytest.fixture(scope="module")
def gui_fw(fw):
    _require_display()
    from test_gui import GuiHarness
    harness = GuiHarness(fw)
    harness.capture_errors()
    yield harness
    harness.destroy()


def _set_prop_table(gui, table_path):
    """
    Drive the propeller-table Browse button, or clear the field when
    `table_path` is None. Uses the real widget path so the loader, the status
    checks and the plots all see the same thing a user would produce.
    """
    widgets = gui.refresh()
    label = next((w for w in widgets if isinstance(w, ttk.Label)
                  and "CSV table" in str(w.cget("text"))), None)
    assert label is not None, "propeller CSV table field not found"

    browse = next((w for w in widgets if isinstance(w, ttk.Button)
                   and "Browse" in str(w.cget("text"))
                   and (w.master is label.master
                        or w.master.master is label.master)), None)
    assert browse is not None, "Browse button for the CSV table not found"

    if table_path is None:
        # Clear via the entry that shares the Browse button's frame. The GUI
        # is module-scoped here, so a leftover path from a previous case would
        # silently turn a "no-table" run into a table run.
        cleared = False
        for entry in widgets:
            if isinstance(entry, ttk.Entry) and entry.master is browse.master:
                try:
                    name = entry.cget("textvariable")
                except Exception:
                    continue
                if name:
                    gui.root.setvar(name, "")
                    cleared = True
        assert cleared, "could not clear the propeller CSV table field"
        gui.pump()
        return

    gui.set_open_dialog(table_path)
    browse.invoke()
    gui.pump()


def _load_and_run(gui, config_path, table_path):
    gui.set_open_dialog(config_path)
    errors = gui.click("Load Config")
    assert errors == [], f"loading {os.path.basename(config_path)}: {errors[:1]}"
    _set_prop_table(gui, table_path)
    return gui.click("Single-Point")


@pytest.mark.parametrize("with_table", [False, True], ids=["no-table", "table"])
@pytest.mark.parametrize("config", MC_CONFIGS, ids=_ids(MC_CONFIGS))
def test_gui_multicopter_configs(gui_mc, config, with_table):
    """
    Regression: this exact combination — a config loaded in the GUI with a
    propeller table — raised "name 'm' is not defined" on every run.
    """
    errors = _load_and_run(gui_mc, config, MC_TABLE if with_table else None)
    assert errors == [], f"{os.path.basename(config)} table={with_table}: {errors[:1]}"


@pytest.mark.parametrize("with_table", [False, True], ids=["no-table", "table"])
@pytest.mark.parametrize("config", FW_CONFIGS, ids=_ids(FW_CONFIGS))
def test_gui_fixedwing_configs(gui_fw, config, with_table):
    errors = _load_and_run(gui_fw, config, FW_TABLE if with_table else None)
    assert errors == [], f"{os.path.basename(config)} table={with_table}: {errors[:1]}"


def _run_mission(gui, config_path, mission_path, table_path):
    gui.set_open_dialog(config_path)
    gui.click("Load Config")
    _set_prop_table(gui, table_path)
    gui.set_open_dialog(mission_path)
    for text, button in list(gui.buttons.items()):
        if "Browse" in text:
            try:
                button.invoke()
            except Exception:
                pass
    gui.pump()
    return gui.click("Run Mission")


@pytest.mark.parametrize("with_table", [False, True], ids=["no-table", "table"])
@pytest.mark.parametrize("mission", MC_MISSIONS, ids=_ids(MC_MISSIONS))
def test_gui_multicopter_missions(gui_mc, mission, with_table):
    errors = _run_mission(gui_mc, MC_CONFIGS[0], mission,
                          MC_TABLE if with_table else None)
    assert errors == [], f"{os.path.basename(mission)} table={with_table}: {errors[:1]}"


@pytest.mark.parametrize("with_table", [False, True], ids=["no-table", "table"])
@pytest.mark.parametrize("mission", FW_MISSIONS, ids=_ids(FW_MISSIONS))
def test_gui_fixedwing_missions(gui_fw, mission, with_table):
    errors = _run_mission(gui_fw, FW_CONFIGS[0], mission,
                          FW_TABLE if with_table else None)
    assert errors == [], f"{os.path.basename(mission)} table={with_table}: {errors[:1]}"


# ======================================================================
# The table must reach the display, not just the model
# ======================================================================

@pytest.mark.parametrize("which,table", [("mc", MC_TABLE), ("fw", FW_TABLE)])
def test_status_reports_table_thrust_range(request, which, table):
    """
    With a table loaded, the Status tab must show the "Table thrust range"
    row. That row is where the crash lived, and it is also the only warning a
    user gets when a table is paired with the wrong propeller.
    """
    gui = request.getfixturevalue("gui_mc" if which == "mc" else "gui_fw")
    configs = MC_CONFIGS if which == "mc" else FW_CONFIGS
    assert _load_and_run(gui, configs[0], table) == []

    rows = []
    for widget in gui.refresh():
        if isinstance(widget, ttk.Treeview):
            for parent in widget.get_children(""):
                rows.append(str(widget.item(parent, "values")))
                for child in widget.get_children(parent):
                    rows.append(str(widget.item(child, "values")))
    assert any("Table thrust range" in r for r in rows), \
        "Status tab does not report the table thrust range"


@pytest.mark.parametrize("which,table", [("mc", MC_TABLE), ("fw", FW_TABLE)])
def test_gui_results_change_when_a_table_is_loaded(request, which, table):
    """The GUI must actually use the table, not merely accept the path."""
    gui = request.getfixturevalue("gui_mc" if which == "mc" else "gui_fw")
    configs = MC_CONFIGS if which == "mc" else FW_CONFIGS

    def output_text():
        for widget in gui.refresh():
            if isinstance(widget, tk.Text):
                try:
                    body = widget.get("1.0", "end")
                except Exception:
                    continue
                if "Flight" in body or "flight" in body:
                    return body
        return ""

    assert _load_and_run(gui, configs[0], None) == []
    plain = output_text()
    assert _load_and_run(gui, configs[0], table) == []
    tabled = output_text()

    assert plain and tabled, "no run output captured"
    assert plain != tabled, "loading a table changed nothing in the output"
