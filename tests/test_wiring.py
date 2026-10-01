"""
Wiring and connector tests, for all three simulators.

The main battery lead and the three connectors are one shared model in
rotorworks_core: the lead's I^2 R loss is paid by the pack, the ESCs see the
pack voltage minus its I*R drop, and Status checks the lead's temperature and
each connector's current and voltage rating. These tests pin that model down
and check each simulator wires it in the same way — and that leaving the
Wiring tab blank changes nothing at all.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

import rotorworks_core as core


def _wiring(**kw):
    """A Wiring-tab field dict, as the GUI would hand it over."""
    return core.wiring_from_fields({k: str(v) for k, v in kw.items()})


# ----------------------------------------------------------------------
# The core model
# ----------------------------------------------------------------------

def test_a_blank_wiring_tab_builds_no_wiring():
    assert core.wiring_from_fields({}) is None
    assert core.wiring_from_fields({"wire_len": "", "wire_awg": "",
                                    "conn_batt": "XT60"}) is None


def test_both_conductors_are_counted():
    # 1 m of 12 AWG one way is 2 m of copper.
    w = _wiring(wire_len=1.0, wire_awg=12)
    assert w.resistance_ohm == pytest.approx(2.0 * core.wire_ohm_per_m(12, None))


def test_a_measured_resistance_overrides_the_gauge():
    w = _wiring(wire_len=0.5, wire_awg=12, wire_ohm_m=0.02)
    assert w.resistance_ohm == pytest.approx(0.02)


def test_loss_and_drop_are_ohms_law():
    w = _wiring(wire_len=0.3, wire_awg=14)
    s = w.summary(30.0, 25.0)
    assert s["drop_V"] == pytest.approx(30.0 * w.resistance_ohm)
    assert s["loss_W"] == pytest.approx(30.0 ** 2 * w.resistance_ohm)


def test_wire_temperature_rises_with_current_and_thinner_wire():
    thick = core.wire_ohm_per_m(12, None)
    thin = core.wire_ohm_per_m(18, None)
    assert core.wire_temperature_C(0.0, thick, 25.0) == 25.0
    assert (core.wire_temperature_C(10.0, thick, 25.0)
            < core.wire_temperature_C(20.0, thick, 25.0))
    assert (core.wire_temperature_C(15.0, thick, 25.0)
            < core.wire_temperature_C(15.0, thin, 25.0))


def test_wire_temperature_runs_away_rather_than_going_negative():
    # Far past what 22 AWG can shed: the model reports runaway, not a number.
    assert math.isinf(core.wire_temperature_C(200.0, core.wire_ohm_per_m(22, None), 25.0))


def test_wire_temperature_is_in_a_sane_range():
    # 16 AWG silicone at 20 A in still air runs hot but not absurd.
    t = core.wire_temperature_C(20.0, core.wire_ohm_per_m(16, None), 25.0)
    assert 60.0 < t < 160.0


def _rows(wiring, pack_A=40.0, esc_A=10.0, full_V=25.2, ambient=25.0):
    return {r[1]: r for r in core.wiring_status_rows(wiring, pack_A, esc_A, full_V, ambient)}


def test_status_rows_cover_drop_temperature_and_every_connector():
    w = _wiring(wire_len=0.3, wire_awg=12,
                conn_batt_cont=60, conn_batt_max=120, conn_batt_volt=500,
                conn_esc_cont=40, conn_esc_max=80,
                conn_motor_cont=30, conn_motor_max=60)
    rows = _rows(w)
    for name in ("Main wire voltage drop", "Wire temperature (est)",
                 "Battery connector current", "Battery connector voltage",
                 "ESC connector current", "ESC connector voltage",
                 "Motor connector current", "Motor connector voltage"):
        assert name in rows, f"missing Status row {name!r}"
    assert rows["Battery connector voltage"][4] == "ok"
    assert rows["ESC connector voltage"][4] == "na", "no rating entered is not a pass"
    # The battery connector sits with the battery; the others on a motor channel.
    assert rows["Battery connector current"][0] == "battery"
    assert rows["Motor connector current"][0] == "motor"


def test_each_connector_sees_its_own_current():
    w = _wiring(conn_batt_cont=100, conn_esc_cont=100, conn_motor_cont=100)
    rows = _rows(w, pack_A=40.0, esc_A=10.0)
    assert rows["Battery connector current"][2] == "40.0 A"
    assert rows["ESC connector current"][2] == "10.0 A"
    assert rows["Motor connector current"][2] == "11.5 A"


def test_connector_over_its_ratings_is_flagged():
    w = _wiring(conn_batt_cont=30, conn_batt_max=60, conn_batt_volt=12)
    rows = _rows(w, pack_A=45.0, full_V=25.2)
    assert rows["Battery connector current"][4] == "warn"   # over continuous
    assert rows["Battery connector voltage"][4] == "bad"    # 6S on a 12 V part
    assert _rows(w, pack_A=70.0)["Battery connector current"][4] == "bad"


def test_a_hot_or_lossy_lead_is_flagged():
    cool = _rows(_wiring(wire_len=0.2, wire_awg=10), pack_A=20.0)
    assert cool["Wire temperature (est)"][4] == "ok"
    assert cool["Main wire voltage drop"][4] == "ok"
    hot = _rows(_wiring(wire_len=1.5, wire_awg=20, wire_temp_limit=105), pack_A=40.0)
    assert hot["Wire temperature (est)"][4] == "bad"
    assert hot["Main wire voltage drop"][4] == "bad"


def test_the_temperature_limit_defaults_and_can_be_set():
    assert _wiring(wire_len=1, wire_awg=12).temp_limit_C == core.WIRE_TEMP_LIMIT_C
    assert _wiring(wire_len=1, wire_awg=12, wire_temp_limit=90).temp_limit_C == 90.0


def test_connector_voltage_presets():
    assert core.connector_voltage_default("XT60") == 500.0
    assert core.connector_voltage_default("no such connector") is None


# ----------------------------------------------------------------------
# Multicopter
# ----------------------------------------------------------------------

LEAD = dict(wire_len=0.6, wire_awg=18)


def test_multicopter_blank_wiring_changes_nothing(mc, mc_quad):
    base = mc.compute_operating_metrics(mc_quad, 0.0, "hover")
    mc_quad.wiring = None
    again = mc.compute_operating_metrics(mc_quad, 0.0, "hover")
    assert again["total_power_W"] == base["total_power_W"]
    assert again["wire_loss_W"] == 0.0 and again["wire_drop_V"] == 0.0


@pytest.mark.parametrize("mode,speed", [("hover", 0.0), ("forward", 10.0)])
def test_multicopter_pays_for_the_lead(mc, mc_quad, mode, speed):
    base = mc.compute_operating_metrics(mc_quad, speed, mode)
    mc_quad.wiring = _wiring(**LEAD)
    m = mc.compute_operating_metrics(mc_quad, speed, mode)
    r = mc_quad.wiring.resistance_ohm
    assert m["wire_resistance_ohm"] == pytest.approx(r)
    assert m["wire_loss_W"] == pytest.approx(m["pack_current_A"] ** 2 * r, rel=0.02)
    assert m["wire_drop_V"] == pytest.approx(m["pack_current_A"] * r, rel=0.02)
    # The pack pays the lead's heat, and a little more: the ESC works harder
    # at the lower voltage the lead leaves it.
    assert m["total_power_W"] >= base["total_power_W"] + 0.9 * m["wire_loss_W"]
    assert m["esc_input_voltage_V"] < base["esc_input_voltage_V"]
    assert m["wire_temp_C"] > m.get("ambient_temp_C", 15.0)


def test_multicopter_thinner_wire_costs_more(mc, mc_quad):
    mc_quad.wiring = _wiring(wire_len=0.6, wire_awg=14)
    thick = mc.compute_operating_metrics(mc_quad, 0.0, "hover")
    mc_quad.wiring = _wiring(wire_len=0.6, wire_awg=20)
    thin = mc.compute_operating_metrics(mc_quad, 0.0, "hover")
    assert thin["wire_loss_W"] > thick["wire_loss_W"]
    assert thin["total_power_W"] > thick["total_power_W"]
    assert thin["wire_temp_C"] > thick["wire_temp_C"]


# ----------------------------------------------------------------------
# Fixed-wing
# ----------------------------------------------------------------------

def test_fixedwing_blank_wiring_changes_nothing(fw, fw_plane):
    base = fw.compute_metrics(fw_plane, 19.0)
    fw_plane.wiring = None
    again = fw.compute_metrics(fw_plane, 19.0)
    assert again["total_power_W"] == base["total_power_W"]
    assert again["wire_loss_W"] == 0.0


def test_fixedwing_pays_for_the_lead(fw, fw_plane):
    base = fw.compute_metrics(fw_plane, 19.0)
    fw_plane.wiring = _wiring(wire_len=1.0, wire_awg=20)
    m = fw.compute_metrics(fw_plane, 19.0)
    r = fw_plane.wiring.resistance_ohm
    assert m["wire_loss_W"] == pytest.approx(m["pack_current_A"] ** 2 * r, rel=0.02)
    assert m["total_power_W"] >= base["total_power_W"] + 0.9 * m["wire_loss_W"]
    assert m["esc_input_voltage_V"] == pytest.approx(m["v_load_V"] - m["wire_drop_V"])
    # Less voltage at the ESC: the same power needs more motor current.
    assert m["motor_I_per_esc_A"] > base["motor_I_per_esc_A"]


# ----------------------------------------------------------------------
# VTOL
# ----------------------------------------------------------------------

def test_vtol_builds_its_wiring_through_the_core(vtol):
    cfg = vtol.config_from_fields({"wire_len": "0.5", "wire_awg": "16",
                                   "conn_batt_cont": "60", "conn_batt_volt": "500"})
    assert cfg.wiring is not None
    assert cfg.wire_resistance_ohm == pytest.approx(cfg.wiring.resistance_ohm)
    assert cfg.wiring.connectors["Battery"] == (60.0, None, 500.0)
    assert vtol.config_from_fields({}).wiring is None


def test_vtol_lead_costs_power_and_throttle(vtol):
    base = vtol.compute_metrics(vtol.config_from_fields({}))
    m = vtol.compute_metrics(vtol.config_from_fields({"wire_len": "1.0", "wire_awg": "18"}))
    assert m["hover_power_W"] > base["hover_power_W"]
    # Less voltage reaches the ESCs, so the same motor needs more throttle.
    assert m["hover_lift_throttle"] > base["hover_lift_throttle"]


def test_vtol_blank_wiring_changes_nothing(vtol):
    a = vtol.compute_metrics(vtol.config_from_fields({}))
    b = vtol.compute_metrics(vtol.config_from_fields({"wire_len": "", "wire_awg": ""}))
    assert a["hover_power_W"] == b["hover_power_W"]
    assert a["hover_lift_throttle"] == b["hover_lift_throttle"]


# ----------------------------------------------------------------------
# One set of flags and field names everywhere
# ----------------------------------------------------------------------

def test_vtol_and_batch_use_the_core_wiring_names(vtol, rw):
    for key, flag in core.WIRING_FIELD_TO_CLI.items():
        assert vtol.FIELD_TO_CLI.get(key) == flag, key
        assert rw.GUI_TO_CLI_MULTICOPTER.get(key) == flag, key
        assert rw.GUI_TO_CLI_FIXEDWING.get(key) == flag, key


def test_the_cli_flags_build_the_same_wiring_as_the_fields():
    import argparse
    parser = argparse.ArgumentParser()
    core.add_wiring_arguments(parser)
    args = parser.parse_args(["--wire_length", "0.4", "--wire_awg", "16",
                              "--wire_temp_limit", "105",
                              "--connector_esc_cont", "40", "--connector_esc_volt", "500"])
    from_cli = core.wiring_from_args(args)
    from_gui = _wiring(wire_len=0.4, wire_awg=16, wire_temp_limit=105,
                       conn_esc_cont=40, conn_esc_volt=500)
    assert from_cli.resistance_ohm == from_gui.resistance_ohm
    assert from_cli.temp_limit_C == from_gui.temp_limit_C
    assert from_cli.connectors == from_gui.connectors


def _run(script, args):
    return subprocess.run([sys.executable, script] + args, capture_output=True,
                          text=True, timeout=300)


@pytest.mark.slow
@pytest.mark.parametrize("which", ["multicopter", "fixedwing"])
def test_cli_reports_the_lead_only_when_one_is_given(paths, which):
    from test_cli import FW_BASE, MC_BASE
    script = paths[which]
    base = MC_BASE if which == "multicopter" else FW_BASE
    plain = _run(script, base)
    wired = _run(script, base + ["--wire_length", "0.5", "--wire_awg", "18"])
    assert plain.returncode == 0, plain.stderr
    assert wired.returncode == 0, wired.stderr
    assert "Main lead" not in plain.stdout
    assert "Main lead" in wired.stdout


@pytest.mark.slow
def test_vtol_cli_accepts_the_new_wiring_flags(paths):
    script = os.path.join(paths["root"], "vtol-power-sim-gui.py")
    r = _run(script, ["--wire_length", "0.5", "--wire_awg", "16",
                      "--wire_temp_limit", "105", "--connector_batt_volt", "500"])
    assert r.returncode == 0, r.stderr
