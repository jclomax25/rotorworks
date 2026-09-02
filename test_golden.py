"""
Golden-output regression.

A refactor must not change a single number. This module computes a broad set
of metrics from fixed configurations and compares them against a stored
snapshot, so any behavioural drift during the shared-core extraction shows up
immediately and precisely ("flight_time_min moved by 0.3") rather than as a
vague downstream failure.

Regenerate the snapshot ONLY when a change to the physics is intended:

    python tests/test_golden.py --update

and read the resulting diff carefully before committing it.
"""

from __future__ import annotations

import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, "golden_snapshot.json")

# Metrics compared for each operating point. Restricted to values that are
# meaningful across both simulators and stable run to run.
MC_KEYS = [
    "total_power_W", "motor_power_W", "esc_loss_W", "pack_current_A",
    "v_load_V", "thrust_total_N", "thrust_per_motor_N", "tilt_required_deg",
    "disk_loading_N_m2", "hover_efficiency_gW", "figure_of_merit",
    "motor_temp_est_C", "reserve_margin_Wh",
]

FW_KEYS = [
    "total_power_W", "motor_power_W", "esc_loss_W", "pack_current_A",
    "v_load_V", "thrust_required_N", "thrust_available_N",
    "CL", "CD", "LD_ratio", "stall_speed_mps", "flight_time_min",
    "flight_range_km", "wing_loading_N_m2", "glide_ratio",
    "glide_distance_m", "takeoff_dist_m", "landing_dist_m",
    "reynolds_number", "specific_range_m_per_Wh",
]


def _build_multicopter(mc):
    """Two contrasting multicopters: a light quad and a heavy coaxial X8."""
    out = {}

    batt = mc.BatteryConfig(
        chemistry="LiPo", operating_voltage_min=3.3,
        operating_voltage_nominal=3.7, operating_voltage_max=4.2,
        unit_mode="pack", pack_capacity_mAh=5200, pack_weight_g=520,
        series_units=1, parallel_units=1, cells_series_per_unit=4,
        discharge_percent=80, resistance_cell_mOhm=4.0, discharge_c_cont=25,
    )
    quad = mc.DroneConfig(
        num_motors=4, battery=batt,
        motor=mc.MotorConfig(kv=920, idle_current=0.5, idle_voltage=10,
                             rated_voltage=4, resistance=0.115,
                             max_current=18, max_power=250, weight_g=56),
        propeller=mc.PropellerConfig(diameter_in=10, pitch_in=4.5, max_rpm=0,
                                     max_thrust_g=1100, blades=2, weight_g=13),
        drone_weight_g=1800,
        profile_drag_coefficient=None, profile_area=None,
        parasite_drag_coefficient=None, parasite_area=None, frontal_area=None,
        cruise_speed=10.0, periph_current=0.0,
        body_length_m=0.22, body_width_m=0.16, body_height_m=0.09,
        arm_length_m=0.225, arm_width_m=0.016,
    )
    quad.air_density = mc.compute_air_density(0)
    quad.derive_drag_from_geometry_if_missing()
    out["quad_4s"] = quad

    heavy_batt = mc.BatteryConfig(
        chemistry="LiPo", operating_voltage_min=3.3,
        operating_voltage_nominal=3.7, operating_voltage_max=4.2,
        unit_mode="pack", pack_capacity_mAh=16000, pack_weight_g=2100,
        series_units=2, parallel_units=1, cells_series_per_unit=6,
        discharge_percent=80, resistance_cell_mOhm=3.0, discharge_c_cont=15,
    )
    x8 = mc.DroneConfig(
        num_motors=8, battery=heavy_batt,
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
        arm_length_m=0.40, arm_width_m=0.030,
    )
    x8.air_density = mc.compute_air_density(500)
    x8.derive_drag_from_geometry_if_missing()
    out["x8_12s_coaxial"] = x8
    return out


def _build_fixedwing(fw):
    """Two contrasting aircraft: a draggy foam trainer and a clean glider."""
    out = {}

    trainer = fw.FixedWingConfig(
        aircraft_weight_g=1050,
        airframe=fw.AirframeConfig(
            wing_span_m=1.5, wing_area_m2=0.32, CD0=0.042, oswald=0.80,
            CL_max=1.05, prop_efficiency=0.72, mu_roll=0.06, mu_brake=0.20,
            CL_takeoff=0.85),
        battery=fw.BatteryConfig(
            chemistry="LiPo", operating_voltage_min=3.3,
            operating_voltage_nominal=3.7, operating_voltage_max=4.2,
            unit_mode="cell", cell_capacity_mAh=2200, cell_weight_g=62,
            series_units=3, parallel_units=1,
            discharge_percent=80, resistance_cell_mOhm=8.0),
        motor=fw.MotorConfig(kv=1000, idle_current=0.7, idle_voltage=10,
                             rated_voltage=3, resistance=0.10,
                             max_current=25, max_power=280, weight_g=78),
        propeller=fw.PropellerConfig(diameter_in=9, pitch_in=6, blades=2,
                                     weight_g=12),
        esc=fw.ESCConfig(voltage_rating=3, continuous_current_A=30,
                         max_current_A=40, idle_current_A=0.05,
                         resistance=0.003, weight_g=28),
        cruise_speed_mps=16.0, air_density=1.225,
        reference_altitude_m=100, cruise_altitude_m=250,
    )
    out["foam_trainer_3s"] = trainer

    glider = fw.FixedWingConfig(
        aircraft_weight_g=4200,
        airframe=fw.AirframeConfig(
            wing_span_m=3.0, wing_area_m2=0.60, CD0=0.019, oswald=0.92,
            CL_max=1.30, prop_efficiency=0.80, mu_roll=0.05, mu_brake=0.25,
            CL_takeoff=0.90),
        battery=fw.BatteryConfig(
            chemistry="Li-ion", operating_voltage_min=2.8,
            operating_voltage_nominal=3.6, operating_voltage_max=4.2,
            unit_mode="cell", cell_capacity_mAh=3500, cell_weight_g=48,
            series_units=6, parallel_units=4,
            discharge_percent=90, resistance_cell_mOhm=30.0),
        motor=fw.MotorConfig(kv=430, idle_current=0.5, idle_voltage=10,
                             rated_voltage=6, resistance=0.09,
                             max_current=30, max_power=550, weight_g=185),
        propeller=fw.PropellerConfig(diameter_in=14, pitch_in=8, blades=2,
                                     weight_g=28),
        cruise_speed_mps=17.0, air_density=1.225,
        reference_altitude_m=150, cruise_altitude_m=600,
    )
    out["endurance_6s_liion"] = glider
    return out


def _collect(mc, fw):
    """Compute the full snapshot dictionary."""
    snap = {}

    for name, drone in _build_multicopter(mc).items():
        for speed, orientation in [(0.0, "hover"), (5.0, "forward"),
                                   (10.0, "forward"), (18.0, "forward")]:
            metrics = mc.compute_operating_metrics(drone, speed, orientation)
            row = {}
            for key in MC_KEYS:
                val = metrics.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    row[key] = round(float(val), 6)
            row["flight_time_min"] = round(
                mc.estimate_flight_time_minutes(drone, speed, orientation=orientation), 6)
            snap[f"mc/{name}/{orientation}@{speed}"] = row

        be, bt, br, bd = mc.find_optimal_speeds(drone, min_speed=0.5, max_speed=25.0)
        snap[f"mc/{name}/optimal"] = {
            "best_endurance_speed": round(be, 4),
            "best_endurance_min": round(bt, 4),
            "best_range_speed": round(br, 4),
            "best_range_km": round(bd, 4),
        }

    for name, cfg in _build_fixedwing(fw).items():
        for speed in (12.0, 16.0, 22.0, 28.0):
            metrics = fw.compute_metrics(cfg, speed)
            row = {}
            for key in FW_KEYS:
                val = metrics.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if math.isfinite(float(val)):
                        row[key] = round(float(val), 6)
            row["prop_efficiency_at_speed"] = round(
                fw.propeller_efficiency_at_speed(cfg, speed), 6)
            snap[f"fw/{name}/@{speed}"] = row

    # Battery pack arithmetic across topologies, from both simulators.
    for label, module in (("mc", mc), ("fw", fw)):
        for series, parallel in ((1, 1), (2, 1), (1, 2), (2, 2)):
            pack = module.BatteryConfig(
                chemistry="LiPo", operating_voltage_min=3.0,
                operating_voltage_nominal=3.7, operating_voltage_max=4.2,
                unit_mode="pack", series_units=series, parallel_units=parallel,
                cells_series_per_unit=6, cells_parallel_per_unit=1,
                pack_capacity_mAh=5000, pack_weight_g=700,
                discharge_percent=80, resistance_cell_mOhm=3.0)
            snap[f"{label}/battery/{series}S{parallel}P"] = {
                "capacity_mAh": round(pack.capacity_mAh, 6),
                "capacity_Wh": round(pack.capacity_Wh, 6),
                "weight_g": round(pack.weight_g, 6),
                "vmax_pack": round(pack.vmax_pack, 6),
                "pack_resistance": round(pack.pack_resistance, 9),
                "ocv_at_50pct": round(pack.ocv_at_soc(0.5), 6),
                "r_at_10pct": round(pack.resistance_at_soc(0.10), 9),
            }

    # Atmosphere, both simulators.
    for altitude, temperature, pressure in [
        (0, None, None), (500, None, None), (2000, None, None),
        (500, 35, None), (500, None, 95000), (500, 35, 95000),
    ]:
        key = f"atmos/{altitude}/{temperature}/{pressure}"
        snap[key] = {
            "mc": round(mc.compute_air_density(altitude, temperature, pressure), 9),
            "fw": round(fw.isa_density(altitude, temperature, pressure), 9),
        }

    return snap


def test_outputs_match_golden_snapshot(mc, fw):
    """
    Every stored value must still be produced exactly.

    If this fails after a refactor, the refactor changed behaviour. If it
    fails after an intended physics change, regenerate the snapshot with
    `python tests/test_golden.py --update` and review the diff.
    """
    if not os.path.exists(SNAPSHOT):
        pytest.skip("no snapshot yet; run: python tests/test_golden.py --update")

    with open(SNAPSHOT, encoding="utf-8") as handle:
        expected = json.load(handle)
    actual = _collect(mc, fw)

    missing = sorted(set(expected) - set(actual))
    assert not missing, f"snapshot cases no longer produced: {missing[:5]}"

    drift = []
    for case, values in expected.items():
        for key, want in values.items():
            got = actual[case].get(key)
            if got is None:
                drift.append(f"{case}.{key}: missing")
            elif want == 0:
                if abs(got) > 1e-9:
                    drift.append(f"{case}.{key}: {want} -> {got}")
            elif abs(got - want) / max(abs(want), 1e-12) > 1e-9:
                drift.append(f"{case}.{key}: {want} -> {got}")

    assert not drift, (
        f"{len(drift)} value(s) changed:\n  " + "\n  ".join(drift[:25])
    )


if __name__ == "__main__":
    import sys

    if "--update" not in sys.argv:
        print(__doc__)
        raise SystemExit("pass --update to regenerate the snapshot")

    # The simulators import rotorworks_core from beside themselves, so the
    # repository root has to be importable when running this file directly.
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    from conftest import FIXEDWING, MULTICOPTER, _load, _stub_tkinter

    try:
        import tkinter  # noqa: F401
    except ImportError:
        _stub_tkinter()

    mc_mod = _load(MULTICOPTER, "rw_multicopter")
    fw_mod = _load(FIXEDWING, "rw_fixedwing")
    data = _collect(mc_mod, fw_mod)

    with open(SNAPSHOT, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)

    total = sum(len(v) for v in data.values())
    print(f"wrote {SNAPSHOT}: {len(data)} cases, {total} values")
