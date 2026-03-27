#!/usr/bin/env python3
"""
fixed_wing_power_sim.py
=======================

Fixed-wing aircraft performance simulator with both CLI and Tkinter GUI modes.

Design goals
------------
1. Reuse as much of the structure of the uploaded multicopter simulator as is sensible:
   - BatteryConfig
   - MotorConfig
   - ESCConfig
   - AvionicsConfig
   - ISA / air-density helpers
   - CSV motor-table interpolation helpers
   - optional GUI + CLI architecture
2. Support two propulsion-model modes:
   - "ecalc" style electrical + prop coefficient model
   - uploaded motor/prop test table model (such as T-Motor test data)
3. Produce the same *kind* of outputs that an eCalc-style fixed-wing calculator exposes:
   - battery metrics
   - motor metrics at optimum efficiency and max power
   - prop metrics
   - total-drive metrics
   - aircraft metrics (stall speed, best range speed, best endurance speed, max level speed,
     climb rate, takeoff estimate, 3D / vertical performance estimate, etc.)
   - graphs for power required / power available / climb performance / motor behavior
4. Offer both CLI and GUI operation.

Notes on fidelity
-----------------
This script aims to be a transparent engineering calculator, not a black-box clone of eCalc.
Several aerodynamic and propeller effects are modeled with standard first-order approximations.
The code is heavily commented so you can replace any block with a more detailed sub-model later.

Main assumptions
----------------
- Quasi-steady fixed-wing flight.
- Wing drag uses a standard polar:
      C_D = C_D0 + k * C_L^2
  where k = 1 / (pi * AR * e)
- Level-flight power required is:
      P_req = D * V
- Climb rate is based on excess power:
      ROC = (P_avail - P_req) / W
- Propeller static / dynamic behavior is approximated from either:
      (a) coefficient model with C_T and C_P, or
      (b) uploaded static test table with simple unloading corrections in forward flight.
- Motor electrical model uses the common brushless DC approximation:
      K_t = 60 / (2*pi*K_v)
      Q = K_t (I - I_0)
      V = I R_m + omega / K_v_rad
- Controller/ESC losses are modeled as I^2 R + idle-current overhead.
- Battery sag is modeled with a pack internal resistance.

Example CLI usage
-----------------
# eCalc-style coefficient / electrical model
python fixed_wing_power_sim.py \
    --weight_g 850 \
    --wingspan_mm 1270 \
    --wing_area_dm2 50 \
    --cd0 0.03 \
    --oswald 0.8 \
    --cl_max 1.3 \
    --battery_unit_mode cell --battery_series_units 3 --battery_parallel_units 1 \
    --battery_cell_capacity 5000 --battery_cell_weight 45 \
    --battery_operating_voltage_min 3.2 --battery_operating_voltage_nominal 3.7 --battery_operating_voltage_max 4.2 \
    --battery_resistance_cell 12 --battery_discharge_c_cont 20 --battery_discharge_c_max 40 \
    --motor_kv 700 --motor_idle_current 1.1 --motor_idle_voltage 10 --motor_resistance 0.06 \
    --motor_max_current 60 --motor_max_power 800 \
    --esc_cont_current 60 --esc_max_current 80 --esc_resistance 0.003 \
    --prop_diameter 10 --prop_pitch 4.7 --prop_blades 2 --prop_tconst 0.11 --prop_pconst 0.055 \
    --motor_count 1 --speed_kmh 60 --plot

# Table-driven mode using a T-Motor-style CSV
python fixed_wing_power_sim.py \
    --weight_g 850 --wingspan_mm 1270 --wing_area_dm2 50 --cd0 0.03 --oswald 0.8 --cl_max 1.3 \
    --battery_unit_mode cell --battery_series_units 12 --battery_parallel_units 1 \
    --battery_cell_capacity 5000 --battery_cell_weight 80 \
    --battery_operating_voltage_min 3.2 --battery_operating_voltage_nominal 3.7 --battery_operating_voltage_max 4.2 \
    --battery_resistance_cell 4 --battery_discharge_c_cont 15 --battery_discharge_c_max 30 \
    --prop_table my_tmotor_table.csv --prop_diameter 21 --prop_pitch 6.3 --prop_blades 2 \
    --speed_kmh 80 --plot

# GUI
python fixed_wing_power_sim.py --gui
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
AIR_DENSITY = 1.225  # kg/m^3 at sea level ISA
R = 287.05          # J/kg/K
T0 = 288.15         # K
P0 = 101325.0       # Pa
L = 0.0065          # K/m
G0 = 9.80665        # m/s^2


# --------------------------------------------------------------------------------------
# Shared utility models (adapted from the uploaded multicopter simulator)
# --------------------------------------------------------------------------------------
class BatteryConfig:
    """
    Simple battery pack model with support for either:
      - cell-based entry, or
      - pack-based entry.

    Voltage model:
        V_load = V_max_pack - I * R_pack
    clamped to a minimum operating voltage.

    Capacity model:
        capacity_Ah = capacity_cell_Ah * parallel_cells
    for cell mode, or derived from pack capacity in pack mode.
    """
    def __init__(self,
                 chemistry: Optional[str],
                 operating_voltage_min: float,
                 operating_voltage_nominal: float,
                 operating_voltage_max: float,
                 unit_mode: str = "cell",
                 series_units: int = 1,
                 parallel_units: int = 1,
                 cells_series_per_unit: int = 1,
                 cells_parallel_per_unit: int = 1,
                 pack_weight_g: Optional[float] = None,
                 cell_weight_g: Optional[float] = None,
                 cell_capacity_mAh: Optional[float] = None,
                 pack_capacity_mAh: Optional[float] = None,
                 unit_energy_density: Optional[float] = None,
                 charge_current_max: Optional[float] = None,
                 discharge_cont_A: Optional[float] = None,
                 discharge_max_A: Optional[float] = None,
                 discharge_c_cont: Optional[float] = None,
                 discharge_c_max: Optional[float] = None,
                 discharge_percent: float = 100.0,
                 resistance_cell_mOhm: float = 0.0):
        self.chemistry = chemistry
        self.operating_voltage_min = float(operating_voltage_min)
        self.operating_voltage_nominal = float(operating_voltage_nominal)
        self.operating_voltage_max = float(operating_voltage_max)
        self.unit_mode = str(unit_mode).strip().lower()
        self.series_units = int(series_units)
        self.parallel_units = int(parallel_units)

        if self.unit_mode == "cell":
            cells_series_per_unit = 1
            cells_parallel_per_unit = 1

        self.cells_series_per_unit = int(cells_series_per_unit)
        self.cells_parallel_per_unit = int(cells_parallel_per_unit)

        self.series_cells = self.series_units * self.cells_series_per_unit
        self.parallel_cells = self.parallel_units * self.cells_parallel_per_unit
        self.total_cells = self.series_cells * self.parallel_cells

        self.vmin_pack = self.operating_voltage_min * self.series_cells
        self.vnom_pack = self.operating_voltage_nominal * self.series_cells
        self.vmax_pack = self.operating_voltage_max * self.series_cells

        self.pack_weight_g = float(pack_weight_g) if pack_weight_g is not None else None
        self.cell_weight_g = float(cell_weight_g) if cell_weight_g is not None else None
        self.cell_capacity_mAh = float(cell_capacity_mAh) if cell_capacity_mAh is not None else None
        self.pack_capacity_mAh = float(pack_capacity_mAh) if pack_capacity_mAh is not None else None
        self.charge_current_max = float(charge_current_max) if charge_current_max is not None else None
        self.resistance_cell = float(resistance_cell_mOhm) / 1000.0
        self.discharge_percent = max(0.0, min(100.0, float(discharge_percent)))
        self.usable_fraction = self.discharge_percent / 100.0

        if self.unit_mode == "cell":
            if self.cell_capacity_mAh is None:
                raise ValueError("cell_capacity_mAh is required in cell mode")
            self.capacity_mAh = self.cell_capacity_mAh * self.parallel_cells
        elif self.unit_mode == "pack":
            if self.pack_capacity_mAh is None:
                raise ValueError("pack_capacity_mAh is required in pack mode")
            self.capacity_mAh = self.pack_capacity_mAh * self.parallel_units
        else:
            raise ValueError("battery unit_mode must be 'cell' or 'pack'")

        self.capacity_Ah = self.capacity_mAh / 1000.0

        if self.unit_mode == "cell":
            if self.cell_weight_g is not None:
                self.weight_g = self.cell_weight_g * self.total_cells
            else:
                self.weight_g = None
        else:
            if self.pack_weight_g is not None:
                self.weight_g = self.pack_weight_g * self.series_units * self.parallel_units
            else:
                self.weight_g = None

        if discharge_cont_A is not None:
            self.discharge_cont_A = float(discharge_cont_A)
        elif discharge_c_cont is not None:
            self.discharge_cont_A = float(discharge_c_cont) * self.capacity_Ah
        else:
            self.discharge_cont_A = float("inf")

        if discharge_max_A is not None:
            self.discharge_max_A = float(discharge_max_A)
        elif discharge_c_max is not None:
            self.discharge_max_A = float(discharge_c_max) * self.capacity_Ah
        else:
            self.discharge_max_A = self.discharge_cont_A

        self.discharge_c_cont = (
            float(discharge_c_cont) if discharge_c_cont is not None
            else (self.discharge_cont_A / self.capacity_Ah if self.capacity_Ah > 0 else None)
        )
        self.discharge_c_max = (
            float(discharge_c_max) if discharge_c_max is not None
            else (self.discharge_max_A / self.capacity_Ah if self.capacity_Ah > 0 else None)
        )

        if unit_energy_density is not None:
            self.energy_density_Wh_per_kg = float(unit_energy_density)
        elif self.weight_g is not None and self.weight_g > 0:
            self.energy_density_Wh_per_kg = self.capacity_Wh / (self.weight_g / 1000.0)
        else:
            self.energy_density_Wh_per_kg = None

    @property
    def pack_resistance(self) -> float:
        # Series resistances add; parallel branches divide resistance.
        return self.resistance_cell * self.series_cells / max(self.parallel_cells, 1)

    @property
    def capacity_Wh(self) -> float:
        return self.capacity_Ah * self.vnom_pack

    @property
    def usable_Wh(self) -> float:
        return self.capacity_Wh * self.usable_fraction


def battery_voltage_under_load(battery: BatteryConfig, current_A: float) -> float:
    v = battery.vmax_pack - current_A * battery.pack_resistance
    return max(v, battery.vmin_pack)


def solve_pack_voltage_and_current(battery: BatteryConfig, total_power_W: float, iters: int = 12) -> Tuple[float, float]:
    """
    Solve for pack voltage and current for a near-constant-power load.

    Fixed-point iteration:
        I = P / V
        V = Vmax - I*Rpack
    """
    if total_power_W <= 0:
        return battery.vmax_pack, 0.0

    v = battery.vnom_pack
    i = total_power_W / max(v, 1e-9)
    for _ in range(max(1, iters)):
        v = battery_voltage_under_load(battery, i)
        i = total_power_W / max(v, 1e-9)
    return float(v), float(i)


class MotorConfig:
    def __init__(self,
                 kv: Optional[float],
                 idle_current: float,
                 idle_voltage: float,
                 rated_voltage: Optional[int],
                 resistance: float,
                 max_current: float,
                 max_power: float,
                 pole_count: Optional[int] = None,
                 weight_g: Optional[float] = None,
                 size_mm: Optional[str] = None):
        self.kv = None if kv in (None, "") else float(kv)     # RPM/V
        self.idle_current = float(idle_current)
        self.idle_voltage = float(idle_voltage)
        self.rated_voltage = int(rated_voltage) if rated_voltage not in (None, "") else None
        self.resistance = float(resistance)
        self.max_current = float(max_current)
        self.max_power = float(max_power)
        self.pole_count = pole_count
        self.weight_g = weight_g
        self.size_mm = size_mm


class ESCConfig:
    def __init__(self,
                 voltage_rating: Optional[int],
                 continuous_current_A: float,
                 max_current_A: float,
                 idle_current_A: float,
                 resistance: float,
                 weight_g: Optional[float] = None):
        self.voltage_rating = int(voltage_rating) if voltage_rating not in (None, "") else None
        self.continuous_rating_A = float(continuous_current_A)
        self.max_current_A = float(max_current_A)
        self.idle_current_A = float(idle_current_A)
        self.resistance = float(resistance)
        self.weight_g = weight_g


class AvionicsConfig:
    def __init__(self, voltage_tree: Optional[dict] = None):
        self.voltage_tree = voltage_tree or {}


def parse_voltage_tree(spec: Optional[str]) -> Dict[float, Tuple[float, float]]:
    """Parse strings like '5:(2,0.9), 12:(1.5,0.85)' into {V: (I, eff)}."""
    if spec is None:
        return {}
    if isinstance(spec, dict):
        out = {}
        for k, v in spec.items():
            out[float(k)] = (float(v[0]), float(v[1]))
        return out

    s = str(spec).strip()
    if not s:
        return {}

    out: Dict[float, Tuple[float, float]] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)\s*$", p)
        if m:
            v, i, eff = map(float, m.groups())
        else:
            m2 = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*([0-9]*\.?[0-9]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$", p)
            if not m2:
                raise ValueError("Invalid avionics voltage tree format.")
            v, i, eff = map(float, m2.groups())
        out[v] = (i, eff)
    return out


def avionics_input_power_W(avionics: Optional[AvionicsConfig]) -> float:
    if avionics is None:
        return 0.0
    total = 0.0
    for v, (i, eff) in avionics.voltage_tree.items():
        total += (float(v) * float(i)) / max(float(eff), 1e-9)
    return float(total)


def esc_loss_and_checks(esc: Optional[ESCConfig], motor_input_power_total_W: float, v_pack: float, motor_count: int) -> Tuple[float, str, float]:
    if esc is None:
        return 0.0, "", 0.0

    p_per_motor = motor_input_power_total_W / max(motor_count, 1)
    i_motor = p_per_motor / max(v_pack, 1e-9)
    p_loss_cond = (i_motor ** 2) * max(esc.resistance, 0.0)
    p_loss_idle = max(esc.idle_current_A, 0.0) * v_pack
    total_loss = motor_count * (p_loss_cond + p_loss_idle)

    note = ""
    if i_motor > esc.max_current_A:
        note = f"ESC OVER MAX: {i_motor:.1f}A > {esc.max_current_A:.1f}A"
    elif i_motor > esc.continuous_rating_A:
        note = f"ESC over continuous: {i_motor:.1f}A > {esc.continuous_rating_A:.1f}A"
    return float(total_loss), note, float(i_motor)


def total_power_with_esc(battery: BatteryConfig,
                         esc: Optional[ESCConfig],
                         motor_power_W: float,
                         periph_power_W: float,
                         motor_count: int,
                         iters: int = 6) -> Tuple[float, float, float, str, float, float]:
    total_power = float(motor_power_W) + float(periph_power_W)
    v_load = battery.vnom_pack
    pack_current = total_power / max(v_load, 1e-9)
    esc_note = ""
    i_esc = 0.0
    esc_loss_W = 0.0

    for _ in range(max(1, iters)):
        v_load, pack_current = solve_pack_voltage_and_current(battery, total_power)
        esc_loss_W, esc_note, i_esc = esc_loss_and_checks(esc, motor_power_W, v_load, motor_count)
        total_power = float(motor_power_W) + float(periph_power_W) + float(esc_loss_W)
    return float(total_power), float(v_load), float(pack_current), esc_note, float(i_esc), float(esc_loss_W)


# --------------------------------------------------------------------------------------
# CSV table parsing (adapted from the multicopter simulator)
# --------------------------------------------------------------------------------------
def load_motor_prop_table_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        df0 = pd.read_csv(path)
    except Exception:
        df0 = pd.DataFrame()

    def _looks_good(df: pd.DataFrame) -> bool:
        cols = {str(c).strip().lower() for c in df.columns}
        return ("thrust_g" in cols) or ("thrust (g)" in cols) or ("power (w)" in cols) or ("rpm" in cols)

    if len(df0.columns) > 1 and _looks_good(df0):
        df = df0
    else:
        raw = pd.read_csv(path, header=None)
        header_row = None
        for i in range(min(len(raw), 30)):
            row = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
            if any("thrust" in x for x in row) and any("power" in x for x in row):
                header_row = i
                break
        if header_row is None:
            raise ValueError(f"Could not locate usable header row in CSV: {path}")
        df = pd.read_csv(path, header=header_row)

    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("thrust_g", "thrust (g)"):
            rename_map[c] = "Thrust_g"
        elif cl in ("power_w", "power (w)"):
            rename_map[c] = "Power_W"
        elif cl in ("current_a", "current (a)"):
            rename_map[c] = "Current_A"
        elif cl in ("voltage_v", "voltage (v)"):
            rename_map[c] = "Voltage_V"
        elif cl == "rpm":
            rename_map[c] = "RPM"
        elif cl.startswith("throttle"):
            rename_map[c] = "Throttle_pct"
        elif "efficiency" in cl:
            rename_map[c] = "Efficiency_gW"
        elif "temperature" in cl:
            rename_map[c] = "Temp_C"
        elif "torque" in cl:
            rename_map[c] = "Torque_Nm"
        elif "propeller" == cl:
            rename_map[c] = "Propeller"

    df = df.rename(columns=rename_map)

    def _coerce(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")

    for col in ("Thrust_g", "Power_W", "Current_A", "Voltage_V", "RPM", "Throttle_pct", "Efficiency_gW", "Temp_C", "Torque_Nm"):
        if col in df.columns:
            df[col] = _coerce(df[col])

    if "Thrust_g" not in df.columns or "Power_W" not in df.columns:
        raise ValueError("The CSV must contain thrust and power columns, e.g. 'Thrust (g)' and 'Power (W)'.")

    df = df.dropna(subset=["Thrust_g", "Power_W"]).copy()
    df = df.sort_values("Thrust_g").drop_duplicates(subset=["Thrust_g"], keep="last").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------------
# Fixed-wing-specific configuration
# --------------------------------------------------------------------------------------
class PropellerConfig:
    def __init__(self,
                 diameter_in: float,
                 pitch_in: float,
                 blades: int = 2,
                 table_csv: Optional[str] = None,
                 TConst: Optional[float] = None,
                 PConst: Optional[float] = None,
                 gear_ratio: float = 1.0,
                 weight_g: Optional[float] = None):
        self.diameter_in = float(diameter_in)
        self.pitch_in = float(pitch_in)
        self.blades = int(blades)
        self.table_csv = table_csv
        self.TConst = None if TConst in (None, "") else float(TConst)
        self.PConst = None if PConst in (None, "") else float(PConst)
        self.gear_ratio = float(gear_ratio)
        self.weight_g = weight_g
        self.table = load_motor_prop_table_csv(table_csv) if table_csv else None


class AirframeConfig:
    def __init__(self,
                 weight_g: float,
                 wingspan_m: float,
                 wing_area_m2: float,
                 cd0: float,
                 oswald: float,
                 cl_max: float,
                 drag_mode: str = "advanced",
                 custom_cd: Optional[float] = None,
                 rolling_friction: float = 0.04,
                 reserve_thrust_fraction_for_takeoff: float = 1.0,
                 name: str = "Fixed Wing"):
        self.weight_g = float(weight_g)
        self.wingspan_m = float(wingspan_m)
        self.wing_area_m2 = float(wing_area_m2)
        self.cd0 = float(cd0)
        self.oswald = float(oswald)
        self.cl_max = float(cl_max)
        self.drag_mode = drag_mode
        self.custom_cd = None if custom_cd in (None, "") else float(custom_cd)
        self.rolling_friction = float(rolling_friction)
        self.reserve_thrust_fraction_for_takeoff = float(reserve_thrust_fraction_for_takeoff)
        self.name = str(name)

    @property
    def weight_N(self) -> float:
        return (self.weight_g / 1000.0) * G0

    @property
    def aspect_ratio(self) -> float:
        # AR = b^2 / S
        return (self.wingspan_m ** 2) / max(self.wing_area_m2, 1e-9)

    @property
    def induced_k(self) -> float:
        # Induced drag factor in the drag polar.
        return 1.0 / max(math.pi * self.aspect_ratio * self.oswald, 1e-9)


class FixedWingConfig:
    def __init__(self,
                 airframe: AirframeConfig,
                 battery: BatteryConfig,
                 motor: MotorConfig,
                 propeller: PropellerConfig,
                 motor_count: int = 1,
                 esc: Optional[ESCConfig] = None,
                 avionics: Optional[AvionicsConfig] = None,
                 altitude_m: float = 0.0,
                 temperature_C: Optional[float] = None,
                 pressure_Pa: Optional[float] = None):
        self.airframe = airframe
        self.battery = battery
        self.motor = motor
        self.propeller = propeller
        self.motor_count = int(motor_count)
        self.esc = esc
        self.avionics = avionics
        self.altitude_m = float(altitude_m)
        self.temperature_C = None if temperature_C in (None, "") else float(temperature_C)
        self.pressure_Pa = None if pressure_Pa in (None, "") else float(pressure_Pa)
        self.air_density = compute_air_density(self.altitude_m, self.temperature_C, self.pressure_Pa)


# --------------------------------------------------------------------------------------
# Standard atmosphere
# --------------------------------------------------------------------------------------
def compute_air_density(altitude_m: float, temperature_C: Optional[float] = None, pressure_Pa: Optional[float] = None) -> float:
    if temperature_C is None:
        temperature_C = (T0 - L * altitude_m) - 273.15
    T = temperature_C + 273.15
    if pressure_Pa is None:
        P = P0 * (1.0 - (L * altitude_m) / T0) ** (G0 / (R * L))
    else:
        P = pressure_Pa
    return P / (R * T)


# --------------------------------------------------------------------------------------
# Fixed-wing aerodynamics
# --------------------------------------------------------------------------------------
def dynamic_pressure(rho: float, V: float) -> float:
    # q = 1/2 * rho * V^2
    return 0.5 * rho * V * V


def stall_speed_mps(cfg: FixedWingConfig) -> float:
    """
    Stall speed from the lift equation.

    Lift equation:
        L = 1/2 rho V^2 S C_L

    At stall, lift must equal weight while C_L = C_Lmax:
        W = 1/2 rho V_stall^2 S C_Lmax

    Therefore:
        V_stall = sqrt(2W / (rho S C_Lmax))
    """
    W = cfg.airframe.weight_N
    rho = cfg.air_density
    S = cfg.airframe.wing_area_m2
    cl_max = cfg.airframe.cl_max
    return math.sqrt((2.0 * W) / max(rho * S * cl_max, 1e-9))


def lift_coefficient_for_level_flight(cfg: FixedWingConfig, V: float) -> float:
    W = cfg.airframe.weight_N
    q = dynamic_pressure(cfg.air_density, V)
    return W / max(q * cfg.airframe.wing_area_m2, 1e-9)


def drag_coefficient(cfg: FixedWingConfig, cl: float) -> float:
    """
    Standard drag polar:
        C_D = C_D0 + k C_L^2

    If the user wants a simplified fixed Cd, they can force drag_mode='simplified'.
    """
    if cfg.airframe.drag_mode == "simplified" and cfg.airframe.custom_cd is not None:
        return cfg.airframe.custom_cd
    return cfg.airframe.cd0 + cfg.airframe.induced_k * (cl ** 2)


def drag_force_N(cfg: FixedWingConfig, V: float) -> float:
    cl = lift_coefficient_for_level_flight(cfg, V)
    cd = drag_coefficient(cfg, cl)
    q = dynamic_pressure(cfg.air_density, V)
    return q * cfg.airframe.wing_area_m2 * cd


def power_required_level_W(cfg: FixedWingConfig, V: float) -> float:
    """
    Level-flight mechanical power requirement.

    For steady level flight:
        thrust required = drag

    Mechanical power required is force times speed:
        P_req = D * V
    """
    return drag_force_N(cfg, V) * V


def pitch_speed_mps(prop: PropellerConfig, rpm: float) -> float:
    """
    Ideal helical pitch speed ignoring slip.

    pitch_speed = pitch * rev_per_sec
    where pitch is converted to meters per revolution.
    """
    pitch_m = prop.pitch_in * 0.0254
    return pitch_m * (rpm / 60.0)


def prop_tip_speed_mps(prop: PropellerConfig, rpm: float) -> float:
    D = prop.diameter_in * 0.0254
    return math.pi * D * (rpm / 60.0)


# --------------------------------------------------------------------------------------
# Propulsion models
# --------------------------------------------------------------------------------------
def interpolate_motor_table_static_point(prop: PropellerConfig, target_thrust_N: float) -> dict:
    if prop.table is None:
        raise ValueError("No prop table is loaded.")

    thrust_g = target_thrust_N * 1000.0 / G0
    df = prop.table

    if thrust_g <= float(df["Thrust_g"].min()):
        row = df.iloc[0]
        return {k: float(row[k]) for k in df.columns if pd.notna(row[k]) and k in ("Thrust_g", "Power_W", "Current_A", "Voltage_V", "RPM", "Throttle_pct", "Efficiency_gW", "Temp_C", "Torque_Nm")}
    if thrust_g >= float(df["Thrust_g"].max()):
        row = df.iloc[-1]
        return {k: float(row[k]) for k in df.columns if pd.notna(row[k]) and k in ("Thrust_g", "Power_W", "Current_A", "Voltage_V", "RPM", "Throttle_pct", "Efficiency_gW", "Temp_C", "Torque_Nm")}

    lower = df[df["Thrust_g"] <= thrust_g].iloc[-1]
    upper = df[df["Thrust_g"] >= thrust_g].iloc[0]
    frac = (thrust_g - lower["Thrust_g"]) / max(upper["Thrust_g"] - lower["Thrust_g"], 1e-9)

    out = {}
    for col in ("Power_W", "Current_A", "Voltage_V", "RPM", "Throttle_pct", "Efficiency_gW", "Temp_C", "Torque_Nm"):
        if col in df.columns and pd.notna(lower[col]) and pd.notna(upper[col]):
            out[col] = float(lower[col] + (upper[col] - lower[col]) * frac)
    return out


def estimate_prop_operating_point_coeff_model(cfg: FixedWingConfig, throttle: float, airspeed_mps: float = 0.0) -> dict:
    """
    eCalc-style coefficient model.

    The propeller relations in dimensional form are:
        T = C_T * rho * n^2 * D^4
        P_mech = C_P * rho * n^3 * D^5

    where:
        n = revolutions per second
        D = prop diameter [m]
        rho = air density [kg/m^3]

    Forward speed reduces thrust and power because the propeller operates at a nonzero advance ratio.
    A full model would use C_T(J) and C_P(J). Here we apply a transparent first-order unloading factor:
        J = V / (n D)
        unload = max(0.25, 1 - a*J)
    and then scale:
        T_dynamic = T_static * unload
        P_dynamic = P_static * (0.70 + 0.30*unload)

    Electrical model:
        K_t = 60 / (2*pi*K_v)
        Q = P_mech / omega
        I = Q / K_t + I_0
        V = I*R_m + omega/K_v_rad
    """
    motor = cfg.motor
    prop = cfg.propeller
    batt = cfg.battery

    if motor.kv is None:
        raise ValueError("motor.kv is required for the coefficient/electrical propulsion model")
    if prop.TConst is None or prop.PConst is None:
        raise ValueError("prop_tconst and prop_pconst are required for the coefficient propulsion model")

    throttle = max(0.0, min(1.0, float(throttle)))
    gear = max(prop.gear_ratio, 1e-9)

    v_guess = batt.vnom_pack * throttle
    # No-load speed approximation, then back out by electrical and prop loading.
    rpm_noload = motor.kv * v_guess / gear
    rpm = max(100.0, 0.92 * rpm_noload)

    D = prop.diameter_in * 0.0254
    rho = cfg.air_density

    # Iterate because prop torque and electrical torque must match.
    for _ in range(25):
        n = rpm / 60.0
        J = airspeed_mps / max(n * D, 1e-9)
        unload = max(0.25, 1.0 - 0.55 * J)

        thrust_N = prop.TConst * rho * (n ** 2) * (D ** 4) * unload
        mech_power_W = prop.PConst * rho * (n ** 3) * (D ** 5) * (0.70 + 0.30 * unload)
        omega = 2.0 * math.pi * n
        torque_Nm = mech_power_W / max(omega, 1e-9)

        # BLDC motor constants.
        Kt = 60.0 / (2.0 * math.pi * motor.kv)
        current_A = torque_Nm / max(Kt, 1e-9) + motor.idle_current

        # Motor terminal equation in SI-friendly form.
        kv_rad = motor.kv * (2.0 * math.pi / 60.0)  # rad/s per volt
        v_back_emf = omega / max(kv_rad, 1e-9)
        v_terminal = v_back_emf + current_A * motor.resistance

        # Limit by available battery/throttle voltage.
        v_available = batt.vnom_pack * throttle
        if v_terminal > v_available:
            rpm *= max(0.3, v_available / max(v_terminal, 1e-9))
        else:
            # relax toward the electrical operating point
            rpm *= 1.01

    n = rpm / 60.0
    J = airspeed_mps / max(n * D, 1e-9)
    unload = max(0.25, 1.0 - 0.55 * J)
    thrust_N = prop.TConst * rho * (n ** 2) * (D ** 4) * unload
    mech_power_W = prop.PConst * rho * (n ** 3) * (D ** 5) * (0.70 + 0.30 * unload)
    omega = 2.0 * math.pi * n
    torque_Nm = mech_power_W / max(omega, 1e-9)
    Kt = 60.0 / (2.0 * math.pi * motor.kv)
    current_A = torque_Nm / max(Kt, 1e-9) + motor.idle_current
    electrical_power_W = min(batt.vnom_pack * current_A, motor.max_power)
    current_A = min(current_A, motor.max_current)

    return {
        "rpm": float(rpm),
        "thrust_N": float(thrust_N),
        "torque_Nm": float(torque_Nm),
        "current_A": float(current_A),
        "electrical_power_W": float(electrical_power_W),
        "mechanical_power_W": float(mech_power_W),
        "efficiency": float(mech_power_W / max(electrical_power_W, 1e-9)),
        "advance_ratio": float(J),
        "pitch_speed_mps": float(pitch_speed_mps(prop, rpm)),
        "tip_speed_mps": float(prop_tip_speed_mps(prop, rpm)),
        "throttle_pct": float(throttle * 100.0),
    }


def estimate_prop_operating_point_table_model(cfg: FixedWingConfig, throttle: float, airspeed_mps: float = 0.0) -> dict:
    """
    Table-driven static model with a simple forward-flight unloading correction.

    We treat the uploaded test table as static test data. In forward flight, the propeller is unloaded,
    so current and input power usually decrease while pitch-speed-related effects become important.

    The correction used here is intentionally simple and easy to edit:
        J ~ V / (nD)
        unload = max(0.30, 1 - 0.60*J)
        T_dynamic = T_static * unload
        P_dynamic = P_static * (0.65 + 0.35*unload)

    For exact propeller maps you would replace this with C_T(J), C_P(J) data.
    """
    prop = cfg.propeller
    if prop.table is None:
        raise ValueError("A prop test table CSV is required for table mode.")

    throttle = max(0.0, min(1.0, float(throttle)))
    df = prop.table

    if "Throttle_pct" in df.columns and df["Throttle_pct"].notna().sum() >= 2:
        x = float(throttle * 100.0)
        xvals = df["Throttle_pct"].to_numpy()
        row_lo = df.iloc[(abs(xvals - x)).argmin()]
        # linear interpolation using surrounding points if possible
        lower_df = df[df["Throttle_pct"] <= x]
        upper_df = df[df["Throttle_pct"] >= x]
        if len(lower_df) and len(upper_df):
            lower = lower_df.iloc[-1]
            upper = upper_df.iloc[0]
            frac = 0.0 if upper["Throttle_pct"] == lower["Throttle_pct"] else (x - lower["Throttle_pct"]) / max(upper["Throttle_pct"] - lower["Throttle_pct"], 1e-9)
            interp = {}
            for col in ("Thrust_g", "Power_W", "Current_A", "Voltage_V", "RPM", "Throttle_pct", "Efficiency_gW", "Temp_C", "Torque_Nm"):
                if col in df.columns and pd.notna(lower[col]) and pd.notna(upper[col]):
                    interp[col] = float(lower[col] + (upper[col] - lower[col]) * frac)
        else:
            interp = {col: float(row_lo[col]) for col in ("Thrust_g", "Power_W", "Current_A", "Voltage_V", "RPM", "Throttle_pct", "Efficiency_gW", "Temp_C", "Torque_Nm") if col in df.columns and pd.notna(row_lo[col])}
    else:
        # Fallback: map throttle to table index fraction.
        idx = int(round(throttle * (len(df) - 1)))
        row = df.iloc[idx]
        interp = {col: float(row[col]) for col in ("Thrust_g", "Power_W", "Current_A", "Voltage_V", "RPM", "Throttle_pct", "Efficiency_gW", "Temp_C", "Torque_Nm") if col in df.columns and pd.notna(row[col])}

    rpm = float(interp.get("RPM", 0.0))
    D = prop.diameter_in * 0.0254
    n = rpm / 60.0
    J = airspeed_mps / max(n * D, 1e-9) if rpm > 1.0 else 0.0
    unload = max(0.30, 1.0 - 0.60 * J)

    thrust_N = (interp.get("Thrust_g", 0.0) / 1000.0) * G0 * unload
    electrical_power_W = interp.get("Power_W", 0.0) * (0.65 + 0.35 * unload)
    current_A = interp.get("Current_A", electrical_power_W / max(cfg.battery.vnom_pack, 1e-9)) * (0.70 + 0.30 * unload)
    torque_Nm = interp.get("Torque_Nm", (electrical_power_W / max(2 * math.pi * n, 1e-9)) if rpm > 1 else 0.0)
    mech_eff = 0.80  # conservative placeholder when shaft power is not directly measured
    mechanical_power_W = electrical_power_W * mech_eff

    return {
        "rpm": float(rpm),
        "thrust_N": float(thrust_N),
        "torque_Nm": float(torque_Nm),
        "current_A": float(current_A),
        "electrical_power_W": float(electrical_power_W),
        "mechanical_power_W": float(mechanical_power_W),
        "efficiency": float(mechanical_power_W / max(electrical_power_W, 1e-9)),
        "advance_ratio": float(J),
        "pitch_speed_mps": float(pitch_speed_mps(prop, rpm)),
        "tip_speed_mps": float(prop_tip_speed_mps(prop, rpm)),
        "throttle_pct": float(interp.get("Throttle_pct", throttle * 100.0)),
        "temperature_C": float(interp.get("Temp_C", 25.0)),
    }


def estimate_prop_operating_point(cfg: FixedWingConfig, throttle: float, airspeed_mps: float = 0.0) -> dict:
    if cfg.propeller.table is not None:
        return estimate_prop_operating_point_table_model(cfg, throttle, airspeed_mps)
    return estimate_prop_operating_point_coeff_model(cfg, throttle, airspeed_mps)


def find_throttle_for_required_thrust(cfg: FixedWingConfig, required_thrust_N: float, airspeed_mps: float) -> dict:
    """Solve throttle so propulsive thrust matches required thrust in steady flight."""
    lo, hi = 0.0, 1.0
    best = estimate_prop_operating_point(cfg, 1.0, airspeed_mps)
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        pt = estimate_prop_operating_point(cfg, mid, airspeed_mps)
        total_thrust = pt["thrust_N"] * cfg.motor_count
        if total_thrust < required_thrust_N:
            lo = mid
        else:
            hi = mid
            best = pt
    best = dict(best)
    best["throttle"] = float(0.5 * (lo + hi))
    best["total_thrust_N"] = float(best["thrust_N"] * cfg.motor_count)
    return best


def max_propulsion_at_speed(cfg: FixedWingConfig, airspeed_mps: float) -> dict:
    pt = estimate_prop_operating_point(cfg, 1.0, airspeed_mps)
    total_motor_input_W = pt["electrical_power_W"] * cfg.motor_count
    periph_power_W = avionics_input_power_W(cfg.avionics)
    total_input_W, v_load, pack_current_A, esc_note, i_esc, esc_loss_W = total_power_with_esc(
        cfg.battery, cfg.esc, total_motor_input_W, periph_power_W, cfg.motor_count
    )
    mech_propulsive_W = pt["mechanical_power_W"] * cfg.motor_count
    total_thrust = pt["thrust_N"] * cfg.motor_count
    return {
        **pt,
        "total_thrust_N": float(total_thrust),
        "motor_input_total_W": float(total_motor_input_W),
        "mech_propulsive_total_W": float(mech_propulsive_W),
        "periph_power_W": float(periph_power_W),
        "total_input_power_W": float(total_input_W),
        "v_load_V": float(v_load),
        "pack_current_A": float(pack_current_A),
        "esc_note": esc_note,
        "motor_I_per_esc_A": float(i_esc),
        "esc_loss_W": float(esc_loss_W),
    }


# --------------------------------------------------------------------------------------
# Performance calculations
# --------------------------------------------------------------------------------------
def operating_point_for_speed(cfg: FixedWingConfig, speed_mps: float) -> dict:
    """
    Find the steady level-flight operating point at the requested speed.

    Required thrust in steady level flight is approximately the drag force.
    The propulsion solver then finds the throttle needed to generate that thrust.
    """
    if speed_mps <= 0:
        raise ValueError("Speed must be > 0")

    Vstall = stall_speed_mps(cfg)
    cl = lift_coefficient_for_level_flight(cfg, speed_mps)
    if cl > cfg.airframe.cl_max * 1.001:
        return {"feasible": False, "reason": "Below stall for requested weight/wing/CLmax", "speed_mps": float(speed_mps)}

    drag_N = drag_force_N(cfg, speed_mps)
    preq_mech_W = drag_N * speed_mps
    prop_pt = find_throttle_for_required_thrust(cfg, drag_N, speed_mps)

    motor_input_total_W = prop_pt["electrical_power_W"] * cfg.motor_count
    mech_propulsive_total_W = prop_pt["mechanical_power_W"] * cfg.motor_count
    periph_power_W = avionics_input_power_W(cfg.avionics)
    total_input_W, v_load, pack_current_A, esc_note, i_esc, esc_loss_W = total_power_with_esc(
        cfg.battery, cfg.esc, motor_input_total_W, periph_power_W, cfg.motor_count
    )

    if pack_current_A > cfg.battery.discharge_max_A:
        return {"feasible": False, "reason": "Battery current exceeds max discharge", "speed_mps": float(speed_mps)}
    if v_load < cfg.battery.vmin_pack:
        return {"feasible": False, "reason": "Battery voltage under minimum operating voltage", "speed_mps": float(speed_mps)}
    if cfg.esc is not None and i_esc > cfg.esc.max_current_A:
        return {"feasible": False, "reason": "ESC current exceeds max rating", "speed_mps": float(speed_mps)}

    shaft_eff = mech_propulsive_total_W / max(motor_input_total_W, 1e-9)
    total_elec_eff = preq_mech_W / max(total_input_W, 1e-9)

    return {
        "feasible": True,
        "speed_mps": float(speed_mps),
        "speed_kmh": float(speed_mps * 3.6),
        "stall_speed_mps": float(Vstall),
        "stall_speed_kmh": float(Vstall * 3.6),
        "cl": float(cl),
        "cd": float(drag_coefficient(cfg, cl)),
        "drag_N": float(drag_N),
        "thrust_required_N": float(drag_N),
        "throttle": float(prop_pt["throttle"]),
        "throttle_pct": float(100.0 * prop_pt["throttle"]),
        "rpm": float(prop_pt["rpm"]),
        "motor_input_total_W": float(motor_input_total_W),
        "propulsive_mech_total_W": float(mech_propulsive_total_W),
        "level_power_required_W": float(preq_mech_W),
        "pack_input_power_W": float(total_input_W),
        "pack_voltage_V": float(v_load),
        "pack_current_A": float(pack_current_A),
        "motor_current_A": float(prop_pt["current_A"]),
        "motor_torque_Nm": float(prop_pt["torque_Nm"]),
        "shaft_efficiency": float(shaft_eff),
        "total_electrical_efficiency": float(total_elec_eff),
        "advance_ratio": float(prop_pt["advance_ratio"]),
        "pitch_speed_mps": float(prop_pt["pitch_speed_mps"]),
        "pitch_speed_kmh": float(prop_pt["pitch_speed_mps"] * 3.6),
        "tip_speed_mps": float(prop_pt["tip_speed_mps"]),
        "periph_power_W": float(periph_power_W),
        "esc_loss_W": float(esc_loss_W),
        "esc_note": esc_note,
        "motor_I_per_esc_A": float(i_esc),
        "flight_time_min": float((cfg.battery.usable_Wh / max(total_input_W, 1e-9)) * 60.0),
        "range_km": float((cfg.battery.usable_Wh / max(total_input_W, 1e-9)) * speed_mps * 3.6),
    }


def climb_point_for_speed(cfg: FixedWingConfig, speed_mps: float) -> dict:
    """
    Climb from excess propulsive power.

    Let:
        P_excess = P_avail_mech - P_req_level

    Then rate of climb is:
        ROC = P_excess / W

    Because power is force times velocity and W is a force, ROC comes out in m/s.
    """
    max_pt = max_propulsion_at_speed(cfg, speed_mps)
    level_preq = power_required_level_W(cfg, speed_mps)
    P_excess = max_pt["mech_propulsive_total_W"] - level_preq
    roc = P_excess / max(cfg.airframe.weight_N, 1e-9)
    roc = max(-50.0, roc)
    gamma_deg = math.degrees(math.asin(max(-0.999, min(0.999, roc / max(speed_mps, 1e-9))))) if speed_mps > 1e-9 else 0.0
    return {
        **max_pt,
        "speed_mps": float(speed_mps),
        "speed_kmh": float(speed_mps * 3.6),
        "level_power_required_W": float(level_preq),
        "excess_power_W": float(P_excess),
        "roc_mps": float(roc),
        "climb_angle_deg": float(gamma_deg),
        "time_to_500m_s": float(500.0 / roc) if roc > 1e-9 else float("inf"),
    }


def static_vertical_performance(cfg: FixedWingConfig) -> dict:
    """
    Treat the airplane like a 3D pull-up / vertical climb point using static thrust.

    Net vertical acceleration:
        a = (T_static - W) / m

    Ideal sustained vertical speed from excess propulsive power:
        V_vert ~ P_excess / W
    where P_excess uses static propulsive power minus the power needed to merely hold weight.

    This is only a rough 3D aerobatic indicator, not a full post-stall flight model.
    """
    max_static = max_propulsion_at_speed(cfg, 0.0)
    T = max_static["total_thrust_N"]
    W = cfg.airframe.weight_N
    m = cfg.airframe.weight_g / 1000.0
    accel = (T - W) / max(m, 1e-9)
    vertical_capability = T / max(W, 1e-9)
    v_vert = max(0.0, (max_static["mech_propulsive_total_W"] - 0.0) / max(W, 1e-9))
    return {
        **max_static,
        "vertical_accel_mps2": float(accel),
        "vertical_speed_mps": float(v_vert if T > W else 0.0),
        "three_d_ratio": float(vertical_capability),
        "can_hover_like_3d": bool(T > W),
    }


def estimate_takeoff_distance_m(cfg: FixedWingConfig, safety_factor_speed: float = 1.2) -> float:
    """
    Crude ground-run estimate using constant average acceleration.

    Target takeoff speed:
        V_to = 1.2 * V_stall

    Average net force on the ground:
        F_net = T_avg - mu (W - L_avg) - D_avg

    We approximate L_avg ~ 0.3W during the roll and D_avg at 0.7 V_to.
    Then:
        a_avg = F_net / m
        s = V_to^2 / (2 a_avg)
    """
    Vto = safety_factor_speed * stall_speed_mps(cfg)
    Vavg = 0.7 * Vto
    max_pt = max_propulsion_at_speed(cfg, Vavg)
    Tavg = cfg.airframe.reserve_thrust_fraction_for_takeoff * max_pt["total_thrust_N"]
    W = cfg.airframe.weight_N
    Davg = drag_force_N(cfg, Vavg)
    rolling = cfg.airframe.rolling_friction * max(W - 0.3 * W, 0.0)
    m = cfg.airframe.weight_g / 1000.0
    a = (Tavg - Davg - rolling) / max(m, 1e-9)
    if a <= 1e-9:
        return float("inf")
    return (Vto ** 2) / (2.0 * a)


def scan_speed_performance(cfg: FixedWingConfig,
                           vmin_mps: Optional[float] = None,
                           vmax_mps: Optional[float] = None,
                           npts: int = 140) -> pd.DataFrame:
    Vstall = stall_speed_mps(cfg)
    if vmin_mps is None:
        vmin_mps = max(3.0, 1.05 * Vstall)
    if vmax_mps is None:
        vmax_mps = max(vmin_mps + 1.0, 3.5 * Vstall)

    rows = []
    for i in range(npts):
        V = vmin_mps + (vmax_mps - vmin_mps) * i / max(npts - 1, 1)
        op = operating_point_for_speed(cfg, V)
        cl = climb_point_for_speed(cfg, V)
        row = {
            "speed_mps": V,
            "speed_kmh": V * 3.6,
            "stall_speed_mps": Vstall,
            "stall_speed_kmh": Vstall * 3.6,
            "feasible": bool(op.get("feasible", False)),
            "reason": op.get("reason", ""),
            "cl": float(op.get("cl", float("nan"))),
            "cd": float(op.get("cd", float("nan"))),
            "drag_N": float(op.get("drag_N", drag_force_N(cfg, V))),
            "throttle_pct": float(op.get("throttle_pct", float("nan"))),
            "rpm": float(op.get("rpm", cl.get("rpm", float("nan")))),
            "pack_input_power_W": float(op.get("pack_input_power_W", float("nan"))),
            "level_power_required_W": float(op.get("level_power_required_W", power_required_level_W(cfg, V))),
            "motor_input_total_W": float(cl.get("motor_input_total_W", float("nan"))),
            "mech_propulsive_total_W": float(cl.get("mech_propulsive_total_W", float("nan"))),
            "excess_power_W": float(cl.get("excess_power_W", float("nan"))),
            "roc_mps": float(cl.get("roc_mps", float("nan"))),
            "climb_angle_deg": float(cl.get("climb_angle_deg", float("nan"))),
            "time_to_500m_s": float(cl.get("time_to_500m_s", float("nan"))),
            "range_km": float(op.get("range_km", float("nan"))),
            "flight_time_min": float(op.get("flight_time_min", float("nan"))),
            "pack_current_A": float(op.get("pack_current_A", cl.get("pack_current_A", float("nan")))),
            "pack_voltage_V": float(op.get("pack_voltage_V", cl.get("v_load_V", float("nan")))),
            "motor_current_A": float(op.get("motor_current_A", cl.get("current_A", float("nan")))),
            "torque_Nm": float(op.get("motor_torque_Nm", cl.get("torque_Nm", float("nan")))),
            "pitch_speed_kmh": float(op.get("pitch_speed_kmh", cl.get("pitch_speed_mps", float("nan")) * 3.6 if math.isfinite(float(cl.get("pitch_speed_mps", float("nan")))) else float("nan"))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_performance(cfg: FixedWingConfig, speed_kmh: float) -> dict:
    V = speed_kmh / 3.6
    curve = scan_speed_performance(cfg)
    curve_feas = curve[curve["feasible"]].copy()
    if curve_feas.empty:
        raise RuntimeError("No feasible operating points found. Check battery/motor/prop sizing.")

    op = operating_point_for_speed(cfg, V)
    static3d = static_vertical_performance(cfg)

    idx_end = curve_feas["flight_time_min"].idxmax()
    idx_range = curve_feas["range_km"].idxmax()
    idx_climb = curve_feas["roc_mps"].idxmax()
    idx_vmax = curve_feas[curve_feas["roc_mps"] >= 0.0]["speed_mps"].idxmax() if (curve_feas["roc_mps"] >= 0.0).any() else curve_feas["speed_mps"].idxmax()

    best_end = curve_feas.loc[idx_end].to_dict()
    best_range = curve_feas.loc[idx_range].to_dict()
    best_climb = curve_feas.loc[idx_climb].to_dict()
    vmax_row = curve_feas.loc[idx_vmax].to_dict()

    # Estimate motor optimum efficiency point as the feasible point with minimum electrical input power.
    idx_opt_motor = curve_feas["pack_input_power_W"].idxmin()
    opt_motor = curve_feas.loc[idx_opt_motor].to_dict()

    # Maximum-power point from speed sweep.
    idx_motor_max = curve_feas["motor_input_total_W"].idxmax()
    max_motor = curve_feas.loc[idx_motor_max].to_dict()

    takeoff_m = estimate_takeoff_distance_m(cfg)
    drive_weight_g = sum(x for x in [cfg.motor.weight_g, cfg.propeller.weight_g, cfg.battery.weight_g, (cfg.esc.weight_g if cfg.esc else 0.0)] if x is not None)

    result = {
        "selected_speed": op,
        "curve": curve,
        "battery": {
            "load_C": op["pack_current_A"] / max(cfg.battery.capacity_Ah, 1e-9),
            "voltage_V": op["pack_voltage_V"],
            "rated_voltage_V": cfg.battery.vnom_pack,
            "energy_Wh": cfg.battery.capacity_Wh,
            "usable_energy_Wh": cfg.battery.usable_Wh,
            "total_capacity_mAh": cfg.battery.capacity_mAh,
            "used_capacity_mAh": (op["pack_input_power_W"] / max(op["pack_voltage_V"], 1e-9)) * (op["flight_time_min"] / 60.0) * 1000.0,
            "flight_time_min": op["flight_time_min"],
            "mixed_flight_time_min": best_end["flight_time_min"],
            "weight_g": cfg.battery.weight_g,
        },
        "motor_optimum": {
            "current_A": opt_motor["motor_current_A"],
            "voltage_V": opt_motor["pack_voltage_V"],
            "rpm": opt_motor["rpm"],
            "electrical_power_W": opt_motor["motor_input_total_W"],
            "mechanical_power_W": opt_motor["level_power_required_W"],
            "efficiency_percent": 100.0 * (opt_motor["level_power_required_W"] / max(opt_motor["pack_input_power_W"], 1e-9)),
        },
        "motor_max": {
            "current_A": max_motor["motor_current_A"],
            "voltage_V": max_motor["pack_voltage_V"],
            "rpm": max_motor["rpm"],
            "electrical_power_W": max_motor["motor_input_total_W"],
            "mechanical_power_W": max_motor["mech_propulsive_total_W"],
            "efficiency_percent": 100.0 * (max_motor["mech_propulsive_total_W"] / max(max_motor["motor_input_total_W"], 1e-9)),
            "temperature_C": None,
            "rm_mohm": cfg.motor.resistance * 1000.0,
        },
        "propeller": {
            "static_thrust_g": static3d["total_thrust_N"] * 1000.0 / G0,
            "rpm": static3d["rpm"],
            "available_thrust_g_at_selected_speed": op["thrust_required_N"] * 1000.0 / G0,
            "pitch_speed_kmh": static3d["pitch_speed_mps"] * 3.6,
            "tip_speed_kmh": static3d["tip_speed_mps"] * 3.6,
            "specific_thrust_g_per_W": (static3d["total_thrust_N"] * 1000.0 / G0) / max(static3d["motor_input_total_W"], 1e-9),
            "n100W_rpm": static3d["rpm"] * math.sqrt(100.0 / max(static3d["motor_input_total_W"], 1e-9)),
            "n10N_rpm": static3d["rpm"] * math.sqrt(10.0 / max(static3d["total_thrust_N"], 1e-9)),
        },
        "total_drive": {
            "drive_weight_g": drive_weight_g,
            "power_to_weight_Wkg": max_motor["motor_input_total_W"] / max(cfg.airframe.weight_g / 1000.0, 1e-9),
            "thrust_to_weight": static3d["three_d_ratio"],
            "current_at_max_A": max_motor["pack_current_A"],
            "pin_at_max_W": max_motor["pack_input_power_W"],
            "pout_at_max_W": max_motor["mech_propulsive_total_W"],
            "efficiency_at_max_percent": 100.0 * (max_motor["mech_propulsive_total_W"] / max(max_motor["pack_input_power_W"], 1e-9)),
            "torque_Nm": max_motor["torque_Nm"],
            "climb_capacity_m": max(best_climb["roc_mps"], 0.0) * op["flight_time_min"] * 60.0,
        },
        "airplane": {
            "all_up_weight_g": cfg.airframe.weight_g,
            "wing_load_gdm2": cfg.airframe.weight_g / max(cfg.airframe.wing_area_m2 * 100.0, 1e-9),
            "cubic_wing_loading": cfg.airframe.weight_g / max((cfg.airframe.wing_area_m2 * 10000.0) ** 1.5 / 1000.0, 1e-9),
            "stall_speed_kmh": stall_speed_mps(cfg) * 3.6,
            "best_endurance_speed_kmh": best_end["speed_kmh"],
            "best_range_speed_kmh": best_range["speed_kmh"],
            "best_efficiency_Whkm": cfg.battery.usable_Wh / max(best_range["range_km"], 1e-9),
            "max_speed_kmh": vmax_row["speed_kmh"],
            "max_rate_of_climb_mps": best_climb["roc_mps"],
            "time_to_500m_s": best_climb["time_to_500m_s"],
            "takeoff_distance_m": takeoff_m,
            "pitch_speed_kmh": op["pitch_speed_kmh"],
            "zero_thrust_speed_kmh": static3d["pitch_speed_mps"] * 3.6,
            "prop_unstall_speed_kmh": 0.85 * static3d["pitch_speed_mps"] * 3.6,
            "max_vertical_speed_mps": static3d["vertical_speed_mps"],
            "three_d_ratio": static3d["three_d_ratio"],
            "vertical_accel_mps2": static3d["vertical_accel_mps2"],
        },
    }
    return result


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def make_power_diagram_figure(curve: pd.DataFrame):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    good = curve[curve["feasible"]].copy()
    ax.plot(good["speed_kmh"], good["level_power_required_W"], label="Min. Power for Level Flight [W]")
    ax.plot(good["speed_kmh"], good["mech_propulsive_total_W"], label="Dynamic Propeller Power Available [W]")
    ax.plot(good["speed_kmh"], good["motor_input_total_W"], label="Electrical Motor Input [W]")
    ax.set_xlabel("Air Speed [km/h]")
    ax.set_ylabel("Power [W]")
    ax.set_title("Power Diagram in Level Flight")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def make_climb_figure(curve: pd.DataFrame):
    good = curve[curve["feasible"]].copy()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(good["speed_kmh"], good["roc_mps"], label="Rate of Climb [m/s]")
    ax.plot(good["speed_kmh"], good["climb_angle_deg"], label="Angle of Climb [deg]")
    ax2 = ax.twinx()
    ax2.plot(good["speed_kmh"], good["time_to_500m_s"], label="Time to 500m [s]", linestyle="--")
    ax.set_xlabel("Air Speed [km/h]")
    ax.set_ylabel("ROC / climb angle")
    ax2.set_ylabel("Time to 500m [s]")
    ax.set_title("Climb Performance")
    ax.grid(True)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    return fig


def make_motor_characteristic_figure(curve: pd.DataFrame):
    good = curve[curve["feasible"]].copy()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(good["speed_kmh"], good["motor_input_total_W"], label="Electrical Power [W]")
    ax.plot(good["speed_kmh"], good["motor_current_A"], label="Current [A]")
    ax.plot(good["speed_kmh"], good["rpm"] / 100.0, label="RPM / 100")
    ax.plot(good["speed_kmh"], good["torque_Nm"] * 100.0, label="Torque x100 [Nm]")
    ax.set_xlabel("Air Speed [km/h]")
    ax.set_ylabel("Motor quantities")
    ax.set_title("Motor Characteristic at Full-Load Envelope")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def make_vertical_performance_figure(cfg: FixedWingConfig):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    data = static_vertical_performance(cfg)
    labels = ["3D Capability", "Acceleration [m/s²]", "Vertical Speed [m/s]"]
    values = [data["three_d_ratio"], data["vertical_accel_mps2"], data["vertical_speed_mps"]]
    ax.bar(labels, values)
    ax.set_title("Static Vertical / 3D Performance")
    ax.set_ylabel("Metric value")
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig


def plot_all_figures(summary: dict, cfg: FixedWingConfig):
    curve = summary["curve"]
    figs = [
        make_power_diagram_figure(curve),
        make_climb_figure(curve),
        make_motor_characteristic_figure(curve),
        make_vertical_performance_figure(cfg),
    ]
    plt.show()
    return figs


# --------------------------------------------------------------------------------------
# CLI reporting helpers
# --------------------------------------------------------------------------------------
def print_summary(summary: dict):
    sel = summary["selected_speed"]
    batt = summary["battery"]
    mopt = summary["motor_optimum"]
    mmax = summary["motor_max"]
    prop = summary["propeller"]
    drive = summary["total_drive"]
    air = summary["airplane"]

    print("\n=== Selected Operating Point ===")
    print(f"Speed: {sel['speed_kmh']:.1f} km/h")
    print(f"Throttle: {sel['throttle_pct']:.1f} %")
    print(f"Pack input power: {sel['pack_input_power_W']:.1f} W")
    print(f"Pack current: {sel['pack_current_A']:.2f} A")
    print(f"Pack voltage: {sel['pack_voltage_V']:.2f} V")
    print(f"RPM: {sel['rpm']:.0f}")
    print(f"Flight time: {sel['flight_time_min']:.2f} min")
    print(f"Range: {sel['range_km']:.2f} km")

    print("\n=== Battery ===")
    for k, v in batt.items():
        print(f"{k}: {v}")

    print("\n=== Motor @ Optimum Efficiency ===")
    for k, v in mopt.items():
        print(f"{k}: {v}")

    print("\n=== Motor @ Maximum ===")
    for k, v in mmax.items():
        print(f"{k}: {v}")

    print("\n=== Propeller ===")
    for k, v in prop.items():
        print(f"{k}: {v}")

    print("\n=== Total Drive ===")
    for k, v in drive.items():
        print(f"{k}: {v}")

    print("\n=== Airplane ===")
    for k, v in air.items():
        print(f"{k}: {v}")


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------
def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("Fixed-Wing Power Simulator")
    root.geometry("1450x980")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    tab_inputs = ttk.Frame(notebook)
    tab_summary = ttk.Frame(notebook)
    tab_plots = ttk.Frame(notebook)
    notebook.add(tab_inputs, text="Inputs")
    notebook.add(tab_summary, text="Summary")
    notebook.add(tab_plots, text="Plots")

    # ---------------- Input variables ----------------
    vars_ = {
        "weight_g": tk.StringVar(value="850"),
        "wingspan_mm": tk.StringVar(value="1270"),
        "wing_area_dm2": tk.StringVar(value="50"),
        "cd0": tk.StringVar(value="0.03"),
        "oswald": tk.StringVar(value="0.80"),
        "cl_max": tk.StringVar(value="1.30"),
        "drag_mode": tk.StringVar(value="advanced"),
        "custom_cd": tk.StringVar(value="0.06"),
        "motor_count": tk.StringVar(value="1"),
        "altitude_m": tk.StringVar(value="500"),
        "temperature_C": tk.StringVar(value="25"),
        "pressure_hpa": tk.StringVar(value="1013"),
        "battery_unit_mode": tk.StringVar(value="cell"),
        "battery_series_units": tk.StringVar(value="3"),
        "battery_parallel_units": tk.StringVar(value="1"),
        "battery_cells_series_per_unit": tk.StringVar(value="1"),
        "battery_cells_parallel_per_unit": tk.StringVar(value="1"),
        "battery_cell_capacity": tk.StringVar(value="5000"),
        "battery_pack_capacity": tk.StringVar(value="5000"),
        "battery_cell_weight": tk.StringVar(value="45"),
        "battery_pack_weight": tk.StringVar(value="400"),
        "battery_operating_voltage_min": tk.StringVar(value="3.2"),
        "battery_operating_voltage_nominal": tk.StringVar(value="3.7"),
        "battery_operating_voltage_max": tk.StringVar(value="4.2"),
        "battery_resistance_cell": tk.StringVar(value="12"),
        "battery_discharge_c_cont": tk.StringVar(value="20"),
        "battery_discharge_c_max": tk.StringVar(value="40"),
        "battery_discharge_percent": tk.StringVar(value="85"),
        "motor_kv": tk.StringVar(value="700"),
        "motor_idle_current": tk.StringVar(value="1.1"),
        "motor_idle_voltage": tk.StringVar(value="10"),
        "motor_resistance": tk.StringVar(value="0.06"),
        "motor_max_current": tk.StringVar(value="60"),
        "motor_max_power": tk.StringVar(value="800"),
        "motor_weight_g": tk.StringVar(value="70"),
        "esc_cont_current": tk.StringVar(value="60"),
        "esc_max_current": tk.StringVar(value="80"),
        "esc_idle_current": tk.StringVar(value="0.03"),
        "esc_resistance": tk.StringVar(value="0.003"),
        "esc_weight_g": tk.StringVar(value="20"),
        "prop_diameter": tk.StringVar(value="10"),
        "prop_pitch": tk.StringVar(value="4.7"),
        "prop_blades": tk.StringVar(value="2"),
        "prop_tconst": tk.StringVar(value="0.11"),
        "prop_pconst": tk.StringVar(value="0.055"),
        "gear_ratio": tk.StringVar(value="1.0"),
        "prop_weight_g": tk.StringVar(value="15"),
        "prop_table": tk.StringVar(value=""),
        "avionics_voltage_tree": tk.StringVar(value="5:(0.8,0.9)"),
        "speed_kmh": tk.StringVar(value="60"),
    }

    def add_row(parent, r, c, label, key, width=12, combo_vals=None):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=3)
        if combo_vals is None:
            e = ttk.Entry(parent, textvariable=vars_[key], width=width)
        else:
            e = ttk.Combobox(parent, textvariable=vars_[key], values=combo_vals, width=width - 1, state="readonly")
        e.grid(row=r, column=c + 1, sticky="ew", padx=4, pady=3)
        return e

    input_container = ttk.Frame(tab_inputs, padding=8)
    input_container.pack(fill="both", expand=True)

    lf_general = ttk.LabelFrame(input_container, text="General / Airframe", padding=8)
    lf_general.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    lf_battery = ttk.LabelFrame(input_container, text="Battery", padding=8)
    lf_battery.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
    lf_motor = ttk.LabelFrame(input_container, text="Motor / ESC / Prop", padding=8)
    lf_motor.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
    lf_runtime = ttk.LabelFrame(input_container, text="Runtime / Files", padding=8)
    lf_runtime.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

    for i in range(2):
        input_container.columnconfigure(i, weight=1)

    add_row(lf_general, 0, 0, "Weight [g]", "weight_g")
    add_row(lf_general, 1, 0, "Wingspan [mm]", "wingspan_mm")
    add_row(lf_general, 2, 0, "Wing area [dm²]", "wing_area_dm2")
    add_row(lf_general, 3, 0, "Cd0", "cd0")
    add_row(lf_general, 4, 0, "Oswald e", "oswald")
    add_row(lf_general, 5, 0, "CL max", "cl_max")
    add_row(lf_general, 6, 0, "Drag mode", "drag_mode", combo_vals=["advanced", "simplified"])
    add_row(lf_general, 7, 0, "Fixed Cd (simplified)", "custom_cd")
    add_row(lf_general, 8, 0, "# Motors", "motor_count")
    add_row(lf_general, 9, 0, "Field elevation [m]", "altitude_m")
    add_row(lf_general, 10, 0, "Air temp [C]", "temperature_C")
    add_row(lf_general, 11, 0, "Pressure [hPa]", "pressure_hpa")

    add_row(lf_battery, 0, 0, "Unit mode", "battery_unit_mode", combo_vals=["cell", "pack"])
    add_row(lf_battery, 1, 0, "Series units", "battery_series_units")
    add_row(lf_battery, 2, 0, "Parallel units", "battery_parallel_units")
    add_row(lf_battery, 3, 0, "Cells/pack series", "battery_cells_series_per_unit")
    add_row(lf_battery, 4, 0, "Cells/pack parallel", "battery_cells_parallel_per_unit")
    add_row(lf_battery, 5, 0, "Cell capacity [mAh]", "battery_cell_capacity")
    add_row(lf_battery, 6, 0, "Pack capacity [mAh]", "battery_pack_capacity")
    add_row(lf_battery, 7, 0, "Cell weight [g]", "battery_cell_weight")
    add_row(lf_battery, 8, 0, "Pack weight [g]", "battery_pack_weight")
    add_row(lf_battery, 9, 0, "Vmin/cell [V]", "battery_operating_voltage_min")
    add_row(lf_battery, 10, 0, "Vnom/cell [V]", "battery_operating_voltage_nominal")
    add_row(lf_battery, 11, 0, "Vmax/cell [V]", "battery_operating_voltage_max")
    add_row(lf_battery, 12, 0, "Cell resistance [mΩ]", "battery_resistance_cell")
    add_row(lf_battery, 13, 0, "Continuous C-rate", "battery_discharge_c_cont")
    add_row(lf_battery, 14, 0, "Burst C-rate", "battery_discharge_c_max")
    add_row(lf_battery, 15, 0, "Usable discharge [%]", "battery_discharge_percent")

    add_row(lf_motor, 0, 0, "Motor KV [rpm/V]", "motor_kv")
    add_row(lf_motor, 1, 0, "Motor idle current [A]", "motor_idle_current")
    add_row(lf_motor, 2, 0, "Idle-voltage ref [V]", "motor_idle_voltage")
    add_row(lf_motor, 3, 0, "Motor resistance [Ω]", "motor_resistance")
    add_row(lf_motor, 4, 0, "Motor max current [A]", "motor_max_current")
    add_row(lf_motor, 5, 0, "Motor max power [W]", "motor_max_power")
    add_row(lf_motor, 6, 0, "Motor weight [g]", "motor_weight_g")
    add_row(lf_motor, 7, 0, "ESC cont current [A]", "esc_cont_current")
    add_row(lf_motor, 8, 0, "ESC max current [A]", "esc_max_current")
    add_row(lf_motor, 9, 0, "ESC idle current [A]", "esc_idle_current")
    add_row(lf_motor, 10, 0, "ESC resistance [Ω]", "esc_resistance")
    add_row(lf_motor, 11, 0, "ESC weight [g]", "esc_weight_g")
    add_row(lf_motor, 12, 0, "Prop diameter [in]", "prop_diameter")
    add_row(lf_motor, 13, 0, "Prop pitch [in]", "prop_pitch")
    add_row(lf_motor, 14, 0, "Blades", "prop_blades")
    add_row(lf_motor, 15, 0, "Prop TConst", "prop_tconst")
    add_row(lf_motor, 16, 0, "Prop PConst", "prop_pconst")
    add_row(lf_motor, 17, 0, "Gear ratio", "gear_ratio")
    add_row(lf_motor, 18, 0, "Prop weight [g]", "prop_weight_g")

    add_row(lf_runtime, 0, 0, "Avionics rails", "avionics_voltage_tree", width=25)
    add_row(lf_runtime, 1, 0, "Selected speed [km/h]", "speed_kmh")
    add_row(lf_runtime, 2, 0, "Prop table CSV", "prop_table", width=25)

    def choose_csv():
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            vars_["prop_table"].set(p)

    ttk.Button(lf_runtime, text="Browse CSV", command=choose_csv).grid(row=2, column=2, padx=4, pady=3)

    # ---------------- Summary widgets ----------------
    summary_text = tk.Text(tab_summary, wrap="word", font=("Consolas", 10))
    summary_text.pack(fill="both", expand=True)

    plot_host = ttk.Frame(tab_plots, padding=6)
    plot_host.pack(fill="both", expand=True)
    plot_host.columnconfigure(0, weight=1)
    plot_host.rowconfigure(0, weight=1)

    current_figs = []
    current_canvases = []

    def clear_plots():
        nonlocal current_figs, current_canvases
        for c in current_canvases:
            try:
                c.get_tk_widget().destroy()
            except Exception:
                pass
        for f in current_figs:
            try:
                plt.close(f)
            except Exception:
                pass
        current_figs = []
        current_canvases = []

    def build_config_from_gui() -> FixedWingConfig:
        def f(key):
            return float(vars_[key].get())

        airframe = AirframeConfig(
            weight_g=f("weight_g"),
            wingspan_m=f("wingspan_mm") / 1000.0,
            wing_area_m2=f("wing_area_dm2") / 100.0,
            cd0=f("cd0"),
            oswald=f("oswald"),
            cl_max=f("cl_max"),
            drag_mode=vars_["drag_mode"].get(),
            custom_cd=f("custom_cd") if vars_["custom_cd"].get().strip() else None,
        )
        battery = BatteryConfig(
            chemistry=None,
            operating_voltage_min=f("battery_operating_voltage_min"),
            operating_voltage_nominal=f("battery_operating_voltage_nominal"),
            operating_voltage_max=f("battery_operating_voltage_max"),
            unit_mode=vars_["battery_unit_mode"].get(),
            series_units=int(float(vars_["battery_series_units"].get())),
            parallel_units=int(float(vars_["battery_parallel_units"].get())),
            cells_series_per_unit=int(float(vars_["battery_cells_series_per_unit"].get())),
            cells_parallel_per_unit=int(float(vars_["battery_cells_parallel_per_unit"].get())),
            pack_weight_g=f("battery_pack_weight") if vars_["battery_pack_weight"].get().strip() else None,
            cell_weight_g=f("battery_cell_weight") if vars_["battery_cell_weight"].get().strip() else None,
            cell_capacity_mAh=f("battery_cell_capacity") if vars_["battery_cell_capacity"].get().strip() else None,
            pack_capacity_mAh=f("battery_pack_capacity") if vars_["battery_pack_capacity"].get().strip() else None,
            discharge_c_cont=f("battery_discharge_c_cont") if vars_["battery_discharge_c_cont"].get().strip() else None,
            discharge_c_max=f("battery_discharge_c_max") if vars_["battery_discharge_c_max"].get().strip() else None,
            discharge_percent=f("battery_discharge_percent"),
            resistance_cell_mOhm=f("battery_resistance_cell"),
        )
        motor = MotorConfig(
            kv=f("motor_kv") if vars_["motor_kv"].get().strip() else None,
            idle_current=f("motor_idle_current"),
            idle_voltage=f("motor_idle_voltage"),
            rated_voltage=None,
            resistance=f("motor_resistance"),
            max_current=f("motor_max_current"),
            max_power=f("motor_max_power"),
            weight_g=f("motor_weight_g") if vars_["motor_weight_g"].get().strip() else None,
        )
        esc = ESCConfig(
            voltage_rating=None,
            continuous_current_A=f("esc_cont_current"),
            max_current_A=f("esc_max_current"),
            idle_current_A=f("esc_idle_current"),
            resistance=f("esc_resistance"),
            weight_g=f("esc_weight_g") if vars_["esc_weight_g"].get().strip() else None,
        )
        prop = PropellerConfig(
            diameter_in=f("prop_diameter"),
            pitch_in=f("prop_pitch"),
            blades=int(float(vars_["prop_blades"].get())),
            table_csv=(vars_["prop_table"].get().strip() or None),
            TConst=f("prop_tconst") if vars_["prop_tconst"].get().strip() else None,
            PConst=f("prop_pconst") if vars_["prop_pconst"].get().strip() else None,
            gear_ratio=f("gear_ratio"),
            weight_g=f("prop_weight_g") if vars_["prop_weight_g"].get().strip() else None,
        )
        avionics = AvionicsConfig(parse_voltage_tree(vars_["avionics_voltage_tree"].get().strip()))
        cfg = FixedWingConfig(
            airframe=airframe,
            battery=battery,
            motor=motor,
            propeller=prop,
            motor_count=int(float(vars_["motor_count"].get())),
            esc=esc,
            avionics=avionics,
            altitude_m=f("altitude_m"),
            temperature_C=f("temperature_C") if vars_["temperature_C"].get().strip() else None,
            pressure_Pa=(f("pressure_hpa") * 100.0) if vars_["pressure_hpa"].get().strip() else None,
        )
        return cfg

    def write_summary(summary: dict):
        summary_text.delete("1.0", "end")
        sel = summary["selected_speed"]
        batt = summary["battery"]
        mopt = summary["motor_optimum"]
        mmax = summary["motor_max"]
        prop = summary["propeller"]
        drive = summary["total_drive"]
        air = summary["airplane"]

        sections = [
            ("Selected Operating Point", sel),
            ("Battery", batt),
            ("Motor @ Optimum Efficiency", mopt),
            ("Motor @ Maximum", mmax),
            ("Propeller", prop),
            ("Total Drive", drive),
            ("Airplane", air),
        ]
        for title, data in sections:
            summary_text.insert("end", f"=== {title} ===\n")
            for k, v in data.items():
                summary_text.insert("end", f"{k}: {v}\n")
            summary_text.insert("end", "\n")

    def render_plots(summary: dict, cfg: FixedWingConfig):
        clear_plots()
        figs = [
            make_power_diagram_figure(summary["curve"]),
            make_climb_figure(summary["curve"]),
            make_motor_characteristic_figure(summary["curve"]),
            make_vertical_performance_figure(cfg),
        ]
        current_figs[:] = figs

        grid = ttk.Frame(plot_host)
        grid.grid(row=0, column=0, sticky="nsew")
        for r in range(2):
            grid.rowconfigure(r, weight=1)
        for c in range(2):
            grid.columnconfigure(c, weight=1)

        for idx, fig in enumerate(figs):
            canvas = FigureCanvasTkAgg(fig, master=grid)
            canvas.draw()
            canvas.get_tk_widget().grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=4, pady=4)
            current_canvases.append(canvas)

    def run_sim():
        try:
            cfg = build_config_from_gui()
            summary = summarize_performance(cfg, float(vars_["speed_kmh"].get()))
            write_summary(summary)
            render_plots(summary, cfg)
            notebook.select(tab_summary)
        except Exception as e:
            messagebox.showerror("Simulation error", str(e))

    def save_json():
        try:
            p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if not p:
                return
            payload = {k: v.get() for k, v in vars_.items()}
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def load_json():
        try:
            p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            if not p:
                return
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for k, v in payload.items():
                if k in vars_:
                    vars_[k].set(str(v))
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    btn_bar = ttk.Frame(tab_inputs, padding=8)
    btn_bar.pack(fill="x")
    ttk.Button(btn_bar, text="Run simulation", command=run_sim).pack(side="left", padx=4)
    ttk.Button(btn_bar, text="Load config", command=load_json).pack(side="left", padx=4)
    ttk.Button(btn_bar, text="Save config", command=save_json).pack(side="left", padx=4)
    ttk.Button(btn_bar, text="Quit", command=root.destroy).pack(side="right", padx=4)

    root.mainloop()


# --------------------------------------------------------------------------------------
# CLI argument parser
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed-wing performance simulator (CLI + GUI)")
    p.add_argument("--gui", action="store_true", help="Launch Tkinter GUI")

    # Airframe
    p.add_argument("--weight_g", type=float)
    p.add_argument("--wingspan_mm", type=float)
    p.add_argument("--wing_area_dm2", type=float)
    p.add_argument("--cd0", type=float, default=0.03)
    p.add_argument("--oswald", type=float, default=0.80)
    p.add_argument("--cl_max", type=float, default=1.3)
    p.add_argument("--drag_mode", type=str, default="advanced", choices=["advanced", "simplified"])
    p.add_argument("--custom_cd", type=float, default=None)
    p.add_argument("--motor_count", type=int, default=1)
    p.add_argument("--altitude_m", type=float, default=0.0)
    p.add_argument("--temperature_C", type=float, default=None)
    p.add_argument("--pressure_hpa", type=float, default=None)

    # Battery
    p.add_argument("--battery_unit_mode", type=str, default="cell", choices=["cell", "pack"])
    p.add_argument("--battery_series_units", type=int, default=1)
    p.add_argument("--battery_parallel_units", type=int, default=1)
    p.add_argument("--battery_cells_series_per_unit", type=int, default=1)
    p.add_argument("--battery_cells_parallel_per_unit", type=int, default=1)
    p.add_argument("--battery_cell_capacity", type=float, default=None)
    p.add_argument("--battery_pack_capacity", type=float, default=None)
    p.add_argument("--battery_cell_weight", type=float, default=None)
    p.add_argument("--battery_pack_weight", type=float, default=None)
    p.add_argument("--battery_operating_voltage_min", type=float, default=3.2)
    p.add_argument("--battery_operating_voltage_nominal", type=float, default=3.7)
    p.add_argument("--battery_operating_voltage_max", type=float, default=4.2)
    p.add_argument("--battery_resistance_cell", type=float, default=0.0)
    p.add_argument("--battery_discharge_cont_A", type=float, default=None)
    p.add_argument("--battery_discharge_max_A", type=float, default=None)
    p.add_argument("--battery_discharge_c_cont", type=float, default=None)
    p.add_argument("--battery_discharge_c_max", type=float, default=None)
    p.add_argument("--battery_discharge_percent", type=float, default=85.0)

    # Motor / ESC / prop
    p.add_argument("--motor_kv", type=float, default=None)
    p.add_argument("--motor_idle_current", type=float, default=0.8)
    p.add_argument("--motor_idle_voltage", type=float, default=10.0)
    p.add_argument("--motor_resistance", type=float, default=0.05)
    p.add_argument("--motor_max_current", type=float, default=50.0)
    p.add_argument("--motor_max_power", type=float, default=800.0)
    p.add_argument("--motor_weight_g", type=float, default=None)
    p.add_argument("--esc_cont_current", type=float, default=50.0)
    p.add_argument("--esc_max_current", type=float, default=60.0)
    p.add_argument("--esc_idle_current", type=float, default=0.02)
    p.add_argument("--esc_resistance", type=float, default=0.003)
    p.add_argument("--esc_weight_g", type=float, default=None)
    p.add_argument("--prop_diameter", type=float, default=10.0)
    p.add_argument("--prop_pitch", type=float, default=4.7)
    p.add_argument("--prop_blades", type=int, default=2)
    p.add_argument("--prop_tconst", type=float, default=None)
    p.add_argument("--prop_pconst", type=float, default=None)
    p.add_argument("--gear_ratio", type=float, default=1.0)
    p.add_argument("--prop_weight_g", type=float, default=None)
    p.add_argument("--prop_table", type=str, default=None)

    # Runtime
    p.add_argument("--avionics_voltage_tree", type=str, default="")
    p.add_argument("--speed_kmh", type=float, default=60.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--export_csv", type=str, default=None, help="Optional path to export the speed sweep CSV")
    return p


def validate_required_cli_args(args):
    if args.gui:
        return
    required = ["weight_g", "wingspan_mm", "wing_area_dm2"]
    missing = [k for k in required if getattr(args, k) in (None, "")]
    if missing:
        raise SystemExit("Missing required CLI args: " + ", ".join("--" + x for x in missing))


# --------------------------------------------------------------------------------------
# Build config from CLI args
# --------------------------------------------------------------------------------------
def build_cfg_from_args(args) -> FixedWingConfig:
    airframe = AirframeConfig(
        weight_g=args.weight_g,
        wingspan_m=args.wingspan_mm / 1000.0,
        wing_area_m2=args.wing_area_dm2 / 100.0,
        cd0=args.cd0,
        oswald=args.oswald,
        cl_max=args.cl_max,
        drag_mode=args.drag_mode,
        custom_cd=args.custom_cd,
    )
    battery = BatteryConfig(
        chemistry=None,
        operating_voltage_min=args.battery_operating_voltage_min,
        operating_voltage_nominal=args.battery_operating_voltage_nominal,
        operating_voltage_max=args.battery_operating_voltage_max,
        unit_mode=args.battery_unit_mode,
        series_units=args.battery_series_units,
        parallel_units=args.battery_parallel_units,
        cells_series_per_unit=args.battery_cells_series_per_unit,
        cells_parallel_per_unit=args.battery_cells_parallel_per_unit,
        pack_weight_g=args.battery_pack_weight,
        cell_weight_g=args.battery_cell_weight,
        cell_capacity_mAh=args.battery_cell_capacity,
        pack_capacity_mAh=args.battery_pack_capacity,
        discharge_cont_A=args.battery_discharge_cont_A,
        discharge_max_A=args.battery_discharge_max_A,
        discharge_c_cont=args.battery_discharge_c_cont,
        discharge_c_max=args.battery_discharge_c_max,
        discharge_percent=args.battery_discharge_percent,
        resistance_cell_mOhm=args.battery_resistance_cell,
    )
    motor = MotorConfig(
        kv=args.motor_kv,
        idle_current=args.motor_idle_current,
        idle_voltage=args.motor_idle_voltage,
        rated_voltage=None,
        resistance=args.motor_resistance,
        max_current=args.motor_max_current,
        max_power=args.motor_max_power,
        weight_g=args.motor_weight_g,
    )
    esc = ESCConfig(
        voltage_rating=None,
        continuous_current_A=args.esc_cont_current,
        max_current_A=args.esc_max_current,
        idle_current_A=args.esc_idle_current,
        resistance=args.esc_resistance,
        weight_g=args.esc_weight_g,
    )
    prop = PropellerConfig(
        diameter_in=args.prop_diameter,
        pitch_in=args.prop_pitch,
        blades=args.prop_blades,
        table_csv=args.prop_table,
        TConst=args.prop_tconst,
        PConst=args.prop_pconst,
        gear_ratio=args.gear_ratio,
        weight_g=args.prop_weight_g,
    )
    avionics = AvionicsConfig(parse_voltage_tree(args.avionics_voltage_tree))
    return FixedWingConfig(
        airframe=airframe,
        battery=battery,
        motor=motor,
        propeller=prop,
        motor_count=args.motor_count,
        esc=esc,
        avionics=avionics,
        altitude_m=args.altitude_m,
        temperature_C=args.temperature_C,
        pressure_Pa=(args.pressure_hpa * 100.0 if args.pressure_hpa is not None else None),
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    validate_required_cli_args(args)
    cfg = build_cfg_from_args(args)
    summary = summarize_performance(cfg, args.speed_kmh)
    print_summary(summary)

    if args.export_csv:
        summary["curve"].to_csv(args.export_csv, index=False)
        print(f"\nExported speed sweep to: {args.export_csv}")

    if args.plot:
        plot_all_figures(summary, cfg)


if __name__ == "__main__":
    main()
