"""
multicopter-power-sim.py
------------------
Multicopter performance simulator with three modeling modes:
1. Motor test table (CSV input with thrust vs power).
2. Motor electrical model (using KV, idle current, resistance, limits).
3. Theoretical induced velocity model (simplified).

Features:
- Power consumption (hover + forward flight)
- Flight time & distance
- Best endurance & best range speeds
- Plotting of performance curves
- Mission profile simulation (JSON)
- NEW: Optional Tkinter GUI for entering inputs and viewing plots in one window.
      CLI interface is preserved.

Examples (CLI):
    python multicopter-power-sim.py --num_motors 4 --weight 1.5 --area 0.05 \
        --battery_operating_voltage_min 3.0 --battery_operating_voltage_max 4.2 \
        --battery_capacity 5000 --battery_weight 400 --battery_energy_density 200 \
        --battery_charge_current_max 5 --battery_discharge_cont 60 --battery_resistance_cell 20 \
        --battery_cell_count 4 --battery_chemistry LiIon \
        --motor_kv 650 --motor_idle_current 0.5 --motor_resistance 0.2 --motor_max_current 20 --motor_max_power 200 \
        --prop_diameter 12 --prop_pitch 6 \
        --speed 10 --plot

    # Use motor/prop test table for power interpolation:
    python multicopter-power-sim.py ... --prop_table motor_data.csv --plot

    # Run GUI:
    python multicopter-power-sim.py --gui
"""

from __future__ import annotations

from html import parser
import math
import argparse
import json
import os
import re
from dataclasses import dataclass
import sys
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Constants
# -------------------------------
AIR_DENSITY = 1.225  # kg/m^3 (sea level ISA)

# Gas constant / ISA params
R = 287.05   # J/kg/K (specific gas constant for dry air)
T0 = 288.15  # K (sea level standard temp, 15°C)
P0 = 101325  # Pa (sea level standard pressure)
L = 0.0065   # K/m (temperature lapse rate)
g0 = 9.80665 # m/s^2


# -------------------------------
# Battery Model
# -------------------------------
class BatteryConfig:
    """
    Simple pack model:
      - nominal voltage taken as Vmax_cell * Ncells
      - internal resistance modeled as Rcell * Ncells (series)
      - under-load voltage V = Vnom - I*Rpack (clamped at Vmin_cell*Ncells)
    """
    def __init__(self,
                 chemistry: Optional[str],
                 operating_voltage_min: float,  # V per cell
                 operating_voltage_nominal: float,  # V per cell
                 operating_voltage_max: float,  # V per cell
                 unit_mode: str = "cell",  # "cell" or "pack" (for voltage inputs)
                 series_units: int = 1,  # number of units (cells or packs) in series
                 parallel_units: int = 1,  # number of parallel units (cells or packs)
                 cells_series_per_unit: int = 1,  # for "pack" mode: how many cells in series per pack
                 cells_parallel_per_unit: int = 1,  # for "pack" mode: how many cells in parallel per pack
                 pack_weight_g: float = None,
                 cell_weight_g: float = None,
                 cell_capacity_mAh: float = None,
                 pack_capacity_mAh: float = None,
                 unit_energy_density: float = None,
                 max_operating_temperature_C: Optional[float] = None,
                 min_operating_temperature_C: Optional[float] = None,
                 charge_current_max: float = None,
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
        self.unit_mode = str(unit_mode).strip().lower() if unit_mode is not None else "cell"
        self.series_units = int(series_units)
        self.parallel_units = int(parallel_units)

        if self.unit_mode not in ("cell", "pack"):
            # Fallback if the GUI or args provide a non-standard string.
            self.unit_mode = "pack" if pack_weight_g is not None else "cell"

        if self.unit_mode == "cell":
            cells_series_per_unit = 1
            cells_parallel_per_unit = 1

        self.cells_series_per_unit = int(cells_series_per_unit)
        self.cells_parallel_per_unit = int(cells_parallel_per_unit)

        # Final pack layout in CELLS
        self.series_cells = self.series_units * self.cells_series_per_unit
        self.parallel_cells = self.parallel_units * self.cells_parallel_per_unit
        self.total_cells = self.series_cells * self.parallel_cells

        # Derived pack voltage
        self.vmin_pack = self.operating_voltage_min * self.series_cells
        self.vmax_pack = self.operating_voltage_max * self.series_cells
        self.vnom_pack = self.operating_voltage_nominal * self.series_cells

        self.pack_weight_g = float(pack_weight_g) if pack_weight_g is not None else None
        self.cell_weight_g = float(cell_weight_g) if cell_weight_g is not None else None
        self.cell_capacity_mAh = float(cell_capacity_mAh) if cell_capacity_mAh is not None else None
        self.pack_capacity_mAh = float(pack_capacity_mAh) if pack_capacity_mAh is not None else None
        self.max_operating_temperature_C = float(max_operating_temperature_C) if max_operating_temperature_C is not None else None
        self.min_operating_temperature_C = float(min_operating_temperature_C) if min_operating_temperature_C is not None else None
        self.charge_current_max = float(charge_current_max) if charge_current_max is not None else 0.0

        # Capacity in Ah — must be computed before energy_density_Wh_per_kg
        if unit_mode == "cell":
            self.capacity_mAh = (self.cell_capacity_mAh or 0.0) * self.parallel_cells
        elif unit_mode == "pack":
            self.capacity_mAh = (self.pack_capacity_mAh or 0.0) * self.parallel_units * self.series_units
        else:
            self.capacity_mAh = 0.0
        self.capacity_Ah = self.capacity_mAh / 1000.0

        if unit_mode == "cell":
            self.weight_g = (self.cell_weight_g or 0.0) * self.total_cells
        elif unit_mode == "pack":
            self.weight_g = (self.pack_weight_g or 0.0) * self.parallel_units * self.series_units
        else:
            self.weight_g = 0.0

        # Energy density — must follow capacity_Ah and weight_g
        if unit_energy_density is not None:
            self.energy_density_Wh_per_kg = float(unit_energy_density)
        else:
            wkg = self.weight_g / 1000.0
            self.energy_density_Wh_per_kg = (self.capacity_Wh / wkg) if wkg > 0 else 0.0

        # Discharge limits:
        # - If you provide discharge_cont_A (legacy), we use it directly (amps).
        # - Otherwise, if you provide discharge_c_cont, we compute I_cont = C_cont * capacity_Ah.
        # - If you provide discharge_c_max, we compute I_max = C_max * capacity_Ah (burst limit).
        if discharge_cont_A is not None:
            self.discharge_cont_A = float(discharge_cont_A)
        elif discharge_c_cont is not None:
            self.discharge_cont_A = float(discharge_c_cont) * self.capacity_Ah
        else:
            self.discharge_cont_A = float("inf")  # No limit specified (not recommended)

        if discharge_max_A is not None:
            self.discharge_max_A = float(discharge_max_A)
        elif discharge_c_max is not None:
            self.discharge_max_A = float(discharge_c_max) * self.capacity_Ah
        else:
            # Default: max equals continuous if not specified
            self.discharge_max_A = float(self.discharge_cont_A)

        # Store C-rates (if derivable)
        if discharge_c_cont is not None:
            self.discharge_c_cont = float(discharge_c_cont)
        else:
            self.discharge_c_cont = (self.discharge_cont_A / self.capacity_Ah) if self.capacity_Ah > 0 else None

        if discharge_c_max is not None:
            self.discharge_c_max = float(discharge_c_max)
        else:
            self.discharge_c_max = (self.discharge_max_A / self.capacity_Ah) if self.capacity_Ah > 0 else None

        # Usable fraction of the pack energy (e.g., 80% -> stop at 20% remaining)
        self.discharge_percent = float(discharge_percent)
        self.discharge_percent = min(max(self.discharge_percent, 0.0), 100.0)
        self.usable_fraction = self.discharge_percent / 100.0

        self.resistance_cell = float(resistance_cell_mOhm) / 1000.0  # Ω

    @property
    def pack_resistance(self) -> float:
        return self.resistance_cell * self.series_cells / self.parallel_cells

    @property
    def capacity_Wh(self) -> float:
        return self.capacity_Ah * self.vnom_pack

    @property
    def usable_Wh(self) -> float:
        return self.capacity_Wh * self.usable_fraction


def battery_voltage_under_load(battery: BatteryConfig, current_A: float) -> float:
    v = battery.vmax_pack - current_A * battery.pack_resistance
    return max(v, battery.vmin_pack)


# -------------------------------
# Motor Model
# -------------------------------
class MotorConfig:
    def __init__(self,
                 kv: Optional[float],
                 idle_current: float,
                 idle_voltage: float,
                 rated_voltage: int, #S rating for ESC compatibility checks
                 resistance: float,
                 max_current: float,
                 max_power: float,
                 pole_count: Optional[int] = None,
                 weight_g: Optional[float] = None,
                 size_mm: Optional[str] = None):
        self.kv = None if kv is None else float(kv)   # RPM/V
        self.idle_current = float(idle_current)       # A
        self.idle_voltage = float(idle_voltage)
        self.resistance = float(resistance)           # Ω
        self.max_current = float(max_current)         # A
        self.max_power = float(max_power)             # W
        self.rated_voltage = int(rated_voltage)
        self.pole_count = pole_count
        self.weight_g = weight_g
        self.size_mm = size_mm

# -------------------------------
# ESC Model
# -------------------------------
class ESCConfig:
    def __init__(self,
                 voltage_rating: int,  # S rating for ESC compatibility checks
                 continuous_current_A: float,
                 max_current_A: float,
                 idle_current_A: float,
                 resistance: float,
                 weight_g: Optional[float] = None):
        self.voltage_rating = int(voltage_rating)
        self.continuous_rating_A = float(continuous_current_A)
        self.max_current_A = float(max_current_A)
        self.idle_current_A = float(idle_current_A)
        self.weight_g = weight_g
        self.resistance = float(resistance)           # Ω

# -------------------------------
# Avionics/Peripherals Model
# -------------------------------
class AvionicsConfig:
    def __init__(self,
                voltage_tree: Optional[dict] = None):
        self.voltage_tree = voltage_tree or {}  # e.g., {5.0: (2, 0.9), 12.0: (1.5, 0.85)} translates to 5V with 2A load, 90% BEC efficiency, and 12V with 1.5A load, 85% BEC efficiency}


def parse_voltage_tree(spec: Optional[str]) -> dict:
    """Parse an avionics voltage tree specification into {V_rail: (I_rail, eff)}.

    Accepted formats (comma-separated rails, commas inside parentheses are ignored):
      - "5.0:(2,0.9), 12.0:(1.5,0.85)"
      - "5.0:2:0.9, 12.0:1.5:0.85"

    Returns:
      dict[float, tuple[float, float]]
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        out = {}
        for k, v in spec.items():
            try:
                vk = float(k)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    out[vk] = (float(v[0]), float(v[1]))
                else:
                    raise ValueError
            except Exception as e:
                raise ValueError(f"Invalid voltage_tree entry {k!r}: {v!r}") from e
        return out

    s = str(spec).strip()
    if not s:
        return {}

    # Split on commas that are NOT inside parentheses
    parts, depth, buf = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1; buf += ch
        elif ch == ")":
            depth -= 1; buf += ch
        elif ch == "," and depth == 0:
            parts.append(buf.strip()); buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())

    out: dict = {}
    for p in parts:
        if not p:
            continue
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)\s*$", p)
        if m:
            v, i, eff = map(float, m.groups())
        else:
            m2 = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*([0-9]*\.?[0-9]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$", p)
            if not m2:
                raise ValueError(
                    "Invalid --avionics_voltage_tree format. "
                    "Use e.g. \"5.0:(2,0.9), 12.0:(1.5,0.85)\" or \"5.0:2:0.9, 12.0:1.5:0.85\"."
                )
            v, i, eff = map(float, m2.groups())

        if v <= 0:
            raise ValueError(f"Avionics rail voltage must be > 0, got {v}.")
        if i < 0:
            raise ValueError(f"Avionics rail current must be >= 0, got {i}.")
        if eff <= 0 or eff > 1.0:
            raise ValueError(f"BEC efficiency must be in (0, 1], got {eff}.")
        out[float(v)] = (float(i), float(eff))

    return out
def avionics_input_power_W(avionics: Optional[AvionicsConfig]) -> float:
    """Return total input power drawn from the battery to supply avionics rails (W).

    For each rail:
      P_in = (V_rail * I_rail) / efficiency
    """
    if avionics is None or not getattr(avionics, "voltage_tree", None):
        return 0.0
    total = 0.0
    for v, (i, eff) in avionics.voltage_tree.items():
        total += (float(v) * float(i)) / max(float(eff), 1e-9)
    return float(total)


def esc_loss_and_checks(config: "DroneConfig", v_pack: float, motor_power_total_W: float) -> tuple[float, str, float]:
    """Compute ESC electrical loss and simple current-limit checks.

    Returns:
      (esc_loss_W, status_note, motor_current_per_esc_A)

    Model:
      - Conduction loss: I^2 * R for each ESC
      - Idle/overhead draw: I_idle * V for each ESC (if provided)
    Assumption:
      Motor/ESC sees pack voltage (no separate motor rail).
    """
    esc = getattr(config, "esc", None)
    if esc is None:
        return 0.0, "", 0.0

    v = max(float(v_pack), 1e-9)
    p_per_motor = float(motor_power_total_W) / max(int(config.num_motors), 1)
    i_motor = p_per_motor / v  # A per ESC (approx)

    # Losses per ESC
    p_loss_cond = (i_motor ** 2) * max(float(esc.resistance), 0.0)
    p_loss_idle = max(float(esc.idle_current_A), 0.0) * v
    esc_loss_total = (p_loss_cond + p_loss_idle) * int(config.num_motors)

    note_parts = []
    if i_motor > float(esc.max_current_A):
        note_parts.append(f"ESC OVER MAX: {i_motor:.1f}A > {esc.max_current_A:.1f}A")
    elif i_motor > float(esc.continuous_rating_A):
        note_parts.append(f"ESC over continuous: {i_motor:.1f}A > {esc.continuous_rating_A:.1f}A")

    return float(esc_loss_total), ("; ".join(note_parts) if note_parts else ""), float(i_motor)


def total_power_with_esc(config: "DroneConfig",
                         motor_power_W: float,
                         periph_power_W: float,
                         iters: int = 6) -> tuple[float, float, float, str, float]:
    """Iteratively solve pack voltage/current while accounting for ESC loss.

    Returns:
      (total_power_W, v_load_V, pack_current_A, esc_note, motor_current_per_esc_A)
    """
    total_power = float(motor_power_W) + float(periph_power_W)
    v_load = float(config.battery.vnom_pack)
    pack_current = total_power / max(v_load, 1e-9)
    esc_note = ""
    i_motor = 0.0

    # Fixed-point iteration (ESC loss depends on v_load)
    for _ in range(max(int(iters), 1)):
        v_load, pack_current = solve_pack_voltage_and_current(config.battery, total_power)
        esc_loss_W, esc_note, i_motor = esc_loss_and_checks(config, v_load, motor_power_W)
        total_power = float(motor_power_W) + float(periph_power_W) + float(esc_loss_W)

    return float(total_power), float(v_load), float(pack_current), esc_note, float(i_motor)


def solve_pack_voltage_and_current(battery: "BatteryConfig", total_power_W: float, iters: int = 12) -> tuple[float, float]:
    """Solve V_load and I_pack for a load that draws a (roughly) constant electrical power.

    We iterate:
      I = P / V
      V = Vmax - I*Rpack (clamped at Vmin)

    Returns (V_load, I_pack).
    """
    if total_power_W <= 0:
        return (battery.vmax_pack, 0.0)

    v = float(battery.vnom_pack)
    i = total_power_W / max(v, 1e-9)
    for _ in range(max(1, int(iters))):
        v = battery_voltage_under_load(battery, i)
        i = total_power_W / max(v, 1e-9)
    return (float(v), float(i))


# -------------------------------
# Propeller Model
# -------------------------------
def load_motor_prop_table_csv(path: str) -> pd.DataFrame:
    """Load a motor/prop test table CSV.

    Supports two formats:
    1) Simple header with columns like: Thrust_g, Power_W, Current_A, RPM, Voltage_V, Throttle_pct, Efficiency_gW, Temp_C
    2) eCalc-style exported CSV where the *real* header is on a later row (e.g. first row says 'Test Data') and
       columns are named like 'Thrust (g)', 'Power (W)', 'Current (A)', 'RPM', 'Voltage (V)', 'Throttle', etc.

    Returns a cleaned DataFrame sorted by Thrust_g with standardized column names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try normal header first
    try:
        df0 = pd.read_csv(path)
    except Exception:
        df0 = pd.DataFrame()

    def _looks_good(df: pd.DataFrame) -> bool:
        cols = {str(c).strip().lower() for c in df.columns}
        return ("thrust_g" in cols) or ("thrust (g)" in cols) or ("thrust" in cols and "(g)" in " ".join(cols))

    if df0 is not None and len(df0.columns) > 1 and _looks_good(df0):
        df = df0
    else:
        # Read without header, find the header row
        raw = pd.read_csv(path, header=None)
        header_row = None
        for i in range(min(len(raw), 25)):
            row = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
            if any("throttle" == x or x.startswith("throttle") for x in row) and any("thrust" in x for x in row):
                header_row = i
                break
            # Some exports put "Voltage (V)" and "Thrust (g)" on same line
            if any("voltage" in x for x in row) and any("thrust" in x for x in row) and any("power" in x for x in row):
                header_row = i
                break
        if header_row is None:
            raise ValueError(f"Could not locate header row in CSV: {path}")

        df = pd.read_csv(path, header=header_row)

    # Standardize column names
    rename_map = {}
    for c in df.columns:
        c0 = str(c).strip()
        cl = c0.lower()

        if cl in ("thrust_g", "thrust (g)"):
            rename_map[c] = "Thrust_g"
        elif cl in ("power_w", "power (w)"):
            rename_map[c] = "Power_W"
        elif cl in ("current_a", "current (a)"):
            rename_map[c] = "Current_A"
        elif cl in ("voltage_v", "voltage (v)"):
            rename_map[c] = "Voltage_V"
        elif cl in ("rpm",):
            rename_map[c] = "RPM"
        elif cl in ("efficiency (g/w)", "efficiency_gw", "efficiency (g/w) "):
            rename_map[c] = "Efficiency_gW"
        elif "operating temperature" in cl or "temperature" in cl:
            rename_map[c] = "Temp_C"
        elif cl.startswith("throttle"):
            rename_map[c] = "Throttle_pct"
        elif cl == "propeller":
            rename_map[c] = "Propeller"
        elif cl == "type":
            rename_map[c] = "Type"
        elif "torque" in cl:
            rename_map[c] = "Torque_Nm"

    df = df.rename(columns=rename_map)

    # Clean / coerce types
    def _to_float_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")

    for col in ("Thrust_g", "Power_W", "Current_A", "Voltage_V", "RPM", "Efficiency_gW", "Temp_C", "Throttle_pct", "Torque_Nm"):
        if col in df.columns:
            df[col] = _to_float_series(df[col])

    # Drop rows without thrust/power
    if "Thrust_g" not in df.columns or "Power_W" not in df.columns:
        raise ValueError("Motor/prop CSV must contain thrust and power columns (e.g., 'Thrust (g)' and 'Power (W)')")

    df = df.dropna(subset=["Thrust_g", "Power_W"]).copy()

    # Sort & de-duplicate by thrust (keep last)
    df = df.sort_values("Thrust_g")
    df = df.drop_duplicates(subset=["Thrust_g"], keep="last").reset_index(drop=True)

    return df

class PropellerConfig:
    def __init__(self,
                 diameter_in: float,
                 pitch_in: float,
                 max_rpm: int,
                 max_thrust_g: float,
                 blades: int = 2,
                 table_csv: Optional[str] = None,
                 TConst: Optional[float] = None,
                 PConst: Optional[float] = None,
                 weight_g: Optional[float] = None):
        self.diameter_in = float(diameter_in)
        self.pitch_in = float(pitch_in)
        self.blades = int(blades)
        self.max_rpm = int(max_rpm)
        self.max_thrust_g = float(max_thrust_g)
        self.table_csv = table_csv
        self.TConst = TConst
        self.PConst = PConst
        self.weight_g = weight_g

        self.table: Optional[pd.DataFrame] = None
        if table_csv:
            self.table = load_motor_prop_table_csv(table_csv)


# -------------------------------
# Drone Config
# -------------------------------
class DroneConfig:
    def __init__(self,
                 num_motors: int,
                 battery: BatteryConfig,
                 motor: MotorConfig,
                 propeller: PropellerConfig,
                 drone_weight_g: float,
                 profile_drag_coefficient: float,
                 profile_area: float,
                 parasite_drag_coefficient: float,
                 parasite_area: float,
                 frontal_area: float,
                 cruise_speed: float,
                 periph_current: float,
                 esc: Optional[ESCConfig] = None,
                 avionics: Optional[AvionicsConfig] = None,
                 air_density: float = AIR_DENSITY,
                 # --- Vehicle geometry / mechanical params (new) ---
                 body_length_m: Optional[float] = None,
                 body_width_m: Optional[float] = None,
                 body_height_m: Optional[float] = None,
                 arm_length_m: Optional[float] = None,
                 arm_width_m: Optional[float] = None,
                 coaxial_spacing_m: Optional[float] = None,
                 max_tilt_deg: Optional[float] = None,
                 motor_configuration: str = "flat"  # "flat" or "coaxial"
                 ):
        self.num_motors = int(num_motors)
        self.battery = battery
        self.motor = motor
        self.propeller = propeller
        self.drone_weight_g = float(drone_weight_g)

        # Drag parameters (may be overridden / derived from geometry if not provided)
        self.profile_drag_coefficient = float(profile_drag_coefficient) if profile_drag_coefficient is not None else 0.0
        self.profile_area = float(profile_area) if profile_area is not None else 0.0
        self.parasite_drag_coefficient = float(parasite_drag_coefficient) if parasite_drag_coefficient is not None else 0.0
        self.parasite_area = float(parasite_area) if parasite_area is not None else 0.0
        self.frontal_area = float(frontal_area) if frontal_area is not None else 0.0

        self.cruise_speed = float(cruise_speed)
        self.periph_current = float(periph_current)
        self.esc = esc
        self.avionics = avionics
        self.air_density = float(air_density)

        # Geometry / limits
        self.body_length_m = float(body_length_m) if body_length_m not in (None, "") else None
        self.body_width_m = float(body_width_m) if body_width_m not in (None, "") else None
        self.body_height_m = float(body_height_m) if body_height_m not in (None, "") else None
        self.arm_length_m = float(arm_length_m) if arm_length_m not in (None, "") else None
        self.arm_width_m = float(arm_width_m) if arm_width_m not in (None, "") else None
        self.coaxial_spacing_m = float(coaxial_spacing_m) if coaxial_spacing_m not in (None, "") else None
        self.max_tilt_deg = float(max_tilt_deg) if max_tilt_deg not in (None, "") else None
        self.motor_configuration = (motor_configuration or "flat").strip().lower()
        if self.motor_configuration not in ("flat", "coaxial"):
            self.motor_configuration = "flat"

        # Internal flag so we only derive drag once (unless user edits values)
        self._derived_drag_from_geometry = False

    def derive_drag_from_geometry_if_missing(self) -> None:
        """
        If the user did not provide drag parameters (or they are zero/negative),
        approximate drag using a simple box body + square-tube arm model.

        This is intentionally simple and meant as a fallback, not a substitute
        for measured CdA.
        """
        # Consider drag "not provided" if all are <= 0
        provided = any(x > 0 for x in (
            self.profile_drag_coefficient, self.profile_area,
            self.parasite_drag_coefficient, self.parasite_area,
            self.frontal_area
        ))
        if provided or self._derived_drag_from_geometry:
            return

        # Need enough geometry to do anything useful
        if self.body_width_m is None or self.body_height_m is None:
            return
        if self.arm_length_m is None:
            return

        # --- Assumptions (fallback constants) ---
        # Rectangular box Cd normal to flow ~ 1.0–1.2; use 1.05
        CD_BOX = 1.05
        # Square tube / cylinder-ish boom Cd ~ 1.0–1.3; use 1.1
        CD_ARM = 1.10
        # Arm tube width (square tube outer width). User-configurable; fallback to 20 mm.
        ARM_TUBE_SIDE_M = float(self.arm_width_m) if self.arm_width_m not in (None,) else 0.02

        # Determine number of arms (coaxial usually has fewer arms for the same motor count)
        if self.motor_configuration == "coaxial":
            num_arms = max(1, self.num_motors // 2)
        else:
            num_arms = self.num_motors

        # Body frontal area for forward flight (assume length is forward axis)
        A_body = self.body_width_m * self.body_height_m

        # Arms projected area (very rough). Each arm contributes ~ (tube_side * arm_length)
        # with a projection factor to account for non-perfect alignment (X layout etc).
        PROJ = 0.7
        A_arms = num_arms * (ARM_TUBE_SIDE_M * self.arm_length_m) * PROJ

        # Parasite drag: body + arms
        A_total = A_body + A_arms
        CdA_total = CD_BOX * A_body + CD_ARM * A_arms
        if A_total > 0:
            self.parasite_area = A_total
            self.parasite_drag_coefficient = CdA_total / A_total
        else:
            self.parasite_area = 0.0
            self.parasite_drag_coefficient = 0.0

        # Profile drag term in this simplified model represents rotor/arm "profile".
        # We approximate it with arms only so hover-side drag isn't zero.
        self.profile_area = A_arms
        self.profile_drag_coefficient = CD_ARM if A_arms > 0 else 0.0

        # Frontal area stored separately if you want to use it elsewhere
        self.frontal_area = A_body

        self._derived_drag_from_geometry = True


# -------------------------------
# Mission Profile
# -------------------------------
@dataclass
class MissionPhase:
    name: str
    speed: float
    duration: Optional[float] = None  # seconds
    distance: Optional[float] = None  # meters
    altitude: float = 0.0             # meters


class MissionProfile:
    def __init__(self, phases: List[MissionPhase]):
        self.phases = phases

    @staticmethod
    def from_json(path: str) -> "MissionProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        phases = []
        for p in data.get("phases", []):
            phases.append(MissionPhase(
                name=p["name"],
                speed=float(p["speed"]),
                duration=p.get("duration"),
                distance=p.get("distance"),
                altitude=float(p.get("altitude", 0.0)),
            ))
        return MissionProfile(phases)


# -------------------------------
# Physics Helpers
# -------------------------------

def drag_force_required(config: DroneConfig, speed_mps: float, orientation: str) -> float:
    """Compute aerodynamic drag force (N) for the given airspeed and orientation."""
    # If drag parameters were not provided, attempt to derive them from geometry.
    config.derive_drag_from_geometry_if_missing()

    if orientation == "hover":
        return 0.5 * config.air_density * speed_mps**2 * config.profile_area * config.profile_drag_coefficient

    parasite = 0.5 * config.air_density * speed_mps**2 * config.parasite_area * config.parasite_drag_coefficient
    profile = 0.5 * config.air_density * speed_mps**2 * config.profile_area * config.profile_drag_coefficient
    return parasite + profile


def required_tilt_deg(config: DroneConfig, speed_mps: float, orientation: str) -> float:
    """Approximate tilt angle (deg) required in forward flight to balance drag."""
    if orientation != "forward":
        return 0.0
    weight_force = config.drone_weight_g * 9.81 / 1000.0  # Convert grams to kg
    drag_force = drag_force_required(config, speed_mps, orientation="forward")
    return math.degrees(math.atan2(drag_force, max(weight_force, 1e-9)))

def disk_area(diameter_in: float) -> float:
    d_m = diameter_in * 0.0254
    return math.pi * (d_m / 2.0) ** 2


def compute_air_density(altitude_m: float, temperature_C: Optional[float] = None, pressure_Pa: Optional[float] = None) -> float:
    """
    altitude_m: meters
    temperature_C: °C; if None, ISA lapse rate is used
    pressure_Pa: Pa; if None, barometric formula is used (troposphere)
    """
    if temperature_C is None:
        temperature_C = (T0 - L * altitude_m) - 273.15

    T = temperature_C + 273.15

    if pressure_Pa is None:
        P = P0 * (1.0 - (L * altitude_m) / T0) ** (g0 / (R * L))
    else:
        P = pressure_Pa

    return P / (R * T)


def thrust_required(config: DroneConfig, speed_mps: float, orientation: str) -> float:
    """
    Simplified force model.

    Hover:
      Required thrust ≈ weight + (small profile drag term if you model it)

    Forward flight:
      Drag is primarily horizontal and weight is vertical, so required thrust magnitude is:
          T = sqrt(weight^2 + drag^2)
      and required tilt is:
          tilt = atan(drag / weight)

    This function returns total thrust magnitude (N).
    """
    weight_force = config.drone_weight_g * 9.81 / 1000.0  # Convert grams to kg

    drag_force = drag_force_required(config, speed_mps, orientation)

    if orientation == "hover":
        return weight_force + drag_force

    # forward flight (vector sum)
    return math.sqrt(weight_force**2 + drag_force**2)


# -------------------------------
# Power Models
# -------------------------------
def _fit_propeller_curve(thrust_g_data: np.ndarray, y_data: np.ndarray, degree: int = 2) -> tuple:
    """Fit a polynomial to propeller data for extrapolation.
    
    Args:
        thrust_g_data: Thrust values in grams (x-axis)
        y_data: Property values (Power_W, Current_A, etc.)
        degree: Polynomial degree (1 or 2 recommended)
    
    Returns:
        (coeffs, min_thrust, max_thrust) where coeffs are polyfit coefficients
    """
    # Filter out NaN values
    valid_idx = ~(np.isnan(thrust_g_data) | np.isnan(y_data))
    if valid_idx.sum() < degree + 1:
        # Not enough valid points, return None
        return None
    
    x_valid = thrust_g_data[valid_idx]
    y_valid = y_data[valid_idx]
    
    try:
        # Fit polynomial (highest degree first)
        coeffs = np.polyfit(x_valid, y_valid, degree)
        return (coeffs, x_valid.min(), x_valid.max())
    except:
        return None


def _eval_poly(coeffs: np.ndarray, x: float) -> float:
    """Evaluate polynomial with coefficients from np.polyfit at point x."""
    if coeffs is None:
        return None
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (x ** (len(coeffs) - 1 - i))
    return result


def _extrapolate_motor_value(df: pd.DataFrame, thrust_g: float, column: str) -> Optional[float]:
    """Extrapolate a motor property value using fitted curve if thrust is below minimum.
    
    Args:
        df: Propeller table dataframe
        thrust_g: Target thrust in grams
        column: Column name to extrapolate (Power_W, Current_A, etc.)
    
    Returns:
        Extrapolated value or None if not available
    """
    if column not in df.columns:
        return None
    
    thrust_g_data = df["Thrust_g"].values
    try:
        y_data = pd.to_numeric(df[column], errors='coerce').values
    except:
        return None
    
    min_thrust = thrust_g_data.min()
    
    # If we're within data range, don't extrapolate
    if thrust_g >= min_thrust:
        return None
    
    # Fit a 2nd degree polynomial to the data
    fit_result = _fit_propeller_curve(thrust_g_data, y_data, degree=2)
    if fit_result is None:
        return None
    
    coeffs, _, _ = fit_result
    extrapolated = _eval_poly(coeffs, thrust_g)
    
    # For some properties, clamp to minimum reasonable values
    if column == "Current_A" and extrapolated is not None:
        extrapolated = max(extrapolated, 0.0)
    elif column == "Power_W" and extrapolated is not None:
        extrapolated = max(extrapolated, 0.0)
    elif column == "Efficiency_gW" and extrapolated is not None:
        extrapolated = max(extrapolated, 0.0)
    elif column == "RPM" and extrapolated is not None:
        extrapolated = max(extrapolated, 0.0)
    
    return extrapolated


def interpolate_motor_point(config: DroneConfig, thrust_per_motor_N: float) -> dict:
    """Interpolate a full operating point from a motor/prop test table.

    Returns a dict with (when available):
      Power_W, RPM, Current_A, Voltage_V, Throttle_pct, Efficiency_gW, Temp_C, Torque_Nm

    The lookup key is thrust (grams-force). Interpolation is linear between nearest points.
    """
    if config.propeller.table is None:
        raise ValueError("No prop_table loaded, cannot interpolate.")

    thrust_g = thrust_per_motor_N * 1000.0 / 9.81
    df = config.propeller.table

    if "Thrust_g" not in df.columns or "Power_W" not in df.columns:
        raise ValueError("prop_table CSV must have columns: Thrust_g, Power_W (after parsing)")

    min_thrust = float(df["Thrust_g"].min())
    max_thrust = float(df["Thrust_g"].max())
    
    # Handle below minimum: extrapolate using curve fit
    if thrust_g < min_thrust:
        out = {}
        for col in ("Power_W","RPM","Current_A","Voltage_V","Throttle_pct","Efficiency_gW","Temp_C","Torque_Nm"):
            # Try to extrapolate
            extrapolated = _extrapolate_motor_value(df, thrust_g, col)
            if extrapolated is not None:
                out[col] = extrapolated
            elif col in df.columns:
                # Fall back to first row value
                row = df.iloc[0]
                if pd.notna(row[col]):
                    out[col] = float(row[col])
        return out

    if thrust_g >= max_thrust:
        row = df.iloc[-1]
        return {k: float(row[k]) for k in df.columns if k in (
            "Thrust_g","Power_W","RPM","Current_A","Voltage_V","Throttle_pct","Efficiency_gW","Temp_C","Torque_Nm"
        ) and pd.notna(row[k])}

    lower = df[df["Thrust_g"] <= thrust_g].iloc[-1]
    upper = df[df["Thrust_g"] >= thrust_g].iloc[0]

    denom = float(upper["Thrust_g"] - lower["Thrust_g"])
    frac = 0.0 if denom == 0 else float((thrust_g - lower["Thrust_g"]) / denom)

    out = {}
    for col in ("Power_W","RPM","Current_A","Voltage_V","Throttle_pct","Efficiency_gW","Temp_C","Torque_Nm"):
        if col in df.columns and pd.notna(lower[col]) and pd.notna(upper[col]):
            out[col] = float(lower[col] + (upper[col] - lower[col]) * frac)
        elif col in df.columns and pd.notna(lower[col]) and pd.isna(upper[col]):
            out[col] = float(lower[col])
        elif col in df.columns and pd.isna(lower[col]) and pd.notna(upper[col]):
            out[col] = float(upper[col])
    return out


def interpolate_motor_power(config: DroneConfig, thrust_per_motor_N: float) -> float:
    """Interpolate electrical input power from a motor/prop test table."""
    pt = interpolate_motor_point(config, thrust_per_motor_N)
    if "Power_W" not in pt:
        raise ValueError("prop_table interpolation failed to produce Power_W")
    return float(pt["Power_W"])

def motor_power_from_params(config: DroneConfig, thrust_per_motor_N: float) -> float:
    """
    Estimate electrical input power required for a given thrust per motor.

    Two options:
      - If propeller TConst/PConst provided: solve for RPM via thrust model:
            T = C_T * rho * n^2 * D^4
        Then mechanical shaft power:
            P_mech = C_P * rho * n^3 * D^5
      - Else: fallback to momentum theory induced power:
            vi = sqrt(T / (2*rho*A))
            P_mech ≈ T * vi

    Electrical conversion (very simplified):
      - torque constant: Kt = 60 / (2π Kv)  [Nm/A]
      - approximate motor torque from mech power and omega
      - current ≈ torque/Kt + I0
      - copper loss via motor resistance
      - clamp by motor max current/power
    """
    if config.motor.kv is None:
        raise ValueError("motor_kv must be set to use motor electrical model.")

    D = config.propeller.diameter_in * 0.0254  # meters
    rho = config.air_density

    # ---- Mechanical shaft power ----
    if config.propeller.TConst and config.propeller.PConst:
        low, high = 100.0, 40000.0  # RPM bounds
        rpm_solution = None

        for _ in range(40):
            mid = 0.5 * (low + high)
            n = mid / 60.0  # rev/s
            thrust = config.propeller.TConst * rho * (n**2) * (D**4)
            if thrust < thrust_per_motor_N:
                low = mid
            else:
                high = mid
                rpm_solution = mid

        if rpm_solution is None:
            return 0.0

        n = rpm_solution / 60.0
        mech_power_W = config.propeller.PConst * rho * (n**3) * (D**5)
        omega = 2.0 * math.pi * n
        torque_Nm = mech_power_W / max(omega, 1e-9)
    else:
        A = disk_area(config.propeller.diameter_in)
        vi = math.sqrt(thrust_per_motor_N / (2.0 * rho * A))
        mech_power_W = thrust_per_motor_N * vi
        # crude omega/torque estimate from Kv and voltage
        omega = (config.battery.vnom_pack * config.motor.kv) * (2.0 * math.pi / 60.0)
        torque_Nm = mech_power_W / max(omega, 1e-9)

    # ---- Electrical model ----
    kt = 60.0 / (2.0 * math.pi * config.motor.kv)  # Nm/A
    current_A = torque_Nm / max(kt, 1e-9) + config.motor.idle_current

    # motor terminal voltage ~ pack voltage - I*R (motor copper)
    v_drop = current_A * config.motor.resistance
    v_eff = max(config.battery.vnom_pack - v_drop, 0.0)
    input_power_W = v_eff * current_A

    # Enforce current/power limits
    if config.motor.max_current and current_A > config.motor.max_current:
        current_A = config.motor.max_current
        input_power_W = config.battery.vnom_pack * current_A
    if config.motor.max_power and input_power_W > config.motor.max_power:
        input_power_W = config.motor.max_power

    return float(input_power_W)



def motor_configuration_power_multiplier(config: DroneConfig, orientation: str) -> float:
    """Return a multiplier applied to per-motor electrical power based on motor configuration.

    Coaxial (stacked) rotors suffer aerodynamic interference that depends strongly on the
    vertical spacing between rotors (relative to rotor diameter). This models that effect
    as a smooth penalty multiplier on *per-motor* electrical power for a given required thrust.

    - For motor_configuration == "flat": returns 1.0.
    - For motor_configuration == "coaxial": returns > 1.0 depending on spacing.

    The intent is to provide a reasonable default without requiring detailed coaxial aero data.
    """
    cfg = getattr(config, "motor_configuration", "flat")
    if cfg != "coaxial":
        return 1.0

    # Rotor diameter (m) from prop config (inches -> meters)
    try:
        diameter_m = float(config.propeller.diameter_in) * 0.0254
    except Exception:
        diameter_m = 0.0

    # If we can't compute a ratio, fall back to a conservative default.
    if diameter_m <= 0.0:
        return 1.18 if orientation == "hover" else 1.12

    # Use user spacing if provided; otherwise assume a typical ~0.2D spacing.
    spacing_m = getattr(config, "coaxial_spacing_m", None)
    if spacing_m is None or float(spacing_m) <= 0.0:
        spacing_m = 0.20 * diameter_m

    spacing_ratio = max(0.0, float(spacing_m) / diameter_m)

    # Penalty model:
    # - Higher penalty at small spacing, diminishing as spacing increases.
    # - Hover interference tends to be worse than in forward flight.
    # Tuning targets: ~1.18 at ~0.2D in hover; smaller in forward flight.
    inc = 0.25 * math.exp(-3.0 * spacing_ratio) + 0.03  # 3% floor + spacing-dependent term
    if orientation != "hover":
        inc *= 0.70  # reduced penalty in forward flight

    return 1.0 + inc

def power_required(config: DroneConfig, speed_mps: float, orientation: str) -> float:
    """
    Total electrical power for all motors (W), not including peripheral current.
    """
    total_thrust_N = thrust_required(config, speed_mps, orientation)
    thrust_per_motor_N = total_thrust_N / config.num_motors

    if config.propeller.table is not None:
        motor_power_W = interpolate_motor_power(config, thrust_per_motor_N)
    elif config.motor.kv is not None:
        motor_power_W = motor_power_from_params(config, thrust_per_motor_N)
    else:
        A = disk_area(config.propeller.diameter_in)
        vi = math.sqrt(thrust_per_motor_N / (2.0 * config.air_density * A))
        motor_power_W = (thrust_per_motor_N * vi) / 0.85

    # Apply motor-configuration penalty (e.g., coaxial interference)
    motor_power_W *= motor_configuration_power_multiplier(config, orientation)

    return motor_power_W * config.num_motors


# -------------------------------
# Flight Performance
# -------------------------------
def estimate_flight_time_minutes(config: DroneConfig, speed_mps: float, orientation: str = "forward") -> float:
    """
    Returns minutes of flight time based on:
      usable_energy_Wh / total_power_W

    Total power includes motor power plus avionics/peripheral draw.

    Avionics draw can be provided either as:
      - Legacy: config.periph_current (A at pack input), OR
      - Voltage rails: config.avionics.voltage_tree = {Vrail: (Irail, eff), ...}

    When voltage rails are used, we treat the avionics *output* loads as constant,
    so the BEC input power is:
        P_avionics_in = sum(Vrail * Irail / eff)
    and the pack current increases as pack voltage sags.
    """
    # Enforce max tilt in forward flight (if provided)
    if orientation == "forward" and getattr(config, "max_tilt_deg", None) is not None:
        tilt_req = required_tilt_deg(config, speed_mps, orientation="forward")
        if tilt_req > float(config.max_tilt_deg) + 1e-9:
            return 0.0
    motor_power_W = power_required(config, speed_mps, orientation)
    periph_power_W = avionics_input_power_W(getattr(config, "avionics", None))
    if periph_power_W <= 0.0:
        # Legacy behavior: constant current draw at the pack input
        periph_power_W = config.battery.vnom_pack * max(config.periph_current, 0.0)

    total_power_W, v_load, pack_current_A, esc_note, motor_I_esc_A = total_power_with_esc(
        config,
        motor_power_W=motor_power_W,
        periph_power_W=periph_power_W,
    )

    if total_power_W <= 0:
        return 0.0

    # Discharge limit checks (battery)
    if pack_current_A > config.battery.discharge_max_A:
        return 0.0
    
    if v_load < config.battery.vmin_pack:
        return 0.0
    
    # ESC limit checks
    if getattr(config, "esc", None) is not None and motor_I_esc_A > config.esc.max_current_A:
        return 0.0

    time_h = config.battery.usable_Wh / total_power_W
    return float(time_h * 60.0)


def estimate_flight_distance_km(config: DroneConfig, speed_mps: float, orientation: str = "forward") -> float:
    t_min = estimate_flight_time_minutes(config, speed_mps, orientation)
    return float((speed_mps * (t_min * 60.0)) / 1000.0)


def find_optimal_speeds(config: DroneConfig, min_speed: float = 1.0, max_speed: float = 30.0, step: float = 0.5):
    """
    Returns:
      - best_endurance_speed (max minutes)
      - best_range_speed (max km)
    """
    best_endurance_speed, best_minutes = 0.0, -1.0
    best_range_speed, best_km = 0.0, -1.0

    n_steps = int((max_speed - min_speed) / step) + 1
    for i in range(n_steps):
        v = min_speed + i * step
        t = estimate_flight_time_minutes(config, v, orientation="forward")
        d = estimate_flight_distance_km(config, v, orientation="forward")
        if t > best_minutes:
            best_minutes = t
            best_endurance_speed = v
        if d > best_km:
            best_km = d
            best_range_speed = v

    return best_endurance_speed, best_minutes, best_range_speed, best_km


def simulate_mission(config: DroneConfig,
                     mission: MissionProfile,
                     orientation: str = "forward",
                     temperature_C: Optional[float] = None,
                     pressure_Pa: Optional[float] = None,
                     wind_mps: float = 0.0) -> Tuple[List[Tuple[str, float, float, str]], Optional[dict], Optional[dict]]:
    """
    Simulate mission phases, draining remaining energy (Wh).

    Wind sign convention:
      +wind_mps = headwind (airspeed > groundspeed)
      -wind_mps = tailwind

    Each result tuple:
      (phase_name, time_minutes, distance_km, status_string)

    Returns:
      (results, worst_metrics) where worst_metrics aggregates worst-case values across phases
      for limit/status checks.
    """
    remaining_wh = config.battery.usable_Wh
    results: List[Tuple[str, float, float, str]] = []
    worst_metrics: Optional[dict] = None
    mission_series: dict = {
        't_s': [],
        'phase': [],
        'airspeed_mps': [],
        'groundspeed_mps': [],
        'distance_km': [],
        'altitude_m': [],
        'tilt_deg': [],
        'battery_voltage_V': [],
        'battery_current_A': [],
        'battery_energy_Wh': [],
        'battery_capacity_mAh': [],
        'total_power_W': [],
        'motor_power_W': [],
        'motor_power_per_motor_W': [],
        'motor_current_A': [],
        'motor_rpm': [],
        'motor_thrust_N': [],
        'thrust_total_N': [],
        'periph_power_W': [],
        'esc_loss_W': [],
    }

    t_s = 0.0
    dist_km = 0.0

    def _append_point(phase_name: str, phase_alt_m: float, m: dict, t_s_now: float, dist_km_now: float, remaining_wh_now: float):
        # Note: many metrics are phase-steady; distance/energy change across the phase.
        mission_series['t_s'].append(float(t_s_now))
        mission_series['phase'].append(str(phase_name))
        mission_series['airspeed_mps'].append(float(m.get('airspeed_mps', 0.0)))
        mission_series['groundspeed_mps'].append(float(phase.speed))
        mission_series['distance_km'].append(float(dist_km_now))
        mission_series['altitude_m'].append(float(phase_alt_m))
        mission_series['tilt_deg'].append(float(m.get('tilt_required_deg', 0.0)))
        mission_series['battery_voltage_V'].append(float(m.get('v_load_V', 0.0)))
        mission_series['battery_current_A'].append(float(m.get('pack_current_A', 0.0)))
        mission_series['battery_energy_Wh'].append(float(remaining_wh_now))
        vnom = float(config.battery.vnom_pack) if float(config.battery.vnom_pack) > 1e-9 else 1.0
        mission_series['battery_capacity_mAh'].append(float(remaining_wh_now) * 1000.0 / vnom)
        mission_series['total_power_W'].append(float(m.get('total_power_W', 0.0)))
        mp = float(m.get('motor_power_W', 0.0))
        mission_series['motor_power_W'].append(mp)
        mission_series['motor_power_per_motor_W'].append(mp / max(int(config.num_motors), 1))
        mission_series['motor_current_A'].append(float(m.get('motor_I_per_esc_A', 0.0)))
        mission_series['motor_rpm'].append(float(m.get('prop_rpm')) if m.get('prop_rpm') is not None else float('nan'))
        mission_series['motor_thrust_N'].append(float(m.get('thrust_per_motor_N', 0.0)))
        mission_series['thrust_total_N'].append(float(m.get('thrust_total_N', 0.0)))
        mission_series['periph_power_W'].append(float(m.get('periph_power_W', 0.0)))
        mission_series['esc_loss_W'].append(float(m.get('esc_loss_W', 0.0)))


    def _merge_worst(worst: Optional[dict], m: dict) -> dict:
        if worst is None:
            return dict(m)
        # Max-style metrics
        for k in ("pack_current_A", "total_power_W", "motor_power_W", "periph_power_W",
                  "esc_loss_W", "motor_I_per_esc_A", "thrust_total_N", "thrust_per_motor_N"):
            worst[k] = max(float(worst.get(k, 0.0)), float(m.get(k, 0.0)))
        # Min-style metrics
        worst["v_load_V"] = min(float(worst.get("v_load_V", 1e9)), float(m.get("v_load_V", 1e9)))
        # RPM if available
        if worst.get("prop_rpm") is None:
            worst["prop_rpm"] = m.get("prop_rpm")
        elif m.get("prop_rpm") is not None:
            worst["prop_rpm"] = max(float(worst["prop_rpm"]), float(m["prop_rpm"]))
        # Keep notes (best-effort)
        if str(m.get("esc_note", "")).strip():
            worst["esc_note"] = (str(worst.get("esc_note", "")) + "; " + str(m.get("esc_note", ""))).strip("; ")
        return worst

    for phase in mission.phases:
        rho = compute_air_density(altitude_m=phase.altitude, temperature_C=temperature_C, pressure_Pa=pressure_Pa)
        config.air_density = rho

        # Compute operating metrics for this phase, then fold into worst-case metrics.
        m = compute_operating_metrics(config, speed_mps=float(phase.speed), orientation=orientation, wind_mps=wind_mps)
        worst_metrics = _merge_worst(worst_metrics, m)

        # Append phase-start sample for mission plots
        _append_point(phase.name, float(phase.altitude), m, t_s, dist_km, remaining_wh)

        airspeed = float(m.get("airspeed_mps", 0.0))
        total_power_W = float(m.get("total_power_W", 0.0))
        pack_current_A = float(m.get("pack_current_A", 0.0))
        v_load = float(m.get("v_load_V", 0.0))
        motor_I_esc_A = float(m.get("motor_I_per_esc_A", 0.0))
        esc_note = str(m.get("esc_note", "")).strip()

        if pack_current_A > config.battery.discharge_max_A:
            results.append((phase.name, 0.0, 0.0, "Battery depleted (discharge limit exceeded)"))
            break

        if v_load < config.battery.vmin_pack:
            results.append((phase.name, 0.0, 0.0, "Battery depleted (voltage under load)"))
            break

        if getattr(config, "esc", None) is not None and motor_I_esc_A > config.esc.max_current_A:
            results.append((phase.name, 0.0, 0.0, f"ESC over max current: {motor_I_esc_A:.1f}A > {config.esc.max_current_A:.1f}A"))
            break

        status_ok = "OK"
        if esc_note:
            status_ok += f", {esc_note}"

        if phase.duration is not None:
            duration_s = float(phase.duration)
            energy_used_Wh = total_power_W * (duration_s / 3600.0)
            if energy_used_Wh > remaining_wh:
                actual_time_min = (remaining_wh / total_power_W) * 60.0
                actual_dist_km = airspeed * (actual_time_min * 60.0) / 1000.0
                # advance and append final point at depletion
                t_s += actual_time_min * 60.0
                dist_km += actual_dist_km
                remaining_wh = 0.0
                _append_point(phase.name, float(phase.altitude), m, t_s, dist_km, remaining_wh)
                # advance and append final point at depletion
                t_s += actual_time_min * 60.0
                dist_km += actual_dist_km
                remaining_wh = 0.0
                _append_point(phase.name, float(phase.altitude), m, t_s, dist_km, remaining_wh)
                results.append((phase.name, actual_time_min, actual_dist_km, "Battery depleted"))
                break
            remaining_wh -= energy_used_Wh
            t_s += duration_s
            dist_km += airspeed * duration_s / 1000.0
            _append_point(phase.name, float(phase.altitude), m, t_s, dist_km, remaining_wh)
            results.append((phase.name, duration_s / 60.0, airspeed * duration_s / 1000.0, status_ok))

        elif phase.distance is not None:
            dist_m = float(phase.distance)
            if airspeed <= 1e-9:
                results.append((phase.name, 0.0, 0.0, "Invalid: zero airspeed with distance phase"))
                break

            time_s = dist_m / airspeed
            energy_used_Wh = total_power_W * (time_s / 3600.0)

            if energy_used_Wh > remaining_wh:
                actual_time_min = (remaining_wh / total_power_W) * 60.0
                actual_dist_km = airspeed * (actual_time_min * 60.0) / 1000.0
                # advance and append final point at depletion
                t_s += actual_time_min * 60.0
                dist_km += actual_dist_km
                remaining_wh = 0.0
                _append_point(phase.name, float(phase.altitude), m, t_s, dist_km, remaining_wh)
                results.append((phase.name, actual_time_min, actual_dist_km, "Battery depleted"))
                break

            remaining_wh -= energy_used_Wh
            t_s += time_s
            dist_km += dist_m / 1000.0
            _append_point(phase.name, float(phase.altitude), m, t_s, dist_km, remaining_wh)
            results.append((phase.name, time_s / 60.0, dist_m / 1000.0, status_ok))
        else:
            results.append((phase.name, 0.0, 0.0, "Invalid: phase missing duration/distance"))
            break

    return results, worst_metrics, mission_series



# -------------------------------
# Plotting
# -------------------------------
def make_performance_figure(config: DroneConfig,
                             max_speed: float = 30.0,
                             figsize: tuple = (15, 9)):
    """
    Four-panel multicopter performance figure:
      1. Flight Time & Range vs Speed  (forward)
      2. Power Required vs Speed       (hover + forward)
      3. Thrust Required vs Speed      (hover + forward)
      4. Power Breakdown bar chart     (at cruise speed)
    """
    v_lo  = 0.5
    steps = 200
    speeds = [v_lo + (max_speed - v_lo) * i / steps for i in range(steps + 1)]

    times_fwd, dists_fwd = [], []
    pwr_fwd, pwr_hov     = [], []
    thr_fwd, thr_hov     = [], []
    cruise = float(config.cruise_speed)

    for V in speeds:
        times_fwd.append(estimate_flight_time_minutes(config, V, orientation="forward"))
        dists_fwd.append(estimate_flight_distance_km(config, V, orientation="forward"))
        pwr_fwd.append(power_required(config, V, "forward"))
        pwr_hov.append(power_required(config, V, "hover"))
        thr_fwd.append(thrust_required(config, V, "forward"))
        thr_hov.append(thrust_required(config, V, "hover"))

    be_v, be_t, br_v, br_d = find_optimal_speeds(config, min_speed=v_lo, max_speed=max_speed)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Multicopter Performance", fontsize=13, fontweight="bold")

    # 1. Time & Range
    ax = axes[0, 0]
    ax2 = ax.twinx()
    l1, = ax.plot(speeds, times_fwd, color="royalblue",  label="Flight Time (min)")
    l2, = ax2.plot(speeds, dists_fwd, color="darkorange", label="Range (km)", linestyle="--")
    ax.axvline(be_v,  color="royalblue",  linestyle=":", linewidth=1.2)
    ax.axvline(br_v,  color="darkorange", linestyle=":", linewidth=1.2)
    ax.axvline(cruise,color="gray",       linestyle="-.", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Speed (m/s)"); ax.set_ylabel("Time (min)"); ax2.set_ylabel("Range (km)")
    ax.set_title("Flight Time & Range vs Speed")
    ax.legend(handles=[l1, l2], loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.4)

    # 2. Power
    ax = axes[0, 1]
    ax.plot(speeds, [p/1000 for p in pwr_fwd], color="crimson",   label="Forward Flight")
    ax.plot(speeds, [p/1000 for p in pwr_hov], color="steelblue", label="Hover", linestyle="--")
    ax.axvline(cruise, color="gray", linestyle="-.", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Speed (m/s)"); ax.set_ylabel("Power (kW)")
    ax.set_title("Power Required vs Speed")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # 3. Thrust
    ax = axes[1, 0]
    ax.plot(speeds, thr_fwd, color="teal",      label="Forward Flight")
    ax.plot(speeds, thr_hov, color="slategray", label="Hover", linestyle="--")
    ax.axvline(cruise, color="gray", linestyle="-.", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Speed (m/s)"); ax.set_ylabel("Thrust (N)")
    ax.set_title("Thrust Required vs Speed")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # 4. Power breakdown at cruise
    ax = axes[1, 1]
    try:
        motor_P  = power_required(config, cruise, "forward")
        periph_P = avionics_input_power_W(getattr(config, "avionics", None))
        if periph_P <= 0.0:
            periph_P = config.battery.vnom_pack * max(config.periph_current, 0.0)
        total_P, _, _, _, _ = total_power_with_esc(
            config, motor_power_W=motor_P, periph_power_W=periph_P)
        esc_P = max(0.0, total_P - motor_P - periph_P)
        labels = ["Motor", "ESC Loss", "Avionics"]
        values = [motor_P, esc_P, periph_P]
        colors = ["#4c72b0", "#dd8452", "#55a868"]
        valid  = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0.1]
        if valid:
            ls, vs, cs = zip(*valid)
            bars = ax.bar(ls, [v/1000 for v in vs], color=cs, edgecolor="white", linewidth=0.8)
            ax.bar_label(bars, fmt=lambda x: f"{x*1000:.0f} W", padding=3, fontsize=8)
    except Exception:
        pass
    ax.set_ylabel("Power (kW)"); ax.set_title(f"Power Breakdown @ {cruise:.1f} m/s")
    ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_performance(config: DroneConfig, max_speed: float = 30.0):
    fig = make_performance_figure(config, max_speed=max_speed)
    plt.show()


def make_motor_operating_point_figure(config: DroneConfig, metrics: dict, figsize: tuple = (12, 8)):
    """
    Create a figure showing motor/propeller operating curves with the current operating point marked.
    Uses propeller table data if available.
    
    Two subplots:
      1. Thrust vs Power (left), Thrust vs Current (right) on same plot
      2. Thrust vs Efficiency g/W (left), Thrust vs RPM (right) on same plot
    """
    if config.propeller.table is None:
        # Return empty figure if no propeller table
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "Propeller table not available\nCannot plot operating curves",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return fig
    
    df = config.propeller.table
    thrust_pm_N = float(metrics.get("thrust_per_motor_N", 0.0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get data from propeller table
    thrust_g = df["Thrust_g"].values if "Thrust_g" in df.columns else []
    
    # Subplot 1: Thrust vs Power & Thrust vs Current
    ax1_1 = ax1
    ax1_2 = None
    if "Power_W" in df.columns and len(thrust_g) > 0:
        power_W = df["Power_W"].values
        ax1_1.plot(thrust_g, power_W, "b-", linewidth=2, label="Power")
        ax1_1.set_xlabel("Thrust per motor (gf)", fontsize=10)
        ax1_1.set_ylabel("Power (W)", fontsize=10, color="b")
        ax1_1.tick_params(axis="y", labelcolor="b")
        ax1_1.grid(True, alpha=0.3)
    
    if "Current_A" in df.columns and len(thrust_g) > 0:
        current_A = df["Current_A"].values
        ax1_2 = ax1.twinx()
        ax1_2.plot(thrust_g, current_A, "r--", linewidth=2, label="Current")
        ax1_2.set_ylabel("Current (A)", fontsize=10, color="r")
        ax1_2.tick_params(axis="y", labelcolor="r")
    
    # Mark operating point on subplot 1 (both y-axes)
    thrust_g_op = thrust_pm_N * 1000.0 / 9.81
    point = interpolate_motor_point(config, thrust_pm_N)
    
    if "Power_W" in df.columns and len(thrust_g) > 0:
        power_op = point.get("Power_W", 0.0)
        ax1_1.plot(thrust_g_op, power_op, "b*", markersize=15, label=f"Pow: {power_op:.1f}W", markeredgewidth=0.5, markeredgecolor="darkblue")
    
    if "Current_A" in df.columns and len(thrust_g) > 0 and ax1_2:
        current_op = point.get("Current_A", 0.0)
        ax1_2.plot(thrust_g_op, current_op, "r*", markersize=15, label=f"Cur: {current_op:.2f}A", markeredgewidth=0.5, markeredgecolor="darkred")
    
    ax1.set_title("Thrust vs Power & Current", fontsize=11, fontweight="bold")
    # Combine legends from both axes
    lines1, labels1 = ax1_1.get_legend_handles_labels()
    lines2, labels2 = (ax1_2.get_legend_handles_labels() if ax1_2 else ([], []))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    
    # Subplot 2: Thrust vs Efficiency & Thrust vs RPM
    ax2_1 = ax2
    ax2_2 = None
    if "Efficiency_gW" in df.columns and len(thrust_g) > 0:
        eff_gW = df["Efficiency_gW"].values
        ax2_1.plot(thrust_g, eff_gW, "g-", linewidth=2, label="Efficiency")
        ax2_1.set_xlabel("Thrust per motor (gf)", fontsize=10)
        ax2_1.set_ylabel("Efficiency (g/W)", fontsize=10, color="g")
        ax2_1.tick_params(axis="y", labelcolor="g")
        ax2_1.grid(True, alpha=0.3)
    
    if "RPM" in df.columns and len(thrust_g) > 0:
        rpm = df["RPM"].values
        ax2_2 = ax2.twinx()
        ax2_2.plot(thrust_g, rpm, "m--", linewidth=2, label="RPM")
        ax2_2.set_ylabel("RPM", fontsize=10, color="m")
        ax2_2.tick_params(axis="y", labelcolor="m")
    
    # Mark operating point on subplot 2 (both y-axes)
    if "Efficiency_gW" in df.columns and len(thrust_g) > 0:
        eff_op = point.get("Efficiency_gW", 0.0)
        ax2_1.plot(thrust_g_op, eff_op, "g*", markersize=15, label=f"Eff: {eff_op:.2f}g/W", markeredgewidth=0.5, markeredgecolor="darkgreen")
    
    if "RPM" in df.columns and len(thrust_g) > 0 and ax2_2:
        rpm_op = point.get("RPM", 0.0)
        ax2_2.plot(thrust_g_op, rpm_op, "m*", markersize=15, label=f"RPM: {rpm_op:.0f}", markeredgewidth=0.5, markeredgecolor="darkmagenta")
    
    ax2.set_title("Thrust vs Efficiency & RPM", fontsize=11, fontweight="bold")
    # Combine legends from both axes
    lines1, labels1 = ax2_1.get_legend_handles_labels()
    lines2, labels2 = (ax2_2.get_legend_handles_labels() if ax2_2 else ([], []))
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    
    fig.suptitle(f"Motor/Propeller Operating Curves (Thrust/Motor: {thrust_g_op:.0f}g)", 
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# -------------------------------
# Config builders
# -------------------------------
def build_drone_from_args(args) -> DroneConfig:
    battery = BatteryConfig(
        operating_voltage_min=args.battery_operating_voltage_min,
        operating_voltage_nominal=args.battery_operating_voltage_nominal,
        operating_voltage_max=args.battery_operating_voltage_max,
        unit_energy_density=args.battery_energy_density,
        chemistry=args.battery_chemistry,
        charge_current_max=args.battery_charge_current_max,
        discharge_cont_A=args.battery_discharge_cont_A,
        discharge_max_A=args.battery_discharge_max_A,
        discharge_c_cont=args.battery_discharge_c_cont,
        discharge_c_max=args.battery_discharge_c_max,
        discharge_percent=args.battery_discharge_percent,
        resistance_cell_mOhm=args.battery_resistance_cell,
        unit_mode=args.battery_unit_mode,
        series_units=args.battery_series_units,
        parallel_units=args.battery_parallel_units,
        cells_series_per_unit=args.battery_cells_series_per_unit,
        cells_parallel_per_unit=args.battery_cells_parallel_per_unit,
        pack_weight_g=args.battery_pack_weight,
        cell_weight_g=args.battery_cell_weight,
        cell_capacity_mAh=args.battery_cell_capacity,
        pack_capacity_mAh=args.battery_pack_capacity,
    )

    motor = MotorConfig(
        kv=args.motor_kv,
        idle_current=args.motor_idle_current,
        idle_voltage=args.motor_idle_voltage,
        rated_voltage=args.motor_rated_voltage,
        resistance=args.motor_resistance,
        max_current=args.motor_max_current,
        max_power=args.motor_max_power,
        pole_count=args.motor_pole_count,
        weight_g=args.motor_weight,
        size_mm=args.motor_size_mm,
    )

    esc = ESCConfig(
        voltage_rating=args.esc_voltage_rating,
        continuous_current_A=args.esc_continuous_current_A,
        max_current_A=args.esc_max_current_A,
        idle_current_A=args.esc_idle_current_A,
        resistance=args.esc_resistance,
        weight_g=args.esc_weight,
    )

    prop = PropellerConfig(
        diameter_in=args.prop_diameter,
        pitch_in=args.prop_pitch,
        max_rpm=args.prop_max_rpm,
        max_thrust_g=args.prop_max_thrust_g,
        blades=args.prop_blades,
        table_csv=args.prop_table,
        PConst=args.prop_pconst,
        TConst=args.prop_tconst,
        weight_g=args.prop_weight,
    )

    avionics = AvionicsConfig(
        voltage_tree=parse_voltage_tree(args.avionics_voltage_tree),
    )

    esc = None
    if any(x is not None for x in [args.esc_voltage_rating, args.esc_cont_current, args.esc_max_current, args.esc_idle_current, args.esc_resistance, args.esc_weight]):
        # Provide safe defaults if some fields omitted
        esc = ESCConfig(
            voltage_rating=int(args.esc_voltage_rating) if args.esc_voltage_rating is not None else int(args.series),
            continuous_current_A=float(args.esc_cont_current) if args.esc_cont_current is not None else 0.0,
            max_current_A=float(args.esc_max_current) if args.esc_max_current is not None else float(args.esc_cont_current or 0.0),
            idle_current_A=float(args.esc_idle_current) if args.esc_idle_current is not None else 0.0,
            resistance=float(args.esc_resistance) if args.esc_resistance is not None else 0.0,
            weight_g=float(args.esc_weight) if args.esc_weight is not None else None,
        )

    drone = DroneConfig(
        num_motors=args.num_motors,
        battery=battery,
        motor=motor,
        propeller=prop,
        drone_weight_g=args.weight,
        profile_drag_coefficient=args.profile_drag,
        profile_area=args.profile_area,
        parasite_drag_coefficient=args.parasite_drag,
        parasite_area=args.parasite_area,
        frontal_area=args.area,
        cruise_speed=args.speed,
        periph_current=args.periph_current,
        esc=esc,
        avionics=avionics,
        air_density=AIR_DENSITY,
        body_length_m=args.body_length_m,
        body_width_m=args.body_width_m,
        body_height_m=args.body_height_m,
        arm_length_m=args.arm_length_m,
        arm_width_m=args.arm_width_m,
        coaxial_spacing_m=args.coaxial_spacing_m,
        max_tilt_deg=args.max_tilt_deg,
        motor_configuration=args.motor_configuration,
    )

    # Initialize air density at user-specified conditions
    drone.air_density = compute_air_density(
        altitude_m=args.altitude,
        temperature_C=args.temperature,
        pressure_Pa=args.pressure,
    )
    return drone


# -------------------------------
# GUI
# -------------------------------

# -------------------------------
# GUI
# -------------------------------
# ============================================================
# SHARED REPORTING / EXPORT UTILITIES
# ============================================================
import csv as _csv
import io  as _io
import datetime as _dt
import tempfile as _tmpfile

def _fmt_g(x, nd=2):
    try:    return f"{float(x):.{nd}f}"
    except: return "n/a"

def _extract_weight_budget(cfg) -> list:
    rows = []
    total_g = float(getattr(cfg, "drone_weight_g",
                    getattr(cfg, "aircraft_weight_g", 0.0)))
    num_motors = int(getattr(cfg, "num_motors",
                    getattr(getattr(cfg, "airframe", None), "num_motors", 1)))
    batt  = getattr(cfg, "battery",   None)
    motor = getattr(cfg, "motor",     None)
    esc   = getattr(cfg, "esc",       None)
    prop  = getattr(cfg, "propeller", None)
    accounted = 0.0
    # Battery: weight_g already includes all series/parallel units
    if batt:
        w = float(getattr(batt, "weight_g", 0.0) or 0.0)
        rows.append(("Battery", w, 1, w)); accounted += w
    # Motor: per-motor weight, multiply by num_motors
    if motor:
        w = float(getattr(motor, "weight_g", 0.0) or 0.0)
        rows.append(("Motor", w, num_motors, w * num_motors)); accounted += w * num_motors
    # ESC: per-ESC weight, multiply by num_motors
    if esc:
        w = float(getattr(esc, "weight_g", 0.0) or 0.0)
        rows.append(("ESC", w, num_motors, w * num_motors)); accounted += w * num_motors
    # Propeller: per-propeller weight, multiply by num_motors
    if prop:
        w = float(getattr(prop, "weight_g", 0.0) or 0.0)
        rows.append(("Propeller", w, num_motors, w * num_motors)); accounted += w * num_motors
    airframe_g = max(0.0, total_g - accounted)
    rows.append(("Airframe / Structure", airframe_g, 1, airframe_g))
    rows.append(("TOTAL", total_g, 1, total_g))
    return rows

def _export_csv_file(path: str, sweep: dict, metrics: list) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["[Performance Sweep]"])
        if sweep:
            headers = list(sweep.keys())
            w.writerow(headers)
            n = max(len(v) for v in sweep.values())
            for i in range(n):
                w.writerow([sweep[h][i] if i < len(sweep[h]) else "" for h in headers])
        w.writerow([])
        w.writerow(["[Metrics]"])
        w.writerow(["Metric", "Value"])
        for label, value in metrics:
            w.writerow([label, value])

def _export_excel_file(path: str, sweep: dict, metrics: list, weight_budget: list) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook()
    ws = wb.active; ws.title = "Performance Sweep"
    if sweep:
        headers = list(sweep.keys())
        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=1, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1F3864")
        n = max(len(v) for v in sweep.values())
        for ri in range(n):
            for ci, h in enumerate(headers, 1):
                ws.cell(row=ri+2, column=ci, value=sweep[h][ri] if ri < len(sweep[h]) else None)
    ws2 = wb.create_sheet("Metrics")
    ws2.append(["Metric", "Value"])
    for r in [ws2["A1"], ws2["B1"]]: r.font = Font(bold=True)
    for label, value in metrics: ws2.append([label, value])
    ws3 = wb.create_sheet("Weight Budget")
    ws3.append(["Component", "Unit Weight (g)", "Count", "Total Weight (g)", "% of Total"])
    for c in ws3[1]: c.font = Font(bold=True)
    total_g = weight_budget[-1][3] if weight_budget else 1.0
    for label, uw, cnt, tw in weight_budget[:-1]:
        pct = round(tw/total_g*100, 1) if total_g > 0 else 0
        ws3.append([label, round(uw,1), cnt, round(tw,1), pct])
    if weight_budget:
        label, uw, cnt, tw = weight_budget[-1]
        ws3.append([label, "", "", round(tw,1), 100.0])
        for c in list(ws3.rows)[-1]: c.font = Font(bold=True)
    wb.save(path)

def _generate_pdf_report(path: str, report_title: str,
                          inputs_rows: list, metrics_rows: list,
                          status_sections: list, log_text: str,
                          figures: list, weight_budget: list) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, PageBreak, Image,
                                    HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units  import cm
    from reportlab.lib        import colors
    from reportlab.lib.enums  import TA_CENTER
    import io
    PAGE_W, PAGE_H = A4
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=2.0*cm,  bottomMargin=2.0*cm)
    styles = getSampleStyleSheet()
    story  = []
    NAVY  = colors.HexColor("#1F3864")
    TEAL  = colors.HexColor("#2E75B6")
    LGREY = colors.HexColor("#F2F2F2")
    DGREY = colors.HexColor("#595959")
    sTitle = ParagraphStyle("rTitle", fontSize=22, textColor=NAVY,
                             spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold")
    sSub   = ParagraphStyle("rSub", fontSize=11, textColor=DGREY,
                             spaceAfter=20, alignment=TA_CENTER, fontName="Helvetica")
    sH1    = ParagraphStyle("rH1", fontSize=13, textColor=NAVY,
                             spaceBefore=12, spaceAfter=4, fontName="Helvetica-Bold")
    def _ts(header_bg=TEAL):
        return TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), header_bg),
            ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGREY]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.lightgrey),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
            ("TOPPADDING",    (0,0),(-1,-1), 2),
            ("BOTTOMPADDING", (0,0),(-1,-1), 2),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ])
    usable_w = PAGE_W - 3.6*cm
    # Title page
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph(report_title, sTitle))
    story.append(Paragraph(
        _dt.datetime.now().strftime("Generated %d %B %Y  %H:%M"), sSub))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=12))
    if weight_budget:
        story.append(Paragraph("Component Weight Summary", sH1))
        total_g = weight_budget[-1][3] if weight_budget else 1.0
        wb_data = [["Component", "Unit (g)", "Qty", "Total (g)", "%"]]
        for label, uw, cnt, tw in weight_budget[:-1]:
            pct = f"{tw/total_g*100:.1f}" if total_g > 0 else "0"
            wb_data.append([label, f"{uw:.1f}", str(cnt), f"{tw:.1f}", pct])
        last = weight_budget[-1]
        wb_data.append(["TOTAL", "", "", f"{last[3]:.1f}", "100.0"])
        ts = _ts()
        ts.add("FONTNAME",(0,len(wb_data)-1),(-1,len(wb_data)-1),"Helvetica-Bold")
        t = Table(wb_data, colWidths=[usable_w*0.38, usable_w*0.15,
                                       usable_w*0.1, usable_w*0.17, usable_w*0.1])
        t.setStyle(ts); story.append(t)
    story.append(PageBreak())
    # Inputs
    if inputs_rows:
        story.append(Paragraph("Design Inputs", sH1))
        mid = (len(inputs_rows)+1)//2
        left = inputs_rows[:mid]; right = inputs_rows[mid:]
        while len(right) < len(left): right.append(("",""))
        rows_data = [["Parameter","Value","Parameter","Value"]]
        for (la,va),(lb,vb) in zip(left,right): rows_data.append([la,va,lb,vb])
        col_w = usable_w/4
        t = Table(rows_data, colWidths=[col_w*1.4,col_w*0.6]*2)
        t.setStyle(_ts()); story.append(t)
    story.append(PageBreak())
    # Metrics
    if metrics_rows:
        story.append(Paragraph("Performance Metrics", sH1))
        mid = (len(metrics_rows)+1)//2
        left = metrics_rows[:mid]; right = metrics_rows[mid:]
        while len(right) < len(left): right.append(("",""))
        rows_data = [["Metric","Value","Metric","Value"]]
        for (la,va),(lb,vb) in zip(left,right): rows_data.append([la,va,lb,vb])
        col_w = usable_w/4
        t = Table(rows_data, colWidths=[col_w*1.4,col_w*0.6]*2)
        t.setStyle(_ts()); story.append(t)
    story.append(PageBreak())
    # Status
    if status_sections:
        story.append(Paragraph("Status Checks", sH1))
        for sec_title, sec_rows in status_sections:
            story.append(Paragraph(sec_title, ParagraphStyle("secH", fontSize=10,
                textColor=TEAL, spaceBefore=6, spaceAfter=2, fontName="Helvetica-Bold")))
            tdata = [["Metric","Value","Limit","Notes"]]
            ts = _ts()
            for ri,(metric,val,lim,note,tag) in enumerate(sec_rows,1):
                tdata.append([metric,val,lim,note])
                bg = {"ok":colors.HexColor("#D9F2D9"),"warn":colors.HexColor("#FFF2CC"),
                      "bad":colors.HexColor("#F8D7DA")}.get(tag, colors.white)
                ts.add("BACKGROUND",(0,ri),(-1,ri),bg)
            cw = [usable_w*0.28,usable_w*0.20,usable_w*0.20,usable_w*0.32]
            t = Table(tdata, colWidths=cw); t.setStyle(ts); story.append(t)
            story.append(Spacer(1,4))
    story.append(PageBreak())
    # Plots
    if figures:
        story.append(Paragraph("Performance Plots", sH1))
        for fig in figures:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            img_w = usable_w
            img_h = img_w * (fig.get_figheight() / max(fig.get_figwidth(), 0.01))
            max_h = PAGE_H - 5*cm
            if img_h > max_h:
                img_h = max_h
                img_w = img_h * (fig.get_figwidth() / max(fig.get_figheight(), 0.01))
            story.append(Image(buf, width=img_w, height=img_h))
            story.append(Spacer(1, 8)); story.append(PageBreak())
    # Log
    if log_text.strip():
        story.append(Paragraph("Simulation Output Log", sH1))
        mono = ParagraphStyle("mono", fontName="Courier", fontSize=7.5, leading=10, spaceAfter=2)
        for line in log_text.splitlines():
            safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe or " ", mono))
    doc.build(story)

def launch_gui():
    """
    Tkinter GUI — styled to match the fixed-wing simulator:
      - Scrollable input tabs (Drone / Battery / Motor / ESC / Avionics / Propeller / Mission+Env)
      - Right panel: Plots | Status | Metrics | Mission Plots
      - View menu with Window Scale, Plot Size, UI Font Size, Plot Font Size, Quick Presets
      - Save / Load config (JSON)
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # ------------------------------------------------------------------ #
    #  HELPERS                                                            #
    # ------------------------------------------------------------------ #
    def parse_float(name, s):
        try:
            return float(s)
        except Exception:
            raise ValueError(f"Invalid {name}: {s!r}")

    def parse_int(name, s):
        try:
            return int(float(s))
        except Exception:
            raise ValueError(f"Invalid {name}: {s!r}")

    def safe_float(val, default=0.0):
        try:
            return float(str(val).strip())
        except Exception:
            return default

    def choose_file(var, filetypes):
        p = filedialog.askopenfilename(filetypes=filetypes)
        if p:
            var.set(p)

    def exit_app():
        try:
            plt.close("all")
        except Exception:
            pass
        root.quit()
        root.destroy()

    # ------------------------------------------------------------------ #
    #  ROOT WINDOW                                                        #
    # ------------------------------------------------------------------ #
    root = tk.Tk()
    root.title("Multicopter Power Simulator")
    root.minsize(1100, 700)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ------------------------------------------------------------------ #
    #  VIEW STATE  (shared by all scaling callbacks)                      #
    # ------------------------------------------------------------------ #
    try:
        _base_tk_scale = float(root.tk.call("tk", "scaling"))
    except Exception:
        _base_tk_scale = 1.333

    _view = {
        "scale_pct":    100,
        "plot_w":       15.0,
        "plot_h":        9.0,
        "mpl_fontsize":  9,
        "ui_fontsize":   9,
    }
    _last_run: dict = {}   # cached after each single-point run for View re-renders

    # ---- scaling helpers ----
    def _apply_tk_scale(pct: int) -> None:
        _view["scale_pct"] = pct
        factor = _base_tk_scale * (pct / 100.0)
        try:
            root.tk.call("tk", "scaling", factor)
        except Exception:
            pass
        root.minsize(int(1100 * pct / 100), int(700 * pct / 100))
        root.update_idletasks()

    def _apply_ui_font(size: int) -> None:
        _view["ui_fontsize"] = size
        sty = ttk.Style()
        font_spec = ("TkDefaultFont", size)
        for ws in ("TLabel", "TButton", "TEntry", "TCombobox",
                   "TNotebook.Tab", "Treeview", "Treeview.Heading",
                   "TLabelframe.Label", "TLabelframe"):
            try:
                sty.configure(ws, font=font_spec)
            except Exception:
                pass
        try:
            sty.configure("Treeview", rowheight=max(18, size + 8))
        except Exception:
            pass
        root.update_idletasks()

    def _apply_mpl_font(size: int) -> None:
        _view["mpl_fontsize"] = size
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.size":        size,
            "axes.titlesize":   size + 1,
            "axes.labelsize":   size,
            "xtick.labelsize":  size - 1,
            "ytick.labelsize":  size - 1,
            "legend.fontsize":  size - 1,
        })
        _rerender_if_possible()

    def _apply_plot_size(w: float, h: float) -> None:
        _view["plot_w"] = w
        _view["plot_h"] = h
        _rerender_if_possible()

    def _rerender_if_possible() -> None:
        if not _last_run:
            return
        try:
            drone   = _last_run["drone"]
            max_spd = _last_run["max_spd"]
            metrics = _last_run.get("metrics", {})
            fig = make_performance_figure(
                drone,
                max_speed=max_spd,
                figsize=(_view["plot_w"], _view["plot_h"]),
            )
            # Generate motor operating point figure if available
            motor_fig = None
            if drone.propeller.table is not None and metrics:
                try:
                    motor_fig = make_motor_operating_point_figure(drone, metrics, figsize=(_view["plot_w"], 6))
                except Exception:
                    pass
            # Display both figures
            if motor_fig:
                _show_figure([fig, motor_fig])
            else:
                _show_figure(fig)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  MENU BAR                                                           #
    # ------------------------------------------------------------------ #
    menubar   = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)

    # -- Window Scale --
    _scale_var = tk.IntVar(value=100)
    scale_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Window Scale", menu=scale_menu)
    for pct, lbl in [(75,  "75 %  – Compact"),
                     (90,  "90 %  – Smaller"),
                     (100, "100 % – Default"),
                     (115, "115 % – Slightly Larger"),
                     (125, "125 % – Large"),
                     (150, "150 % – Extra Large"),
                     (175, "175 % – Very Large"),
                     (200, "200 % – Max")]:
        scale_menu.add_radiobutton(label=lbl, variable=_scale_var, value=pct,
                                   command=lambda p=pct: _apply_tk_scale(p))
    view_menu.add_separator()

    # -- Plot Size --
    _plot_size_var = tk.StringVar(value="medium")
    plot_size_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Plot Size", menu=plot_size_menu)
    for key, lbl, (w, h) in [
        ("small",   "Small  (12 × 7)",          (12.0,  7.0)),
        ("medium",  "Medium (15 × 9)  ← default",(15.0,  9.0)),
        ("large",   "Large  (18 × 11)",          (18.0, 11.0)),
        ("xlarge",  "X-Large (22 × 13)",         (22.0, 13.0)),
        ("xxlarge", "XX-Large (26 × 15)",        (26.0, 15.0)),
    ]:
        plot_size_menu.add_radiobutton(label=lbl, variable=_plot_size_var, value=key,
                                       command=lambda pw=w, ph=h: _apply_plot_size(pw, ph))
    view_menu.add_separator()

    # -- UI Font Size --
    _ui_font_var = tk.IntVar(value=9)
    ui_font_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="UI Font Size", menu=ui_font_menu)
    for sz, lbl in [(8, "8 pt  – Tiny"), (9, "9 pt  – Default"),
                    (10,"10 pt – Comfortable"), (11,"11 pt – Large"),
                    (13,"13 pt – Extra Large"), (15,"15 pt – Accessibility")]:
        ui_font_menu.add_radiobutton(label=lbl, variable=_ui_font_var, value=sz,
                                     command=lambda s=sz: _apply_ui_font(s))
    view_menu.add_separator()

    # -- Plot Font Size --
    _mpl_font_var = tk.IntVar(value=9)
    mpl_font_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Plot Font Size", menu=mpl_font_menu)
    for sz, lbl in [(7, "7 pt  – Tiny"), (8, "8 pt  – Small"),
                    (9, "9 pt  – Default"), (10,"10 pt – Medium"),
                    (12,"12 pt – Large"), (14,"14 pt – Extra Large")]:
        mpl_font_menu.add_radiobutton(label=lbl, variable=_mpl_font_var, value=sz,
                                      command=lambda s=sz: _apply_mpl_font(s))
    view_menu.add_separator()

    # -- Quick Presets --
    def _preset_compact():
        _scale_var.set(85);       _apply_tk_scale(85)
        _ui_font_var.set(8);      _apply_ui_font(8)
        _mpl_font_var.set(8);     _apply_mpl_font(8)
        _plot_size_var.set("small"); _apply_plot_size(12.0, 7.0)

    def _preset_default():
        _scale_var.set(100);      _apply_tk_scale(100)
        _ui_font_var.set(9);      _apply_ui_font(9)
        _mpl_font_var.set(9);     _apply_mpl_font(9)
        _plot_size_var.set("medium"); _apply_plot_size(15.0, 9.0)

    def _preset_presentation():
        _scale_var.set(140);      _apply_tk_scale(140)
        _ui_font_var.set(12);     _apply_ui_font(12)
        _mpl_font_var.set(12);    _apply_mpl_font(12)
        _plot_size_var.set("large"); _apply_plot_size(18.0, 11.0)

    def _preset_accessibility():
        _scale_var.set(160);      _apply_tk_scale(160)
        _ui_font_var.set(14);     _apply_ui_font(14)
        _mpl_font_var.set(13);    _apply_mpl_font(13)
        _plot_size_var.set("xlarge"); _apply_plot_size(22.0, 13.0)

    presets_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Quick Presets", menu=presets_menu)
    presets_menu.add_command(label="🗜  Compact",       command=_preset_compact)
    presets_menu.add_command(label="⚙  Default",        command=_preset_default)
    presets_menu.add_command(label="📊  Presentation",  command=_preset_presentation)
    presets_menu.add_command(label="♿  Accessibility",  command=_preset_accessibility)
    view_menu.add_separator()
    view_menu.add_command(label="Reset All to Default", command=_preset_default)

    root.config(menu=menubar)

    # ------------------------------------------------------------------ #
    #  STRING VARS (with sensible defaults)                               #
    # ------------------------------------------------------------------ #
    def sv(val=""):
        return tk.StringVar(value=str(val))

    # Drone
    v_num_motors        = sv(4)
    v_weight            = sv(1500)
    v_area              = sv(0.05)
    v_speed             = sv(10)
    v_periph_current    = sv(0.0)
    v_profile_drag      = sv(0.02)
    v_profile_area      = sv(0.01)
    v_parasite_drag     = sv(0.9)
    v_parasite_area     = sv(0.05)
    v_body_length_m     = sv("")
    v_body_width_m      = sv("")
    v_body_height_m     = sv("")
    v_arm_length_m      = sv("")
    v_arm_width_m       = sv("")
    v_coaxial_spacing_m = sv("")
    v_max_tilt_deg      = sv("")
    v_motor_configuration = sv("flat")

    # Battery
    v_batt_vmin         = sv(3.0)
    v_batt_vnom         = sv(3.7)
    v_batt_vmax         = sv(4.2)
    v_batt_unit_mode    = sv("cell")
    v_batt_cell_capacity= sv(5000)
    v_batt_pack_capacity= sv(5000)
    v_batt_energy_density = sv(200)
    v_batt_chg          = sv(5)
    v_batt_a_cont       = sv(50)
    v_batt_a_max        = sv(100)
    v_batt_c_cont       = sv(15)
    v_batt_c_max        = sv(25)
    v_batt_dischg_pct   = sv(80)
    v_batt_r            = sv(20)
    v_batt_chem         = sv("LiPo")
    v_batt_series       = sv(4)
    v_batt_parallel     = sv(1)
    v_batt_cells_series = sv(1)
    v_batt_cells_parallel = sv(1)
    v_batt_pack_weight  = sv(0)
    v_batt_cell_weight  = sv(0)

    # Motor
    v_motor_kv          = sv(650)
    v_motor_i0          = sv(0.5)
    v_motor_v0          = sv(10)
    v_motor_rated_v     = sv(6)
    v_motor_r           = sv(0.2)
    v_motor_imax        = sv(20)
    v_motor_pmax        = sv(200)
    v_motor_pole_count  = sv(14)
    v_motor_weight      = sv(168)
    v_motor_size        = sv("28x28mm")

    # ESC
    v_esc_voltage_rating= sv(6)
    v_esc_cont_current  = sv(30)
    v_esc_max_current   = sv(60)
    v_esc_idle_current  = sv(0.5)
    v_esc_r             = sv(0.01)
    v_esc_weight        = sv(36)

    # Avionics (string var kept in sync with the treeview)
    v_avionics_voltage_tree = sv("5.0:(2,0.9), 12.0:(1.5,0.85)")

    # Prop
    v_prop_d            = sv(12)
    v_prop_pitch        = sv(6)
    v_prop_max_rpm      = sv(10000)
    v_prop_max_thrust   = sv(3000)
    v_prop_blades       = sv(2)
    v_prop_table        = sv("")
    v_prop_tconst       = sv("")
    v_prop_pconst       = sv("")
    v_prop_weight       = sv(20)

    # Mission / env
    v_mission           = sv("")
    v_alt               = sv(0)
    v_temp              = sv("")
    v_press             = sv("")
    v_wind              = sv(0)
    v_orientation       = sv("forward")
    v_max_speed_plot    = sv(30)

    # config_vars collected for save/load  (populated after tab widgets are built)
    config_vars: dict = {}

    # ------------------------------------------------------------------ #
    #  MAIN LAYOUT                                                        #
    # ------------------------------------------------------------------ #
    main = ttk.Frame(root, padding=8)
    main.grid(sticky="nsew")
    main.columnconfigure(0, weight=1, minsize=370)
    main.columnconfigure(1, weight=3)
    main.rowconfigure(0, weight=1)

    # ===== LEFT: scrollable input notebook =====
    left = ttk.Frame(main)
    left.grid(row=0, column=0, sticky="nsew")
    left.columnconfigure(0, weight=1)
    left.rowconfigure(0, weight=1)

    input_nb = ttk.Notebook(left)
    input_nb.grid(row=0, column=0, sticky="nsew")

    # --- Scrollable-tab registry so the single wheel binding can find the
    #     active canvas without iterating or calling bind_all repeatedly.
    _tab_canvases: list = []

    def make_scrollable_tab(nb, title):
        """
        Wrap a notebook tab in a Canvas + vertical Scrollbar.

        Key fixes vs the original:
          1. canvas.<Configure> — fires when the canvas itself is resized
             (e.g. when the tab is first shown).  Without this, winfo_width()
             returns 1 at startup so the inner frame is rendered 1 px wide
             and appears blank until the user switches tabs twice.
          2. bind_all is NOT used here — a single wheel binding is attached
             to the notebook after all tabs are created (see below), routing
             scroll events only to the currently-visible canvas.  Calling
             bind_all inside a loop creates N overlapping handlers on every
             widget in the application, which stalls the Tk event loop and
             causes the multi-minute render delay.
        """
        outer = ttk.Frame(nb)
        nb.add(outer, text=title)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        cv  = tk.Canvas(outer, highlightthickness=0)
        sb  = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        inn = ttk.Frame(cv)
        cv.configure(yscrollcommand=sb.set)
        cv.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        win_id = cv.create_window((0, 0), window=inn, anchor="nw")

        def _resize_inner(width):
            """Set the embedded-window width to match the canvas width."""
            if width > 1:                     # ignore the startup 1-px dummy value
                cv.itemconfig(win_id, width=width)

        def _on_inner_configure(evt):
            """Called when widgets are added to / removed from the inner frame."""
            cv.configure(scrollregion=cv.bbox("all"))
            _resize_inner(cv.winfo_width())

        def _on_canvas_configure(evt):
            """
            Called when the canvas is resized — critically, this fires when
            the tab is *first shown*, giving us the real pixel width.
            """
            cv.configure(scrollregion=cv.bbox("all"))
            _resize_inner(evt.width)

        inn.bind("<Configure>", _on_inner_configure)
        cv.bind("<Configure>",  _on_canvas_configure)   # <- the essential new line

        inn.columnconfigure(0, weight=1)
        inn.columnconfigure(1, weight=1)
        _tab_canvases.append(cv)   # register so wheel handler can find it
        return inn

    tab_drone    = make_scrollable_tab(input_nb, "Drone")
    tab_batt     = make_scrollable_tab(input_nb, "Battery")
    tab_motor    = make_scrollable_tab(input_nb, "Motor")
    tab_esc      = make_scrollable_tab(input_nb, "ESC")
    tab_avionics = make_scrollable_tab(input_nb, "Avionics")
    tab_prop     = make_scrollable_tab(input_nb, "Propeller")
    tab_mission  = make_scrollable_tab(input_nb, "Mission/Environment")

    # --- Single notebook-level mouse-wheel binding ----------------------------
    # Replaces the old bind_all-per-tab pattern.  We find whichever canvas
    # belongs to the currently-selected tab and scroll only that one.
    def _on_nb_mousewheel(evt):
        try:
            idx = input_nb.index(input_nb.select())
            if 0 <= idx < len(_tab_canvases):
                _tab_canvases[idx].yview_scroll(
                    int(-1 * (evt.delta / 120)), "units")
        except Exception:
            pass
    input_nb.bind("<MouseWheel>", _on_nb_mousewheel)
    # Also bind directly to each canvas so hovering over tab content works.
    for _cv in _tab_canvases:
        _cv.bind("<MouseWheel>", lambda evt, c=_cv:
                 c.yview_scroll(int(-1 * (evt.delta / 120)), "units"))
    # --------------------------------------------------------------------------


    def add_row(parent, r, label, var, **kw):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=3)
        e = ttk.Entry(parent, textvariable=var, width=14, **kw)
        e.grid(row=r, column=1, sticky="ew", padx=6, pady=3)
        return e

    # ===== DRONE TAB =====
    ttk.Label(tab_drone, text="Motor Configuration:").grid(
        row=0, column=0, sticky="w", padx=6, pady=4)
    motor_cfg_cb = ttk.Combobox(tab_drone, textvariable=v_motor_configuration,
                                values=["flat","coaxial"], state="readonly", width=12)
    motor_cfg_cb.grid(row=0, column=1, sticky="w", padx=6, pady=4)

    r = 1
    add_row(tab_drone, r, "Num Motors",              v_num_motors);        r += 1
    add_row(tab_drone, r, "Weight (g)",              v_weight);            r += 1
    add_row(tab_drone, r, "Frontal Area (m²)",       v_area);              r += 1
    add_row(tab_drone, r, "Cruise Speed (m/s)",      v_speed);             r += 1
    add_row(tab_drone, r, "Peripheral Current (A)",  v_periph_current);    r += 1

    ttk.Separator(tab_drone, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    ttk.Label(tab_drone, text="── Drag Parameters ──",
              foreground="gray").grid(row=r, column=0, columnspan=2, sticky="w", padx=6); r += 1
    add_row(tab_drone, r, "Profile Cd",              v_profile_drag);      r += 1
    add_row(tab_drone, r, "Profile Area (m²)",       v_profile_area);      r += 1
    add_row(tab_drone, r, "Parasite Cd",             v_parasite_drag);     r += 1
    add_row(tab_drone, r, "Parasite Area (m²)",      v_parasite_area);     r += 1

    ttk.Separator(tab_drone, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    ttk.Label(tab_drone, text="── Body Geometry (optional) ──",
              foreground="gray").grid(row=r, column=0, columnspan=2, sticky="w", padx=6); r += 1
    add_row(tab_drone, r, "Body Length (m)",         v_body_length_m);     r += 1
    add_row(tab_drone, r, "Body Width (m)",          v_body_width_m);      r += 1
    add_row(tab_drone, r, "Body Height (m)",         v_body_height_m);     r += 1
    add_row(tab_drone, r, "Arm Length (m)",          v_arm_length_m);      r += 1
    add_row(tab_drone, r, "Arm Width (m)",           v_arm_width_m);       r += 1
    add_row(tab_drone, r, "Max Tilt (deg)",          v_max_tilt_deg);      r += 1
    coax_entry = add_row(tab_drone, r, "Coaxial Spacing (m)", v_coaxial_spacing_m); r += 1
    coax_entry.configure(state="disabled")

    def _update_coax_state(event=None):
        if v_motor_configuration.get().strip().lower() == "coaxial":
            coax_entry.configure(state="normal")
        else:
            v_coaxial_spacing_m.set("")
            coax_entry.configure(state="disabled")
    motor_cfg_cb.bind("<<ComboboxSelected>>", _update_coax_state)

    # ===== BATTERY TAB =====
    ttk.Label(tab_batt, text="Unit mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    unit_mode_cb = ttk.Combobox(tab_batt, textvariable=v_batt_unit_mode,
                                values=["cell","pack"], state="readonly", width=10)
    unit_mode_cb.grid(row=0, column=1, sticky="w", padx=6, pady=4)

    r = 1
    add_row(tab_batt, r, "Vmin/cell (V)",            v_batt_vmin);         r += 1
    add_row(tab_batt, r, "Vnom/cell (V)",            v_batt_vnom);         r += 1
    add_row(tab_batt, r, "Vmax/cell (V)",            v_batt_vmax);         r += 1
    cell_cap_e = add_row(tab_batt, r, "Cell Capacity (mAh)",   v_batt_cell_capacity); r += 1
    pack_cap_e = add_row(tab_batt, r, "Pack Capacity (mAh)",   v_batt_pack_capacity); r += 1
    cell_wt_e  = add_row(tab_batt, r, "Cell Weight (g)",       v_batt_cell_weight);   r += 1
    pack_wt_e  = add_row(tab_batt, r, "Pack Weight (g)",       v_batt_pack_weight);   r += 1
    add_row(tab_batt, r, "Energy Density (Wh/kg)",   v_batt_energy_density); r += 1
    add_row(tab_batt, r, "Max Charge Current (A)",   v_batt_chg);          r += 1
    add_row(tab_batt, r, "Cont Discharge (A)",       v_batt_a_cont);       r += 1
    add_row(tab_batt, r, "Max Discharge (A)",        v_batt_a_max);        r += 1
    add_row(tab_batt, r, "Cont C-rate",              v_batt_c_cont);       r += 1
    add_row(tab_batt, r, "Max C-rate",               v_batt_c_max);        r += 1
    add_row(tab_batt, r, "Usable Discharge (%)",     v_batt_dischg_pct);   r += 1
    add_row(tab_batt, r, "Rcell (mΩ)",               v_batt_r);            r += 1
    add_row(tab_batt, r, "Series Cells/Packs",       v_batt_series);       r += 1
    add_row(tab_batt, r, "Parallel Cells/Packs",     v_batt_parallel);     r += 1
    cells_s_e = add_row(tab_batt, r, "Cells in Series/Pack",  v_batt_cells_series);  r += 1
    cells_p_e = add_row(tab_batt, r, "Cells in Parallel/Pack",v_batt_cells_parallel); r += 1
    add_row(tab_batt, r, "Chemistry",                v_batt_chem);         r += 1

    def on_unit_mode_change(event=None):
        mode = v_batt_unit_mode.get()
        if mode == "cell":
            cell_cap_e.configure(state="normal");  cell_wt_e.configure(state="normal")
            pack_cap_e.configure(state="disabled"); pack_wt_e.configure(state="disabled")
            cells_s_e.configure(state="disabled");  cells_p_e.configure(state="disabled")
        else:
            cell_cap_e.configure(state="disabled"); cell_wt_e.configure(state="disabled")
            pack_cap_e.configure(state="normal");   pack_wt_e.configure(state="normal")
            cells_s_e.configure(state="normal");    cells_p_e.configure(state="normal")
    unit_mode_cb.bind("<<ComboboxSelected>>", on_unit_mode_change)
    on_unit_mode_change()

    # ===== MOTOR TAB =====
    r = 0
    add_row(tab_motor, r, "Kv (RPM/V)",          v_motor_kv);         r += 1
    add_row(tab_motor, r, "Idle Current I0 (A)",  v_motor_i0);         r += 1
    add_row(tab_motor, r, "Idle Voltage V0 (V)",  v_motor_v0);         r += 1
    add_row(tab_motor, r, "Rated Voltage (V)",    v_motor_rated_v);    r += 1
    add_row(tab_motor, r, "Resistance Rm (Ω)",    v_motor_r);          r += 1
    add_row(tab_motor, r, "Max Current (A)",      v_motor_imax);       r += 1
    add_row(tab_motor, r, "Max Power (W)",        v_motor_pmax);       r += 1
    add_row(tab_motor, r, "Pole Count",           v_motor_pole_count); r += 1
    add_row(tab_motor, r, "Weight (g)",           v_motor_weight);     r += 1
    add_row(tab_motor, r, "Size (e.g. 28x28mm)",  v_motor_size);       r += 1

    # ===== ESC TAB =====
    r = 0
    add_row(tab_esc, r, "Voltage Rating (V)",     v_esc_voltage_rating); r += 1
    add_row(tab_esc, r, "Continuous Current (A)",  v_esc_cont_current);   r += 1
    add_row(tab_esc, r, "Max Current (A)",         v_esc_max_current);    r += 1
    add_row(tab_esc, r, "Idle Current (A)",        v_esc_idle_current);   r += 1
    add_row(tab_esc, r, "Resistance (Ω)",          v_esc_r);              r += 1
    add_row(tab_esc, r, "Weight (g)",              v_esc_weight);         r += 1

    # ===== AVIONICS TAB =====
    tab_avionics.columnconfigure(0, weight=1)
    tab_avionics.rowconfigure(1, weight=1)

    ttk.Label(
        tab_avionics,
        text="BEC / Avionics voltage rails  —  one row per regulated output bus.\n"
             "Double-click any cell to edit it in-place.",
        wraplength=340, justify="left", foreground="#555555",
    ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))

    av_tree_frame = ttk.Frame(tab_avionics)
    av_tree_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0, 4))
    av_tree_frame.columnconfigure(0, weight=1)
    av_tree_frame.rowconfigure(0, weight=1)

    _AV_COLS = ("voltage", "current", "efficiency")
    avionics_tree = ttk.Treeview(
        av_tree_frame, columns=_AV_COLS, show="headings", height=8, selectmode="browse")
    for col, heading, width in [
        ("voltage",    "Rail Voltage (V)",      130),
        ("current",    "Rail Current (A)",      130),
        ("efficiency", "BEC Efficiency (0–1]",  150),
    ]:
        avionics_tree.heading(col, text=heading)
        avionics_tree.column(col, width=width, anchor="center", stretch=True)
    avionics_tree.grid(row=0, column=0, sticky="nsew")

    av_sb = ttk.Scrollbar(av_tree_frame, orient="vertical", command=avionics_tree.yview)
    av_sb.grid(row=0, column=1, sticky="ns")
    avionics_tree.configure(yscrollcommand=av_sb.set)

    # ---- Helpers for the avionics table ----
    def _canonical_voltage_tree_string(d: dict) -> str:
        items = sorted(((float(v), float(i), float(e)) for v, (i, e) in d.items()),
                       key=lambda t: t[0])
        return ", ".join(f"{v:.3g}:({i:.3g},{e:.3g})" for v, i, e in items)

    def _sync_voltage_tree_var_from_table() -> None:
        d = {}
        for iid in avionics_tree.get_children():
            v_s, i_s, e_s = avionics_tree.item(iid, "values")
            try:
                d[float(v_s)] = (float(i_s), float(e_s))
            except Exception:
                continue
        v_avionics_voltage_tree.set(_canonical_voltage_tree_string(d))

    def _get_voltage_tree_from_table() -> dict:
        d = {}
        for iid in avionics_tree.get_children():
            v_s, i_s, e_s = avionics_tree.item(iid, "values")
            v, i, e = float(v_s), float(i_s), float(e_s)
            if v <= 0:   raise ValueError(f"Rail voltage must be > 0, got {v}.")
            if i < 0:    raise ValueError(f"Rail current must be ≥ 0, got {i}.")
            if e <= 0 or e > 1.0: raise ValueError(f"BEC efficiency must be in (0,1], got {e}.")
            d[v] = (i, e)
        return d

    def _av_load_rows(rows: list) -> None:
        avionics_tree.delete(*avionics_tree.get_children())
        for r in sorted(rows, key=lambda x: float(x.get("voltage", 0))):
            try:
                v = float(r.get("voltage")); i = float(r.get("current")); e = float(r.get("eff"))
                avionics_tree.insert("", "end", values=(f"{v:g}", f"{i:g}", f"{e:g}"))
            except Exception:
                continue
        _sync_voltage_tree_var_from_table()

    # ---- Double-click in-place editing ----
    def _av_begin_edit(event):
        region = avionics_tree.identify("region", event.x, event.y)
        if region != "cell": return
        row_id = avionics_tree.identify_row(event.y)
        col_id = avionics_tree.identify_column(event.x)
        if not row_id or not col_id: return
        col_idx = int(col_id.replace("#", "")) - 1
        bbox = avionics_tree.bbox(row_id, col_id)
        if not bbox: return
        x, y, w, h = bbox
        old_vals = list(avionics_tree.item(row_id, "values"))

        ed = tk.Entry(avionics_tree, justify="center")
        ed.insert(0, old_vals[col_idx])
        ed.select_range(0, tk.END)
        ed.focus_set()
        ed.place(x=x, y=y, width=w, height=h)

        def _commit(_evt=None):
            old_vals[col_idx] = ed.get().strip()
            avionics_tree.item(row_id, values=tuple(old_vals))
            ed.destroy()
            _sync_voltage_tree_var_from_table()

        def _cancel(_evt=None): ed.destroy()

        ed.bind("<Return>",   _commit)
        ed.bind("<Tab>",      _commit)
        ed.bind("<FocusOut>", _commit)
        ed.bind("<Escape>",   _cancel)

    avionics_tree.bind("<Double-1>", _av_begin_edit)

    # ---- Click row → populate entry fields ----
    _av_v_var = tk.StringVar()
    _av_i_var = tk.StringVar()
    _av_e_var = tk.StringVar(value="0.90")

    def _av_on_select(event):
        sel = avionics_tree.selection()
        if not sel: return
        vals = avionics_tree.item(sel[0], "values")
        try: _av_v_var.set(vals[0]); _av_i_var.set(vals[1]); _av_e_var.set(vals[2])
        except Exception: pass
    avionics_tree.bind("<<TreeviewSelect>>", _av_on_select)

    # ---- Entry fields for add/update ----
    av_entry_frame = ttk.Frame(tab_avionics)
    av_entry_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 2))
    for c in range(6): av_entry_frame.columnconfigure(c, weight=1)

    ttk.Label(av_entry_frame, text="Voltage (V):").grid(row=0, column=0, sticky="e", padx=(0,2))
    ttk.Entry(av_entry_frame, textvariable=_av_v_var, width=7).grid(row=0, column=1, sticky="ew", padx=(0,6))
    ttk.Label(av_entry_frame, text="Current (A):").grid(row=0, column=2, sticky="e", padx=(0,2))
    ttk.Entry(av_entry_frame, textvariable=_av_i_var, width=7).grid(row=0, column=3, sticky="ew", padx=(0,6))
    ttk.Label(av_entry_frame, text="Efficiency:").grid(row=0, column=4, sticky="e", padx=(0,2))
    ttk.Entry(av_entry_frame, textvariable=_av_e_var, width=7).grid(row=0, column=5, sticky="ew")

    # ---- Buttons ----
    av_btn_frame = ttk.Frame(tab_avionics)
    av_btn_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))

    def _av_add_or_update():
        try:
            v = float(_av_v_var.get()); i = float(_av_i_var.get()); e = float(_av_e_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Voltage, Current and Efficiency must be numbers.")
            return
        if v <= 0:
            messagebox.showerror("Invalid input", "Rail voltage must be > 0 V."); return
        if i < 0:
            messagebox.showerror("Invalid input", "Rail current must be ≥ 0 A."); return
        if not (0 < e <= 1.0):
            messagebox.showerror("Invalid input", "BEC efficiency must be in (0, 1]."); return
        for iid in avionics_tree.get_children():
            try:
                if abs(float(avionics_tree.item(iid,"values")[0]) - v) < 1e-9:
                    avionics_tree.item(iid, values=(f"{v:g}", f"{i:g}", f"{e:g}"))
                    _sync_voltage_tree_var_from_table()
                    _av_v_var.set(""); _av_i_var.set(""); _av_e_var.set("0.90")
                    return
            except Exception: pass
        avionics_tree.insert("", "end", values=(f"{v:g}", f"{i:g}", f"{e:g}"))
        _sync_voltage_tree_var_from_table()
        _av_v_var.set(""); _av_i_var.set(""); _av_e_var.set("0.90")

    def _av_remove():
        sel = avionics_tree.selection()
        for iid in sel: avionics_tree.delete(iid)
        _sync_voltage_tree_var_from_table()

    def _av_clear():
        avionics_tree.delete(*avionics_tree.get_children())
        _sync_voltage_tree_var_from_table()

    ttk.Button(av_btn_frame, text="➕  Add / Update Rail",
               command=_av_add_or_update).grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Button(av_btn_frame, text="🗑  Remove Selected",
               command=_av_remove).grid(row=0, column=1, sticky="w", padx=(0, 6))
    ttk.Button(av_btn_frame, text="✖  Clear All",
               command=_av_clear).grid(row=0, column=2, sticky="w")

    # Seed the table from the default string var
    try:
        _initial_tree = parse_voltage_tree(v_avionics_voltage_tree.get())
        _av_load_rows([{"voltage": v, "current": ci[0], "eff": ci[1]}
                       for v, ci in sorted(_initial_tree.items())])
    except Exception:
        _av_clear()

    # ===== PROPELLER TAB =====
    r = 0
    add_row(tab_prop, r, "Diameter (in)",         v_prop_d);          r += 1
    add_row(tab_prop, r, "Pitch (in)",            v_prop_pitch);      r += 1
    add_row(tab_prop, r, "Blades",                v_prop_blades);     r += 1
    add_row(tab_prop, r, "Max RPM",               v_prop_max_rpm);    r += 1
    add_row(tab_prop, r, "Max Thrust (g)",         v_prop_max_thrust); r += 1
    ttk.Separator(tab_prop, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    ttk.Label(tab_prop, text="Prop/Motor CSV table (optional)").grid(
        row=r, column=0, sticky="w", padx=6, pady=2)
    frow = ttk.Frame(tab_prop); frow.grid(row=r, column=1, sticky="ew")
    frow.columnconfigure(0, weight=1)
    ttk.Entry(frow, textvariable=v_prop_table).grid(row=0, column=0, sticky="ew", padx=(6,4))
    ttk.Button(frow, text="Browse…",
               command=lambda: choose_file(v_prop_table, [("CSV","*.csv"),("All","*.*")])).grid(
        row=0, column=1, padx=(0,6)); r += 1
    add_row(tab_prop, r, "TConst (optional)",     v_prop_tconst);     r += 1
    add_row(tab_prop, r, "PConst (optional)",     v_prop_pconst);     r += 1
    add_row(tab_prop, r, "Weight (g)",            v_prop_weight);     r += 1

    # ===== MISSION / ENV TAB =====
    r = 0
    ttk.Label(tab_mission, text="Mission JSON (optional)").grid(
        row=r, column=0, sticky="w", padx=6, pady=2)
    mrow = ttk.Frame(tab_mission); mrow.grid(row=r, column=1, sticky="ew")
    mrow.columnconfigure(0, weight=1)
    ttk.Entry(mrow, textvariable=v_mission).grid(row=0, column=0, sticky="ew", padx=(6,4))
    ttk.Button(mrow, text="Browse…",
               command=lambda: choose_file(v_mission, [("JSON","*.json"),("All","*.*")])).grid(
        row=0, column=1, padx=(0,6)); r += 1
    ttk.Separator(tab_mission, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1

    orientation_cb = ttk.Combobox(tab_mission, textvariable=v_orientation,
                                  values=["forward","hover"], state="readonly", width=12)
    ttk.Label(tab_mission, text="Orientation:").grid(row=r, column=0, sticky="w", padx=6, pady=3)
    orientation_cb.grid(row=r, column=1, sticky="w", padx=6, pady=3); r += 1

    add_row(tab_mission, r, "Altitude (m)",              v_alt);             r += 1
    add_row(tab_mission, r, "Temperature (°C, optional)",v_temp);            r += 1
    add_row(tab_mission, r, "Pressure (Pa, optional)",   v_press);           r += 1
    add_row(tab_mission, r, "Wind (m/s, + = headwind)",  v_wind);            r += 1
    add_row(tab_mission, r, "Max Speed for Plot (m/s)",  v_max_speed_plot);  r += 1

    # ------------------------------------------------------------------ #
    #  Collect config_vars for save/load                                  #
    # ------------------------------------------------------------------ #
    config_vars = {
        "num_motors": v_num_motors, "weight": v_weight, "area": v_area,
        "speed": v_speed, "periph_current": v_periph_current,
        "profile_drag": v_profile_drag, "profile_area": v_profile_area,
        "parasite_drag": v_parasite_drag, "parasite_area": v_parasite_area,
        "body_length_m": v_body_length_m, "body_width_m": v_body_width_m,
        "body_height_m": v_body_height_m, "arm_length_m": v_arm_length_m,
        "arm_width_m": v_arm_width_m, "coaxial_spacing_m": v_coaxial_spacing_m,
        "max_tilt_deg": v_max_tilt_deg, "motor_configuration": v_motor_configuration,
        "batt_vmin": v_batt_vmin, "batt_vnom": v_batt_vnom, "batt_vmax": v_batt_vmax,
        "batt_unit_mode": v_batt_unit_mode,
        "batt_cell_capacity": v_batt_cell_capacity, "batt_pack_capacity": v_batt_pack_capacity,
        "batt_energy_density": v_batt_energy_density,
        "batt_chg": v_batt_chg, "batt_a_cont": v_batt_a_cont, "batt_a_max": v_batt_a_max,
        "batt_c_cont": v_batt_c_cont, "batt_c_max": v_batt_c_max,
        "batt_dischg_pct": v_batt_dischg_pct, "batt_r": v_batt_r, "batt_chem": v_batt_chem,
        "batt_series": v_batt_series, "batt_parallel": v_batt_parallel,
        "batt_cells_series": v_batt_cells_series, "batt_cells_parallel": v_batt_cells_parallel,
        "batt_pack_weight": v_batt_pack_weight, "batt_cell_weight": v_batt_cell_weight,
        "motor_kv": v_motor_kv, "motor_i0": v_motor_i0, "motor_v0": v_motor_v0,
        "motor_rated_v": v_motor_rated_v, "motor_r": v_motor_r,
        "motor_imax": v_motor_imax, "motor_pmax": v_motor_pmax,
        "motor_pole_count": v_motor_pole_count, "motor_weight": v_motor_weight,
        "motor_size": v_motor_size,
        "esc_voltage_rating": v_esc_voltage_rating, "esc_cont_current": v_esc_cont_current,
        "esc_max_current": v_esc_max_current, "esc_idle_current": v_esc_idle_current,
        "esc_r": v_esc_r, "esc_weight": v_esc_weight,
        "avionics_voltage_tree": v_avionics_voltage_tree,
        "prop_d": v_prop_d, "prop_pitch": v_prop_pitch, "prop_blades": v_prop_blades,
        "prop_max_rpm": v_prop_max_rpm, "prop_max_thrust": v_prop_max_thrust,
        "prop_table": v_prop_table, "prop_tconst": v_prop_tconst,
        "prop_pconst": v_prop_pconst, "prop_weight": v_prop_weight,
        "mission": v_mission, "alt": v_alt, "temp": v_temp,
        "press": v_press, "wind": v_wind, "orientation": v_orientation,
        "max_speed_plot": v_max_speed_plot,
    }

    # ================================================================== #
    #  RIGHT PANEL: output notebooks                                      #
    # ================================================================== #
    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    right.columnconfigure(0, weight=1)
    right.rowconfigure(0, weight=2)
    right.rowconfigure(1, weight=1)

    display_nb = ttk.Notebook(right)
    display_nb.grid(row=0, column=0, sticky="nsew")

    tab_plot_out         = ttk.Frame(display_nb, padding=0)
    tab_status_out       = ttk.Frame(display_nb, padding=0)
    tab_metrics_out      = ttk.Frame(display_nb, padding=0)
    tab_mission_plots_out= ttk.Frame(display_nb, padding=0)
    for t in (tab_plot_out, tab_status_out, tab_metrics_out, tab_mission_plots_out):
        t.columnconfigure(0, weight=1); t.rowconfigure(0, weight=1)
    display_nb.add(tab_plot_out,          text="Plots")
    display_nb.add(tab_status_out,        text="Status")
    display_nb.add(tab_metrics_out,       text="Metrics")
    display_nb.add(tab_mission_plots_out, text="Mission Plots")
    tab_weight_budget_out = ttk.Frame(display_nb, padding=0)
    tab_weight_budget_out.columnconfigure(0, weight=1)
    tab_weight_budget_out.rowconfigure(0, weight=1)
    display_nb.add(tab_weight_budget_out, text="Weight Budget")


    # ---- Plots panel ----
    plot_frame = ttk.LabelFrame(tab_plot_out, text="Performance Plots", padding=4)
    plot_frame.grid(row=0, column=0, sticky="nsew")
    plot_frame.columnconfigure(0, weight=1)
    plot_frame.rowconfigure(0, weight=1)

    plot_canvas = tk.Canvas(plot_frame, highlightthickness=0)
    plot_canvas.grid(row=0, column=0, sticky="nsew")
    plot_scroll = ttk.Scrollbar(plot_frame, orient="vertical", command=plot_canvas.yview)
    plot_scroll.grid(row=0, column=1, sticky="ns")
    plot_canvas.configure(yscrollcommand=plot_scroll.set)

    plot_inner = ttk.Frame(plot_canvas)
    plot_inner_id = plot_canvas.create_window((0, 0), window=plot_inner, anchor="nw")
    plot_inner.columnconfigure(0, weight=1)

    def _update_plot_scrollregion(event=None):
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

    def _match_plot_inner_width(event):
        plot_canvas.itemconfigure(plot_inner_id, width=event.width)

    plot_inner.bind("<Configure>", lambda event: _update_plot_scrollregion(event))
    plot_canvas.bind("<Configure>", lambda event: _match_plot_inner_width(event))
    plot_inner.bind("<Enter>", lambda _: plot_canvas.bind_all("<MouseWheel>", _on_plot_mousewheel))
    plot_inner.bind("<Leave>", lambda _: plot_canvas.unbind_all("<MouseWheel>"))

    _plot_canvases = []  # Track multiple figure canvases
    _current_plot_figs = []  # Track multiple figures

    def _on_plot_mousewheel(evt):
        plot_canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        return "break"

    plot_canvas.bind("<Enter>", lambda _: plot_canvas.bind_all("<MouseWheel>", _on_plot_mousewheel))
    plot_canvas.bind("<Leave>", lambda _: plot_canvas.unbind_all("<MouseWheel>"))

    def _show_figure(fig_or_figs):
        """Display one or more figures in the scrollable plot area."""
        # Clear previous figures
        for canvas in _plot_canvases:
            try:
                canvas.get_tk_widget().destroy()
            except:
                pass
        for fig in _current_plot_figs:
            try:
                plt.close(fig)
            except:
                pass
        _plot_canvases.clear()
        _current_plot_figs.clear()
        
        # Handle both single figure and list of figures
        figures = fig_or_figs if isinstance(fig_or_figs, list) else [fig_or_figs]
        
        for idx, fig in enumerate(figures):
            if fig is None:
                continue
            fc = FigureCanvasTkAgg(fig, master=plot_inner)
            fc.draw()
            fc.get_tk_widget().grid(row=idx, column=0, sticky="nsew")
            _plot_canvases.append(fc)
            _current_plot_figs.append(fig)
        
        # Update scroll region
        plot_inner.update_idletasks()
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))


    # Weight Budget panel
    _wb_canvas_ref = [None]
    def _draw_weight_chart(rows):
        data_rows = [r for r in rows if r[0] != "TOTAL"]
        if not data_rows: return
        labels = [r[0] for r in data_rows]
        totals = [r[3] for r in data_rows]
        grand  = sum(totals)
        COLORS = ["#2E75B6","#ED7D31","#A9D18E","#FFC000","#5B9BD5","#FF7F7F"]
        fig, axes = plt.subplots(1, 2, figsize=(7, max(3, len(labels)*0.6+1)))
        fig.patch.set_facecolor("white")
        ax = axes[0]
        left_ = 0.0
        pcts = [t/grand*100 if grand>0 else 0 for t in totals]
        for i,(lbl,pct) in enumerate(zip(labels,pcts)):
            ax.barh(0, pct, left=left_, color=COLORS[i%len(COLORS)],
                    label=lbl, edgecolor="white", linewidth=0.5)
            if pct > 5:
                ax.text(left_+pct/2, 0, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=7.5, color="white")
            left_ += pct
        ax.set_xlim(0,100); ax.set_yticks([])
        ax.set_xlabel("% of total weight"); ax.set_title("Weight Distribution")
        ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=2,fontsize=7,frameon=False)
        ax.grid(axis="x",alpha=0.3)
        ax2 = axes[1]
        _,_, ats = ax2.pie(totals,labels=None,autopct="%1.0f%%",
            colors=COLORS[:len(labels)],startangle=90,pctdistance=0.75,
            wedgeprops=dict(edgecolor="white",linewidth=0.8))
        for at in ats: at.set_fontsize(7)
        ax2.set_title(f"Total: {grand:.0f} g")
        ax2.legend(labels,loc="lower center",bbox_to_anchor=(0.5,-0.22),ncol=2,fontsize=7,frameon=False)
        fig.tight_layout()
        if _wb_canvas_ref[0]:
            try: _wb_canvas_ref[0].get_tk_widget().destroy()
            except Exception: pass
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fc = FigureCanvasTkAgg(fig, master=wb_right)
        fc.draw(); fc.get_tk_widget().grid(row=0,column=0,sticky="nsew")
        _wb_canvas_ref[0] = fc; plt.close(fig)
    # Weight Budget layout
    wb_outer = ttk.Frame(tab_weight_budget_out, padding=4)
    wb_outer.grid(row=0, column=0, sticky="nsew")
    wb_outer.columnconfigure(0, weight=1); wb_outer.columnconfigure(1, weight=3)
    wb_outer.rowconfigure(0, weight=1)
    wb_left = ttk.LabelFrame(wb_outer, text="Component Weights", padding=4)
    wb_left.grid(row=0, column=0, sticky="nsew", padx=(0,6))
    wb_left.columnconfigure(0, weight=1); wb_left.rowconfigure(1, weight=1)
    wb_ph = ttk.Label(wb_left, text="Run a simulation to populate.",
                     foreground="#888888", wraplength=200)
    wb_ph.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(0,4))
    _wb_tv_cols = ("component","unit_w","count","total_w","pct")
    wb_tv = ttk.Treeview(wb_left, columns=_wb_tv_cols, show="headings", height=10)
    for col, heading, width in [
        ("component","Component",160),("unit_w","Unit (g)",70),
        ("count","Qty",40),("total_w","Total (g)",75),("pct","% of Total",70)]:
        wb_tv.heading(col, text=heading)
        wb_tv.column(col, width=width,
                     anchor="w" if col=="component" else "center", stretch=True)
    wb_tv.grid(row=1, column=0, sticky="nsew")
    wb_sb = ttk.Scrollbar(wb_left, orient="vertical", command=wb_tv.yview)
    wb_sb.grid(row=1, column=1, sticky="ns")
    wb_tv.configure(yscrollcommand=wb_sb.set)
    wb_tv.tag_configure("total", font=("TkDefaultFont", 9, "bold"))
    wb_right = ttk.LabelFrame(wb_outer, text="Weight Distribution", padding=4)
    wb_right.grid(row=0, column=1, sticky="nsew")
    wb_right.columnconfigure(0, weight=1); wb_right.rowconfigure(0, weight=1)
    def update_weight_budget(cfg):
        rows = _extract_weight_budget(cfg)
        for iid in wb_tv.get_children(): wb_tv.delete(iid)
        total_g = rows[-1][3] if rows else 1.0
        for label, unit_w, count, total_w in rows[:-1]:
            pct = f"{total_w/total_g*100:.1f}%" if total_g > 0 else "0%"
            wb_tv.insert("","end",values=(label,f"{unit_w:.1f}",
                         str(count),f"{total_w:.1f}",pct))
        if rows:
            label,unit_w,count,total_w = rows[-1]
            wb_tv.insert("","end",values=(label,"","",
                         f"{total_w:.1f}","100%"),tags=("total",))
        _draw_weight_chart(rows)

    # ---- Status panel ----
    sty = ttk.Style()
    try: sty.theme_use(sty.theme_use())
    except Exception: pass

    def _status_color_tag(value, limit, kind):
        try: v = float(value); L = float(limit)
        except Exception: return "na"
        if kind == "max": return "bad" if v > L else ("warn" if v > 0.9*L else "ok")
        else:             return "bad" if v < L else ("warn" if v < 1.1*L else "ok")

    def _make_status_tv(parent, title):
        lf = ttk.LabelFrame(parent, text=title, padding=4)
        lf.pack(fill="both", expand=True, padx=4, pady=4)
        cols = ("metric","value","limit","note")
        tv = ttk.Treeview(lf, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (200, 120, 120, 220)):
            tv.heading(c, text=c.capitalize())
            tv.column(c, width=w, anchor="w" if c in ("metric","note") else "center")
        tv.pack(fill="both", expand=True)
        for tag, bg in [("ok","#d9f2d9"),("warn","#fff2cc"),("bad","#f8d7da"),("na","#efefef")]:
            tv.tag_configure(tag, background=bg)
        return tv

    status_scroll = ttk.Frame(tab_status_out)
    status_scroll.grid(row=0, column=0, sticky="nsew")
    status_scroll.columnconfigure(0, weight=1)

    batt_table    = _make_status_tv(status_scroll, "Battery Status")
    motor_table   = _make_status_tv(status_scroll, "Motor / ESC Status")
    prop_table_tv = _make_status_tv(status_scroll, "Propeller Status")

    def _clear_status_tables():
        for tv in (batt_table, motor_table, prop_table_tv):
            for iid in tv.get_children(): tv.delete(iid)

    def _insert_status_row(tv, metric, val_str, lim_str, tag, note=""):
        tv.insert("", "end", values=(metric, val_str, lim_str, note), tags=(tag,))

    def update_status_tables_from_metrics(config: DroneConfig, metrics: dict):
        _clear_status_tables()
        Ipack = float(metrics.get("pack_current_A", 0.0))
        Vload = float(metrics.get("v_load_V", 0.0))
        Ptot  = float(metrics.get("total_power_W", 0.0))

        Vmin = float(getattr(config.battery, "vmin_pack", 0.0))
        tag_v = _status_color_tag(Vload, Vmin, "min") if Vmin > 0 else "na"
        _insert_status_row(batt_table, "Pack voltage (loaded)", f"{Vload:.2f} V", f">= {Vmin:.2f} V", tag_v)

        if getattr(config, 'max_tilt_deg', None) is not None and metrics.get('tilt_required_deg') is not None:
            tilt_req = float(metrics['tilt_required_deg'])
            tilt_lim = float(config.max_tilt_deg)
            _insert_status_row(batt_table, "Tilt req vs max",
                f"{tilt_req:.1f}° / {tilt_lim:.1f}°", f"<= {tilt_lim:.1f}°",
                _status_color_tag(tilt_req, tilt_lim, 'max'))

        Icont = float(getattr(config.battery, "discharge_cont_A", float("inf")))
        Imax  = float(getattr(config.battery, "discharge_max_A",  float("inf")))
        if math.isfinite(Icont):
            _insert_status_row(batt_table, "Pack current", f"{Ipack:.2f} A",
                f"<= {Icont:.2f} A (cont)", _status_color_tag(Ipack, Icont, "max"))
        else:
            _insert_status_row(batt_table, "Pack current", f"{Ipack:.2f} A", "cont: n/a", "na")
        if math.isfinite(Imax):
            _insert_status_row(batt_table, "Pack current (max)", f"{Ipack:.2f} A",
                f"<= {Imax:.2f} A (max)", _status_color_tag(Ipack, Imax, "max"))
        _insert_status_row(batt_table, "Total electrical power", f"{Ptot:.0f} W", "", "na")

        Iesc      = float(metrics.get("motor_I_per_esc_A", 0.0))
        Pmotor    = float(metrics.get("motor_power_W", 0.0))
        Pmotor_pm = Pmotor / max(int(config.num_motors), 1)
        _insert_status_row(motor_table, "Motor power / motor", f"{Pmotor_pm:.0f} W", "", "na")
        if getattr(config.motor, "max_current", None) is not None:
            _insert_status_row(motor_table, "Motor current / motor (est)", f"{Iesc:.2f} A",
                f"<= {float(config.motor.max_current):.2f} A",
                _status_color_tag(Iesc, float(config.motor.max_current), "max"))
        if getattr(config.motor, "max_power", None) is not None:
            _insert_status_row(motor_table, "Motor power / motor", f"{Pmotor_pm:.0f} W",
                f"<= {float(config.motor.max_power):.0f} W",
                _status_color_tag(Pmotor_pm, float(config.motor.max_power), "max"))
        esc = getattr(config, "esc", None)
        if esc is not None:
            _insert_status_row(motor_table, "ESC current (cont)", f"{Iesc:.2f} A",
                f"<= {float(esc.continuous_rating_A):.2f} A",
                _status_color_tag(Iesc, float(esc.continuous_rating_A), "max"))
            _insert_status_row(motor_table, "ESC current (max)", f"{Iesc:.2f} A",
                f"<= {float(esc.max_current_A):.2f} A",
                _status_color_tag(Iesc, float(esc.max_current_A), "max"))
            _insert_status_row(motor_table, "ESC loss (total)",
                f"{float(metrics.get('esc_loss_W',0)):.0f} W", "", "na")

        thrust_pm_N = float(metrics.get("thrust_per_motor_N", 0.0))
        thrust_pm_g = thrust_pm_N * 1000.0 / 9.81
        max_thr_g   = float(getattr(config.propeller, "max_thrust_g", 0.0))
        if max_thr_g > 0:
            _insert_status_row(prop_table_tv, "Thrust / motor", f"{thrust_pm_g:.0f} g",
                f"<= {max_thr_g:.0f} g", _status_color_tag(thrust_pm_g, max_thr_g, "max"))
        else:
            _insert_status_row(prop_table_tv, "Thrust / motor", f"{thrust_pm_g:.0f} g", "n/a", "na")
        rpm = metrics.get("prop_rpm", None)
        mr  = getattr(config.propeller, "max_rpm", None)
        if rpm is not None and float(rpm) > 0 and mr is not None:
            _insert_status_row(prop_table_tv, "Prop RPM (est)", f"{float(rpm):.0f} rpm",
                f"<= {float(mr):.0f} rpm", _status_color_tag(float(rpm), float(mr), "max"))
        else:
            _insert_status_row(prop_table_tv, "Prop RPM (est)", "n/a",
                f"<= {float(mr):.0f} rpm" if mr else "n/a", "na")

    # ---- Metrics panel ----
    metrics_container = ttk.Frame(tab_metrics_out, padding=4)
    metrics_container.grid(row=0, column=0, sticky="nsew")
    metrics_container.columnconfigure(0, weight=1)
    metrics_container.rowconfigure(0, weight=1)

    metrics_tv = ttk.Treeview(metrics_container, columns=("metric","value"),
                               show="headings", height=28)
    metrics_tv.heading("metric", text="Metric")
    metrics_tv.heading("value",  text="Value")
    metrics_tv.column("metric", width=280, anchor="w")
    metrics_tv.column("value",  width=260, anchor="w")
    metrics_sb = ttk.Scrollbar(metrics_container, orient="vertical", command=metrics_tv.yview)
    metrics_tv.configure(yscrollcommand=metrics_sb.set)
    metrics_tv.grid(row=0, column=0, sticky="nsew")
    metrics_sb.grid(row=0, column=1, sticky="ns")
    try:
        ttk.Style().configure("Metrics.Treeview", rowheight=24)
        metrics_tv.configure(style="Metrics.Treeview")
        metrics_tv.tag_configure("section", font=("TkDefaultFont", 10, "bold"))
    except Exception:
        pass

    def _metrics_clear():
        for iid in metrics_tv.get_children(): metrics_tv.delete(iid)

    def _metrics_add_section(title: str):
        metrics_tv.insert("", "end", values=(f"── {title} ──", ""), tags=("section",))

    def _metrics_add(metric: str, value: str):
        metrics_tv.insert("", "end", values=(metric, value))

    def update_metrics_tab(drone: DroneConfig, metrics: dict, speed_mps: float, orientation: str):
        _metrics_clear()
        def fmt(x, nd=2):
            try: return f"{float(x):.{nd}f}"
            except Exception: return "n/a"

        batt = drone.battery
        nm   = max(int(drone.num_motors), 1)
        g0   = 9.80665

        v_load       = float(metrics.get("v_load_V",         float("nan")))
        I_pack       = float(metrics.get("pack_current_A",   float("nan")))
        P_total      = float(metrics.get("total_power_W",    float("nan")))
        P_motor_total= float(metrics.get("motor_power_W",    0.0))
        P_periph     = float(metrics.get("periph_power_W",   0.0))
        P_esc_loss   = float(metrics.get("esc_loss_W",       0.0))

        cap_mAh     = float(batt.capacity_mAh)
        cap_Ah      = cap_mAh / 1000.0 if cap_mAh else 0.0
        usable_frac = max(0.0, min(1.0, float(getattr(batt, "discharge_percent", 100.0)) / 100.0))
        usable_mAh  = cap_mAh * usable_frac
        usable_Wh   = float(batt.capacity_Wh) * usable_frac
        load_C      = (I_pack / cap_Ah) if cap_Ah > 0 else float("nan")

        t_min   = estimate_flight_time_minutes(drone, speed_mps, orientation=orientation)
        range_m = (t_min * 60.0) * float(speed_mps)
        range_mi= (range_m / 1000.0) * 0.621371

        I_motor = float(metrics.get("motor_I_per_esc_A", float("nan")))
        rpm     = metrics.get("prop_rpm", None)
        rpm     = float(rpm) if rpm is not None else float("nan")
        kv      = getattr(drone.motor, "kv", None)
        Rm      = float(getattr(drone.motor, "resistance", 0.0))
        I0      = float(getattr(drone.motor, "idle_current", 0.0))

        V_emf = float("nan")
        if kv and kv > 0 and rpm == rpm:
            V_emf = rpm / float(kv)
        V_motor = v_load
        if V_emf == V_emf and I_motor == I_motor:
            V_motor = V_emf + I_motor * Rm

        throttle_linear = (V_motor / v_load) if (v_load == v_load and v_load > 1e-6 and V_motor == V_motor) else float("nan")
        if throttle_linear == throttle_linear:
            throttle_linear = max(0.0, min(1.2, throttle_linear))
        throttle_log = float("nan")
        if throttle_linear == throttle_linear:
            throttle_log = math.log10(1 + 9 * max(0.0, min(1.0, throttle_linear)))

        P_elec_motor = (I_motor * v_load) if (I_motor == I_motor and v_load == v_load) else float("nan")
        P_mech_motor = float("nan")
        if V_emf == V_emf and I_motor == I_motor:
            P_mech_motor = V_emf * max(0.0, I_motor - I0)
        motor_eff = (P_mech_motor / P_elec_motor) if (P_mech_motor==P_mech_motor and P_elec_motor==P_elec_motor and P_elec_motor > 0) else float("nan")

        T_est = float("nan")
        try:
            Imax = float(getattr(drone.motor, "max_current", 0.0))
            if Imax > 0 and I_motor == I_motor:
                copper = (I_motor**2) * Rm
                T_est  = 25.0 + 20.0 * (copper / max(1e-6, (Imax**2) * Rm))
        except Exception: pass

        thrust_total_N = float(metrics.get("thrust_total_N",    float("nan")))
        thrust_pm_N    = float(metrics.get("thrust_per_motor_N",float("nan")))
        weight_kg      = float(drone.drone_weight_g) / 1000.0
        twr            = (thrust_total_N / (weight_kg * g0)) if weight_kg > 0 else float("nan")
        thrust_pm_g    = (thrust_pm_N / g0) * 1000.0 if thrust_pm_N == thrust_pm_N else float("nan")
        spec_thrust    = (thrust_pm_g / P_elec_motor) if (P_elec_motor == P_elec_motor and P_elec_motor > 0 and thrust_pm_g == thrust_pm_g) else float("nan")

        drive_g = 0.0
        if getattr(drone.motor, "weight_g", None): drive_g += float(drone.motor.weight_g) * nm
        if getattr(drone, "esc", None) and getattr(drone.esc,"weight_g",None): drive_g += float(drone.esc.weight_g) * nm
        if getattr(drone.propeller, "weight_g", None): drive_g += float(drone.propeller.weight_g) * nm

        D_m = float(getattr(drone.propeller,"diameter_in",0.0)) * 0.0254
        A_total = math.pi * (D_m/2)**2 * nm

        p2w_Wkg = (P_total / weight_kg) if weight_kg > 0 else float("nan")
        P_out   = max(0.0, P_total - P_esc_loss - P_periph)
        eff_tot = (P_out / P_total) if P_total > 0 else float("nan")
        tilt    = float(metrics.get("tilt_required_deg", float("nan")))

        _metrics_add_section("Battery")
        _metrics_add("Load",              f"{fmt(load_C,2)} C")
        _metrics_add("Voltage (loaded)",  f"{fmt(v_load,2)} V")
        _metrics_add("Rated Voltage",     f"{fmt(batt.vmax_pack,2)} V")
        _metrics_add("Energy (usable)",   f"{fmt(usable_Wh,1)} Wh")
        _metrics_add("Energy (total)",    f"{fmt(float(batt.capacity_Wh),1)} Wh")
        _metrics_add("Total Capacity",    f"{fmt(cap_mAh,0)} mAh")
        _metrics_add("Usable Capacity",   f"{fmt(usable_mAh,0)} mAh")
        _metrics_add("Flight Time",       f"{fmt(t_min,2)} min")
        _metrics_add("Battery Weight",    f"{fmt(batt.weight_g,0)} g")

        _metrics_add_section("Motor @ Operating Point")
        _metrics_add("Current",            f"{fmt(I_motor,2)} A")
        _metrics_add("Voltage",            f"{fmt(V_motor,2)} V")
        _metrics_add("RPM",                f"{fmt(rpm,0)} rpm")
        _metrics_add("Thrust (per motor)", f"{fmt(thrust_pm_g,0)} g")
        _metrics_add("Thrust (total)",     f"{fmt((thrust_total_N/g0)*1000,0)} g")
        _metrics_add("Electric Power",     f"{fmt(P_elec_motor,1)} W")
        _metrics_add("Mechanical Power",   f"{fmt(P_mech_motor,1)} W")
        _metrics_add("Throttle (log)",     f"{fmt(throttle_log*100,0)} %")
        _metrics_add("Throttle (linear)",  f"{fmt(throttle_linear*100,0)} %")
        _metrics_add("Efficiency",         f"{fmt(motor_eff*100,1)} %")
        _metrics_add("Resistance Rm",      f"{fmt(Rm*1000,1)} mΩ")
        _metrics_add("Specific Thrust",    f"{fmt(spec_thrust,2)} g/W")
        _metrics_add("Est. Temperature",   f"{fmt(T_est,0)} °C")

        _metrics_add_section("Total Drive")
        _metrics_add("Drive Weight",       f"{fmt(drive_g,0)} g")
        _metrics_add("Thrust-Weight Ratio",f"{fmt(twr,2)} : 1")
        _metrics_add("Total Current",      f"{fmt(I_pack,2)} A")
        _metrics_add("P(in)",              f"{fmt(P_total,1)} W")
        _metrics_add("P(out)",             f"{fmt(P_out,1)} W")
        _metrics_add("Total Efficiency",   f"{fmt(eff_tot*100,1)} %")
        _metrics_add("Power / Weight",     f"{fmt(p2w_Wkg,1)} W/kg")

        _metrics_add_section("Multicopter")
        _metrics_add("Vehicle Weight",     f"{fmt(weight_kg*1000,0)} g")
        _metrics_add("Tilt Angle",         f"{fmt(tilt,1)} °")
        _metrics_add("Speed",              f"{fmt(speed_mps*3.6,1)} km/h  ({fmt(speed_mps*2.237,1)} mph)")
        _metrics_add("Estimated Range",    f"{fmt(range_m,0)} m  ({fmt(range_mi,2)} mi)")
        _metrics_add("Total Disc Area",    f"{fmt(A_total*1e4,0)} cm²  ({fmt(A_total/(0.0254**2),0)} in²)")
        max_pay_g = (thrust_total_N/g0 - weight_kg)*1000 if thrust_total_N==thrust_total_N else float("nan")
        _metrics_add("Max Extra Payload",  f"{fmt(max_pay_g,0)} g")

    # ---- Mission Plots panel ----
    mission_container = ttk.Frame(tab_mission_plots_out, padding=4)
    mission_container.grid(row=0, column=0, sticky="nsew")
    mission_container.columnconfigure(0, weight=0)
    mission_container.columnconfigure(1, weight=1)
    mission_container.rowconfigure(0, weight=1)

    mission_controls  = ttk.LabelFrame(mission_container, text="Y-axis variables", padding=4)
    mission_controls.grid(row=0, column=0, sticky="ns", padx=(0,8))
    mission_plot_frame= ttk.LabelFrame(mission_container, text="Mission plot", padding=4)
    mission_plot_frame.grid(row=0, column=1, sticky="nsew")
    mission_plot_frame.columnconfigure(0, weight=1)
    mission_plot_frame.rowconfigure(0, weight=1)

    ttk.Label(mission_controls, text="Select variables to plot vs mission time.").grid(
        row=0, column=0, sticky="w")
    mission_var_list = tk.Listbox(mission_controls, selectmode="extended", height=16, exportselection=False)
    mission_var_list.grid(row=1, column=0, sticky="nsew", pady=(4,4))
    mission_controls.rowconfigure(1, weight=1)
    mission_controls.columnconfigure(0, weight=1)
    ml_sb = ttk.Scrollbar(mission_controls, orient="vertical", command=mission_var_list.yview)
    ml_sb.grid(row=1, column=1, sticky="ns", pady=(4,4))
    mission_var_list.configure(yscrollcommand=ml_sb.set)

    mission_btns = ttk.Frame(mission_controls)
    mission_btns.grid(row=2, column=0, columnspan=2, sticky="ew")
    ttk.Button(mission_btns, text="Plot selected", command=lambda: _update_mission_plot()).grid(row=0, column=0, padx=(0,4))
    ttk.Button(mission_btns, text="Clear", command=lambda: _clear_mission_plot()).grid(row=0, column=1)

    mission_canvas_ref = [None]
    last_mission_series = [None]

    MISSION_VARS = [
        ("airspeed_mps",        "Vehicle airspeed",              "m/s"),
        ("groundspeed_mps",     "Ground speed",                  "m/s"),
        ("distance_km",         "Distance traveled",             "km"),
        ("altitude_m",          "Altitude",                      "m"),
        ("tilt_deg",            "Tilt angle",                    "deg"),
        ("battery_voltage_V",   "Battery voltage (loaded)",      "V"),
        ("battery_current_A",   "Battery current",               "A"),
        ("battery_energy_Wh",   "Battery energy remaining",      "Wh"),
        ("battery_capacity_mAh","Battery capacity remaining",    "mAh"),
        ("total_power_W",       "Total power",                   "W"),
        ("motor_power_W",       "Motor power (total)",           "W"),
        ("motor_power_per_motor_W","Motor power (per motor)",    "W"),
        ("motor_current_A",     "Motor/ESC current (per ESC)",   "A"),
        ("motor_rpm",           "Motor RPM",                     "rpm"),
        ("motor_thrust_N",      "Motor thrust (per motor)",      "N"),
        ("thrust_total_N",      "Total thrust",                  "N"),
        ("periph_power_W",      "Avionics/peripherals power",    "W"),
        ("esc_loss_W",          "ESC loss power",                "W"),
    ]
    _mission_items = []
    for k, lbl, unit in MISSION_VARS:
        mission_var_list.insert(tk.END, f"{lbl} ({unit})")
        _mission_items.append((k, lbl, unit))

    def _clear_mission_plot():
        for w in mission_plot_frame.winfo_children(): w.destroy()
        mission_canvas_ref[0] = None

    def _update_mission_plot():
        ms = last_mission_series[0]
        if ms is None:
            messagebox.showinfo("Mission plot", "Run a mission first."); return
        sel = list(mission_var_list.curselection())
        if not sel:
            messagebox.showinfo("Mission plot", "Select at least one variable."); return

        t_min = [x/60.0 for x in ms.get('t_s', [])]
        if not t_min: return

        fig = plt.Figure(figsize=(7.5, 4.5), dpi=100)
        ax0 = fig.add_subplot(111)
        selected = [_mission_items[i] for i in sel]
        by_unit = {}
        for k, lbl, unit in selected:
            by_unit.setdefault(unit, []).append((k, lbl))

        axes = [(ax0, list(by_unit.keys())[0])]
        ax0.set_ylabel(list(by_unit.keys())[0])
        for ui, unit in enumerate(list(by_unit.keys())[1:], 1):
            axn = ax0.twinx()
            axn.spines['right'].set_position(('outward', 55*(ui-0)))
            axn.set_ylabel(unit)
            axes.append((axn, unit))

        lines, labels = [], []
        for ax, unit in axes:
            for key, lbl in by_unit.get(unit, []):
                y = ms.get(key, [])
                if y is None: continue
                yy = []
                for v in y:
                    try: yy.append(float("nan") if isinstance(v,float) and v!=v else float(v))
                    except Exception: yy.append(float("nan"))
                ln, = ax.plot(t_min, yy, label=lbl)
                lines.append(ln); labels.append(lbl)

        ax0.set_xlabel("Mission time (min)")
        ax0.grid(True)
        fig.suptitle("Mission variables vs time")
        if lines: ax0.legend(lines, labels, loc="best")

        for w in mission_plot_frame.winfo_children(): w.destroy()
        mc = FigureCanvasTkAgg(fig, master=mission_plot_frame)
        mc.draw()
        mc.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        mission_canvas_ref[0] = mc

    def _on_list_wheel(evt):
        if evt.delta: mission_var_list.yview_scroll(int(-1*(evt.delta/120)), "units")
        return "break"
    mission_var_list.bind("<MouseWheel>", _on_list_wheel)

    # ---- Output text ----
    out_frame = ttk.LabelFrame(right, text="Output", padding=4)
    out_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
    out_frame.columnconfigure(0, weight=1)
    out_frame.rowconfigure(0, weight=1)

    out_text = tk.Text(out_frame, height=8, wrap="word", state="disabled")
    out_text.grid(row=0, column=0, sticky="nsew")
    out_sb = ttk.Scrollbar(out_frame, orient="vertical", command=out_text.yview)
    out_sb.grid(row=0, column=1, sticky="ns")
    out_text.configure(yscrollcommand=out_sb.set)

    def out_print(msg: str):
        out_text.configure(state="normal")
        out_text.delete("1.0", "end")
        out_text.insert("end", msg)
        out_text.configure(state="disabled")

    def out_append(msg: str):
        out_text.configure(state="normal")
        out_text.insert("end", msg + "\n")
        out_text.see("end")
        out_text.configure(state="disabled")

    def clear_log():
        out_text.configure(state="normal")
        out_text.delete("1.0", "end")
        out_text.configure(state="disabled")

    # ================================================================== #
    #  BUILD CONFIG FROM GUI                                              #
    # ================================================================== #
    def build_config_from_gui() -> DroneConfig:
        batt = BatteryConfig(
            chemistry              = v_batt_chem.get().strip() or None,
            operating_voltage_min  = parse_float("Vmin/cell", v_batt_vmin.get()),
            operating_voltage_nominal = parse_float("Vnom/cell", v_batt_vnom.get()),
            operating_voltage_max  = parse_float("Vmax/cell", v_batt_vmax.get()),
            cell_capacity_mAh      = parse_float("Cell capacity", v_batt_cell_capacity.get()),
            pack_capacity_mAh      = parse_float("Pack capacity", v_batt_pack_capacity.get()),
            cell_weight_g          = parse_float("Cell weight", v_batt_cell_weight.get()),
            pack_weight_g          = parse_float("Pack weight", v_batt_pack_weight.get()),
            unit_energy_density    = parse_float("Energy density", v_batt_energy_density.get()),
            charge_current_max     = parse_float("Max charge current", v_batt_chg.get()),
            discharge_cont_A       = parse_float("Cont discharge", v_batt_a_cont.get()),
            discharge_max_A        = parse_float("Max discharge", v_batt_a_max.get()),
            discharge_c_cont       = parse_float("Cont C-rate", v_batt_c_cont.get()),
            discharge_c_max        = parse_float("Max C-rate", v_batt_c_max.get()),
            discharge_percent      = parse_float("Discharge %", v_batt_dischg_pct.get()),
            resistance_cell_mOhm   = parse_float("Rcell", v_batt_r.get()),
            unit_mode              = v_batt_unit_mode.get().strip().lower(),
            series_units           = parse_int("Series units", v_batt_series.get()),
            parallel_units         = parse_int("Parallel units", v_batt_parallel.get()),
            cells_series_per_unit  = parse_int("Cells series/unit", v_batt_cells_series.get()),
            cells_parallel_per_unit= parse_int("Cells parallel/unit", v_batt_cells_parallel.get()),
        )
        motor = MotorConfig(
            kv            = parse_float("Kv", v_motor_kv.get()),
            idle_current  = parse_float("Idle current", v_motor_i0.get()),
            idle_voltage  = parse_float("Idle voltage", v_motor_v0.get()),
            rated_voltage = parse_int("Rated voltage", v_motor_rated_v.get()),
            resistance    = parse_float("Motor resistance", v_motor_r.get()),
            max_current   = parse_float("Motor max current", v_motor_imax.get()),
            max_power     = parse_float("Motor max power", v_motor_pmax.get()),
            pole_count    = parse_int("Pole count", v_motor_pole_count.get()),
            weight_g      = parse_float("Motor weight", v_motor_weight.get()),
            size_mm       = v_motor_size.get().strip() or None,
        )
        esc = None
        _ef = [v_esc_voltage_rating.get().strip(), v_esc_cont_current.get().strip(),
               v_esc_max_current.get().strip(), v_esc_idle_current.get().strip(),
               v_esc_r.get().strip(), v_esc_weight.get().strip()]
        if any(_ef):
            esc = ESCConfig(
                voltage_rating       = parse_int("ESC voltage rating", v_esc_voltage_rating.get()),
                continuous_current_A = parse_float("ESC cont current", v_esc_cont_current.get()),
                max_current_A        = parse_float("ESC max current", v_esc_max_current.get()),
                idle_current_A       = parse_float("ESC idle current", v_esc_idle_current.get()),
                resistance           = parse_float("ESC resistance", v_esc_r.get()),
                weight_g             = parse_float("ESC weight", v_esc_weight.get()),
            )
        avionics = AvionicsConfig(voltage_tree=_get_voltage_tree_from_table())

        prop_table_path = v_prop_table.get().strip() or None
        tc = v_prop_tconst.get().strip(); pc = v_prop_pconst.get().strip()
        prop = PropellerConfig(
            diameter_in  = parse_float("Prop diameter", v_prop_d.get()),
            pitch_in     = parse_float("Prop pitch", v_prop_pitch.get()),
            max_rpm      = parse_float("Prop max RPM", v_prop_max_rpm.get()),
            max_thrust_g = parse_float("Prop max thrust", v_prop_max_thrust.get()),
            blades       = parse_int("Prop blades", v_prop_blades.get()),
            table_csv    = prop_table_path,
            TConst       = float(tc) if tc else None,
            PConst       = float(pc) if pc else None,
            weight_g     = parse_float("Prop weight", v_prop_weight.get()),
        )
        drone = DroneConfig(
            num_motors               = parse_int("Num motors", v_num_motors.get()),
            battery                  = batt,
            motor                    = motor,
            propeller                = prop,
            drone_weight_g           = parse_float("Weight", v_weight.get()),
            profile_drag_coefficient = (parse_float("Profile Cd", v_profile_drag.get()) if v_profile_drag.get().strip() else 0.0),
            profile_area             = (parse_float("Profile area", v_profile_area.get()) if v_profile_area.get().strip() else 0.0),
            parasite_drag_coefficient= (parse_float("Parasite Cd", v_parasite_drag.get()) if v_parasite_drag.get().strip() else 0.0),
            parasite_area            = (parse_float("Parasite area", v_parasite_area.get()) if v_parasite_area.get().strip() else 0.0),
            frontal_area             = (parse_float("Frontal area", v_area.get()) if v_area.get().strip() else 0.0),
            cruise_speed             = parse_float("Cruise speed", v_speed.get()),
            periph_current           = parse_float("Peripheral current", v_periph_current.get()),
            esc                      = esc,
            avionics                 = avionics,
            air_density              = AIR_DENSITY,
            body_length_m            = (parse_float("Body length", v_body_length_m.get()) if v_body_length_m.get().strip() else None),
            body_width_m             = (parse_float("Body width", v_body_width_m.get()) if v_body_width_m.get().strip() else None),
            body_height_m            = (parse_float("Body height", v_body_height_m.get()) if v_body_height_m.get().strip() else None),
            arm_length_m             = (parse_float("Arm length", v_arm_length_m.get()) if v_arm_length_m.get().strip() else None),
            arm_width_m              = (parse_float("Arm width", v_arm_width_m.get()) if v_arm_width_m.get().strip() else None),
            coaxial_spacing_m        = (parse_float("Coaxial spacing", v_coaxial_spacing_m.get()) if v_coaxial_spacing_m.get().strip() else None),
            max_tilt_deg             = (parse_float("Max tilt", v_max_tilt_deg.get()) if v_max_tilt_deg.get().strip() else None),
            motor_configuration      = (v_motor_configuration.get().strip().lower() or "flat"),
        )
        alt  = parse_float("Altitude", v_alt.get())
        temp = v_temp.get().strip()
        pres = v_press.get().strip()
        drone.air_density = compute_air_density(
            altitude_m    = alt,
            temperature_C = float(temp) if temp else None,
            pressure_Pa   = float(pres) if pres else None,
        )
        return drone

    # ================================================================== #
    #  OPERATING METRICS                                                  #
    # ================================================================== #
    def _estimate_prop_rpm(drone: DroneConfig, thrust_per_motor_N: float) -> Optional[float]:
        try:
            if not (drone.propeller.TConst and drone.propeller.PConst):
                return None
            D   = drone.propeller.diameter_in * 0.0254
            rho = drone.air_density
            lo, hi = 100.0, 40000.0
            sol = None
            for _ in range(40):
                mid = 0.5*(lo+hi)
                n   = mid/60.0
                T   = float(drone.propeller.TConst)*rho*(n**2)*(D**4)
                if T < thrust_per_motor_N: lo = mid
                else: hi = mid; sol = mid
            return float(sol) if sol is not None else None
        except Exception:
            return None

    def compute_operating_metrics(drone: DroneConfig, speed_mps: float, orientation: str, wind_mps: float = 0.0) -> dict:
        airspeed = float(speed_mps)
        if orientation == "forward":
            airspeed = max(0.0, float(speed_mps) + float(wind_mps))
        motor_power_W = power_required(drone, airspeed, orientation)
        periph_power_W = avionics_input_power_W(getattr(drone, "avionics", None))
        if periph_power_W <= 0.0:
            periph_power_W = drone.battery.vnom_pack * max(drone.periph_current, 0.0)
        total_power_W, v_load, pack_current_A, esc_note, motor_I_esc_A = total_power_with_esc(
            drone, motor_power_W=motor_power_W, periph_power_W=periph_power_W)
        total_thrust_N     = thrust_required(drone, airspeed, orientation)
        thrust_per_motor_N = total_thrust_N / max(int(drone.num_motors), 1)
        esc_loss_W  = max(0.0, float(total_power_W) - float(motor_power_W) - float(periph_power_W))
        rpm_est     = _estimate_prop_rpm(drone, thrust_per_motor_N)
        motor_table = None
        if getattr(drone, 'propeller', None) is not None and drone.propeller.table is not None:
            try:
                motor_table = interpolate_motor_point(drone, thrust_per_motor_N)
                if 'RPM' in motor_table: rpm_est = float(motor_table['RPM'])
            except Exception: motor_table = None
        tilt_req = required_tilt_deg(drone, airspeed, orientation)
        return {
            "airspeed_mps":        float(airspeed),
            "tilt_required_deg":   float(tilt_req),
            "tilt_limit_deg":      (float(drone.max_tilt_deg) if getattr(drone,'max_tilt_deg',None) is not None else None),
            "motor_power_W":       float(motor_power_W),
            "periph_power_W":      float(periph_power_W),
            "total_power_W":       float(total_power_W),
            "v_load_V":            float(v_load),
            "pack_current_A":      float(pack_current_A),
            "esc_loss_W":          float(esc_loss_W),
            "esc_note":            str(esc_note),
            "motor_I_per_esc_A":   float(motor_I_esc_A),
            "thrust_total_N":      float(total_thrust_N),
            "thrust_per_motor_N":  float(thrust_per_motor_N),
            "prop_rpm":            (float(rpm_est) if rpm_est is not None else None),
            "motor_table_throttle_pct": (float(motor_table["Throttle_pct"]) if motor_table and "Throttle_pct" in motor_table else None),
            "motor_table_eff_gW":  (float(motor_table["Efficiency_gW"])    if motor_table and "Efficiency_gW" in motor_table else None),
            "motor_table_temp_C":  (float(motor_table["Temp_C"])           if motor_table and "Temp_C" in motor_table else None),
        }

    # ================================================================== #
    #  RUN CALLBACKS                                                      #
    # ================================================================== #
    def run_single_point():
        clear_log()
        try:
            drone       = build_config_from_gui()
            orientation = v_orientation.get().strip().lower()
            if orientation not in ("hover","forward"):
                raise ValueError("Orientation must be 'hover' or 'forward'.")
            speed   = parse_float("Speed", v_speed.get())
            t_min   = estimate_flight_time_minutes(drone, speed, orientation=orientation)
            d_km    = estimate_flight_distance_km(drone, speed, orientation=orientation)
            be_v, be_min, br_v, br_km = find_optimal_speeds(drone)
            metrics = compute_operating_metrics(drone, speed_mps=speed, orientation=orientation)
            update_status_tables_from_metrics(drone, metrics)
            update_metrics_tab(drone, metrics, speed_mps=speed, orientation=orientation)

            max_spd = parse_float("Max speed plot", v_max_speed_plot.get())
            _last_run["drone"]   = drone
            _last_run["max_spd"] = max_spd
            # Capture sweep data for CSV/Excel export
            _mc_v = [0.5 + (max_spd-0.5)*i/200 for i in range(201)]
            _last_run_sweep.clear()
            _last_run_sweep.update({
                "Speed (m/s)": _mc_v,
                "Flight Time (min)": [estimate_flight_time_minutes(drone,v,"forward") for v in _mc_v],
                "Range (km)": [estimate_flight_distance_km(drone,v,"forward") for v in _mc_v],
                "Power Fwd (W)": [power_required(drone,v,"forward") for v in _mc_v],
                "Power Hov (W)": [power_required(drone,v,"hover") for v in _mc_v],
                "Thrust Fwd (N)": [thrust_required(drone,v,"forward") for v in _mc_v],
                "Thrust Hov (N)": [thrust_required(drone,v,"hover") for v in _mc_v],
            })
            _last_run_cfg[0] = drone
            update_weight_budget(drone)
            fig = make_performance_figure(
                drone, max_speed=max_spd,
                figsize=(_view["plot_w"], _view["plot_h"]))
            # Store metrics for regeneration during plot scale changes
            _last_run["metrics"] = metrics
            # Generate motor operating point figure if propeller table is available
            motor_fig = None
            if drone.propeller.table is not None:
                try:
                    motor_fig = make_motor_operating_point_figure(drone, metrics, figsize=(_view["plot_w"], 6))
                except Exception:
                    pass
            # Display both figures
            if motor_fig:
                _show_figure([fig, motor_fig])
            else:
                _show_figure(fig)
            display_nb.select(tab_plot_out)

            out_print(
                f"=== Single-Point Run @ {speed:.1f} m/s ({orientation}) ===\n"
                f"Air density     : {drone.air_density:.3f} kg/m³\n"
                f"Flight time     : {t_min:.2f} min\n"
                f"Flight distance : {d_km:.2f} km\n"
                f"Best endurance  : {be_v:.2f} m/s → {be_min:.2f} min\n"
                f"Best range      : {br_v:.2f} m/s → {br_km:.2f} km\n"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_mission():
        clear_log()
        try:
            mission_path = v_mission.get().strip()
            if not mission_path:
                raise ValueError("Select a mission JSON file first (Mission/Env tab).")
            drone       = build_config_from_gui()
            orientation = v_orientation.get().strip().lower()
            if orientation not in ("hover","forward"):
                raise ValueError("Orientation must be 'hover' or 'forward'.")
            temp = v_temp.get().strip()
            pres = v_press.get().strip()
            wind = parse_float("Wind", v_wind.get())

            mission = MissionProfile.from_json(mission_path)
            results, worst_metrics, mission_series = simulate_mission(
                drone, mission, orientation=orientation,
                temperature_C = float(temp) if temp else None,
                pressure_Pa   = float(pres) if pres else None,
                wind_mps      = wind,
            )
            last_mission_series[0] = mission_series

            lines = [f"=== Mission: {os.path.basename(mission_path)} ===",
                     f"Orientation: {orientation}  |  Wind: {wind:.2f} m/s", ""]
            total_t = total_d = 0.0
            for name, t_min, d_km, status in results:
                total_t += t_min; total_d += d_km
                lines.append(f"  {name}: {t_min:.2f} min, {d_km:.2f} km  [{status}]")
            lines += ["", f"TOTAL: {total_t:.2f} min, {total_d:.2f} km"]
            out_print("\n".join(lines))

            if worst_metrics is not None:
                update_status_tables_from_metrics(drone, worst_metrics)

            max_spd = parse_float("Max speed plot", v_max_speed_plot.get())
            _last_run["drone"]   = drone
            _last_run["max_spd"] = max_spd
            _last_run_cfg[0] = drone
            update_weight_budget(drone)
            fig = make_performance_figure(
                drone, max_speed=max_spd,
                figsize=(_view["plot_w"], _view["plot_h"]))
            _show_figure(fig)
            display_nb.select(tab_plot_out)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ================================================================== #
    #  SAVE / LOAD CONFIG                                                 #
    # ================================================================== #
    def _extract_avionics_rails() -> list:
        rails = []
        for iid in avionics_tree.get_children():
            v_s, i_s, e_s = avionics_tree.item(iid, "values")
            try:
                rails.append({"voltage": float(v_s), "current": float(i_s), "eff": float(e_s)})
            except Exception: continue
        rails.sort(key=lambda r: r["voltage"])
        return rails

    def save_config_to_file(path: str) -> None:
        data = {
            "schema": "multicopter_power_sim_gui_config",
            "version": 1,
            "vars": {k: v.get() for k, v in config_vars.items()},
            "avionics_rails": _extract_avionics_rails(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_config_from_file(path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, val in data.get("vars", {}).items():
            if k in config_vars:
                try: config_vars[k].set("" if val is None else str(val))
                except Exception: pass
        rails = data.get("avionics_rails", None)
        if isinstance(rails, list) and rails:
            _av_load_rows(rails)
        else:
            try:
                d = parse_voltage_tree(v_avionics_voltage_tree.get().strip())
                _av_load_rows([{"voltage": v, "current": ci[0], "eff": ci[1]}
                               for v, ci in d.items()])
            except Exception: pass
        on_unit_mode_change()

    def prompt_save_config():
        path = filedialog.asksaveasfilename(
            title="Save configuration", defaultextension=".json",
            filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path: return
        try:
            save_config_to_file(path)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error saving config", str(e))

    def prompt_load_config():
        path = filedialog.askopenfilename(
            title="Load configuration",
            filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path: return
        try:
            load_config_from_file(path)
            messagebox.showinfo("Loaded", f"Loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error loading config", str(e))

    file_menu.add_command(label="Load Config…", command=prompt_load_config)
    file_menu.add_command(label="Save Config…", command=prompt_save_config)

    _status_tv_pairs = [
        (batt_table, "Battery Status"),
        (motor_table, "Motor / ESC Status"),
        (prop_table_tv, "Propeller Status"),
    ]
    _report_title = "Multicopter Power Simulator — Performance Analysis"

    # ================================================================== #
    # EXPORT & REPORT FUNCTIONS
    # ================================================================== #
    _last_run_sweep: dict = {}    # populated after each run
    _last_run_cfg  = [None]       # populated after each run

    def _get_metrics_rows() -> list:
        """Read all rows from the metrics Treeview."""
        rows = []
        for iid in metrics_tv.get_children():
            vals = metrics_tv.item(iid, "values")
            if vals and len(vals) >= 2:
                rows.append((str(vals[0]), str(vals[1])))
        return rows

    def _get_status_sections() -> list:
        """Read all status Treeview tables as (title, [(metric,val,lim,note,tag),...])."""
        result = []
        for tv, title in _status_tv_pairs:
            sec_rows = []
            for iid in tv.get_children():
                vals = tv.item(iid, "values")
                tags = tv.item(iid, "tags")
                tag  = tags[0] if tags else "na"
                if vals and len(vals) >= 4:
                    sec_rows.append((str(vals[0]), str(vals[1]),
                                     str(vals[2]), str(vals[3]), str(tag)))
                elif vals and len(vals) == 3:
                    sec_rows.append((str(vals[0]), str(vals[1]),
                                     str(vals[2]), "", "na"))
            if sec_rows:
                result.append((title, sec_rows))
        return result

    def _get_log_text() -> str:
        try:
            out_text.configure(state="normal")
            t = out_text.get("1.0", "end")
            out_text.configure(state="disabled")
            return t
        except Exception:
            return ""

    def _get_inputs_rows() -> list:
        """Return all GUI input variables as (label, value) pairs."""
        rows = []
        for k, var in config_vars.items():
            try:
                val = var.get()
                if val not in ("", None):
                    rows.append((k.replace("_", " ").title(), str(val)))
            except Exception:
                pass
        return rows

    def _do_export_csv():
        if not _last_run_sweep:
            messagebox.showinfo("No data", "Run a simulation first to generate sweep data.")
            return
        path = filedialog.asksaveasfilename(
            title="Export CSV", defaultextension=".csv",
            filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path: return
        try:
            _export_csv_file(path, _last_run_sweep, _get_metrics_rows())
            messagebox.showinfo("Exported", f"CSV saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _do_export_excel():
        if not _last_run_sweep:
            messagebox.showinfo("No data", "Run a simulation first to generate sweep data.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Excel", defaultextension=".xlsx",
            filetypes=[("Excel files","*.xlsx"),("All files","*.*")])
        if not path: return
        try:
            cfg = _last_run_cfg[0]
            wb  = _extract_weight_budget(cfg) if cfg else []
            _export_excel_file(path, _last_run_sweep, _get_metrics_rows(), wb)
            messagebox.showinfo("Exported", f"Excel saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _do_generate_report():
        if not _last_run_sweep and not _get_metrics_rows():
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save PDF Report", defaultextension=".pdf",
            filetypes=[("PDF files","*.pdf"),("All files","*.*")])
        if not path: return
        try:
            cfg   = _last_run_cfg[0]
            figs  = []
            # Add all current plot figures from the scrollable area
            figs.extend(_current_plot_figs)
            if mission_canvas_ref[0] is not None:
                figs.append(mission_canvas_ref[0].figure)
            for num in plt.get_fignums():
                fig = plt.figure(num)
                if fig not in figs:
                    figs.append(fig)
            wb    = _extract_weight_budget(cfg) if cfg else []
            _generate_pdf_report(
                path         = path,
                report_title = _report_title,
                inputs_rows  = _get_inputs_rows(),
                metrics_rows = _get_metrics_rows(),
                status_sections = _get_status_sections(),
                log_text     = _get_log_text(),
                figures      = figs,
                weight_budget = wb,
            )
            messagebox.showinfo("Report generated", f"PDF report saved to:\n{path}")
        except Exception as e:
            import traceback
            messagebox.showerror("Report error", traceback.format_exc())

    # Wire export menu items
    file_menu.add_separator()
    file_menu.add_command(label="Export CSV…",       command=_do_export_csv)
    file_menu.add_command(label="Export Excel…",     command=_do_export_excel)
    file_menu.add_command(label="Generate PDF Report…", command=_do_generate_report)

    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=exit_app)

    # ================================================================== #
    #  BUTTON ROW                                                         #
    # ================================================================== #
    btn_frame = ttk.Frame(main)
    btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    btn_frame.columnconfigure(1, weight=1)

    ttk.Button(btn_frame, text="▶  Run Single-Point",
               command=run_single_point).grid(row=0, column=0, padx=(0, 6), pady=4)
    ttk.Button(btn_frame, text="📋  Run Mission (JSON)",
               command=run_mission).grid(row=0, column=1, padx=(0, 6), pady=4, sticky="w")
    ttk.Button(btn_frame, text="💾  Save Config",
               command=prompt_save_config).grid(row=0, column=2, padx=4, pady=4)
    ttk.Button(btn_frame, text="📂  Load Config",
               command=prompt_load_config).grid(row=0, column=3, padx=4, pady=4)
    ttk.Button(btn_frame, text="📊  Export CSV",
               command=_do_export_csv).grid(row=0, column=4, padx=4, pady=4)
    ttk.Button(btn_frame, text="📗  Export Excel",
               command=_do_export_excel).grid(row=0, column=5, padx=4, pady=4)
    ttk.Button(btn_frame, text="📄  Generate Report",
               command=_do_generate_report).grid(row=0, column=6, padx=4, pady=4)

    root.protocol("WM_DELETE_WINDOW", exit_app)
    root.mainloop()

# -------------------------------
# CLI
# -------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multicopter Flight Simulator (CLI + optional GUI)")

    parser.add_argument("--gui", action="store_true", help="Launch the GUI instead of CLI run.")

    # Drone
    parser.add_argument("--num_motors", type=int, required=False)
    parser.add_argument("--weight", type=float, required=False, help="Drone weight (g)")
    parser.add_argument("--profile_drag", type=float, default=0.0, help="Profile drag coefficient")
    parser.add_argument("--profile_area", type=float, default=0.0, help="Rotor/arms profile reference area (m^2)")
    parser.add_argument("--parasite_drag", type=float, default=0.0, help="Parasite drag coefficient (fuselage/arms)")
    parser.add_argument("--parasite_area", type=float, default=0.0, help="Parasite reference area (m^2)")
    parser.add_argument("--area", type=float, required=False, help="Frontal area (m^2)")
    parser.add_argument("--body_length_m", type=float, default=None, help="Body length (m) for geometry-based drag fallback")
    parser.add_argument("--body_width_m", type=float, default=None, help="Body width (m) for geometry-based drag fallback")
    parser.add_argument("--body_height_m", type=float, default=None, help="Body height (m) for geometry-based drag fallback")
    parser.add_argument("--arm_length_m", type=float, default=None, help="Arm length (m) (center to motor) for geometry-based drag fallback")
    parser.add_argument("--arm_width_m", type=float, default=None, help="Arm width (m) for square-tube arm drag fallback")
    parser.add_argument("--max_tilt_deg", type=float, default=None, help="Maximum tilt angle (deg) for forward flight")
    parser.add_argument("--motor_configuration", type=str, default="flat", choices=["flat","coaxial"], help="Motor layout: flat or coaxial")
    parser.add_argument("--coaxial_spacing_m", type=float, default=None, help="Vertical spacing between coaxial rotors (m). If omitted, assumes ~0.2D")
    parser.add_argument("--speed", type=float, default=10.0, help="Speed (m/s) for single-point run")
    parser.add_argument("--periph_current", type=float, default=0.0, help="Peripheral current draw (A)")

    # Battery
    parser.add_argument("--battery_operating_voltage_min", type=float, required=False)
    parser.add_argument("--battery_operating_voltage_nominal", type=float, required=False)
    parser.add_argument("--battery_operating_voltage_max", type=float, required=False)
    parser.add_argument("--battery_energy_density", type=float, required=False)
    parser.add_argument("--battery_charge_current_max", type=float, required=False)
    parser.add_argument("--battery_discharge_cont_A", type=float, required=False)
    parser.add_argument("--battery_discharge_max_A", type=float, required=False)
    parser.add_argument("--battery_discharge_c_cont", type=float, required=False,
                        help="Continuous discharge C-rate (e.g., 15 for 15C). Used if --battery_discharge_cont not provided.")
    parser.add_argument("--battery_discharge_c_max", type=float, required=False,
                        help="Max/burst discharge C-rate (e.g., 25 for 25C). Defaults to continuous if omitted.")
    parser.add_argument("--battery_discharge_percent", type=float, default=100.0,
                        help="Percent of pack capacity to use (e.g., 80 means stop at 20% remaining).")
    parser.add_argument("--battery_chemistry", type=str, default=None)

    parser.add_argument("--battery_unit_mode", choices=["cell", "pack"], default="cell") # cells or pack
    parser.add_argument("--battery_series_units", type=int, required=False)    
    parser.add_argument("--battery_parallel_units", type=int, required=False, default=1)

    parser.add_argument("--battery_cells_series_per_unit", type=int, default=1)
    parser.add_argument("--battery_cells_parallel_per_unit", type=int, default=1)

    parser.add_argument("--battery_cell_capacity", type=float, required=False)  # mAh per cell
    parser.add_argument("--battery_resistance_cell", type=float, required=False) # mΩ per cell
    parser.add_argument("--battery_pack_capacity", type=float, required=False)  # mAh per pack
    parser.add_argument("--battery_pack_weight_g", type=float, required=False)
    parser.add_argument("--battery_cell_weight_g", type=float, required=False)

    # Motor
    parser.add_argument("--motor_kv", type=float, required=False)
    parser.add_argument("--motor_idle_current", type=float, required=False)
    parser.add_argument("--motor_idle_voltage", type=float, required=False)
    parser.add_argument("--motor_rated_voltage", type=float, required=False)
    parser.add_argument("--motor_resistance", type=float, required=False)
    parser.add_argument("--motor_max_current", type=float, required=False)
    parser.add_argument("--motor_max_power", type=float, required=False)
    parser.add_argument("--motor_pole_count", type=int, required=False)
    parser.add_argument("--motor_weight", type=float, required=False)
    parser.add_argument("--motor_size", type=str, default=None, help="Motor size/form factor (e.g., 28x28mm)")

    # ESC
    parser.add_argument("--esc_voltage_rating", type=float, required=False)
    parser.add_argument("--esc_cont_current", type=float, required=False)
    parser.add_argument("--esc_max_current", type=float, required=False)
    parser.add_argument("--esc_idle_current", type=float, required=False)
    parser.add_argument("--esc_resistance", type=float, required=False)
    parser.add_argument("--esc_weight", type=float, required=False)

    # Avionics
    parser.add_argument("--avionics_voltage_tree", type=str, default=None, help="Voltage tree for avionics power draw, e.g., '5.0:(2,0.9), 12.0:(1.5,0.85)' means 2A at 5V with 90% efficiency, and 1.5A at 12V with 85% efficiency")

    # Propeller
    parser.add_argument("--prop_diameter", type=float, required=False)
    parser.add_argument("--prop_pitch", type=float, required=False)
    parser.add_argument("--prop_blades", type=int, default=2)
    parser.add_argument("--prop_table", type=str, default=None)
    parser.add_argument("--prop_tconst", type=float, default=None, help="Prop thrust coefficient (C_T-like)")
    parser.add_argument("--prop_pconst", type=float, default=None, help="Prop power coefficient (C_P-like)")

    # Mission
    parser.add_argument("--mission", type=str, default=None, help="Path to mission profile JSON; if omitted, do single-point run.")

    # Environment
    parser.add_argument("--altitude", type=float, default=0.0, help="Altitude above sea level (m)")
    parser.add_argument("--temperature", type=float, default=None, help="Ambient temperature (°C)")
    parser.add_argument("--pressure", type=float, default=None, help="Ambient pressure (Pa)")
    parser.add_argument("--wind", type=float, default=0.0, help="Wind (m/s), + = headwind")

    # Options
    parser.add_argument("--orientation", type=str, default="forward", choices=["hover", "forward"])
    parser.add_argument("--plot", action="store_true", help="Show matplotlib window with performance curves (CLI only)")

    return parser


def validate_required_cli_args(args):
    """
    In CLI mode, we keep the same overall required parameters as before.
    (GUI mode bypasses these.)
    """
    required = [
        "num_motors", "weight", "area",
        "battery_operating_voltage_min", "battery_operating_voltage_max",
        "battery_capacity", "battery_weight", "battery_energy_density",
        "battery_charge_current_max", "battery_discharge_cont",
        "battery_resistance_cell", "battery_cell_count",
        "motor_kv", "motor_idle_current", "motor_resistance", "motor_max_current", "motor_max_power",
        "prop_diameter", "prop_pitch",
    ]
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"Missing required CLI args: {', '.join('--' + m for m in missing)}\n"
                         f"Tip: run with --gui to use the graphical interface.")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    validate_required_cli_args(args)
    drone = build_drone_from_args(args)

    print(f"Using air density = {drone.air_density:.3f} kg/m^3 at {args.altitude:.1f} m altitude")

    if args.mission:
        mission = MissionProfile.from_json(args.mission)
        results, _worst_metrics = simulate_mission(
            drone,
            mission,
            orientation=args.orientation,
            temperature_C=args.temperature,
            pressure_Pa=args.pressure,
            wind_mps=args.wind,
        )
        for name, time_min, dist_km, status in results:
            print(f"{name}: {time_min:.1f} min, {dist_km:.2f} km, {status}")
    else:
        t_min = estimate_flight_time_minutes(drone, args.speed, orientation=args.orientation)
        d_km = estimate_flight_distance_km(drone, args.speed, orientation=args.orientation)
        be_v, be_min, br_v, br_km = find_optimal_speeds(drone)

        print(f"Estimated flight time at {args.speed:.2f} m/s ({args.orientation}): {t_min:.1f} min")
        print(f"Estimated flight distance at {args.speed:.2f} m/s ({args.orientation}): {d_km:.2f} km")
        print(f"Best endurance speed (forward): {be_v:.1f} m/s -> {be_min:.1f} min")
        print(f"Best range speed (forward): {br_v:.1f} m/s -> {br_km:.2f} km")

    if args.plot:
        plot_performance(drone)


if __name__ == "__main__":
    main()
