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
        self.unit_mode = unit_mode
        self.series_units = int(series_units)
        self.parallel_units = int(parallel_units)

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
        self.energy_density_Wh_per_kg = float(unit_energy_density) if unit_energy_density is not None else float(self.capacity_Wh / (self.weight_g / 1000.0))
        self.max_operating_temperature_C = float(max_operating_temperature_C) if max_operating_temperature_C is not None else None
        self.min_operating_temperature_C = float(min_operating_temperature_C) if min_operating_temperature_C is not None else None
        self.charge_current_max = float(charge_current_max)

        # Capacity in Ah for C-rate conversions
        if unit_mode == "cell":
            self.capacity_mAh = self.cell_capacity_mAh * self.parallel_cells
        elif unit_mode == "pack":
            self.capacity_mAh = self.pack_capacity_mAh * self.parallel_units
        
        self.capacity_Ah = self.capacity_mAh / 1000.0

        if unit_mode == "cell":
            self.weight_g = self.cell_weight_g * self.total_cells
        elif unit_mode == "pack":
            self.weight_g = self.pack_weight_g * self.parallel_units * self.series_units

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

    Accepted formats (comma-separated rails):
      - "5.0:(2,0.9), 12.0:(1.5,0.85)"
      - "5.0:2:0.9, 12.0:1.5:0.85"

    Returns:
      dict[float, tuple[float, float]]
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        # Allow passing already-parsed dicts (e.g., from programmatic use).
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

    out: dict[float, tuple[float, float]] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        # Try "V:(I,eff)"
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)\s*$", p)
        if m:
            v, i, eff = map(float, m.groups())
        else:
            # Try "V:I:eff"
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
    # Geometry-derived drag is a fallback only; measured CdA should be preferred for
    # validation studies and performance sizing.
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

    # clamp
    if thrust_g <= float(df["Thrust_g"].min()):
        row = df.iloc[0]
        return {k: float(row[k]) for k in df.columns if k in (
            "Thrust_g","Power_W","RPM","Current_A","Voltage_V","Throttle_pct","Efficiency_gW","Temp_C","Torque_Nm"
        ) and pd.notna(row[k])}

    if thrust_g >= float(df["Thrust_g"].max()):
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

    Legacy mission model note:
      This older solver treats each phase as quasi-steady (single operating point per
      phase) and performs energy integration at phase level, unlike the newer transient
      timestep solver in the main multicopter simulator.
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
def make_performance_figure(config: DroneConfig, max_speed: float = 30.0):
    speeds, times, distances = [], [], []
    for v in range(1, int(max_speed) + 1):
        speeds.append(v)
        times.append(estimate_flight_time_minutes(config, v, orientation="forward"))
        distances.append(estimate_flight_distance_km(config, v, orientation="forward"))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(speeds, times, label="Flight Time (min)")
    ax.plot(speeds, distances, label="Flight Distance (km)")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Performance")
    ax.set_title("Drone Performance vs Speed")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_performance(config: DroneConfig, max_speed: float = 30.0):
    fig = make_performance_figure(config, max_speed=max_speed)
    plt.show()


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
def launch_gui():
    """
    Tkinter GUI that:
      - lets you fill in parameters
      - choose optional CSV/JSON files
      - run a single-point performance calc or mission sim
      - shows the performance plot embedded in the window
      - prints textual outputs in an embedded log panel
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Ensure matplotlib uses a GUI backend
    try:
        import matplotlib
        matplotlib.use("TkAgg")  # safe if already set
    except Exception:
        pass

    root = tk.Tk()
    root.title("Multicopter Power Simulator")

# --- High-DPI / UI scaling helpers ---
    # Enable Windows DPI awareness so Tk uses the correct scaling on high-DPI displays.
    try:
        import ctypes  # type: ignore
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # Base font sizes are captured once so we can scale deterministically.
    import tkinter.font as tkfont
    _BASE_TK_FONTS = {}
    for _fname in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
        try:
            _f = tkfont.nametofont(_fname)
            _BASE_TK_FONTS[_fname] = _f.cget("size")
        except Exception:
            pass

    # Matplotlib base font sizes (used for dynamic scaling)
    _BASE_MPL = {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }

    ui_scale = tk.DoubleVar(value=1.5)  # default; user can change in View menu

    def apply_ui_scale(scale: float) -> None:
        """Apply UI scaling to Tk widgets + Matplotlib plots."""
        try:
            scale = float(scale)
        except Exception:
            return
        if scale <= 0:
            return

        # Tk scaling affects widget geometry (padding, etc.)
        try:
            root.tk.call("tk", "scaling", scale)
        except Exception:
            pass

        # Scale named fonts
        try:
            for fname, base_sz in _BASE_TK_FONTS.items():
                try:
                    f = tkfont.nametofont(fname)
                    # Tk font sizes are in points; round to int for consistency.
                    new_sz = int(round(abs(base_sz) * scale)) * (1 if base_sz >= 0 else -1)
                    # keep sign for negative sizes (some themes use negative to mean pixels)
                    f.configure(size=new_sz)
                except Exception:
                    pass
        except Exception:
            pass

        # Scale Matplotlib defaults (for future plots)
        try:
            import matplotlib as mpl
            mpl.rcParams.update({k: (v * scale) for k, v in _BASE_MPL.items()})
        except Exception:
            pass

        # Also scale the currently displayed figure (if any)
        try:
            if current_fig is not None:
                _apply_matplotlib_scale_to_fig(current_fig, scale)
            if current_canvas is not None:
                current_canvas.draw_idle()
        except Exception:
            pass

        try:
            row_h = int(35 * scale) + 8
            ttk.Style().configure("Avionics.Treeview", rowheight=row_h)
        except Exception:
            pass

    def _apply_matplotlib_scale_to_fig(fig, scale: float) -> None:
        """Best-effort scaling of an existing Matplotlib figure's text."""
        for ax in getattr(fig, "axes", []):
            try:
                ax.title.set_fontsize(_BASE_MPL["axes.titlesize"] * scale)
                ax.xaxis.label.set_fontsize(_BASE_MPL["axes.labelsize"] * scale)
                ax.yaxis.label.set_fontsize(_BASE_MPL["axes.labelsize"] * scale)
                for t in ax.get_xticklabels() + ax.get_yticklabels():
                    t.set_fontsize(_BASE_MPL["xtick.labelsize"] * scale)
                leg = ax.get_legend()
                if leg is not None:
                    for t in leg.get_texts():
                        t.set_fontsize(_BASE_MPL["legend.fontsize"] * scale)
                    if leg.get_title() is not None:
                        leg.get_title().set_fontsize(_BASE_MPL["legend.fontsize"] * scale)
            except Exception:
                pass
        # Suptitle if present
        try:
            st = fig._suptitle
            if st is not None:
                st.set_fontsize(_BASE_MPL["axes.titlesize"] * scale)
        except Exception:
            pass

    # ---------- Menu (Save/Load Config) ----------
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)

    def _set_scale_and_var(v: float) -> None:
        ui_scale.set(float(v))
        apply_ui_scale(float(v))

    # Common UI scaling presets
    for _lbl, _val in [("100%", 1.0), ("125%", 1.25), ("150%", 1.5), ("175%", 1.75), ("200%", 2.0)]:
        view_menu.add_radiobutton(label=_lbl, variable=ui_scale, value=_val, command=lambda vv=_val: _set_scale_and_var(vv))

    view_menu.add_separator()
    view_menu.add_command(label="Reset to 100%", command=lambda: _set_scale_and_var(1.0))

    root.config(menu=menubar)

    # Apply default scaling once at startup
    apply_ui_scale(float(ui_scale.get()))

    # ---------- Helpers ----------
    def add_row(parent, r, label, var, width=14):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=2)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=r, column=1, sticky="ew", padx=6, pady=2)
        return e

    def choose_file(var, filetypes):
        p = filedialog.askopenfilename(filetypes=filetypes)
        if p:
            var.set(p)

    def log(msg: str):
        out.configure(state="normal")
        out.insert("end", msg + "\n")
        out.see("end")
        out.configure(state="disabled")

    def clear_log():
        out.configure(state="normal")
        out.delete("1.0", "end")
        out.configure(state="disabled")

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
        
    def exit_app():
        print("Exiting Application.")
        try:
            plt.close("all")   # stop matplotlib event loops
        except Exception:
            pass

        root.quit()            # stop Tk mainloop
        root.destroy()         # destroy the window
        sys.exit(0)

    # ---------- Variables (defaults) ----------
    # Drone
    v_num_motors = tk.StringVar(value="4")
    v_weight = tk.StringVar(value="1500")
    v_area = tk.StringVar(value="0.05")
    v_speed = tk.StringVar(value="10")
    v_periph_current = tk.StringVar(value="0.0")

    v_profile_drag = tk.StringVar(value="0.02")
    v_profile_area = tk.StringVar(value="0.01")
    v_parasite_drag = tk.StringVar(value="0.9")
    v_parasite_area = tk.StringVar(value="0.05")
    # Vehicle geometry (optional; used for drag fallback when drag params are not provided)
    v_body_length_m = tk.StringVar(value="")
    v_body_width_m = tk.StringVar(value="")
    v_body_height_m = tk.StringVar(value="")
    v_arm_length_m = tk.StringVar(value="")
    v_arm_width_m = tk.StringVar(value="")
    v_coaxial_spacing_m = tk.StringVar(value="")
    v_max_tilt_deg = tk.StringVar(value="")
    v_motor_configuration = tk.StringVar(value="flat")  # flat or coaxial


    # Battery
    v_batt_vmin = tk.StringVar(value="3.0")
    v_batt_vnom = tk.StringVar(value="3.7")
    v_batt_vmax = tk.StringVar(value="4.2")
    v_batt_unit_mode = tk.StringVar(value="cell")
    v_batt_cell_capacity = tk.StringVar(value="5000")
    v_batt_pack_capacity = tk.StringVar(value="5000")
    v_batt_energy_density = tk.StringVar(value="200")
    v_batt_chg = tk.StringVar(value="5")
    v_batt_a_cont = tk.StringVar(value="50")
    v_batt_a_max = tk.StringVar(value="100")
    v_batt_c_cont = tk.StringVar(value="15")
    v_batt_c_max = tk.StringVar(value="25")
    v_batt_dischg_pct = tk.StringVar(value="100")
    v_batt_r = tk.StringVar(value="20")
    v_batt_chem = tk.StringVar(value="LiIon")
    v_batt_series = tk.StringVar(value="1")
    v_batt_parallel = tk.StringVar(value="1")
    v_batt_cells_series = tk.StringVar(value="1")
    v_batt_cells_parallel = tk.StringVar(value="1")
    v_batt_pack_weight = tk.StringVar(value="0")
    v_batt_cell_weight = tk.StringVar(value="0")

    # Motor
    v_motor_kv = tk.StringVar(value="650")
    v_motor_i0 = tk.StringVar(value="0.5")
    v_motor_v0 = tk.StringVar(value="10")
    v_motor_rated_v = tk.StringVar(value="6")
    v_motor_r = tk.StringVar(value="0.2")
    v_motor_imax = tk.StringVar(value="20")
    v_motor_pmax = tk.StringVar(value="200")
    v_motor_pole_count = tk.StringVar(value="14")
    v_motor_weight = tk.StringVar(value="168")
    v_motor_size = tk.StringVar(value="28x28mm")

    # ESC
    v_esc_voltage_rating = tk.StringVar(value="6")
    v_esc_cont_current = tk.StringVar(value="30")
    v_esc_max_current = tk.StringVar(value="60")
    v_esc_idle_current = tk.StringVar(value="0.5")
    v_esc_r = tk.StringVar(value="0.01")
    v_esc_weight = tk.StringVar(value="36")

    # Avionics
    v_avionics_voltage_tree = tk.StringVar(value="5.0:(2,0.9), 12.0:(1.5,0.85)")  # e.g., "5:2:0.9,12:1.5:0.85"

    # Prop
    v_prop_d = tk.StringVar(value="12")
    v_prop_pitch = tk.StringVar(value="6")
    v_prop_max_rpm = tk.StringVar(value="10000")
    v_prop_max_thrust = tk.StringVar(value="3000")
    v_prop_blades = tk.StringVar(value="2")
    v_prop_table = tk.StringVar(value="")
    v_prop_tconst = tk.StringVar(value="")
    v_prop_pconst = tk.StringVar(value="")
    v_prop_weight = tk.StringVar(value="20")

    # Mission / env
    v_mission = tk.StringVar(value="")
    v_alt = tk.StringVar(value="0")
    v_temp = tk.StringVar(value="")
    v_press = tk.StringVar(value="")
    v_wind = tk.StringVar(value="0")
    v_orientation = tk.StringVar(value="forward")
    v_max_speed_plot = tk.StringVar(value="30")

    # ---------- Layout ----------
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    main = ttk.Frame(root, padding=8)
    main.grid(row=0, column=0, sticky="nsew")
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(1, weight=1)

    # Left: Inputs
    inputs = ttk.Notebook(main)
    inputs.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    main.rowconfigure(0, weight=1)

    tab_drone = ttk.Frame(inputs, padding=8)
    tab_batt = ttk.Frame(inputs, padding=8)
    tab_motor = ttk.Frame(inputs, padding=8)
    tab_esc = ttk.Frame(inputs, padding=8)
    tab_avionics = ttk.Frame(inputs, padding=8)
    tab_prop = ttk.Frame(inputs, padding=8)
    tab_mission = ttk.Frame(inputs, padding=8)

    inputs.add(tab_drone, text="Drone")
    inputs.add(tab_batt, text="Battery")
    inputs.add(tab_motor, text="Motor")
    inputs.add(tab_esc, text="ESC")
    inputs.add(tab_avionics, text="Avionics")
    inputs.add(tab_prop, text="Prop")
    inputs.add(tab_mission, text="Mission/Env")

    for t in (tab_drone, tab_batt, tab_motor, tab_esc, tab_avionics, tab_prop, tab_mission):
        t.columnconfigure(1, weight=1)

    # Drone tab
    r = 1
    num_motor_entry = add_row(tab_drone, r, "Num motors", v_num_motors); r += 1
    weight_entry = add_row(tab_drone, r, "Weight (g)", v_weight); r += 1
    area_entry = add_row(tab_drone, r, "Frontal area (m^2)", v_area); r += 1
    speed_entry = add_row(tab_drone, r, "Speed (m/s)", v_speed); r += 1
    periph_current_entry = add_row(tab_drone, r, "Peripheral current (A)", v_periph_current); r += 1

    ttk.Separator(tab_drone, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
    profile_cd_entry = add_row(tab_drone, r, "Profile Cd", v_profile_drag); r += 1
    profile_area_entry = add_row(tab_drone, r, "Profile area (m^2)", v_profile_area); r += 1
    parasite_cd_entry = add_row(tab_drone, r, "Parasite Cd", v_parasite_drag); r += 1
    parasite_area_entry = add_row(tab_drone, r, "Parasite area (m^2)", v_parasite_area); r += 1

    ttk.Separator(tab_drone, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
    add_row(tab_drone, r, "Body length (m)", v_body_length_m); r += 1
    add_row(tab_drone, r, "Body width (m)", v_body_width_m); r += 1
    add_row(tab_drone, r, "Body height (m)", v_body_height_m); r += 1
    add_row(tab_drone, r, "Arm length (m)", v_arm_length_m); r += 1
    add_row(tab_drone, r, "Arm width (m)", v_arm_width_m); r += 1
    add_row(tab_drone, r, "Max tilt (deg)", v_max_tilt_deg); r += 1
    coax_entry = add_row(tab_drone, r, "Coaxial spacing (m)", v_coaxial_spacing_m); r += 1
    coax_entry.configure(state="disabled")  # initially disabled until "coaxial" config selected

    # Motor configuration dropdown
    # after creating v_motor_configuration and v_coaxial_spacing_m StringVars
    ttk.Label(tab_drone, text="Motor Configuration:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    motor_cfg = ttk.Combobox(tab_drone, textvariable=v_motor_configuration,
                            values=["flat", "coaxial"], state="readonly", width=12)
    motor_cfg.grid(row=0, column=1, sticky="w", padx=6, pady=4)

    def _update_coaxial_spacing_enabled(event=None):
        is_coax = (v_motor_configuration.get().strip().lower() == "coaxial")

        if is_coax:
            # enable
            coax_entry.configure(state="normal")
        else:
            # disable + clear value (optional but recommended)
            v_coaxial_spacing_m.set("")
            coax_entry.configure(state="disabled")

    motor_cfg.bind("<<ComboboxSelected>>", _update_coaxial_spacing_enabled)

    # Battery tab
    r = 1
    cell_vmin_entry = add_row(tab_batt, r, "Vmin/cell (V)", v_batt_vmin); r += 1
    cell_vnom_entry = add_row(tab_batt, r, "Vnom/cell (V)", v_batt_vnom); r += 1
    cell_vmax_entry = add_row(tab_batt, r, "Vmax/cell (V)", v_batt_vmax); r += 1
    cell_capacity_entry = add_row(tab_batt, r, "Cell Capacity (mAh)", v_batt_cell_capacity); r += 1
    pack_capacity_entry = add_row(tab_batt, r, "Pack Capacity (mAh)", v_batt_pack_capacity); r += 1
    cell_weight_entry = add_row(tab_batt, r, "Cell Weight (g)", v_batt_cell_weight); r += 1
    pack_weight_entry = add_row(tab_batt, r, "Pack Weight (g)", v_batt_pack_weight); r += 1
    energy_density_entry = add_row(tab_batt, r, "Energy density (Wh/kg)", v_batt_energy_density); r += 1
    max_charge_current_entry = add_row(tab_batt, r, "Max charge current (A)", v_batt_chg); r += 1
    cont_discharge_current_entry = add_row(tab_batt, r, "Cont discharge current (A)", v_batt_a_cont); r += 1
    max_discharge_current_entry = add_row(tab_batt, r, "Max discharge current (A)", v_batt_a_max); r += 1
    cont_discharge_c_rate_entry = add_row(tab_batt, r, "Cont discharge C-rate (C)", v_batt_c_cont); r += 1
    max_discharge_c_rate_entry = add_row(tab_batt, r, "Max discharge C-rate (C)", v_batt_c_max); r += 1
    discharge_usable_entry = add_row(tab_batt, r, "Discharge usable (%)", v_batt_dischg_pct); r += 1
    rcell_entry = add_row(tab_batt, r, "Rcell (mΩ)", v_batt_r); r += 1
    series_units_entry = add_row(tab_batt, r, "Series Cells/Packs", v_batt_series); r += 1
    parallel_units_entry = add_row(tab_batt, r, "Parallel Cells/Packs", v_batt_parallel); r += 1
    cells_in_series_entry = add_row(tab_batt, r, "Cells in series per pack", v_batt_cells_series); r += 1
    cells_in_parallel_entry = add_row(tab_batt, r, "Cells in parallel per pack", v_batt_cells_parallel); r += 1
    chemistry_entry = add_row(tab_batt, r, "Chemistry (text)", v_batt_chem); r += 1

    ttk.Label(tab_batt, text="Unit mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)

    unit_mode_combo = ttk.Combobox(
        tab_batt,
        textvariable=v_batt_unit_mode,
        values=["cell", "pack"],
        state="readonly",     # prevents typing arbitrary values
        width=10
    )
    unit_mode_combo.grid(row=0, column=1, sticky="w", padx=6, pady=4)

    def on_unit_mode_change(event=None):
        mode = v_batt_unit_mode.get()
        if mode == "cell":
            # example: enable per-cell fields, disable pack-only fields
            cell_capacity_entry.configure(state="normal")
            cell_weight_entry.configure(state="normal")
            pack_capacity_entry.configure(state="disabled")
            pack_weight_entry.configure(state="disabled")
            cells_in_parallel_entry.configure(state="disabled")
            cells_in_series_entry.configure(state="disabled")
        else:  # "pack"
            cell_capacity_entry.configure(state="disabled")
            cell_weight_entry.configure(state="disabled")
            pack_capacity_entry.configure(state="normal")
            pack_weight_entry.configure(state="normal")
            cells_in_parallel_entry.configure(state="normal")
            cells_in_series_entry.configure(state="normal")

    unit_mode_combo.bind("<<ComboboxSelected>>", on_unit_mode_change)

    # Motor tab
    r = 0
    kv_entry = add_row(tab_motor, r, "Kv (RPM/V)", v_motor_kv); r += 1
    idle_current_entry = add_row(tab_motor, r, "Idle current I0 (A)", v_motor_i0); r += 1
    idle_voltage_entry = add_row(tab_motor, r, "Idle voltage V0 (V)", v_motor_v0); r += 1
    rated_voltage_entry = add_row(tab_motor, r, "Rated voltage (V)", v_motor_rated_v); r += 1
    resistance_entry = add_row(tab_motor, r, "Resistance (Ω)", v_motor_r); r += 1
    max_current_entry = add_row(tab_motor, r, "Max current (A)", v_motor_imax); r += 1
    max_power_entry = add_row(tab_motor, r, "Max power (W)", v_motor_pmax); r += 1
    pole_count_entry = add_row(tab_motor, r, "Pole count", v_motor_pole_count); r += 1
    motor_weight_entry = add_row(tab_motor, r, "Weight (g)", v_motor_weight); r += 1
    motor_size_entry = add_row(tab_motor, r, "Size (e.g., 28x28mm)", v_motor_size); r += 1

    #ESC tab
    r = 0
    esc_voltage_entry = add_row(tab_esc, r, "Voltage rating (V)", v_esc_voltage_rating); r += 1
    esc_cont_current_entry = add_row(tab_esc, r, "Continuous current (A)", v_esc_cont_current); r += 1
    esc_max_current_entry = add_row(tab_esc, r, "Max current (A)", v_esc_max_current); r += 1
    esc_idle_current_entry = add_row(tab_esc, r, "Idle current (A)", v_esc_idle_current); r += 1
    esc_resistance_entry = add_row(tab_esc, r, "Resistance (Ω)", v_esc_r); r += 1
    esc_weight_entry = add_row(tab_esc, r, "Weight (g)", v_esc_weight); r += 1

    #Avionics tab
    r = 0
    tab_avionics.columnconfigure(0, weight=1)
    tab_avionics.grid_columnconfigure(1, weight=1)
    tab_avionics.grid_rowconfigure(r, weight=1)

    ttk.Label(
        tab_avionics,
        text="Avionics voltage rails (double-click a cell to edit).\n"
             "Each row is: rail voltage (V), rail current (A), and BEC efficiency (0-1]."
    ).grid(row=r, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))
    r += 1

    av_table_frame = ttk.Frame(tab_avionics)
    av_table_frame.grid(row=r, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    av_table_frame.rowconfigure(0, weight=1)
    av_table_frame.columnconfigure(0, weight=1)

    av_cols = ("voltage", "current", "eff")

    style = ttk.Style()
    style.configure("Avionics.Treeview", rowheight=55)

    avionics_tree = ttk.Treeview(av_table_frame, columns=av_cols, show="headings", height=14, style="Avionics.Treeview", selectmode="browse")
    avionics_tree.heading("voltage", text="Rail Voltage (V)")
    avionics_tree.heading("current", text="Rail Current (A)")
    avionics_tree.heading("eff", text="BEC Efficiency (0-1]")
    avionics_tree.column("voltage", width=120, anchor="center")
    avionics_tree.column("current", width=120, anchor="center")
    avionics_tree.column("eff", width=140, anchor="center")
    avionics_tree.grid(row=0, column=0, sticky="nsew")

    av_sb = ttk.Scrollbar(av_table_frame, orient="vertical", command=avionics_tree.yview)
    av_sb.grid(row=0, column=1, sticky="ns")
    avionics_tree.configure(yscrollcommand=av_sb.set)

    av_edit_frame = ttk.Frame(tab_avionics)
    av_edit_frame.grid(row=r+1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
    for c in range(6):
        av_edit_frame.columnconfigure(c, weight=1)

    v_av_row_voltage = tk.StringVar(value="5.0")
    v_av_row_current = tk.StringVar(value="2.0")
    v_av_row_eff = tk.StringVar(value="0.9")

    ttk.Label(av_edit_frame, text="Voltage (V)").grid(row=0, column=0, sticky="w")
    ttk.Entry(av_edit_frame, textvariable=v_av_row_voltage, width=10).grid(row=0, column=1, sticky="ew", padx=(0, 8))
    ttk.Label(av_edit_frame, text="Current (A)").grid(row=0, column=2, sticky="w")
    ttk.Entry(av_edit_frame, textvariable=v_av_row_current, width=10).grid(row=0, column=3, sticky="ew", padx=(0, 8))
    ttk.Label(av_edit_frame, text="Efficiency").grid(row=0, column=4, sticky="w")
    ttk.Entry(av_edit_frame, textvariable=v_av_row_eff, width=10).grid(row=0, column=5, sticky="ew")

    av_btn_frame = ttk.Frame(tab_avionics)
    av_btn_frame.grid(row=r+2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
    av_btn_frame.columnconfigure(0, weight=1)

    def _canonical_voltage_tree_string(d: dict) -> str:
        # Sort for stable display.
        items = sorted(((float(v), float(i), float(e)) for v, (i, e) in d.items()), key=lambda t: t[0])
        return ", ".join([f"{v:.3g}:({i:.3g},{e:.3g})" for v, i, e in items])

    def _sync_voltage_tree_var_from_table() -> None:
        d = {}
        for iid in avionics_tree.get_children():
            v_str, i_str, e_str = avionics_tree.item(iid, "values")
            try:
                v = float(v_str)
                i = float(i_str)
                e = float(e_str)
            except Exception:
                continue
            d[float(v)] = (float(i), float(e))
        v_avionics_voltage_tree.set(_canonical_voltage_tree_string(d))

    def _get_voltage_tree_from_table() -> dict:
        d = {}
        for iid in avionics_tree.get_children():
            v_str, i_str, e_str = avionics_tree.item(iid, "values")
            v = float(v_str); i = float(i_str); e = float(e_str)
            # Validate using existing parser rules
            if v <= 0:
                raise ValueError(f"Avionics rail voltage must be > 0, got {v}.")
            if i < 0:
                raise ValueError(f"Avionics rail current must be >= 0, got {i}.")
            if e <= 0 or e > 1.0:
                raise ValueError(f"BEC efficiency must be in (0, 1], got {e}.")
            d[float(v)] = (float(i), float(e))
        return d

    def _add_or_update_rail() -> None:
        try:
            v = float(v_av_row_voltage.get().strip())
            i = float(v_av_row_current.get().strip())
            e = float(v_av_row_eff.get().strip())
        except Exception:
            messagebox.showerror("Invalid avionics rail", "Voltage/current/efficiency must be numeric.")
            return

        try:
            if v <= 0:
                raise ValueError("Voltage must be > 0.")
            if i < 0:
                raise ValueError("Current must be >= 0.")
            if e <= 0 or e > 1.0:
                raise ValueError("Efficiency must be in (0, 1].")
        except Exception as ex:
            messagebox.showerror("Invalid avionics rail", str(ex))
            return

        # Update existing row if same voltage exists (as float string match tolerant).
        for iid in avionics_tree.get_children():
            vv, _, _ = avionics_tree.item(iid, "values")
            try:
                if abs(float(vv) - v) < 1e-9:
                    avionics_tree.item(iid, values=(f"{v}", f"{i}", f"{e}"))
                    _sync_voltage_tree_var_from_table()
                    return
            except Exception:
                pass

        avionics_tree.insert("", "end", values=(f"{v}", f"{i}", f"{e}"))
        _sync_voltage_tree_var_from_table()

    def _remove_selected_rail() -> None:
        sel = avionics_tree.selection()
        if not sel:
            return
        avionics_tree.delete(sel[0])
        _sync_voltage_tree_var_from_table()

    def _clear_all_rails() -> None:
        for iid in avionics_tree.get_children():
            avionics_tree.delete(iid)
        _sync_voltage_tree_var_from_table()

    ttk.Button(av_btn_frame, text="Add / Update Rail", command=_add_or_update_rail).grid(row=0, column=0, sticky="w")
    ttk.Button(av_btn_frame, text="Remove Selected", command=_remove_selected_rail).grid(row=0, column=1, sticky="w", padx=(8, 0))
    ttk.Button(av_btn_frame, text="Clear", command=_clear_all_rails).grid(row=0, column=2, sticky="w", padx=(8, 0))

    # In-place editing on double-click
    def _begin_cell_edit(event):
        region = avionics_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = avionics_tree.identify_row(event.y)
        col_id = avionics_tree.identify_column(event.x)  # '#1', '#2', '#3'
        if not row_id or not col_id:
            return

        col_index = int(col_id.replace("#", "")) - 1
        x, y, w, h = avionics_tree.bbox(row_id, col_id)
        old_values = list(avionics_tree.item(row_id, "values"))
        if col_index < 0 or col_index >= len(old_values):
            return

        edit = tk.Entry(avionics_tree)
        edit.insert(0, old_values[col_index])
        edit.select_range(0, tk.END)
        edit.focus_set()
        edit.place(x=x, y=y, width=w, height=h)

        def _commit(_evt=None):
            new_val = edit.get().strip()
            old_values[col_index] = new_val
            avionics_tree.item(row_id, values=tuple(old_values))
            edit.destroy()
            try:
                # Validate table (will raise if bad), then sync string var.
                _get_voltage_tree_from_table()
            except Exception as ex:
                messagebox.showerror("Invalid avionics rail", str(ex))
            _sync_voltage_tree_var_from_table()

        def _cancel(_evt=None):
            edit.destroy()

        edit.bind("<Return>", _commit)
        edit.bind("<FocusOut>", _commit)
        edit.bind("<Escape>", _cancel)

    avionics_tree.bind("<Double-1>", _begin_cell_edit)

    # Populate the table from the current string var (so CLI/GUI stay consistent).
    try:
        _initial_tree = parse_voltage_tree(v_avionics_voltage_tree.get())
        for vv, (ii, ee) in sorted(_initial_tree.items(), key=lambda t: float(t[0])):
            avionics_tree.insert("", "end", values=(f"{float(vv)}", f"{float(ii)}", f"{float(ee)}"))
        _sync_voltage_tree_var_from_table()
    except Exception:
        # If the stored string is invalid, leave the table empty.
        _clear_all_rails()

    r += 3


    # Prop tab
    r = 0
    diameter_entry = add_row(tab_prop, r, "Diameter (in)", v_prop_d); r += 1
    pitch_entry = add_row(tab_prop, r, "Pitch (in)", v_prop_pitch); r += 1
    blades_entry = add_row(tab_prop, r, "Blades", v_prop_blades); r += 1
    max_rpm_entry = add_row(tab_prop, r, "Max RPM", v_prop_max_rpm); r += 1
    max_thrust_entry = add_row(tab_prop, r, "Max thrust (g)", v_prop_max_thrust); r += 1

    ttk.Separator(tab_prop, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1

    ttk.Label(tab_prop, text="Prop/motor table CSV (optional)").grid(row=r, column=0, sticky="w", padx=6, pady=2)
    frow = ttk.Frame(tab_prop)
    frow.grid(row=r, column=1, sticky="ew")
    frow.columnconfigure(0, weight=1)
    ttk.Entry(frow, textvariable=v_prop_table).grid(row=0, column=0, sticky="ew", padx=(6, 4))
    ttk.Button(frow, text="Browse…", command=lambda: choose_file(v_prop_table, [("CSV files", "*.csv"), ("All files", "*.*")])).grid(row=0, column=1, padx=(0, 6))
    r += 1

    tconst_entry = add_row(tab_prop, r, "TConst (optional)", v_prop_tconst); r += 1
    pconst_entry = add_row(tab_prop, r, "PConst (optional)", v_prop_pconst); r += 1
    prop_weight_entry = add_row(tab_prop, r, "Weight (g)", v_prop_weight); r += 1

    # Mission tab
    r = 0
    ttk.Label(tab_mission, text="Mission JSON (optional)").grid(row=r, column=0, sticky="w", padx=6, pady=2)
    mrow = ttk.Frame(tab_mission)
    mrow.grid(row=r, column=1, sticky="ew")
    mrow.columnconfigure(0, weight=1)
    ttk.Entry(mrow, textvariable=v_mission).grid(row=0, column=0, sticky="ew", padx=(6, 4))
    ttk.Button(mrow, text="Browse…", command=lambda: choose_file(v_mission, [("JSON files", "*.json"), ("All files", "*.*")])).grid(row=0, column=1, padx=(0, 6))
    r += 1

    ttk.Separator(tab_mission, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
    orientation_entry = add_row(tab_mission, r, "Orientation (hover/forward)", v_orientation); r += 1
    altitude_entry = add_row(tab_mission, r, "Altitude (m)", v_alt); r += 1
    temperature_entry = add_row(tab_mission, r, "Temperature (°C, optional)", v_temp); r += 1
    pressure_entry = add_row(tab_mission, r, "Pressure (Pa, optional)", v_press); r += 1
    wind_entry = add_row(tab_mission, r, "Wind (m/s)", v_wind); r += 1
    max_speed_plot_entry = add_row(tab_mission, r, "Max speed for plot (m/s)", v_max_speed_plot); r += 1

    # Right: plot + output
    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(0, weight=2)
    right.rowconfigure(1, weight=1)

    display_nb = ttk.Notebook(right)
    display_nb.grid(row=0, column=0, sticky="nsew")
    display_nb.columnconfigure(0, weight=1)
    display_nb.rowconfigure(0, weight=1)

    tab_plot_out = ttk.Frame(display_nb, padding=0)
    tab_status_out = ttk.Frame(display_nb, padding=0)
    tab_metrics_out = ttk.Frame(display_nb, padding=0)
    tab_mission_plots_out = ttk.Frame(display_nb, padding=0)
    tab_plot_out.columnconfigure(0, weight=1)
    tab_plot_out.rowconfigure(0, weight=1)
    tab_status_out.columnconfigure(0, weight=1)
    tab_status_out.rowconfigure(0, weight=1)
    tab_metrics_out.columnconfigure(0, weight=1)
    tab_metrics_out.rowconfigure(0, weight=1)
    tab_mission_plots_out.columnconfigure(0, weight=1)
    tab_mission_plots_out.rowconfigure(0, weight=1)

    display_nb.add(tab_plot_out, text="Plots")
    display_nb.add(tab_status_out, text="Status")
    display_nb.add(tab_metrics_out, text="Metrics")
    display_nb.add(tab_mission_plots_out, text="Mission Plots")

    plot_frame = ttk.LabelFrame(tab_plot_out, text="Plots", padding=6)
    plot_frame.grid(row=0, column=0, sticky="nsew")
    plot_frame.columnconfigure(0, weight=1)
    plot_frame.rowconfigure(0, weight=1)



    # ---------- Metrics tab (single-point eCalc-like summary) ----------
    metrics_container = ttk.Frame(tab_metrics_out, padding=6)
    metrics_container.grid(row=0, column=0, sticky="nsew")
    metrics_container.columnconfigure(0, weight=1)
    metrics_container.rowconfigure(0, weight=1)

    metrics_tv = ttk.Treeview(metrics_container, columns=("metric", "value"), show="headings", height=18)
    metrics_tv.heading("metric", text="Metric")
    metrics_tv.heading("value", text="Value")
    metrics_tv.column("metric", width=280, anchor="w")
    metrics_tv.column("value", width=260, anchor="w")

    metrics_sb = ttk.Scrollbar(metrics_container, orient="vertical", command=metrics_tv.yview)
    metrics_tv.configure(yscrollcommand=metrics_sb.set)

    metrics_tv.grid(row=0, column=0, sticky="nsew")
    metrics_sb.grid(row=0, column=1, sticky="ns")

    # Section header styling
    try:
        _mstyle = ttk.Style()
        _mstyle.configure("Metrics.Treeview", rowheight=24)
        metrics_tv.configure(style="Metrics.Treeview")
        metrics_tv.tag_configure("section", font=("TkDefaultFont", 11, "bold"))
    except Exception:
        pass

    def _metrics_clear() -> None:
        for _iid in metrics_tv.get_children():
            metrics_tv.delete(_iid)

    def _metrics_add_section(title: str) -> None:
        metrics_tv.insert("", "end", values=(title, ""), tags=("section",))

    def _metrics_add(metric: str, value: str) -> None:
        metrics_tv.insert("", "end", values=(metric, value))

    def update_metrics_tab(drone: DroneConfig, metrics: dict, speed_mps: float, orientation: str) -> None:
        """Populate the Metrics tab for the last single-point run."""
        _metrics_clear()

        def fmt(x, nd=2):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return "n/a"

        batt = drone.battery
        nm = max(int(drone.num_motors), 1)
        g0 = 9.80665

        v_load = float(metrics.get("v_load_V", float("nan")))
        I_pack = float(metrics.get("pack_current_A", float("nan")))
        P_total = float(metrics.get("total_power_W", float("nan")))
        P_motor_total = float(metrics.get("motor_power_W", 0.0))
        P_periph = float(metrics.get("periph_power_W", 0.0))
        P_esc_loss = float(metrics.get("esc_loss_W", 0.0))

        # Battery
        cap_mAh = float(batt.pack_capacity_mAh) if batt.pack_capacity_mAh is not None else float(batt.capacity_mAh)
        cap_Ah = cap_mAh / 1000.0 if cap_mAh else 0.0
        usable_frac = max(0.0, min(1.0, float(getattr(batt, "discharge_percent", 100.0)) / 100.0))
        usable_mAh = cap_mAh * usable_frac
        usable_Wh = float(batt.capacity_Wh) * usable_frac
        load_C = (I_pack / cap_Ah) if cap_Ah > 0 else float("nan")

        # Flight time + range at this point
        t_min = estimate_flight_time_minutes(drone, speed_mps, orientation=orientation)
        range_m = (t_min * 60.0) * float(speed_mps)
        range_mi = (range_m / 1000.0) * 0.621371

        # Motor estimates per motor
        I_motor = float(metrics.get("motor_I_per_esc_A", float("nan")))
        rpm = metrics.get("prop_rpm", None)
        rpm = float(rpm) if rpm is not None else float("nan")
        kv = getattr(drone.motor, "kv", None)
        Rm = float(getattr(drone.motor, "resistance", 0.0))
        I0 = float(getattr(drone.motor, "idle_current", 0.0))

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
        motor_eff = (P_mech_motor / P_elec_motor) if (P_mech_motor == P_mech_motor and P_elec_motor == P_elec_motor and P_elec_motor > 0) else float("nan")

        # Very rough temperature estimate
        T_est = float("nan")
        try:
            Imax = float(getattr(drone.motor, "max_current", 0.0))
            if Imax > 0 and I_motor == I_motor:
                copper = (I_motor ** 2) * Rm
                copper_max = (Imax ** 2) * Rm
                T_amb = float(getattr(drone, "temperature_C", 25.0))
                T_est = T_amb + 20.0 * (copper / max(1e-6, copper_max))
        except Exception:
            pass

        # Thrust / ratios
        thrust_total_N = float(metrics.get("thrust_total_N", float("nan")))
        thrust_pm_N = float(metrics.get("thrust_per_motor_N", float("nan")))
        weight_kg = float(drone.weight_kg)
        twr = (thrust_total_N / (weight_kg * g0)) if weight_kg > 0 else float("nan")

        thrust_pm_g = (thrust_pm_N / g0) * 1000.0 if thrust_pm_N == thrust_pm_N else float("nan")
        spec_thrust = (thrust_pm_g / P_elec_motor) if (P_elec_motor == P_elec_motor and P_elec_motor > 0 and thrust_pm_g == thrust_pm_g) else float("nan")

        # Drive weight (motors + ESCs + props)
        drive_g = 0.0
        if getattr(drone.motor, "weight_g", None) is not None:
            drive_g += float(drone.motor.weight_g) * nm
        if getattr(drone, "esc", None) is not None and getattr(drone.esc, "weight_g", None) is not None:
            drive_g += float(drone.esc.weight_g) * nm
        if getattr(drone.prop, "weight_g", None) is not None:
            drive_g += float(drone.prop.weight_g) * nm

        # Total disc area
        D_in = float(getattr(drone.prop, "diameter_in", 0.0))
        D_m = D_in * 0.0254
        A_disk = math.pi * (D_m / 2.0) ** 2
        A_total_m2 = A_disk * nm
        A_total_cm2 = A_total_m2 * 1e4
        A_total_in2 = A_total_m2 / (0.0254 ** 2)

        # Total efficiencies / power-to-weight
        p2w_Wkg = (P_total / weight_kg) if weight_kg > 0 else float("nan")
        P_out = max(0.0, P_total - P_esc_loss - P_periph)
        eff_total = (P_out / P_total) if P_total > 0 else float("nan")

        # Speed conversions
        v_kmh = speed_mps * 3.6
        v_mph = v_kmh * 0.621371

        # Tilt
        tilt = float(metrics.get("tilt_required_deg", float("nan")))
        tilt_lim = metrics.get("tilt_limit_deg", None)

        # Max additional payload from thrust margin
        max_payload_kg = (thrust_total_N / g0) - weight_kg if thrust_total_N == thrust_total_N else float("nan")
        max_payload_g = max_payload_kg * 1000.0 if max_payload_kg == max_payload_kg else float("nan")

        # ---- Populate table ----
        _metrics_add_section("Battery")
        _metrics_add("Load", f"{fmt(load_C, 2)} C")
        _metrics_add("Voltage", f"{fmt(v_load, 2)} V")
        _metrics_add("Rated Voltage", f"{fmt(batt.vmax_pack, 2)} V")
        _metrics_add("Energy (usable)", f"{fmt(usable_Wh, 1)} Wh")
        _metrics_add("Energy (total)", f"{fmt(float(batt.capacity_Wh), 1)} Wh")
        _metrics_add("Total Capacity", f"{fmt(cap_mAh, 0)} mAh")
        _metrics_add("Usable Capacity", f"{fmt(usable_mAh, 0)} mAh")
        _metrics_add("Flight Time (this point)", f"{fmt(t_min, 2)} min")
        _metrics_add("Battery Weight", f"{fmt(batt.weight_g, 0)} g")

        _metrics_add_section("Motor @ Operating Point")
        _metrics_add("Current", f"{fmt(I_motor, 2)} A")
        _metrics_add("Voltage", f"{fmt(V_motor, 2)} V")
        _metrics_add("RPM", f"{fmt(rpm, 0)} rpm")
        _metrics_add("Thrust (per motor)", f"{fmt(thrust_pm_g, 0)} g")
        _metrics_add("Thrust (total)", f"{fmt((thrust_total_N/g0)*1000.0, 0)} g")
        _metrics_add("Electric Power", f"{fmt(P_elec_motor, 1)} W")
        _metrics_add("Mechanical Power", f"{fmt(P_mech_motor, 1)} W")
        _metrics_add("Throttle (log)", f"{fmt(throttle_log*100.0, 0)} %")
        _metrics_add("Throttle (linear)", f"{fmt(throttle_linear*100.0, 0)} %")
        if getattr(drone.motor, "weight_g", None):
            _metrics_add("Power/Weight", f"{fmt(P_elec_motor / max(1e-6, (float(drone.motor.weight_g)/1000.0)), 1)} W/kg")
        else:
            _metrics_add("Power/Weight", "n/a")
        _metrics_add("Efficiency", f"{fmt(motor_eff*100.0, 1)} %")
        _metrics_add("Resistance (Rm)", f"{fmt(Rm*1000.0, 1)} mΩ")
        _metrics_add("Specific Thrust", f"{fmt(spec_thrust, 2)} g/W")
        _metrics_add("Est. Temperature", f"{fmt(T_est, 0)} °C")

        _metrics_add_section("Total Drive")
        _metrics_add("Drive Weight", f"{fmt(drive_g, 0)} g")
        _metrics_add("Thrust-Weight", f"{fmt(twr, 2)} : 1")
        _metrics_add("Total Current", f"{fmt(I_pack, 2)} A")
        _metrics_add("P(in)", f"{fmt(P_total, 1)} W")
        _metrics_add("P(out)", f"{fmt(P_out, 1)} W")
        _metrics_add("Total Efficiency", f"{fmt(eff_total*100.0, 1)} %")
        _metrics_add("Total Power/Weight", f"{fmt(p2w_Wkg, 1)} W/kg")

        _metrics_add_section("Multicopter")
        _metrics_add("Vehicle Weight", f"{fmt(weight_kg*1000.0, 0)} g")
        if tilt_lim is not None:
            _metrics_add("Tilt required vs max", f"{fmt(tilt, 1)}° / {fmt(tilt_lim, 1)}°")
        else:
            _metrics_add("Tilt required", f"{fmt(tilt, 1)}°")
        _metrics_add("Speed", f"{fmt(v_kmh, 0)} km/h  ({fmt(v_mph, 0)} mph)")
        _metrics_add("Estimated Range", f"{fmt(range_m, 0)} m  ({fmt(range_mi, 2)} mi)")
        _metrics_add("Total Disc Area", f"{fmt(A_total_cm2, 0)} cm²  ({fmt(A_total_in2, 0)} in²)")
        _metrics_add("Max additional payload", f"{fmt(max_payload_g, 0)} g")
    # ---------- Mission Plots tab (time-series over mission) ----------
    mission_container = ttk.Frame(tab_mission_plots_out, padding=6)
    mission_container.grid(row=0, column=0, sticky="nsew")
    mission_container.columnconfigure(0, weight=0)
    mission_container.columnconfigure(1, weight=1)
    mission_container.rowconfigure(0, weight=1)

    mission_controls = ttk.LabelFrame(mission_container, text="Y-axis variables", padding=6)
    mission_controls.grid(row=0, column=0, sticky="ns", padx=(0, 8))
    mission_plot_frame = ttk.LabelFrame(mission_container, text="Mission plot", padding=6)
    mission_plot_frame.grid(row=0, column=1, sticky="nsew")
    mission_plot_frame.columnconfigure(0, weight=1)
    mission_plot_frame.rowconfigure(0, weight=1)

    ttk.Label(mission_controls, text="Select one or more variables to plot vs mission time.").grid(row=0, column=0, sticky="w")

    mission_var_list = tk.Listbox(mission_controls, selectmode="extended", height=16, exportselection=False)
    mission_var_list.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
    mission_controls.rowconfigure(1, weight=1)
    mission_controls.columnconfigure(0, weight=1)

    mission_list_sb = ttk.Scrollbar(mission_controls, orient="vertical", command=mission_var_list.yview)
    mission_list_sb.grid(row=1, column=1, sticky="ns", pady=(6, 6))
    mission_var_list.configure(yscrollcommand=mission_list_sb.set)

    mission_btns = ttk.Frame(mission_controls)
    mission_btns.grid(row=2, column=0, columnspan=2, sticky="ew")
    ttk.Button(mission_btns, text="Plot selected", command=lambda: update_mission_plot()).grid(row=0, column=0, padx=(0, 6))
    ttk.Button(mission_btns, text="Clear", command=lambda: clear_mission_plot()).grid(row=0, column=1)

    # mission plot canvas state
    mission_canvas = None
    mission_fig = None
    last_mission_series = None

    MISSION_VARS = [
        ("airspeed_mps", "Vehicle airspeed", "m/s"),
        ("groundspeed_mps", "Ground speed", "m/s"),
        ("distance_km", "Distance traveled", "km"),
        ("altitude_m", "Altitude", "m"),
        ("tilt_deg", "Tilt angle", "deg"),
        ("battery_voltage_V", "Battery voltage (loaded)", "V"),
        ("battery_current_A", "Battery current", "A"),
        ("battery_energy_Wh", "Battery energy remaining", "Wh"),
        ("battery_capacity_mAh", "Battery capacity remaining", "mAh"),
        ("total_power_W", "Total power", "W"),
        ("motor_power_W", "Motor power (total)", "W"),
        ("motor_power_per_motor_W", "Motor power (per motor)", "W"),
        ("motor_current_A", "Motor/ESC current (per ESC)", "A"),
        ("motor_rpm", "Motor RPM", "rpm"),
        ("motor_thrust_N", "Motor thrust (per motor)", "N"),
        ("thrust_total_N", "Total thrust", "N"),
        ("periph_power_W", "Avionics/peripherals power", "W"),
        ("esc_loss_W", "ESC loss power", "W"),
    ]

    _mission_display_items = []  # parallel to listbox indices: (key,label,unit)
    for k, lbl, unit in MISSION_VARS:
        mission_var_list.insert(tk.END, f"{lbl} ({unit})")
        _mission_display_items.append((k, lbl, unit))

    def clear_mission_plot():
        nonlocal mission_canvas, mission_fig
        for w in mission_plot_frame.winfo_children():
            w.destroy()
        mission_canvas = None
        mission_fig = None

    def show_mission_figure(fig):
        nonlocal mission_canvas, mission_fig
        for w in mission_plot_frame.winfo_children():
            w.destroy()
        mission_fig = fig
        mission_canvas = FigureCanvasTkAgg(fig, master=mission_plot_frame)
        try:
            _apply_matplotlib_scale_to_fig(fig, float(ui_scale.get()))
        except Exception:
            pass
        mission_canvas.draw()
        mission_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def update_mission_plot():
        nonlocal last_mission_series
        if last_mission_series is None:
            messagebox.showinfo("Mission plot", "Run a mission simulation first to generate mission time series.")
            return
        sel = list(mission_var_list.curselection())
        if not sel:
            messagebox.showinfo("Mission plot", "Select one or more variables to plot.")
            return

        # Build figure with multiple y-axes (grouped by unit)
        import matplotlib.pyplot as plt
        fig = plt.Figure(figsize=(7.5, 4.5), dpi=100)
        ax0 = fig.add_subplot(111)

        t_min = [x / 60.0 for x in last_mission_series.get('t_s', [])]
        if not t_min:
            messagebox.showinfo("Mission plot", "Mission time series is empty.")
            return

        # Group selected variables by unit
        selected_items = [_mission_display_items[i] for i in sel]
        by_unit = {}
        for key, lbl, unit in selected_items:
            by_unit.setdefault(unit, []).append((key, lbl))

        axes = []
        unit_list = list(by_unit.keys())
        if not unit_list:
            return

        # First unit on left axis
        first_unit = unit_list[0]
        axes.append((ax0, first_unit))
        ax0.set_ylabel(first_unit)

        # Additional units on right axes
        for ui, unit in enumerate(unit_list[1:], start=1):
            axn = ax0.twinx()
            # offset spines so multiple y-axes don't overlap
            axn.spines['right'].set_position(('outward', 55 * (ui - 0)))
            axn.set_ylabel(unit)
            axes.append((axn, unit))

        # Plot each unit group
        lines = []
        labels = []
        for ax, unit in axes:
            for key, lbl in by_unit.get(unit, []):
                y = last_mission_series.get(key, [])
                if y is None:
                    continue
                # Replace NaN RPMs with None to avoid breaking autoscale
                yy = []
                for v in y:
                    try:
                        if isinstance(v, float) and (v != v):
                            yy.append(float('nan'))
                        else:
                            yy.append(float(v))
                    except Exception:
                        yy.append(float('nan'))
                ln, = ax.plot(t_min, yy, label=lbl)
                lines.append(ln)
                labels.append(lbl)

        ax0.set_xlabel('Mission time (min)')
        ax0.grid(True)
        fig.suptitle('Mission variables vs time')
        if lines:
            ax0.legend(lines, labels, loc='best')

        show_mission_figure(fig)

    # Scroll wheel for listbox
    def _on_list_wheel(event):
        if event.delta:
            mission_var_list.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        return 'break'
    mission_var_list.bind('<MouseWheel>', _on_list_wheel)

    out_frame = ttk.LabelFrame(right, text="Output", padding=6)
    out_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
    out_frame.columnconfigure(0, weight=1)
    out_frame.rowconfigure(0, weight=1)

    out = tk.Text(out_frame, height=10, wrap="word", state="disabled")
    out.grid(row=0, column=0, sticky="nsew")
    sb = ttk.Scrollbar(out_frame, orient="vertical", command=out.yview)
    sb.grid(row=0, column=1, sticky="ns")
    out.configure(yscrollcommand=sb.set)

    # ---------- Status tab (color-coded limit tables) ----------
    status_container = ttk.Frame(tab_status_out, padding=8)
    status_container.grid(row=0, column=0, sticky="nsew")
    status_container.columnconfigure(0, weight=1)
    status_container.rowconfigure(0, weight=1)
    status_container.rowconfigure(1, weight=1)
    status_container.rowconfigure(2, weight=1)

    # Treeview styling / row tags
    style = ttk.Style()
    try:
        style.theme_use(style.theme_use())
    except Exception:
        pass

    def _status_color_tag(value: float, limit: float, kind: str) -> str:
        """Return tag name for status based on proximity to a max/min limit.

        kind: "max" or "min". Yellow when within 10% of the limit.
        """
        try:
            v = float(value)
            L = float(limit)
        except Exception:
            return "na"
        if kind == "max":
            if v > L:
                return "bad"
            if v > 0.9 * L:
                return "warn"
            return "ok"
        else:  # min
            if v < L:
                return "bad"
            if v < 1.1 * L:
                return "warn"
            return "ok"

    def _apply_tree_tags(tv: ttk.Treeview):
        # Tkinter Treeview row background colors are set via tags.
        tv.tag_configure("ok", background="#d9f2d9")     # light green
        tv.tag_configure("warn", background="#fff2cc")   # light yellow
        tv.tag_configure("bad", background="#f8d7da")    # light red
        tv.tag_configure("na", background="#efefef")     # light gray

    def _make_status_table(parent, title: str) -> ttk.Treeview:
        lf = ttk.LabelFrame(parent, text=title, padding=6)
        lf.grid(sticky="nsew", padx=0, pady=(0, 8))
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)

        cols = ("metric", "value", "limit", "note")
        tv = ttk.Treeview(lf, columns=cols, show="headings", height=6)
        tv.heading("metric", text="Metric")
        tv.heading("value", text="Value")
        tv.heading("limit", text="Limit")
        tv.heading("note", text="Notes")
        tv.column("metric", width=180, anchor="w")
        tv.column("value", width=110, anchor="center")
        tv.column("limit", width=110, anchor="center")
        tv.column("note", width=220, anchor="w")
        tv.grid(row=0, column=0, sticky="nsew")

        sbv = ttk.Scrollbar(lf, orient="vertical", command=tv.yview)
        sbv.grid(row=0, column=1, sticky="ns")
        tv.configure(yscrollcommand=sbv.set)
        _apply_tree_tags(tv)
        return tv

    batt_table = _make_status_table(status_container, "Battery Status")
    status_container.rowconfigure(0, weight=1)
    batt_table.master.grid(row=0, column=0, sticky="nsew")

    motor_table = _make_status_table(status_container, "Motor/ESC Status")
    status_container.rowconfigure(1, weight=1)
    motor_table.master.grid(row=1, column=0, sticky="nsew")

    prop_table_tv = _make_status_table(status_container, "Propeller Status")
    status_container.rowconfigure(2, weight=1)
    prop_table_tv.master.grid(row=2, column=0, sticky="nsew")

    def _clear_status_tables():
        for tv in (batt_table, motor_table, prop_table_tv):
            for iid in tv.get_children():
                tv.delete(iid)

    def _insert_status_row(tv: ttk.Treeview, metric: str, value_str: str, limit_str: str, tag: str, note: str = ""):
        tv.insert("", "end", values=(metric, value_str, limit_str, note), tags=(tag,))

    def update_status_tables_from_metrics(config: DroneConfig, metrics: dict):
        """Populate the Status tab tables based on computed metrics at the last run.

        metrics should include keys like:
          pack_current_A, v_load_V, total_power_W, motor_I_per_esc_A, esc_loss_W,
          thrust_total_N, thrust_per_motor_N, prop_rpm (optional)
        """
        _clear_status_tables()

        # ---- Battery ----
        Ipack = float(metrics.get("pack_current_A", 0.0))
        Vload = float(metrics.get("v_load_V", 0.0))
        Ptot = float(metrics.get("total_power_W", 0.0))

        # Voltage min check
        Vmin = float(getattr(config.battery, "vmin_pack", 0.0))
        tag_v = _status_color_tag(Vload, Vmin, "min") if Vmin > 0 else "na"
        _insert_status_row(batt_table, "Pack voltage (loaded)", f"{Vload:.2f} V", f">= {Vmin:.2f} V", tag_v)

        # Max tilt check (forward flight). If no limit, omit.
        if getattr(config, 'max_tilt_deg', None) is not None and metrics.get('tilt_required_deg', None) is not None:
            tilt_req = float(metrics.get('tilt_required_deg', 0.0))
            tilt_lim = float(config.max_tilt_deg)
            tag_t = _status_color_tag(tilt_req, tilt_lim, 'max')
            _insert_status_row(
                batt_table,
                "Tilt required vs max tilt",
                f"{tilt_req:.1f}° / {tilt_lim:.1f}°",
                f"<= {tilt_lim:.1f}°",
                tag_t,
            )

        # Continuous and max discharge checks (if finite)
        Icont = float(getattr(config.battery, "discharge_cont_A", float("inf")))
        Imax = float(getattr(config.battery, "discharge_max_A", float("inf")))
        if math.isfinite(Icont):
            tag_c = _status_color_tag(Ipack, Icont, "max")
            _insert_status_row(batt_table, "Pack current", f"{Ipack:.2f} A", f"<= {Icont:.2f} A (cont)", tag_c)
        else:
            _insert_status_row(batt_table, "Pack current", f"{Ipack:.2f} A", "cont: n/a", "na")
        if math.isfinite(Imax):
            tag_m = _status_color_tag(Ipack, Imax, "max")
            _insert_status_row(batt_table, "Pack current (max)", f"{Ipack:.2f} A", f"<= {Imax:.2f} A (max)", tag_m)
        else:
            _insert_status_row(batt_table, "Pack current (max)", f"{Ipack:.2f} A", "max: n/a", "na")

        _insert_status_row(batt_table, "Total electrical power", f"{Ptot:.0f} W", "", "na")

        # Optional battery metric limits if user added them (checked dynamically)
        # Convention: any attribute on battery named like "max_<something>" / "min_<something>" with a matching metric in dict.
        for k, v in sorted(metrics.items()):
            if not isinstance(k, str):
                continue
            if k.startswith("battery_"):
                # allow user to pass extra battery metrics (battery_temp_C, etc.)
                base = k[len("battery_"):]
                max_attr = f"max_{base}"
                min_attr = f"min_{base}"
                if hasattr(config.battery, max_attr):
                    L = getattr(config.battery, max_attr)
                    if L is not None:
                        tag = _status_color_tag(float(v), float(L), "max")
                        _insert_status_row(batt_table, base, f"{float(v):.3g}", f"<= {float(L):.3g}", tag)
                if hasattr(config.battery, min_attr):
                    L = getattr(config.battery, min_attr)
                    if L is not None:
                        tag = _status_color_tag(float(v), float(L), "min")
                        _insert_status_row(batt_table, base, f"{float(v):.3g}", f">= {float(L):.3g}", tag)

        # ---- Motor / ESC ----
        Iesc = float(metrics.get("motor_I_per_esc_A", 0.0))
        Pmotor = float(metrics.get("motor_power_W", 0.0))
        Pmotor_per = Pmotor / max(int(config.num_motors), 1)
        _insert_status_row(motor_table, "Motor elec power / motor", f"{Pmotor_per:.0f} W", "", "na")

        # Motor max current/power limits
        if getattr(config.motor, "max_current", None) is not None:
            tag = _status_color_tag(Iesc, float(config.motor.max_current), "max")
            _insert_status_row(motor_table, "Motor current / motor (est)", f"{Iesc:.2f} A", f"<= {float(config.motor.max_current):.2f} A", tag)
        else:
            _insert_status_row(motor_table, "Motor current / motor (est)", f"{Iesc:.2f} A", "n/a", "na")
        if getattr(config.motor, "max_power", None) is not None:
            tag = _status_color_tag(Pmotor_per, float(config.motor.max_power), "max")
            _insert_status_row(motor_table, "Motor power / motor", f"{Pmotor_per:.0f} W", f"<= {float(config.motor.max_power):.0f} W", tag)
        else:
            _insert_status_row(motor_table, "Motor power / motor", f"{Pmotor_per:.0f} W", "n/a", "na")

        # ESC checks
        esc = getattr(config, "esc", None)
        if esc is not None:
            tag_ec = _status_color_tag(Iesc, float(esc.continuous_rating_A), "max")
            _insert_status_row(motor_table, "ESC current / ESC", f"{Iesc:.2f} A", f"<= {float(esc.continuous_rating_A):.2f} A (cont)", tag_ec)
            tag_em = _status_color_tag(Iesc, float(esc.max_current_A), "max")
            _insert_status_row(motor_table, "ESC current / ESC (max)", f"{Iesc:.2f} A", f"<= {float(esc.max_current_A):.2f} A (max)", tag_em)
            Pesc_loss = float(metrics.get("esc_loss_W", 0.0))
            _insert_status_row(motor_table, "ESC loss (total)", f"{Pesc_loss:.0f} W", "", "na")
            note = str(metrics.get("esc_note", ""))
            if note.strip():
                _insert_status_row(motor_table, "ESC note", "", "", "na", note=note)

        # ---- Propeller ----
        thrust_per_N = float(metrics.get("thrust_per_motor_N", 0.0))
        thrust_per_g = thrust_per_N * 1000.0 / 9.81
        max_thrust_g = float(getattr(config.propeller, "max_thrust_g", 0.0))
        if max_thrust_g > 0:
            tag = _status_color_tag(thrust_per_g, max_thrust_g, "max")
            _insert_status_row(prop_table_tv, "Thrust / motor", f"{thrust_per_g:.0f} g", f"<= {max_thrust_g:.0f} g", tag)
        else:
            _insert_status_row(prop_table_tv, "Thrust / motor", f"{thrust_per_g:.0f} g", "n/a", "na")

        rpm = metrics.get("prop_rpm", None)
        if rpm is not None and float(rpm) > 0 and getattr(config.propeller, "max_rpm", None) is not None:
            tag = _status_color_tag(float(rpm), float(config.propeller.max_rpm), "max")
            _insert_status_row(prop_table_tv, "Prop RPM (est)", f"{float(rpm):.0f} rpm", f"<= {float(config.propeller.max_rpm):.0f} rpm", tag)
        else:
            # show max RPM anyway
            mr = getattr(config.propeller, "max_rpm", None)
            _insert_status_row(prop_table_tv, "Prop RPM (est)", "n/a", f"<= {float(mr):.0f} rpm" if mr is not None else "n/a", "na")

        # Any additional prop metrics user provided (dynamic)
        for k, v in sorted(metrics.items()):
            if not isinstance(k, str):
                continue
            if k.startswith("prop_"):
                base = k[len("prop_"):]
                max_attr = f"max_{base}"
                min_attr = f"min_{base}"
                if hasattr(config.propeller, max_attr):
                    L = getattr(config.propeller, max_attr)
                    if L is not None:
                        tag = _status_color_tag(float(v), float(L), "max")
                        _insert_status_row(prop_table_tv, base, f"{float(v):.3g}", f"<= {float(L):.3g}", tag)
                if hasattr(config.propeller, min_attr):
                    L = getattr(config.propeller, min_attr)
                    if L is not None:
                        tag = _status_color_tag(float(v), float(L), "min")
                        _insert_status_row(prop_table_tv, base, f"{float(v):.3g}", f">= {float(L):.3g}", tag)


    # ---------- Save/Load Configuration ----------
    # Collect all Tk variables named v_* so we can persist/restore the GUI state.
    config_vars = {k: v for k, v in locals().items() if k.startswith("v_") and isinstance(v, tk.Variable)}

    def _extract_avionics_rails_from_table() -> list[dict]:
        rails: list[dict] = []
        for iid in avionics_tree.get_children():
            v_str, i_str, e_str = avionics_tree.item(iid, "values")
            try:
                rails.append({
                    "voltage": float(v_str),
                    "current": float(i_str),
                    "eff": float(e_str),
                })
            except Exception:
                continue
        rails.sort(key=lambda r: r["voltage"])
        return rails

    def _populate_avionics_table_from_rails(rails: list[dict]) -> None:
        for iid in avionics_tree.get_children():
            avionics_tree.delete(iid)
        for r in sorted(rails, key=lambda x: float(x.get("voltage", 0.0))):
            try:
                v = float(r.get("voltage"))
                i = float(r.get("current"))
                e = float(r.get("eff"))
            except Exception:
                continue
            avionics_tree.insert("", "end", values=(f"{v:g}", f"{i:g}", f"{e:g}"))
        try:
            _sync_voltage_tree_var_from_table()
        except Exception:
            pass

    def save_config_to_file(path: str) -> None:
        data = {
            "schema": "multicopter_power_sim_gui_config",
            "version": 1,
            "vars": {k: v.get() for k, v in config_vars.items()},
            "avionics_rails": _extract_avionics_rails_from_table(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_config_from_file(path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vars_block = data.get("vars", {}) if isinstance(data, dict) else {}
        if isinstance(vars_block, dict):
            for k, val in vars_block.items():
                if k in config_vars:
                    try:
                        config_vars[k].set("" if val is None else str(val))
                    except Exception:
                        pass

        rails = data.get("avionics_rails", None) if isinstance(data, dict) else None
        if isinstance(rails, list):
            _populate_avionics_table_from_rails(rails)
        else:
            try:
                d = parse_avionics_voltage_tree(v_avionics_voltage_tree.get().strip())
                _populate_avionics_table_from_rails(
                    [{"voltage": v, "current": ci[0], "eff": ci[1]} for v, ci in d.items()]
                )
            except Exception:
                pass

        try:
            on_unit_mode_change()
        except Exception:
            pass

    def prompt_save_config():
        path = filedialog.asksaveasfilename(
            title="Save configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            save_config_to_file(path)
            messagebox.showinfo("Saved", f"Saved configuration to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error saving config", str(e))

    def prompt_load_config():
        path = filedialog.askopenfilename(
            title="Load configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            load_config_from_file(path)
            messagebox.showinfo("Loaded", f"Loaded configuration from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error loading config", str(e))

    # Wire menu items (File -> Load/Save)
    try:
        file_menu.add_command(label="Load Config...", command=prompt_load_config)
        file_menu.add_command(label="Save Config...", command=prompt_save_config)
        file_menu.add_separator()
    except Exception:
        pass

    # Buttons
    btns = ttk.Frame(main)
    btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    btns.columnconfigure(0, weight=1)

    # Matplotlib canvas placeholder
    canvas = None
    current_canvas = None
    current_fig = None

    def build_config_from_gui() -> DroneConfig:
        # Battery
        batt = BatteryConfig(
            chemistry=v_batt_chem.get().strip() or None,
            operating_voltage_min=parse_float("Vmin/cell", v_batt_vmin.get()),
            operating_voltage_nominal=parse_float("Vnom/cell", v_batt_vnom.get()),
            operating_voltage_max=parse_float("Vmax/cell", v_batt_vmax.get()),
            cell_capacity_mAh=parse_float("Cell capacity", v_batt_cell_capacity.get()),
            pack_capacity_mAh=parse_float("Pack capacity", v_batt_pack_capacity.get()),
            cell_weight_g=parse_float("Cell weight", v_batt_cell_weight.get()),
            pack_weight_g=parse_float("Pack weight", v_batt_pack_weight.get()),
            unit_energy_density=parse_float("Energy density", v_batt_energy_density.get()),
            charge_current_max=parse_float("Max charge current", v_batt_chg.get()),
            discharge_cont_A=parse_float("Cont discharge current", v_batt_a_cont.get()),
            discharge_max_A=parse_float("Max discharge current", v_batt_a_max.get()),
            discharge_c_cont=parse_float("Cont C-rate", v_batt_c_cont.get()),
            discharge_c_max=parse_float("Max C-rate", v_batt_c_max.get()),
            discharge_percent=parse_float("Discharge usable %", v_batt_dischg_pct.get()),
            resistance_cell_mOhm=parse_float("Rcell", v_batt_r.get()),
            series_units=parse_int("Series units", v_batt_series.get()),
            parallel_units=parse_int("Parallel units", v_batt_parallel.get()),
            cells_series_per_unit=parse_int("Cells in series per unit", v_batt_cells_series.get()),
            cells_parallel_per_unit=parse_int("Cells in parallel per unit", v_batt_cells_parallel.get()),
        )

        # Motor
        motor = MotorConfig(
            kv=parse_float("Kv", v_motor_kv.get()),
            idle_current=parse_float("Idle current", v_motor_i0.get()),
            idle_voltage=parse_float("Idle voltage", v_motor_v0.get()),
            rated_voltage=parse_int("Rated voltage", v_motor_rated_v.get()),
            resistance=parse_float("Motor resistance", v_motor_r.get()),
            max_current=parse_float("Motor max current", v_motor_imax.get()),
            max_power=parse_float("Motor max power", v_motor_pmax.get()),
            pole_count=parse_int("Motor pole count", v_motor_pole_count.get()),
            weight_g=parse_float("Motor weight", v_motor_weight.get()),
            size_mm=v_motor_size.get().strip() or None,
        )

        # ESC (optional)
        esc = None
        _esc_fields = [
            v_esc_voltage_rating.get().strip(),
            v_esc_cont_current.get().strip(),
            v_esc_max_current.get().strip(),
            v_esc_idle_current.get().strip(),
            v_esc_r.get().strip(),
            v_esc_weight.get().strip(),
        ]
        if any(_esc_fields):
            esc = ESCConfig(
                voltage_rating=parse_int("ESC voltage rating (S)", v_esc_voltage_rating.get()),
                continuous_current_A=parse_float("ESC continuous current", v_esc_cont_current.get()),
                max_current_A=parse_float("ESC max current", v_esc_max_current.get()),
                idle_current_A=parse_float("ESC idle current", v_esc_idle_current.get()),
                resistance=parse_float("ESC resistance", v_esc_r.get()),
                weight_g=parse_float("ESC weight", v_esc_weight.get()),
            )

        # Avionics
        avionics = AvionicsConfig(
            voltage_tree=_get_voltage_tree_from_table(),
        )

        # Prop
        prop_table = v_prop_table.get().strip() or None
        tconst = v_prop_tconst.get().strip()
        pconst = v_prop_pconst.get().strip()
        prop = PropellerConfig(
            diameter_in=parse_float("Prop diameter", v_prop_d.get()),
            pitch_in=parse_float("Prop pitch", v_prop_pitch.get()),
            max_rpm=parse_float("Prop max RPM", v_prop_max_rpm.get()),
            max_thrust_g=parse_float("Prop max thrust", v_prop_max_thrust.get()),
            blades=parse_int("Prop blades", v_prop_blades.get()),
            table_csv=prop_table,
            TConst=float(tconst) if tconst else None,
            PConst=float(pconst) if pconst else None,
            weight_g=parse_float("Prop weight", v_prop_weight.get()),
        )

        drone = DroneConfig(
            num_motors=parse_int("Num motors", v_num_motors.get()),
            battery=batt,
            motor=motor,
            propeller=prop,
            drone_weight_g=parse_float("Weight", v_weight.get()),
            profile_drag_coefficient=(parse_float("Profile Cd", v_profile_drag.get()) if v_profile_drag.get().strip() else 0.0),
            profile_area=(parse_float("Profile area", v_profile_area.get()) if v_profile_area.get().strip() else 0.0),
            parasite_drag_coefficient=(parse_float("Parasite Cd", v_parasite_drag.get()) if v_parasite_drag.get().strip() else 0.0),
            parasite_area=(parse_float("Parasite area", v_parasite_area.get()) if v_parasite_area.get().strip() else 0.0),
            frontal_area=(parse_float("Frontal area", v_area.get()) if v_area.get().strip() else 0.0),
            cruise_speed=parse_float("Cruise speed", v_speed.get()),
            periph_current=parse_float("Peripheral current", v_periph_current.get()),
            esc=esc,
            avionics=avionics,
            air_density=AIR_DENSITY,
            body_length_m=(parse_float("Body length", v_body_length_m.get()) if v_body_length_m.get().strip() else None),
            body_width_m=(parse_float("Body width", v_body_width_m.get()) if v_body_width_m.get().strip() else None),
            body_height_m=(parse_float("Body height", v_body_height_m.get()) if v_body_height_m.get().strip() else None),
            arm_length_m=(parse_float("Arm length", v_arm_length_m.get()) if v_arm_length_m.get().strip() else None),
            arm_width_m=(parse_float("Arm width", v_arm_width_m.get()) if v_arm_width_m.get().strip() else None),
            coaxial_spacing_m=(parse_float("Coaxial spacing", v_coaxial_spacing_m.get()) if v_coaxial_spacing_m.get().strip() else None),
            max_tilt_deg=(parse_float("Max tilt", v_max_tilt_deg.get()) if v_max_tilt_deg.get().strip() else None),
            motor_configuration=(v_motor_configuration.get().strip().lower() or "flat"),
        )

        # Env initial density
        alt = parse_float("Altitude", v_alt.get())
        temp = v_temp.get().strip()
        pres = v_press.get().strip()
        drone.air_density = compute_air_density(
            altitude_m=alt,
            temperature_C=float(temp) if temp else None,
            pressure_Pa=float(pres) if pres else None,
        )
        return drone

    def show_figure(fig):
        nonlocal canvas
        # Clear old
        for w in plot_frame.winfo_children():
            w.destroy()
        nonlocal current_canvas, current_fig
        current_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        current_canvas = canvas
        try:
            _apply_matplotlib_scale_to_fig(fig, float(ui_scale.get()))
        except Exception:
            pass
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")


    def _estimate_prop_rpm_if_possible(drone: DroneConfig, thrust_per_motor_N: float) -> Optional[float]:
        """Estimate prop RPM if we have TConst available (same model used in motor_power_from_params).

        Returns None if not computable.
        """
        try:
            if not (drone.propeller.TConst and drone.propeller.PConst):
                return None
            D = drone.propeller.diameter_in * 0.0254
            rho = drone.air_density
            low, high = 100.0, 40000.0
            rpm_solution = None
            for _ in range(40):
                mid = 0.5 * (low + high)
                n = mid / 60.0
                thrust = float(drone.propeller.TConst) * rho * (n**2) * (D**4)
                if thrust < thrust_per_motor_N:
                    low = mid
                else:
                    high = mid
                    rpm_solution = mid
            return float(rpm_solution) if rpm_solution is not None else None
        except Exception:
            return None

    def compute_operating_metrics(drone: DroneConfig, speed_mps: float, orientation: str, wind_mps: float = 0.0) -> dict:
        """Compute a consistent set of metrics for status/limit checking at an operating point."""
        airspeed = float(speed_mps)
        # simple wind model: +wind is headwind -> higher airspeed for same groundspeed
        if orientation == "forward":
            airspeed = max(0.0, float(speed_mps) + float(wind_mps))
        motor_power_W = power_required(drone, airspeed, orientation)
        periph_power_W = avionics_input_power_W(getattr(drone, "avionics", None))
        if periph_power_W <= 0.0:
            periph_power_W = drone.battery.vnom_pack * max(drone.periph_current, 0.0)
        total_power_W, v_load, pack_current_A, esc_note, motor_I_esc_A = total_power_with_esc(
            drone,
            motor_power_W=motor_power_W,
            periph_power_W=periph_power_W,
        )
        total_thrust_N = thrust_required(drone, airspeed, orientation)
        thrust_per_motor_N = total_thrust_N / max(int(drone.num_motors), 1)
        # ESC loss is total_power - motor - periph
        esc_loss_W = max(0.0, float(total_power_W) - float(motor_power_W) - float(periph_power_W))
        rpm_est = _estimate_prop_rpm_if_possible(drone, thrust_per_motor_N)

        motor_table = None

        if getattr(drone, 'propeller', None) is not None and drone.propeller.table is not None:

            try:

                motor_table = interpolate_motor_point(drone, thrust_per_motor_N)

                if 'RPM' in motor_table:

                    rpm_est = float(motor_table['RPM'])

            except Exception:

                motor_table = None
        tilt_req = required_tilt_deg(drone, airspeed, orientation)
        return {
            "airspeed_mps": float(airspeed),
            "tilt_required_deg": float(tilt_req),
            "tilt_limit_deg": (float(drone.max_tilt_deg) if getattr(drone, 'max_tilt_deg', None) is not None else None),
            "motor_power_W": float(motor_power_W),
            "periph_power_W": float(periph_power_W),
            "total_power_W": float(total_power_W),
            "v_load_V": float(v_load),
            "pack_current_A": float(pack_current_A),
            "esc_loss_W": float(esc_loss_W),
            "esc_note": str(esc_note),
            "motor_I_per_esc_A": float(motor_I_esc_A),
            "thrust_total_N": float(total_thrust_N),
            "thrust_per_motor_N": float(thrust_per_motor_N),
            "prop_rpm": (float(rpm_est) if rpm_est is not None else None),
            "motor_table_throttle_pct": (float(motor_table["Throttle_pct"]) if motor_table and "Throttle_pct" in motor_table else None),
            "motor_table_eff_gW": (float(motor_table["Efficiency_gW"]) if motor_table and "Efficiency_gW" in motor_table else None),
            "motor_table_temp_C": (float(motor_table["Temp_C"]) if motor_table and "Temp_C" in motor_table else None),
            "motor_table_voltage_V": (float(motor_table["Voltage_V"]) if motor_table and "Voltage_V" in motor_table else None),
            "motor_table_current_A": (float(motor_table["Current_A"]) if motor_table and "Current_A" in motor_table else None),
        }
    

    def run_single_point():
        clear_log()
        try:
            drone = build_config_from_gui()
            orientation = v_orientation.get().strip().lower()
            if orientation not in ("hover", "forward"):
                raise ValueError("Orientation must be 'hover' or 'forward'.")

            speed = parse_float("Speed", v_speed.get())
            t_min = estimate_flight_time_minutes(drone, speed, orientation=orientation)
            d_km = estimate_flight_distance_km(drone, speed, orientation=orientation)
            be_v, be_min, br_v, br_km = find_optimal_speeds(drone)

            log(f"Air density: {drone.air_density:.3f} kg/m^3")
            log(f"At {speed:.2f} m/s ({orientation}): time={t_min:.2f} min, distance={d_km:.2f} km")
            log(f"Best endurance (forward): {be_v:.2f} m/s -> {be_min:.2f} min")
            log(f"Best range (forward): {br_v:.2f} m/s -> {br_km:.2f} km")

            metrics = compute_operating_metrics(drone, speed_mps=speed, orientation=orientation, wind_mps=0.0)
            update_status_tables_from_metrics(drone, metrics)
            update_metrics_tab(drone, metrics, speed_mps=speed, orientation=orientation)

            fig = make_performance_figure(drone, max_speed=parse_float("Max speed plot", v_max_speed_plot.get()))
            show_figure(fig)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_mission():
        clear_log()
        try:
            mission_path = v_mission.get().strip()
            if not mission_path:
                raise ValueError("Select a mission JSON file (or switch to 'Single-point' run).")

            drone = build_config_from_gui()
            orientation = v_orientation.get().strip().lower()
            if orientation not in ("hover", "forward"):
                raise ValueError("Orientation must be 'hover' or 'forward'.")

            temp = v_temp.get().strip()
            pres = v_press.get().strip()
            wind = parse_float("Wind", v_wind.get())

            mission = MissionProfile.from_json(mission_path)
            results, worst_metrics, mission_series = simulate_mission(
                drone,
                mission,
                orientation=orientation,
                temperature_C=float(temp) if temp else None,
                pressure_Pa=float(pres) if pres else None,
                wind_mps=wind,
            )

            log(f"Mission: {os.path.basename(mission_path)}")
            log(f"Orientation: {orientation} | Wind: {wind:.2f} m/s")
            log("")
            total_time = 0.0
            total_dist = 0.0
            for name, t_min, d_km, status in results:
                total_time += t_min
                total_dist += d_km
                log(f"{name}: {t_min:.2f} min, {d_km:.2f} km, {status}")

            log("")
            log(f"TOTAL: {total_time:.2f} min, {total_dist:.2f} km")

            if worst_metrics is not None:
                update_status_tables_from_metrics(drone, worst_metrics)

            # Store mission time series for Mission Plots tab
            try:
                nonlocal last_mission_series
                last_mission_series = mission_series
            except Exception:
                pass

            fig = make_performance_figure(drone, max_speed=parse_float("Max speed plot", v_max_speed_plot.get()))
            show_figure(fig)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(btns, text="Run single-point + plot", command=run_single_point).grid(row=0, column=0, padx=6, pady=2, sticky="w")
    ttk.Button(btns, text="Run mission (JSON) + plot", command=run_mission).grid(row=0, column=1, padx=6, pady=2, sticky="w")
    ttk.Button(btns, text="Load config", command=prompt_load_config).grid(row=0, column=2, padx=6, pady=2, sticky="w")
    ttk.Button(btns, text="Save config", command=prompt_save_config).grid(row=0, column=3, padx=6, pady=2, sticky="w")
    ttk.Button(btns, text="Clear output", command=clear_log).grid(row=0, column=4, padx=6, pady=2, sticky="w")
    ttk.Button(btns, text="Quit", command=exit_app).grid(row=0, column=5, padx=6, pady=2, sticky="e")

    root.protocol("WM_DELETE_WINDOW", exit_app)

    # initial empty fig
    try:
        drone0 = build_config_from_gui()
        fig0 = make_performance_figure(drone0, max_speed=30)
        show_figure(fig0)
    except Exception:
        pass

    root.minsize(1050, 650)
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
