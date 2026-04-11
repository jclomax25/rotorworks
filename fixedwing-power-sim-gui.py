"""
fixedwing-power-sim-gui.py
--------------------------
Fixed-Wing UAV performance simulator.

Physics backend is fully rewritten for fixed-wing aerodynamics:
  - Lift / Drag polar model (CD0 + k·CL²)
  - Stall speed, pitch speed, tip speed
  - Angle of attack at cruise
  - Reynolds number
  - Available thrust and specific thrust
  - Rate of climb (instantaneous and max)
  - Max angle of climb
  - Best endurance speed (max L^1.5/D)
  - Best range speed     (max L/D)
  - Takeoff ground roll distance
  - Max propeller power check
  - Flight time and range vs speed curves

Battery / Motor / ESC / Avionics / Propeller models are identical to
the multicopter sim so the same hardware specs can be entered.

Modes:
  - Motor electrical model  (KV, Rm, I0, limits)
  - Motor/prop test table   (CSV with Thrust_g / Power_W columns)
  - Theoretical actuator    (ideal momentum + efficiency)

GUI: Tkinter notebook with tabs:
  Airframe | Battery | Motor | ESC | Avionics | Propeller | Environment
Output panels (right side):
  Plots | Status | Metrics | Mission Plots

CLI usage (all units SI unless noted):
  python fixedwing-power-sim-gui.py --gui
  python fixedwing-power-sim-gui.py --weight 2.5 --wing_area 0.45 --wing_span 1.6 \\
      --CD0 0.028 --CL_max 1.3 --oswald 0.82 \\
      --battery_operating_voltage_min 3.5 --battery_operating_voltage_max 4.2 \\
      --battery_operating_voltage_nominal 3.8 --battery_series_units 4 \\
      --battery_cell_capacity 5000 --battery_cell_weight_g 75 \\
      --battery_charge_current_max 5 --battery_discharge_c_cont 25 \\
      --battery_resistance_cell 5 \\
      --motor_kv 920 --motor_idle_current 0.8 --motor_idle_voltage 7.0 \\
      --motor_rated_voltage 16 --motor_resistance 0.06 \\
      --motor_max_current 40 --motor_max_power 500 \\
      --prop_diameter 10 --prop_pitch 4.5 \\
      --plot
"""

from __future__ import annotations

import math
import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# FigureCanvasTkAgg is imported lazily inside launch_gui() so that
# the physics engine can be used on headless / no-Tkinter systems.

# ============================================================
# CONSTANTS — ISA (International Standard Atmosphere)
# ============================================================
RHO0    = 1.225       # kg/m³  — sea-level air density
R_AIR   = 287.05      # J/kg/K — specific gas constant for dry air
T0      = 288.15      # K      — sea-level temperature (15 °C)
P0      = 101325.0    # Pa     — sea-level pressure
L_LAPSE = 0.0065      # K/m    — temperature lapse rate (troposphere)
G0      = 9.80665     # m/s²   — standard gravity
MU_AIR  = 1.789e-5    # Pa·s   — dynamic viscosity of air at 15 °C (≈ constant for subsonic UAV)


def isa_density(altitude_m: float,
                temperature_C: Optional[float] = None,
                pressure_Pa: Optional[float] = None) -> float:
    """
    Compute air density ρ [kg/m³] from ISA or user-supplied conditions.

    ISA temperature profile (troposphere, h < 11 000 m):
        T(h) = T0 - L·h                  [K]

    ISA pressure profile:
        P(h) = P0·(T(h)/T0)^(g/(R·L))   [Pa]

    Density from ideal gas law:
        ρ = P / (R·T)                    [kg/m³]
    """
    if pressure_Pa is not None and temperature_C is not None:
        T_K = float(temperature_C) + 273.15
        return float(pressure_Pa) / (R_AIR * T_K)

    h = max(float(altitude_m), 0.0)
    T_isa = T0 - L_LAPSE * h                       # K
    P_isa = P0 * (T_isa / T0) ** (G0 / (R_AIR * L_LAPSE))  # Pa
    T_K   = T_isa if temperature_C is None else (float(temperature_C) + 273.15)
    return P_isa / (R_AIR * T_K)


# ============================================================
# BATTERY MODEL  (identical to multicopter sim)
# ============================================================
class BatteryConfig:
    """
    Li-polymer / Li-Ion battery pack model.

    Pack voltage layout:
        V_pack  = V_cell × N_series_cells

    Under-load voltage (Thevenin model):
        V_load = V_max_pack - I_pack × R_pack
        where R_pack = R_cell_series × N_series / N_parallel  [Ω]

    Usable energy:
        E_usable = C_pack_Ah × V_nom_pack × (discharge_percent / 100)  [Wh]
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
        self.operating_voltage_min      = float(operating_voltage_min)
        self.operating_voltage_nominal  = float(operating_voltage_nominal)
        self.operating_voltage_max      = float(operating_voltage_max)
        self.unit_mode = str(unit_mode).strip().lower() if unit_mode is not None else "cell"

        if self.unit_mode not in ("cell", "pack"):
            self.unit_mode = "pack" if pack_weight_g is not None else "cell"

        if self.unit_mode == "cell":
            cells_series_per_unit   = 1
            cells_parallel_per_unit = 1

        self.cells_series_per_unit   = int(cells_series_per_unit)
        self.cells_parallel_per_unit = int(cells_parallel_per_unit)
        self.series_units   = int(series_units)
        self.parallel_units = int(parallel_units)

        self.series_cells   = self.series_units   * self.cells_series_per_unit
        self.parallel_cells = self.parallel_units * self.cells_parallel_per_unit
        self.total_cells    = self.series_cells   * self.parallel_cells

        self.vmin_pack  = self.operating_voltage_min     * self.series_cells
        self.vnom_pack  = self.operating_voltage_nominal * self.series_cells
        self.vmax_pack  = self.operating_voltage_max     * self.series_cells

        self.pack_weight_g    = float(pack_weight_g)    if pack_weight_g    is not None else None
        self.cell_weight_g    = float(cell_weight_g)    if cell_weight_g    is not None else None
        self.cell_capacity_mAh = float(cell_capacity_mAh) if cell_capacity_mAh is not None else None
        self.pack_capacity_mAh = float(pack_capacity_mAh) if pack_capacity_mAh is not None else None

        # Effective pack capacity in mAh
        if self.unit_mode == "cell":
            self.capacity_mAh = (self.cell_capacity_mAh or 0.0) * self.parallel_cells
        else:
            self.capacity_mAh = (self.pack_capacity_mAh or 0.0) * self.parallel_units * self.series_units
        self.capacity_Ah = self.capacity_mAh / 1000.0

        # Weight
        if self.unit_mode == "cell":
            self.weight_g = (self.cell_weight_g or 0.0) * self.total_cells
        else:
            self.weight_g = (self.pack_weight_g or 0.0) * self.series_units * self.parallel_units

        # Derived energy density  [Wh/kg]
        if unit_energy_density is not None:
            self.energy_density_Wh_per_kg = float(unit_energy_density)
        else:
            wkg = self.weight_g / 1000.0
            self.energy_density_Wh_per_kg = (self.capacity_Wh / wkg) if wkg > 0 else 0.0

        self.charge_current_max = float(charge_current_max) if charge_current_max is not None else 0.0

        # Discharge limits — prefer explicit A, fall back to C-rate × capacity
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

        self.discharge_c_cont = (discharge_c_cont if discharge_c_cont is not None
                                 else (self.discharge_cont_A / self.capacity_Ah if self.capacity_Ah > 0 else None))
        self.discharge_c_max  = (discharge_c_max  if discharge_c_max  is not None
                                 else (self.discharge_max_A  / self.capacity_Ah if self.capacity_Ah > 0 else None))

        self.discharge_percent = min(max(float(discharge_percent), 0.0), 100.0)
        self.usable_fraction   = self.discharge_percent / 100.0

        # Internal resistance  [Ω per pack]
        self.resistance_cell = float(resistance_cell_mOhm) / 1000.0   # Ω per cell

    @property
    def pack_resistance(self) -> float:
        """R_pack = R_cell × N_series / N_parallel  [Ω]"""
        return (self.resistance_cell * self.series_cells
                / max(self.parallel_cells, 1))

    @property
    def capacity_Wh(self) -> float:
        return self.capacity_Ah * self.vnom_pack

    @property
    def usable_Wh(self) -> float:
        return self.capacity_Wh * self.usable_fraction

    def voltage_under_load(self, current_A: float) -> float:
        """V_load = V_max - I × R_pack, clamped at V_min."""
        v = self.vmax_pack - float(current_A) * self.pack_resistance
        return max(v, self.vmin_pack)


# ============================================================
# MOTOR MODEL
# ============================================================
class MotorConfig:
    """
    Brushless motor electrical model.

    Back-EMF constant:
        K_e = 1 / KV  [V·s/rad] when KV is in RPM/V

    Shaft power (mechanical):
        P_shaft = (V_motor - I·Rm) × I - I0 × V0
               = η_motor × V_motor × I  (approximate)

    More precisely, using KV model:
        ω  [rad/s] = KV × 2π/60 × V_back_emf
        V_back_emf = V_motor - I × Rm
        I0 = no-load current at idle voltage V0

    We store the raw params and compute shaft power in the physics layer.
    """
    def __init__(self,
                 kv:           Optional[float],
                 idle_current:  float,
                 idle_voltage:  float,
                 rated_voltage: int,
                 resistance:    float,
                 max_current:   float,
                 max_power:     float,
                 pole_count:    Optional[int]   = None,
                 weight_g:      Optional[float] = None,
                 size_mm:       Optional[str]   = None):
        self.kv           = None if kv is None else float(kv)
        self.idle_current = float(idle_current)
        self.idle_voltage = float(idle_voltage)
        self.resistance   = float(resistance)
        self.max_current  = float(max_current)
        self.max_power    = float(max_power)
        self.rated_voltage= int(rated_voltage)
        self.pole_count   = pole_count
        self.weight_g     = weight_g
        self.size_mm      = size_mm


# ============================================================
# ESC MODEL
# ============================================================
class ESCConfig:
    """
    Electronic Speed Controller model.
    Losses:  P_loss = I² × R_esc + I_idle × V  [W per ESC]
    """
    def __init__(self,
                 voltage_rating:      int,
                 continuous_current_A: float,
                 max_current_A:        float,
                 idle_current_A:       float,
                 resistance:           float,
                 weight_g: Optional[float] = None):
        self.voltage_rating     = int(voltage_rating)
        self.continuous_rating_A = float(continuous_current_A)
        self.max_current_A      = float(max_current_A)
        self.idle_current_A     = float(idle_current_A)
        self.resistance         = float(resistance)
        self.weight_g           = weight_g


# ============================================================
# AVIONICS / PERIPHERALS
# ============================================================
class AvionicsConfig:
    """
    BEC-regulated avionics loads.
    voltage_tree: {V_rail: (I_rail [A], BEC_efficiency)} dict

    Input power from battery for each rail:
        P_in = (V_rail × I_rail) / η_bec   [W]
    """
    def __init__(self, voltage_tree: Optional[dict] = None):
        self.voltage_tree = voltage_tree or {}


def avionics_input_power_W(avionics: Optional[AvionicsConfig]) -> float:
    if avionics is None or not avionics.voltage_tree:
        return 0.0
    total = 0.0
    for v, (i, eff) in avionics.voltage_tree.items():
        total += (float(v) * float(i)) / max(float(eff), 1e-9)
    return total


def parse_voltage_tree(spec: Optional[str]) -> dict:
    """Parse 'V:(I,eff), V2:(I2,eff2)' or 'V:I:eff, ...' string.

    Splits on commas that are NOT inside parentheses so that the
    inner 'I,eff' tuple survives the tokenisation step.
    """
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {float(k): (float(v[0]), float(v[1])) for k, v in spec.items()}

    # Split on commas outside parentheses
    parts, depth, buf = [], 0, ""
    for ch in str(spec):
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

    out: Dict[float, Tuple[float, float]] = {}
    for p in parts:
        if not p:
            continue
        m = re.match(r"^([0-9.]+)\s*:\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)$", p)
        if m:
            v, i, e = map(float, m.groups())
        else:
            m2 = re.match(r"^([0-9.]+)\s*:\s*([0-9.]+)\s*:\s*([0-9.]+)$", p)
            if not m2:
                raise ValueError(f"Bad avionics spec token: {p!r}")
            v, i, e = map(float, m2.groups())
        out[v] = (i, e)
    return out


# ============================================================
# PROPELLER MODEL
# ============================================================
def load_prop_table(path: str) -> pd.DataFrame:
    """
    Load motor/prop test CSV. Accepts eCalc-style and simple headers.
    Required columns after renaming: Thrust_g, Power_W
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        df0 = pd.read_csv(path)
    except Exception:
        df0 = pd.DataFrame()

    def looks_ok(df):
        cols = {str(c).strip().lower() for c in df.columns}
        return "thrust_g" in cols or "thrust (g)" in cols

    if len(df0.columns) > 1 and looks_ok(df0):
        df = df0
    else:
        raw = pd.read_csv(path, header=None)
        header_row = None
        for i in range(min(len(raw), 25)):
            row = raw.iloc[i].astype(str).str.lower().tolist()
            if any("thrust" in x for x in row) and any("power" in x for x in row):
                header_row = i
                break
        if header_row is None:
            raise ValueError(f"Cannot locate header row in {path}")
        df = pd.read_csv(path, header=header_row)

    rmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("thrust_g", "thrust (g)"): rmap[c] = "Thrust_g"
        elif cl in ("power_w",  "power (w)"): rmap[c] = "Power_W"
        elif cl in ("current_a","current (a)"): rmap[c] = "Current_A"
        elif cl in ("voltage_v","voltage (v)"): rmap[c] = "Voltage_V"
        elif cl == "rpm":                       rmap[c] = "RPM"
        elif cl.startswith("throttle"):         rmap[c] = "Throttle_pct"
        elif "efficiency" in cl:                rmap[c] = "Efficiency_gW"
    df = df.rename(columns=rmap)

    for col in ("Thrust_g","Power_W","Current_A","Voltage_V","RPM","Efficiency_gW","Throttle_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%","",regex=False).str.strip(), errors="coerce")

    if "Thrust_g" not in df.columns or "Power_W" not in df.columns:
        raise ValueError("CSV must have Thrust and Power columns.")

    df = (df.dropna(subset=["Thrust_g","Power_W"])
            .sort_values("Thrust_g")
            .drop_duplicates(subset=["Thrust_g"], keep="last")
            .reset_index(drop=True))
    return df


class PropellerConfig:
    """
    Propeller geometry used for:
      - Pitch speed  = pitch_m × n  [m/s]  (speed at advance ratio J = 1)
      - Tip speed    = π × D × n   [m/s]  where n [rev/s]
      - Thrust / power from KV-based or table-based motor model
    """
    def __init__(self,
                 diameter_in:  float,
                 pitch_in:     float,
                 blades:       int   = 2,
                 max_rpm:      float = 0.0,
                 max_thrust_g: float = 0.0,
                 table_csv:    Optional[str]   = None,
                 TConst:       Optional[float] = None,
                 PConst:       Optional[float] = None,
                 weight_g:     Optional[float] = None):
        self.diameter_in  = float(diameter_in)
        self.pitch_in     = float(pitch_in)
        self.blades       = int(blades)
        self.max_rpm      = float(max_rpm)
        self.max_thrust_g = float(max_thrust_g)
        self.TConst       = TConst
        self.PConst       = PConst
        self.weight_g     = float(weight_g) if weight_g is not None else 20.0
        self.table: Optional[pd.DataFrame] = None
        if table_csv:
            self.table = load_prop_table(table_csv)

    @property
    def diameter_m(self) -> float:
        return self.diameter_in * 0.0254

    @property
    def pitch_m(self) -> float:
        return self.pitch_in * 0.0254

    def disk_area(self) -> float:
        """Actuator disk area  A = π/4 · D²  [m²]"""
        return math.pi / 4.0 * self.diameter_m ** 2


# ============================================================
# AIRFRAME CONFIG  (fixed-wing specific)
# ============================================================
class AirframeConfig:
    """
    Fixed-wing aerodynamic model using a simple drag polar:

        CD = CD0 + k · CL²

    where:
        k = 1 / (π · AR · e)      (induced drag factor)
        AR = b² / S                (aspect ratio)
        e  = Oswald efficiency     (typically 0.75 – 0.90)

    Lift and drag:
        L = ½ ρ V² S CL
        D = ½ ρ V² S CD

    In straight-level flight L = W, so:
        CL = 2W / (ρ V² S)

    Stall occurs when CL = CL_max:
        V_stall = sqrt(2W / (ρ S CL_max))

    Angle of attack at cruise speed V (relative to zero-lift line):
        AoA = CL / a0  where a0 = lift-curve slope ≈ 2π per radian
        (corrected for AR via Prandtl lifting-line theory)
    """
    def __init__(self,
                 wing_span_m:    float,
                 wing_area_m2:   float,
                 CD0:            float  = 0.030,
                 CL_max:         float  = 1.30,
                 oswald:         float  = 0.80,
                 mu_roll:        float  = 0.04,
                 CL_takeoff:     float  = 0.80,
                 prop_efficiency: float = 0.75,
                 num_motors:     int    = 1):
        self.wing_span_m   = float(wing_span_m)
        self.wing_area_m2  = float(wing_area_m2)
        self.CD0           = float(CD0)
        self.CL_max        = float(CL_max)
        self.oswald        = float(oswald)
        self.mu_roll       = float(mu_roll)       # rolling friction coefficient
        self.CL_takeoff    = float(CL_takeoff)    # CL at take-off rotation
        self.prop_efficiency = float(prop_efficiency)  # η_prop for thrust from power
        self.num_motors    = int(num_motors)

    @property
    def chord_m(self) -> float:
        """Mean aerodynamic chord  c = S / b  [m]"""
        return self.wing_area_m2 / max(self.wing_span_m, 1e-6)

    @property
    def aspect_ratio(self) -> float:
        """AR = b² / S"""
        return self.wing_span_m ** 2 / max(self.wing_area_m2, 1e-6)

    @property
    def k(self) -> float:
        """Induced drag factor  k = 1/(π·AR·e)"""
        return 1.0 / (math.pi * self.aspect_ratio * max(self.oswald, 1e-6))

    def lift_curve_slope(self) -> float:
        """
        Finite-wing lift-curve slope using Prandtl lifting-line theory:
            a = a0 / (1 + a0 / (π · AR))
        where a0 = 2π [1/rad] (thin-aerofoil theory).
        Returns slope in [1/rad].
        """
        a0 = 2.0 * math.pi
        return a0 / (1.0 + a0 / (math.pi * self.aspect_ratio))

    def reynolds_number(self, speed_mps: float, rho: float) -> float:
        """
        Re = ρ · V · c / μ
        Uses chord as characteristic length.
        """
        return rho * speed_mps * self.chord_m / MU_AIR

    def cl_at_speed(self, weight_N: float, speed_mps: float, rho: float) -> float:
        """
        Lift coefficient in straight-level flight (L = W):
            CL = 2W / (ρ V² S)
        """
        q = 0.5 * rho * speed_mps ** 2
        return weight_N / (q * self.wing_area_m2) if (q * self.wing_area_m2) > 0 else 0.0

    def cd_at_cl(self, CL: float) -> float:
        """Drag polar:  CD = CD0 + k · CL²"""
        return self.CD0 + self.k * CL ** 2

    def ld_ratio(self, CL: float) -> float:
        """Lift-to-drag ratio = CL / CD"""
        cd = self.cd_at_cl(CL)
        return CL / cd if cd > 0 else 0.0

    def aoa_deg(self, CL: float) -> float:
        """
        Angle of attack above zero-lift line [degrees]:
            α = CL / a   where a = finite-wing lift-curve slope [1/rad]
        """
        a = self.lift_curve_slope()
        return math.degrees(CL / a) if a > 0 else 0.0


# ============================================================
# FIXED-WING AIRCRAFT CONFIG
# ============================================================
class FixedWingConfig:
    """
    Top-level configuration object passed to all physics functions.
    Combines airframe, propulsion, and battery specs.
    """
    def __init__(self,
                 airframe:        AirframeConfig,
                 battery:         BatteryConfig,
                 motor:           MotorConfig,
                 propeller:       PropellerConfig,
                 aircraft_weight_g: float,
                 cruise_speed_mps:  float,
                 periph_current_A:  float   = 0.0,
                 esc:             Optional[ESCConfig]      = None,
                 avionics:        Optional[AvionicsConfig] = None,
                 air_density:     float     = RHO0):
        self.airframe          = airframe
        self.battery           = battery
        self.motor             = motor
        self.propeller         = propeller
        self.aircraft_weight_g = float(aircraft_weight_g)
        self.cruise_speed_mps  = float(cruise_speed_mps)
        self.periph_current_A  = float(periph_current_A)
        self.esc               = esc
        self.avionics          = avionics
        self.air_density       = float(air_density)
        # Alias for shared code
        self.num_motors        = airframe.num_motors

    @property
    def weight_N(self) -> float:
        """Convert total aircraft weight from grams to Newtons."""
        return (self.aircraft_weight_g / 1000.0) * G0


# ============================================================
# PROPULSION PHYSICS
# ============================================================
def rpm_from_kv_and_throttle(motor: MotorConfig, v_pack: float, throttle: float = 1.0) -> float:
    """
    Approximate no-load RPM at a given throttle fraction:
        RPM_no_load = KV × V_pack × throttle
    """
    if motor.kv is None:
        return 0.0
    return motor.kv * v_pack * max(0.0, min(1.0, throttle))


def prop_thrust_from_rpm(propeller: PropellerConfig, rpm: float, rho: float) -> float:
    """
    Simple momentum-disk thrust estimate from RPM:

        T = CT · ρ · n² · D⁴

    where n = RPM/60 [rev/s] and CT is estimated from pitch ratio.

    Approximation for a standard fixed-pitch prop:
        CT ≈ 0.12 × (pitch/diameter) × (1 – advance_ratio)
    clamped above zero.

    If TConst is supplied, CT is taken directly from it.
    """
    if rpm <= 0:
        return 0.0
    n  = rpm / 60.0          # rev/s
    D  = propeller.diameter_m
    D4 = D ** 4

    if propeller.TConst is not None:
        CT = float(propeller.TConst)
    else:
        # Empirical estimate from pitch/diameter ratio
        ratio = propeller.pitch_m / max(D, 1e-6)
        CT = 0.10 * ratio            # rough approximation

    T = CT * rho * (n ** 2) * D4
    return max(T, 0.0)


def prop_power_from_rpm(propeller: PropellerConfig, rpm: float, rho: float) -> float:
    """
    Propeller shaft power from RPM using power coefficient:

        P_shaft = CP · ρ · n³ · D⁵

    CP is either user-supplied (PConst) or estimated from CT via the
    figure-of-merit relation:  CP ≈ CT × π × pitch / D  (dimensional)
    """
    if rpm <= 0:
        return 0.0
    n  = rpm / 60.0
    D  = propeller.diameter_m

    if propeller.PConst is not None:
        CP = float(propeller.PConst)
    else:
        ratio = propeller.pitch_m / max(D, 1e-6)
        # Approximate: CP ≈ 0.045 × (pitch/diameter) roughly ties with CT model
        CP = 0.045 * ratio

    P = CP * rho * (n ** 3) * D ** 5
    return max(P, 0.0)


def motor_shaft_power_from_thrust(config: FixedWingConfig, thrust_N: float) -> float:
    """
    Electrical input power to the motor to produce required thrust,
    using actuator-disk theory + motor efficiency:

    Step 1: ideal induced velocity from momentum theory
        T = ṁ · Δv  →  T = 2 · ρ · A · vi²
        vi = sqrt(T / (2 · ρ · A))

    Step 2: ideal shaft power
        P_ideal = T · vi

    Step 3: account for motor/prop efficiency
        P_electrical = P_ideal / (η_motor × η_prop)

    If a KV-based motor model is available we invert the RPM-thrust
    relationship using the stored prop thrust model instead.
    """
    prop = config.propeller
    rho  = config.air_density

    if prop.table is not None:
        # --- Test-table mode: interpolate Power_W from Thrust_g with extrapolation ---
        thrust_g = thrust_N * 1000.0 / G0
        df       = prop.table
        min_thrust = float(df["Thrust_g"].iloc[0])
        max_thrust = float(df["Thrust_g"].iloc[-1])
        
        # Below minimum: try to extrapolate
        if thrust_g < min_thrust:
            extrapolated = _extrapolate_motor_value_fw(df, thrust_g, "Power_W")
            if extrapolated is not None:
                return float(extrapolated)
            else:
                return float(df["Power_W"].iloc[0])
        
        # Above maximum: use last value
        if thrust_g >= max_thrust:
            return float(df["Power_W"].iloc[-1])
        
        # Within range: linear interpolation
        return float(pd.Series(df["Thrust_g"]).searchsorted(thrust_g) and
                     _interp1d(df["Thrust_g"].values, df["Power_W"].values, thrust_g))

    if config.motor.kv is not None:
        # --- KV-based electrical model ---
        #
        # Motor power balance:
        #   P_shaft = (V_motor - I·Rm) × I  - I0·V0
        #
        # We use the actuator-disk ideal power and back-calculate the
        # electrical input through motor + prop efficiency.
        A      = prop.disk_area()
        vi     = math.sqrt(max(thrust_N, 0.0) / max(2.0 * rho * A, 1e-9))
        P_prop = thrust_N * vi            # ideal propulsive power [W]
        # Combine prop + motor efficiency from prop_efficiency field
        eta_combined = max(config.airframe.prop_efficiency, 0.10)
        P_elec = P_prop / eta_combined    # electrical input to motor [W]
        return max(P_elec, 0.0)

    # --- Theoretical fallback (no table, no KV) ---
    A    = prop.disk_area()
    vi   = math.sqrt(max(thrust_N, 0.0) / max(2.0 * rho * A, 1e-9))
    return thrust_N * vi / 0.70


def _interp1d(x: "np.ndarray", y: "np.ndarray", xq: float) -> float:
    """Simple numpy-free linear interpolation on sorted x."""
    import bisect
    idx = bisect.bisect_left(x.tolist(), xq)
    idx = max(1, min(idx, len(x) - 1))
    x0, x1 = x[idx-1], x[idx]
    y0, y1 = y[idx-1], y[idx]
    t = (xq - x0) / (x1 - x0) if x1 != x0 else 0.0
    return y0 + t * (y1 - y0)


def max_thrust_N(config: FixedWingConfig) -> float:
    """
    Maximum static thrust available from the motor/prop system.

    For KV-based model at full throttle (advance ratio = 0, static):
        T_max = prop_thrust_from_rpm(RPM_max, rho)
    where RPM_max = KV × V_nom_pack (conservative — full throttle at nominal V).

    Clamped by motor max power:
        T_limited = min(T_max, P_max / V_pitch_speed)
    """
    motor = config.motor
    prop  = config.propeller
    rho   = config.air_density
    batt  = config.battery

    if prop.table is not None:
        # Read peak thrust from table
        return float(prop.table["Thrust_g"].max()) * G0 / 1000.0 * config.num_motors

    if motor.kv is not None:
        rpm_max = motor.kv * batt.vnom_pack
        T_ideal = prop_thrust_from_rpm(prop, rpm_max, rho)

        # Limit by motor max power:
        #   T = P_shaft / v_pitch_speed  (at full throttle pitch speed)
        n_max        = rpm_max / 60.0
        v_pitch      = prop.pitch_m * n_max      # pitch speed [m/s]
        if v_pitch > 1.0:
            T_pmax = motor.max_power / v_pitch
            T_ideal = min(T_ideal, T_pmax)

        return T_ideal * config.num_motors

    # Fallback: actuator-disk at max power
    A     = prop.disk_area()
    # T_max from P_max = T·vi with vi = sqrt(T/2ρA):  T^(3/2) = P·sqrt(2ρA)
    # → T_max = (P_max · sqrt(2ρA))^(2/3)
    factor = (motor.max_power * math.sqrt(2 * rho * A))
    return (factor ** (2.0/3.0)) * config.num_motors


def tip_speed_mps(config: FixedWingConfig, rpm: Optional[float] = None) -> float:
    """
    Propeller tip speed:
        V_tip = π · D · n   where n = RPM/60 [rev/s]
    """
    if rpm is None:
        rpm = config.motor.kv * config.battery.vnom_pack if config.motor.kv else 0.0
    n = rpm / 60.0
    return math.pi * config.propeller.diameter_m * n


def pitch_speed_mps(config: FixedWingConfig, rpm: Optional[float] = None) -> float:
    """
    Pitch speed (speed at advance ratio J = 1):
        V_pitch = pitch_m × n   where n = RPM/60
    This is the airspeed at which the blade sees zero angle of attack
    (theoretical no-thrust speed for an ideal propeller).
    """
    if rpm is None:
        rpm = config.motor.kv * config.battery.vnom_pack if config.motor.kv else 0.0
    n = rpm / 60.0
    return config.propeller.pitch_m * n


def esc_losses_W(config: FixedWingConfig, v_pack: float, motor_P_total_W: float) -> Tuple[float, str]:
    """
    ESC conduction and idle losses:
        P_esc_cond = I_motor² · R_esc
        P_esc_idle = I_idle  · V_pack
    Returns (total_esc_loss_W, status_note).
    """
    esc = config.esc
    if esc is None:
        return 0.0, ""
    v        = max(v_pack, 1.0)
    p_pm     = motor_P_total_W / max(config.num_motors, 1)
    i_motor  = p_pm / v
    p_cond   = i_motor ** 2 * max(esc.resistance, 0.0)
    p_idle   = max(esc.idle_current_A, 0.0) * v
    total    = (p_cond + p_idle) * config.num_motors
    note     = ""
    if i_motor > esc.continuous_rating_A:
        note = f"ESC over cont current ({i_motor:.1f} A > {esc.continuous_rating_A:.1f} A)"
    return total, note


# ============================================================
# AERODYNAMIC PERFORMANCE FUNCTIONS
# ============================================================
def stall_speed(config: FixedWingConfig) -> float:
    """
    Minimum flying speed (stall speed) in straight-level flight:

        V_stall = sqrt(2 · W / (ρ · S · CL_max))

    At V < V_stall the wing cannot generate enough lift.
    """
    rho = config.air_density
    W   = config.weight_N
    S   = config.airframe.wing_area_m2
    CL_max = config.airframe.CL_max
    denom = rho * S * CL_max
    return math.sqrt(2.0 * W / denom) if denom > 0 else 0.0


def drag_N(config: FixedWingConfig, speed_mps: float) -> float:
    """
    Total aerodynamic drag in straight-level flight:

        L = W  →  CL = 2W / (ρ V² S)
        CD = CD0 + k · CL²
        D  = ½ ρ V² S · CD

    The drag equals the thrust required for level flight.
    """
    af  = config.airframe
    rho = config.air_density
    W   = config.weight_N
    V   = max(speed_mps, 0.1)
    q   = 0.5 * rho * V ** 2
    CL  = af.cl_at_speed(W, V, rho)
    CD  = af.cd_at_cl(CL)
    return q * af.wing_area_m2 * CD


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


def _extrapolate_motor_value_fw(df: pd.DataFrame, thrust_g: float, column: str) -> Optional[float]:
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


def power_required_W(config: FixedWingConfig, speed_mps: float) -> float:
    """
    Shaft power required for level flight = D × V  [W].
    This is the propulsive power that must be delivered to the air.
    Electrical input power is higher by (η_motor × η_prop).
    """
    return drag_N(config, speed_mps) * max(speed_mps, 0.1)


def electrical_power_required_W(config: FixedWingConfig, speed_mps: float) -> float:
    """
    Total electrical input power from battery [W]:
        P_elec = P_prop_required / η_combined + P_avionics
    where P_prop_required is the motor shaft + prop power chain input.
    """
    T_req  = drag_N(config, speed_mps)     # thrust = drag for level flight
    P_mech = motor_shaft_power_from_thrust(config, T_req)
    return P_mech


def thrust_available_N(config: FixedWingConfig) -> float:
    """Maximum available thrust from motor/prop system [N]."""
    return max_thrust_N(config)


def specific_thrust(config: FixedWingConfig) -> float:
    """
    Specific thrust = T_available / W  [dimensionless]
    Also called thrust-to-weight ratio.
    """
    return thrust_available_N(config) / max(config.weight_N, 1e-6)


def available_excess_power_W(config: FixedWingConfig, speed_mps: float) -> float:
    """
    Excess propulsive power available for climbing or acceleration:
        P_excess = (T_available - D) × V   [W]
    """
    T_avail = thrust_available_N(config)
    D       = drag_N(config, speed_mps)
    V       = max(speed_mps, 0.1)
    return max((T_avail - D) * V, 0.0)


def rate_of_climb_mps(config: FixedWingConfig, speed_mps: float) -> float:
    """
    Rate of climb at a given airspeed using all available thrust:

        RC = (T_available - D) · V / W   [m/s]

    This is the excess thrust times velocity divided by weight (i.e., excess
    power divided by weight). Clamped to zero if T < D (no climb possible).
    """
    T_avail = thrust_available_N(config)
    D       = drag_N(config, speed_mps)
    W       = config.weight_N
    V       = max(speed_mps, 0.1)
    return max((T_avail - D) * V / W, 0.0)


def max_rate_of_climb_mps(config: FixedWingConfig,
                           v_min: float = 1.0,
                           v_max: float = 80.0,
                           steps: int   = 500) -> Tuple[float, float]:
    """
    Scan airspeed range to find maximum rate of climb.
    Returns (V_at_max_RC [m/s], max_RC [m/s]).
    """
    best_v  = v_min
    best_rc = 0.0
    for i in range(steps + 1):
        V  = v_min + (v_max - v_min) * i / steps
        rc = rate_of_climb_mps(config, V)
        if rc > best_rc:
            best_rc = rc
            best_v  = V
    return best_v, best_rc


def max_angle_of_climb_deg(config: FixedWingConfig, speed_mps: float) -> float:
    """
    Maximum climb angle at a given speed:

        γ_max = arcsin((T - D) / W)   [degrees]

    This is limited to 90° (vertical) and clamped at 0° if T ≤ D.
    """
    T = thrust_available_N(config)
    D = drag_N(config, speed_mps)
    W = max(config.weight_N, 1e-6)
    ratio = (T - D) / W
    ratio = max(-1.0, min(1.0, ratio))
    return math.degrees(math.asin(ratio)) if ratio > 0 else 0.0


def best_angle_of_climb_speed(config: FixedWingConfig,
                               v_min: float = 1.0,
                               v_max: float = 80.0,
                               steps: int   = 500) -> Tuple[float, float]:
    """
    Best angle of climb speed (Vx): maximizes γ = arcsin((T-D)/W).
    Returns (Vx [m/s], γ_max [deg]).
    """
    best_v = v_min
    best_g = 0.0
    for i in range(steps + 1):
        V = v_min + (v_max - v_min) * i / steps
        g = max_angle_of_climb_deg(config, V)
        if g > best_g:
            best_g = g
            best_v = V
    return best_v, best_g


def best_endurance_speed(config: FixedWingConfig,
                          v_stall: float,
                          v_max:   float = 80.0,
                          steps:   int   = 500) -> Tuple[float, float]:
    """
    Best endurance speed for a propeller aircraft:
        Minimise power required  →  Maximise CL^1.5 / CD

    The analytic optimum (from drag polar):
        V_be = (2W/ρS)^0.5 · (k/(3·CD0))^0.25

    Returns (V_be [m/s], P_min [W]).
    """
    af   = config.airframe
    rho  = config.air_density
    W    = config.weight_N
    S    = af.wing_area_m2

    # Analytic guess
    k       = af.k
    CD0     = af.CD0
    V_be_analytic = ((2 * W / (rho * S)) ** 0.5
                     * (k / (3 * CD0)) ** 0.25) if CD0 > 0 else v_stall * 1.3

    # Numerical scan to confirm (also respects stall speed floor)
    v_lo  = max(v_stall * 1.0, 1.0)
    v_hi  = min(v_max, v_lo + 40.0)
    best_v = V_be_analytic
    best_P = 1e12
    for i in range(steps + 1):
        V = v_lo + (v_hi - v_lo) * i / steps
        P = power_required_W(config, V)
        if P < best_P:
            best_P = P
            best_v = V
    return best_v, best_P


def best_range_speed(config: FixedWingConfig,
                     v_stall: float,
                     v_max:   float = 80.0,
                     steps:   int   = 500) -> Tuple[float, float]:
    """
    Best range speed for a propeller aircraft:
        Maximise L/D ratio  →  Minimise CD/CL

    Analytic optimum:
        V_br = (2W/ρS)^0.5 · (k/CD0)^0.25

    Returns (V_br [m/s], L/D at V_br).
    """
    af  = config.airframe
    rho = config.air_density
    W   = config.weight_N
    S   = af.wing_area_m2

    k   = af.k
    CD0 = af.CD0
    V_br_analytic = ((2 * W / (rho * S)) ** 0.5
                     * (k / CD0) ** 0.25) if CD0 > 0 else v_stall * 1.5

    v_lo = max(v_stall * 1.0, 1.0)
    v_hi = min(v_max, v_lo + 40.0)
    best_v = V_br_analytic
    best_ld = 0.0
    for i in range(steps + 1):
        V  = v_lo + (v_hi - v_lo) * i / steps
        CL = af.cl_at_speed(W, V, rho)
        ld = af.ld_ratio(CL)
        if ld > best_ld:
            best_ld = ld
            best_v  = V
    return best_v, best_ld


def takeoff_distance_m(config: FixedWingConfig) -> float:
    """
    Ground roll to lift-off (simplified Raymer method):

        s_g = 1.44 · W² / (g · ρ · S · CL_TO · (T_avg - μr · W))

    where:
        CL_TO = CL at takeoff rotation (user-supplied, typically 0.6–1.0)
        T_avg = average thrust during roll ≈ T_static × 0.75
                (thrust decreases as speed builds up)
        μr    = rolling friction coefficient (paved: 0.02–0.05)

    If the net force (T - μr·W) ≤ 0, no takeoff is possible (∞ distance).
    """
    af    = config.airframe
    rho   = config.air_density
    W     = config.weight_N
    T_max = thrust_available_N(config)
    T_avg = T_max * 0.75        # average over ground roll
    mu_r  = af.mu_roll
    CL_to = max(af.CL_takeoff, 1e-3)
    S     = af.wing_area_m2

    net_force = T_avg - mu_r * W
    if net_force <= 0:
        return float("inf")
    return 1.44 * W ** 2 / (G0 * rho * S * CL_to * net_force)


# ============================================================
# FULL OPERATING METRICS AT A GIVEN SPEED
# ============================================================
def compute_metrics(config: FixedWingConfig, speed_mps: float) -> dict:
    """
    Compute all performance metrics at the given cruise airspeed.
    Returns a dict matching the eCalc-style output columns.
    """
    rho  = config.air_density
    W    = config.weight_N
    af   = config.airframe
    batt = config.battery
    motor= config.motor

    V       = max(speed_mps, stall_speed(config) + 0.01)
    CL      = af.cl_at_speed(W, V, rho)
    CD      = af.cd_at_cl(CL)
    LD      = af.ld_ratio(CL)
    D       = drag_N(config, V)
    P_prop  = power_required_W(config, V)         # shaft / propulsive power  [W]
    T_req   = D                                    # thrust required [N]
    T_avail = thrust_available_N(config)
    P_elec  = motor_shaft_power_from_thrust(config, T_req)  # motor electrical [W]

    # Avionics / peripheral power
    P_avionics = avionics_input_power_W(config.avionics)
    if P_avionics <= 0.0:
        P_avionics = batt.vnom_pack * max(config.periph_current_A, 0.0)

    # ESC losses
    esc_loss, esc_note = esc_losses_W(config, batt.vmax_pack, P_elec)

    P_total = P_elec + esc_loss + P_avionics
    pack_I  = P_total / max(batt.vnom_pack, 1.0)
    V_load  = batt.voltage_under_load(pack_I)

    # RPM estimate
    rpm_est = motor.kv * V_load if motor.kv else 0.0

    # Performance metrics
    V_stall = stall_speed(config)
    V_be, P_be = best_endurance_speed(config, V_stall)
    V_br, LD_br = best_range_speed(config, V_stall)
    v_rc_max, rc_max = max_rate_of_climb_mps(config)
    rc_at_V         = rate_of_climb_mps(config, V)
    v_gamma, gamma  = best_angle_of_climb_speed(config)
    S_to            = takeoff_distance_m(config)
    aoa             = af.aoa_deg(CL)
    Re              = af.reynolds_number(V, rho)
    V_tip           = tip_speed_mps(config, rpm_est)
    V_pitch         = pitch_speed_mps(config, rpm_est)
    spec_thrust     = specific_thrust(config)
    max_prop_P      = motor.max_power * config.num_motors

    # Flight time at current speed
    if P_total > 0 and pack_I <= batt.discharge_max_A and V_load >= batt.vmin_pack:
        t_min = (batt.usable_Wh / P_total) * 60.0
        d_km  = V * (t_min * 60.0) / 1000.0
    else:
        t_min = 0.0
        d_km  = 0.0

    return dict(
        # Speed
        airspeed_mps       = V,
        stall_speed_mps    = V_stall,
        # Aerodynamics
        CL                 = CL,
        CD                 = CD,
        LD_ratio           = LD,
        drag_N             = D,
        aoa_deg            = aoa,
        reynolds_number    = Re,
        # Thrust / power
        thrust_required_N  = T_req,
        thrust_available_N = T_avail,
        specific_thrust    = spec_thrust,
        power_required_W   = P_prop,
        motor_power_W      = P_elec,
        esc_loss_W         = esc_loss,
        avionics_power_W   = P_avionics,
        total_power_W      = P_total,
        max_prop_power_W   = max_prop_P,
        # Battery
        pack_current_A     = pack_I,
        v_load_V           = V_load,
        # Propeller
        rpm_est            = rpm_est,
        tip_speed_mps      = V_tip,
        pitch_speed_mps    = V_pitch,
        # Climb
        rate_of_climb_mps  = rc_at_V,
        max_rc_mps         = rc_max,
        v_max_rc_mps       = v_rc_max,
        max_aoc_deg        = gamma,
        v_max_aoc_mps      = v_gamma,
        # Optimal speeds
        best_endurance_speed_mps  = V_be,
        best_range_speed_mps      = V_br,
        best_ld_ratio             = LD_br,
        # Ground
        takeoff_dist_m     = S_to,
        # Duration
        flight_time_min    = t_min,
        flight_range_km    = d_km,
        # ESC note
        esc_note           = esc_note,
    )


# ============================================================
# FLIGHT TIME / RANGE SWEEP
# ============================================================
def flight_time_min(config: FixedWingConfig, speed_mps: float) -> float:
    """Flight time at constant airspeed [min], respecting all limits."""
    V_stall = stall_speed(config)
    if speed_mps < V_stall:
        return 0.0
    m = compute_metrics(config, speed_mps)
    return m["flight_time_min"]


def flight_range_km(config: FixedWingConfig, speed_mps: float) -> float:
    return compute_metrics(config, speed_mps)["flight_range_km"]


def find_optimal_speeds(config: FixedWingConfig,
                        v_max: float = 80.0) -> Tuple[float, float, float, float]:
    """
    Numerical scan for best endurance and best range speeds.
    Returns (V_be, t_max_min, V_br, d_max_km).
    """
    V_s      = stall_speed(config)
    v_lo     = max(V_s, 1.0)
    n_steps  = 500
    best_t   = -1.0;  best_vt = v_lo
    best_d   = -1.0;  best_vd = v_lo
    for i in range(n_steps + 1):
        V = v_lo + (v_max - v_lo) * i / n_steps
        t = flight_time_min(config, V)
        d = flight_range_km(config, V)
        if t > best_t: best_t = t; best_vt = V
        if d > best_d: best_d = d; best_vd = V
    return best_vt, best_t, best_vd, best_d


# ============================================================
# PERFORMANCE FIGURE (for GUI plot canvas)
# ============================================================
def make_performance_figure(config: FixedWingConfig,
                             max_speed: float = 40.0,
                             figsize: tuple = (15, 9)) -> Figure:
    """
    Six-panel performance figure matching eCalc fixed-wing output layout.

    Panels:
      1. Flight Time & Range vs Speed
      2. Thrust Required vs Available vs Speed
      3. Power Required vs Available vs Speed
      4. Rate of Climb vs Speed
      5. Drag Polar  (CL vs CD)
      6. L/D Ratio vs Speed
    """
    V_stall = stall_speed(config)
    v_lo    = max(V_stall, 1.0)
    v_hi    = max_speed
    speeds  = [v_lo + (v_hi - v_lo) * i / 300 for i in range(301)]

    times, ranges, drags, T_avail_v = [], [], [], []
    powers_req, powers_avail        = [], []
    rcs, CLs, CDs, LDs              = [], [], [], []

    T_av = thrust_available_N(config)
    af   = config.airframe
    rho  = config.air_density
    W    = config.weight_N

    for V in speeds:
        times.append(flight_time_min(config, V))
        ranges.append(flight_range_km(config, V))
        D  = drag_N(config, V)
        drags.append(D)
        T_avail_v.append(T_av)
        powers_req.append(power_required_W(config, V))
        powers_avail.append(T_av * V)
        rcs.append(rate_of_climb_mps(config, V))
        CL = af.cl_at_speed(W, V, rho)
        CD = af.cd_at_cl(CL)
        CLs.append(CL)
        CDs.append(CD)
        LDs.append(af.ld_ratio(CL))

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Fixed-Wing UAV Performance", fontsize=13, fontweight="bold")

    # ---- 1. Time & Range ----
    ax = axes[0, 0]
    ax2 = ax.twinx()
    l1, = ax.plot(speeds, times,  color="royalblue", label="Flight Time (min)")
    l2, = ax2.plot(speeds, ranges, color="darkorange", label="Range (km)", linestyle="--")
    ax.set_xlabel("Airspeed (m/s)"); ax.set_ylabel("Time (min)"); ax2.set_ylabel("Range (km)")
    ax.set_title("Flight Time & Range vs Airspeed")
    ax.axvline(V_stall, color="red", linestyle=":", linewidth=1, label="V_stall")
    ax.legend(handles=[l1, l2], loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.4)

    # ---- 2. Thrust ----
    ax = axes[0, 1]
    ax.plot(speeds, drags,     label="Thrust Required (N)", color="crimson")
    ax.plot(speeds, T_avail_v, label="Thrust Available (N)", color="green", linestyle="--")
    ax.set_xlabel("Airspeed (m/s)"); ax.set_ylabel("Thrust (N)")
    ax.set_title("Thrust Required vs Available")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.axvline(V_stall, color="red", linestyle=":", linewidth=1)

    # ---- 3. Power ----
    ax = axes[0, 2]
    ax.plot(speeds, [p/1000 for p in powers_req],   label="Power Required (kW)", color="crimson")
    ax.plot(speeds, [p/1000 for p in powers_avail], label="Power Available (kW)", color="green", linestyle="--")
    ax.set_xlabel("Airspeed (m/s)"); ax.set_ylabel("Power (kW)")
    ax.set_title("Power Required vs Available")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.axvline(V_stall, color="red", linestyle=":", linewidth=1)

    # ---- 4. Rate of Climb ----
    ax = axes[1, 0]
    ax.plot(speeds, [r*60 for r in rcs], color="teal", label="Rate of Climb (m/min)")
    ax.set_xlabel("Airspeed (m/s)"); ax.set_ylabel("RC (m/min)")
    ax.set_title("Rate of Climb vs Airspeed")
    ax.axhline(0, color="gray", linewidth=0.8); ax.grid(True, alpha=0.4)
    ax.axvline(V_stall, color="red", linestyle=":", linewidth=1)
    ax.legend(fontsize=8)

    # ---- 5. Drag Polar ----
    ax = axes[1, 1]
    ax.plot(CDs, CLs, color="purple")
    ax.set_xlabel("CD (Drag Coefficient)"); ax.set_ylabel("CL (Lift Coefficient)")
    ax.set_title("Drag Polar (CD vs CL)")
    # Mark CL_max
    ax.axhline(config.airframe.CL_max, color="red", linestyle=":", linewidth=1, label="CL_max")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=8)

    # ---- 6. L/D Ratio ----
    ax = axes[1, 2]
    ax.plot(speeds, LDs, color="darkorange")
    ax.set_xlabel("Airspeed (m/s)"); ax.set_ylabel("L/D Ratio")
    ax.set_title("Lift-to-Drag Ratio vs Airspeed")
    ax.axvline(V_stall, color="red", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_motor_operating_point_figure(config: FixedWingConfig, metrics: dict, figsize: tuple = (12, 8)):
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
    thrust_N = float(metrics.get("thrust_required_N", 0.0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get data from propeller table
    thrust_g = df["Thrust_g"].values if "Thrust_g" in df.columns else []
    
    # Subplot 1: Thrust vs Power & Thrust vs Current
    if "Power_W" in df.columns and len(thrust_g) > 0:
        power_W = df["Power_W"].values
        ax1_1 = ax1
        ax1_1.plot(thrust_g, power_W, "b-", linewidth=2, label="Power")
        ax1_1.set_xlabel("Thrust (gf)", fontsize=10)
        ax1_1.set_ylabel("Power (W)", fontsize=10, color="b")
        ax1_1.tick_params(axis="y", labelcolor="b")
        ax1_1.grid(True, alpha=0.3)
    
    if "Current_A" in df.columns and len(thrust_g) > 0:
        current_A = df["Current_A"].values
        ax1_2 = ax1.twinx()
        ax1_2.plot(thrust_g, current_A, "r--", linewidth=2, label="Current")
        ax1_2.set_ylabel("Current (A)", fontsize=10, color="r")
        ax1_2.tick_params(axis="y", labelcolor="r")
    
    # Mark operating point on subplot 1
    thrust_g_op = thrust_N * 1000.0 / 9.81
    if "Power_W" in df.columns and len(thrust_g) > 1:
        try:
            # Simple linear interpolation
            power_W = df["Power_W"].values
            idx = 0
            for i in range(len(thrust_g) - 1):
                if thrust_g[i] <= thrust_g_op <= thrust_g[i + 1]:
                    idx = i
                    break
            frac = (thrust_g_op - thrust_g[idx]) / (thrust_g[idx + 1] - thrust_g[idx]) if thrust_g[idx + 1] != thrust_g[idx] else 0
            power_op = power_W[idx] + frac * (power_W[idx + 1] - power_W[idx])
            ax1.plot(thrust_g_op, power_op, "go", markersize=10, label=f"Operating ({power_op:.1f}W)", markeredgewidth=2, markeredgecolor="darkgreen")
        except:
            pass
    
    ax1.set_title("Thrust vs Power & Current", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    
    # Subplot 2: Thrust vs Efficiency & Thrust vs RPM
    if "Efficiency_gW" in df.columns and len(thrust_g) > 0:
        eff_gW = df["Efficiency_gW"].values
        ax2_1 = ax2
        ax2_1.plot(thrust_g, eff_gW, "g-", linewidth=2, label="Efficiency")
        ax2_1.set_xlabel("Thrust (gf)", fontsize=10)
        ax2_1.set_ylabel("Efficiency (g/W)", fontsize=10, color="g")
        ax2_1.tick_params(axis="y", labelcolor="g")
        ax2_1.grid(True, alpha=0.3)
    
    if "RPM" in df.columns and len(thrust_g) > 0:
        rpm = df["RPM"].values
        ax2_2 = ax2.twinx()
        ax2_2.plot(thrust_g, rpm, "m--", linewidth=2, label="RPM")
        ax2_2.set_ylabel("RPM", fontsize=10, color="m")
        ax2_2.tick_params(axis="y", labelcolor="m")
    
    # Mark operating point on subplot 2
    if "Efficiency_gW" in df.columns and len(thrust_g) > 1:
        try:
            eff_gW = df["Efficiency_gW"].values
            idx = 0
            for i in range(len(thrust_g) - 1):
                if thrust_g[i] <= thrust_g_op <= thrust_g[i + 1]:
                    idx = i
                    break
            frac = (thrust_g_op - thrust_g[idx]) / (thrust_g[idx + 1] - thrust_g[idx]) if thrust_g[idx + 1] != thrust_g[idx] else 0
            eff_op = eff_gW[idx] + frac * (eff_gW[idx + 1] - eff_gW[idx])
            ax2.plot(thrust_g_op, eff_op, "co", markersize=10, label=f"Operating ({eff_op:.2f}g/W)", markeredgewidth=2, markeredgecolor="darkcyan")
        except:
            pass
    
    ax2.set_title("Thrust vs Efficiency & RPM", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    
    fig.suptitle(f"Motor/Propeller Operating Curves (Thrust: {thrust_g_op:.0f}g)", 
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================
# MISSION PROFILE  (fixed-wing version)
# ============================================================
from dataclasses import dataclass, field as _field

@dataclass
class MissionPhase:
    """One leg of a fixed-wing mission.

    Either duration (seconds) or distance (metres) must be supplied.
    altitude_m sets the ISA layer for that leg — air density is recomputed
    per-phase so climb/descent effects are captured.
    """
    name:       str
    speed:      float               # airspeed [m/s]
    duration:   Optional[float] = None   # seconds (mutually exclusive with distance)
    distance:   Optional[float] = None   # metres
    altitude:   float = 0.0         # metres ASL


@dataclass
class MissionProfile:
    phases: list

    @staticmethod
    def from_json(path: str) -> "MissionProfile":
        """Load a mission from a JSON file.

        Expected format (same as multicopter sim):
        {
          "phases": [
            {"name": "Climb",   "speed": 18, "duration": 60,   "altitude": 150},
            {"name": "Cruise",  "speed": 22, "distance": 5000, "altitude": 150},
            {"name": "Return",  "speed": 20, "distance": 5000, "altitude": 80},
            {"name": "Land",    "speed": 16, "duration": 30,   "altitude": 0}
          ]
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        phases = []
        for p in data.get("phases", []):
            phases.append(MissionPhase(
                name     = str(p["name"]),
                speed    = float(p["speed"]),
                duration = float(p["duration"])  if "duration"  in p else None,
                distance = float(p["distance"])  if "distance"  in p else None,
                altitude = float(p.get("altitude", 0.0)),
            ))
        return MissionProfile(phases=phases)


def simulate_fw_mission(
    cfg: "FixedWingConfig",
    mission: MissionProfile,
    wind_mps: float = 0.0,
    temperature_C: Optional[float] = None,
    pressure_Pa:   Optional[float] = None,
) -> tuple:
    """
    Simulate a fixed-wing mission phase-by-phase, draining battery energy.

    Wind sign convention: +wind = headwind (reduces groundspeed and range).

    Returns
    -------
    results      : list of (name, time_min, dist_km, status_str)
    worst_metrics: dict — worst-case single-point metrics across all phases
    series       : dict of lists — time-series data for the Mission Plots tab
    """
    remaining_Wh = cfg.battery.usable_Wh
    results: list  = []
    worst:   dict  = {}
    t_s    = 0.0
    dist_km= 0.0

    series: dict = {
        "t_s":              [],
        "phase":            [],
        "airspeed_mps":     [],
        "groundspeed_mps":  [],
        "altitude_m":       [],
        "battery_voltage_V":[],
        "battery_current_A":[],
        "battery_energy_Wh":[],
        "total_power_W":    [],
        "motor_power_W":    [],
        "drag_N":           [],
        "thrust_avail_N":   [],
        "rate_of_climb_mps":[],
        "lift_drag_ratio":  [],
        "cl_cruise":        [],
    }

    def _append(phase_name, alt_m, m, t_now, d_now, e_now):
        series["t_s"].append(t_now)
        series["phase"].append(phase_name)
        series["airspeed_mps"].append(m.get("airspeed_mps", 0.0))
        gs = max(m.get("airspeed_mps", 0.0) - float(wind_mps), 0.0)
        series["groundspeed_mps"].append(gs)
        series["altitude_m"].append(float(alt_m))
        series["battery_voltage_V"].append(m.get("v_load_V", 0.0))
        series["battery_current_A"].append(m.get("pack_current_A", 0.0))
        series["battery_energy_Wh"].append(e_now)
        series["total_power_W"].append(m.get("total_power_W", 0.0))
        series["motor_power_W"].append(m.get("motor_power_W", 0.0))
        series["drag_N"].append(m.get("drag_N", 0.0))
        series["thrust_avail_N"].append(m.get("thrust_available_N", 0.0))
        series["rate_of_climb_mps"].append(m.get("rate_of_climb_mps", 0.0))
        series["lift_drag_ratio"].append(m.get("LD_ratio", 0.0))
        series["cl_cruise"].append(m.get("CL", 0.0))

    def _merge_worst(w, m):
        if not w:
            return dict(m)
        for k in ("total_power_W","motor_power_W","pack_current_A",
                  "drag_N","thrust_required_N","esc_loss_W"):
            w[k] = max(float(w.get(k,0)), float(m.get(k,0)))
        w["v_load_V"] = min(float(w.get("v_load_V",1e9)),
                            float(m.get("v_load_V",1e9)))
        return w

    for phase in mission.phases:
        # Recompute air density per-phase altitude
        rho = isa_density(float(phase.altitude), temperature_C, pressure_Pa)
        cfg.air_density = rho

        # Airspeed = phase speed; headwind does not change airspeed, only groundspeed
        V_air = max(float(phase.speed), stall_speed(cfg) + 0.01)
        V_gs  = max(V_air - float(wind_mps), 0.1)

        m = compute_metrics(cfg, V_air)
        worst = _merge_worst(worst, m)

        total_P   = m.get("total_power_W", 0.0)
        pack_I    = m.get("pack_current_A", 0.0)
        V_load    = m.get("v_load_V", 0.0)

        # Hard limits
        if pack_I > cfg.battery.discharge_max_A:
            results.append((phase.name, 0.0, 0.0, "Battery limit exceeded (current)"))
            break
        if V_load < cfg.battery.vmin_pack:
            results.append((phase.name, 0.0, 0.0, "Battery limit exceeded (voltage)"))
            break
        if V_air <= stall_speed(cfg):
            results.append((phase.name, 0.0, 0.0, f"Below stall speed ({stall_speed(cfg):.1f} m/s)"))
            break

        _append(phase.name, phase.altitude, m, t_s, dist_km, remaining_Wh)

        if phase.duration is not None:
            dur_s      = float(phase.duration)
            energy_Wh  = total_P * (dur_s / 3600.0)
            if energy_Wh > remaining_Wh:
                # Battery runs out mid-phase
                actual_s   = (remaining_Wh / total_P) * 3600.0 if total_P > 0 else 0.0
                actual_km  = V_gs * actual_s / 1000.0
                t_s       += actual_s;  dist_km += actual_km;  remaining_Wh = 0.0
                _append(phase.name, phase.altitude, m, t_s, dist_km, remaining_Wh)
                results.append((phase.name, actual_s/60.0, actual_km, "Battery depleted"))
                break
            remaining_Wh -= energy_Wh
            t_s          += dur_s
            dist_km      += V_gs * dur_s / 1000.0
            _append(phase.name, phase.altitude, m, t_s, dist_km, remaining_Wh)
            results.append((phase.name, dur_s/60.0, V_gs*dur_s/1000.0, "OK"))

        elif phase.distance is not None:
            dist_m  = float(phase.distance)
            time_s  = dist_m / V_gs
            energy_Wh = total_P * (time_s / 3600.0)
            if energy_Wh > remaining_Wh:
                actual_s  = (remaining_Wh / total_P) * 3600.0 if total_P > 0 else 0.0
                actual_km = V_gs * actual_s / 1000.0
                t_s      += actual_s;  dist_km += actual_km;  remaining_Wh = 0.0
                _append(phase.name, phase.altitude, m, t_s, dist_km, remaining_Wh)
                results.append((phase.name, actual_s/60.0, actual_km, "Battery depleted"))
                break
            remaining_Wh -= energy_Wh
            t_s          += time_s
            dist_km      += dist_m / 1000.0
            _append(phase.name, phase.altitude, m, t_s, dist_km, remaining_Wh)
            results.append((phase.name, time_s/60.0, dist_m/1000.0, "OK"))

        else:
            results.append((phase.name, 0.0, 0.0, "Invalid: no duration or distance"))
            break

    return results, worst, series


# ============================================================
# GUI
# ============================================================
# ============================================================
# SHARED REPORTING / EXPORT UTILITIES
# ============================================================
import csv as _csv_mod
import io  as _io_mod
import datetime as _dt_mod

def _extract_weight_budget(cfg) -> list:
    """Return [(label, unit_g, count, total_g), ...] finishing with a TOTAL row."""
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
    from openpyxl.styles import Font, PatternFill
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
                ws.cell(row=ri+2, column=ci,
                        value=sweep[h][ri] if ri < len(sweep[h]) else None)
    ws2 = wb.create_sheet("Metrics")
    ws2.append(["Metric", "Value"])
    for r in [ws2["A1"], ws2["B1"]]: r.font = Font(bold=True)
    for label, value in metrics: ws2.append([label, value])
    ws3 = wb.create_sheet("Weight Budget")
    ws3.append(["Component", "Unit Weight (g)", "Count", "Total Weight (g)", "% of Total"])
    for c in ws3[1]: c.font = Font(bold=True)
    total_g = weight_budget[-1][3] if weight_budget else 1.0
    for label, uw, cnt, tw in weight_budget[:-1]:
        pct = round(tw / total_g * 100, 1) if total_g > 0 else 0
        ws3.append([label, round(uw, 1), cnt, round(tw, 1), pct])
    if weight_budget:
        label, uw, cnt, tw = weight_budget[-1]
        ws3.append([label, "", "", round(tw, 1), 100.0])
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
    def _ts(hbg=TEAL):
        return TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), hbg),
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
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph(report_title, sTitle))
    story.append(Paragraph(
        _dt_mod.datetime.now().strftime("Generated %d %B %Y  %H:%M"), sSub))
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
    if status_sections:
        story.append(Paragraph("Status Checks", sH1))
        for sec_title, sec_rows in status_sections:
            story.append(Paragraph(sec_title, ParagraphStyle("secH", fontSize=10,
                textColor=TEAL, spaceBefore=6, spaceAfter=2, fontName="Helvetica-Bold")))
            tdata = [["Metric","Value","Limit","Notes"]]
            ts = _ts()
            for ri,(metric,val,lim,note,tag) in enumerate(sec_rows, 1):
                tdata.append([metric,val,lim,note])
                bg = {"ok":colors.HexColor("#D9F2D9"),"warn":colors.HexColor("#FFF2CC"),
                      "bad":colors.HexColor("#F8D7DA")}.get(tag, colors.white)
                ts.add("BACKGROUND",(0,ri),(-1,ri),bg)
            cw = [usable_w*0.28,usable_w*0.20,usable_w*0.20,usable_w*0.32]
            t = Table(tdata, colWidths=cw); t.setStyle(ts); story.append(t)
            story.append(Spacer(1, 4))
    story.append(PageBreak())
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
    if log_text.strip():
        story.append(Paragraph("Simulation Output Log", sH1))
        mono = ParagraphStyle("mono", fontName="Courier", fontSize=7.5, leading=10, spaceAfter=2)
        for line in log_text.splitlines():
            safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe or " ", mono))
    doc.build(story)

def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # ---------- helpers ----------
    def parse_float(label: str, val: str) -> Optional[float]:
        s = str(val).strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"{label}: expected a number, got {s!r}")

    def parse_int(label: str, val: str) -> Optional[int]:
        s = str(val).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            raise ValueError(f"{label}: expected an integer, got {s!r}")

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

    # ---------- root window ----------
    root = tk.Tk()
    root.title("Fixed-Wing UAV Power Simulator")
    root.minsize(1100, 700)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ------------------------------------------------------------------ #
    #  VIEW STATE — mutable container shared by all scaling callbacks     #
    # ------------------------------------------------------------------ #
    # Capture Tk's default DPI scaling so we can multiply from it cleanly.
    try:
        _base_tk_scale = float(root.tk.call("tk", "scaling"))
    except Exception:
        _base_tk_scale = 1.333          # 96 dpi / 72 pt — safe fallback

    _view = {
        "scale_pct":    100,            # window/widget DPI scale  (%)
        "plot_w":       15.0,           # matplotlib figure width  (inches)
        "plot_h":        9.0,           # matplotlib figure height (inches)
        "mpl_fontsize":  9,             # base font size for plots (pt)
        "ui_fontsize":   9,             # ttk widget font size     (pt)
    }

    # Holds the last successful run so we can re-render on plot-size change.
    _last_run: dict = {}   # keys: "cfg", "max_v", "metrics", "wind"

    # ------------------------------------------------------------------ #
    #  SCALING HELPERS                                                    #
    # ------------------------------------------------------------------ #
    def _apply_tk_scale(pct: int) -> None:
        """
        Scale all Tk/ttk widgets by adjusting Tk's DPI scaling factor.

        Tk uses a "scaling" value = pixels-per-point.  The default is
        typically screen_dpi / 72.  Multiplying it by (pct/100) makes
        every widget — fonts, buttons, entries, paddings — proportionally
        larger or smaller without any style surgery.

        We also nudge the root minsize so the window doesn't become
        unusably small at 75 % or clip content at 200 %.
        """
        _view["scale_pct"] = pct
        factor = _base_tk_scale * (pct / 100.0)
        try:
            root.tk.call("tk", "scaling", factor)
        except Exception:
            pass
        # Adjust minimum window size proportionally
        base_w, base_h = 1100, 700
        root.minsize(int(base_w * pct / 100), int(base_h * pct / 100))
        # Force geometry refresh so widgets reflow immediately
        root.update_idletasks()

    def _apply_ui_font(size: int) -> None:
        """
        Override the font size on every ttk style that carries text.
        Uses TkDefaultFont as the family so it follows the OS default.
        """
        _view["ui_fontsize"] = size
        sty = ttk.Style()
        font_spec = ("TkDefaultFont", size)
        for widget_style in (
            "TLabel", "TButton", "TEntry", "TCombobox",
            "TNotebook.Tab", "Treeview", "Treeview.Heading",
            "TLabelframe.Label", "TLabelframe",
        ):
            try:
                sty.configure(widget_style, font=font_spec)
            except Exception:
                pass
        # Treeview row height should track font size
        row_h = max(18, size + 8)
        try:
            sty.configure("Treeview", rowheight=row_h)
        except Exception:
            pass
        root.update_idletasks()

    def _apply_mpl_font(size: int) -> None:
        """
        Set matplotlib's global base font size, then re-render the
        current plot if one exists, so the change is visible immediately.
        """
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
        """Store new figure dimensions and re-render immediately if possible."""
        _view["plot_w"] = w
        _view["plot_h"] = h
        _rerender_if_possible()

    def _rerender_if_possible() -> None:
        """Re-run the plot pipeline with the last known config, if any."""
        if not _last_run:
            return
        try:
            cfg   = _last_run["cfg"]
            max_v = _last_run["max_v"]
            m     = _last_run.get("metrics", {})
            fig   = make_performance_figure(
                cfg,
                max_speed=max_v,
                figsize=(_view["plot_w"], _view["plot_h"]),
            )
            # Generate motor operating point figure if available
            motor_fig = None
            if cfg.propeller.table is not None and m:
                try:
                    motor_fig = make_motor_operating_point_figure(cfg, m, figsize=(_view["plot_w"], 6))
                except Exception:
                    pass
            # Display both figures
            if motor_fig:
                show_figure([fig, motor_fig])
            else:
                show_figure(fig)
        except Exception:
            pass   # silently ignore re-render errors

    # ------------------------------------------------------------------ #
    #  MENU BAR                                                           #
    # ------------------------------------------------------------------ #
    menubar   = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)

    # ---- View menu ----
    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)

    # -- Window Scale sub-menu --
    _scale_var = tk.IntVar(value=100)

    scale_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Window Scale", menu=scale_menu)
    for pct, label in [(75, "75 %  – Compact"),
                       (90, "90 %  – Smaller"),
                       (100,"100 % – Default"),
                       (115,"115 % – Slightly Larger"),
                       (125,"125 % – Large"),
                       (150,"150 % – Extra Large"),
                       (175,"175 % – Very Large"),
                       (200,"200 % – Max")]:
        scale_menu.add_radiobutton(
            label=label,
            variable=_scale_var,
            value=pct,
            command=lambda p=pct: _apply_tk_scale(p),
        )

    view_menu.add_separator()

    # -- Plot Size sub-menu --
    _plot_size_var = tk.StringVar(value="medium")

    plot_size_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Plot Size", menu=plot_size_menu)
    for key, label, (w, h) in [
        ("small",    "Small  (12 × 7)",   (12.0,  7.0)),
        ("medium",   "Medium (15 × 9)  ← default", (15.0,  9.0)),
        ("large",    "Large  (18 × 11)",  (18.0, 11.0)),
        ("xlarge",   "X-Large (22 × 13)", (22.0, 13.0)),
        ("xxlarge",  "XX-Large (26 × 15)",(26.0, 15.0)),
    ]:
        plot_size_menu.add_radiobutton(
            label=label,
            variable=_plot_size_var,
            value=key,
            command=lambda pw=w, ph=h: _apply_plot_size(pw, ph),
        )

    view_menu.add_separator()

    # -- UI Font Size sub-menu --
    _ui_font_var = tk.IntVar(value=9)

    ui_font_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="UI Font Size", menu=ui_font_menu)
    for sz, lbl in [(8,  "8 pt  – Tiny"),
                    (9,  "9 pt  – Default"),
                    (10, "10 pt – Comfortable"),
                    (11, "11 pt – Large"),
                    (13, "13 pt – Extra Large"),
                    (15, "15 pt – Accessibility")]:
        ui_font_menu.add_radiobutton(
            label=lbl,
            variable=_ui_font_var,
            value=sz,
            command=lambda s=sz: _apply_ui_font(s),
        )

    view_menu.add_separator()

    # -- Plot Font Size sub-menu --
    _mpl_font_var = tk.IntVar(value=9)

    mpl_font_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Plot Font Size", menu=mpl_font_menu)
    for sz, lbl in [(7,  "7 pt  – Tiny"),
                    (8,  "8 pt  – Small"),
                    (9,  "9 pt  – Default"),
                    (10, "10 pt – Medium"),
                    (12, "12 pt – Large"),
                    (14, "14 pt – Extra Large")]:
        mpl_font_menu.add_radiobutton(
            label=lbl,
            variable=_mpl_font_var,
            value=sz,
            command=lambda s=sz: _apply_mpl_font(s),
        )

    view_menu.add_separator()

    # -- Quick preset commands --
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
    presets_menu.add_command(label="🗜  Compact",          command=_preset_compact)
    presets_menu.add_command(label="⚙  Default",           command=_preset_default)
    presets_menu.add_command(label="📊  Presentation",     command=_preset_presentation)
    presets_menu.add_command(label="♿  Accessibility",     command=_preset_accessibility)

    view_menu.add_separator()
    view_menu.add_command(label="Reset All to Default",    command=_preset_default)

    root.config(menu=menubar)

    # ---------- main panes ----------
    main = ttk.Frame(root, padding=8)
    main.grid(sticky="nsew")
    main.columnconfigure(0, weight=1, minsize=370)
    main.columnconfigure(1, weight=3)
    main.rowconfigure(0, weight=1)

    # ===== LEFT: input notebook =====
    left = ttk.Frame(main)
    left.grid(row=0, column=0, sticky="nsew")
    left.columnconfigure(0, weight=1)
    left.rowconfigure(0, weight=1)

    input_nb = ttk.Notebook(left)
    input_nb.grid(row=0, column=0, sticky="nsew")

    # Create scrollable tab helper
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
        cv.bind("<Configure>",  _on_canvas_configure)   # ← the essential new line

        inn.columnconfigure(0, weight=1)
        inn.columnconfigure(1, weight=1)
        _tab_canvases.append(cv)   # register so wheel handler can find it
        return inn

    tab_airframe = make_scrollable_tab(input_nb, "Airframe")
    tab_batt     = make_scrollable_tab(input_nb, "Battery")
    tab_motor    = make_scrollable_tab(input_nb, "Motor")
    tab_esc      = make_scrollable_tab(input_nb, "ESC")
    tab_avionics = make_scrollable_tab(input_nb, "Avionics")
    tab_prop     = make_scrollable_tab(input_nb, "Propeller")
    tab_env      = make_scrollable_tab(input_nb, "Mission/Environment")
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
    # Also bind to each outer frame so the wheel works when hovering over
    # the tab content area (not just the notebook header strip).
    for _cv in _tab_canvases:
        _cv.bind("<MouseWheel>", lambda evt, c=_cv:
                 c.yview_scroll(int(-1 * (evt.delta / 120)), "units"))
    # --------------------------------------------------------------------------


    # ---------- StringVars ----------
    def sv(default=""):
        return tk.StringVar(value=str(default))

    # Airframe
    v_weight        = sv(2500)     # grams
    v_num_motors    = sv(1)
    v_wing_span     = sv(1.6)      # m
    v_wing_area     = sv(0.45)     # m²
    v_CD0           = sv(0.028)
    v_CL_max        = sv(1.30)
    v_oswald        = sv(0.82)
    v_mu_roll       = sv(0.04)
    v_CL_takeoff    = sv(0.80)
    v_prop_eff      = sv(0.75)
    v_cruise_speed  = sv(18.0)     # m/s

    # Battery
    v_batt_chem       = sv("LiPo")
    v_batt_vmin       = sv(3.5)
    v_batt_vnom       = sv(3.8)
    v_batt_vmax       = sv(4.2)
    v_batt_cell_cap   = sv(5000)
    v_batt_pack_cap   = sv("")
    v_batt_cell_wt    = sv(75)
    v_batt_pack_wt    = sv("")
    v_batt_dens       = sv("")
    v_batt_chg        = sv(5)
    v_batt_a_cont     = sv("")
    v_batt_a_max      = sv("")
    v_batt_c_cont     = sv(25)
    v_batt_c_max      = sv(50)
    v_batt_dischg_pct = sv(80)
    v_batt_r          = sv(5)
    v_batt_series     = sv(4)
    v_batt_parallel   = sv(1)
    v_batt_cells_s    = sv(1)
    v_batt_cells_p    = sv(1)
    v_batt_unit_mode  = sv("cell")

    # Motor
    v_motor_kv       = sv(920)
    v_motor_i0       = sv(0.8)
    v_motor_v0       = sv(7.0)
    v_motor_rated_v  = sv(16)
    v_motor_r        = sv(0.06)
    v_motor_imax     = sv(40)
    v_motor_pmax     = sv(500)
    v_motor_poles    = sv(14)
    v_motor_wt       = sv(120)
    v_motor_size     = sv("")

    # ESC
    v_esc_vrating  = sv("")
    v_esc_cont     = sv("")
    v_esc_max      = sv("")
    v_esc_idle     = sv("")
    v_esc_r        = sv("")
    v_esc_wt       = sv("")

    # Avionics
    v_avionics_str = sv("")

    # Propeller
    v_prop_d        = sv(10)
    v_prop_pitch    = sv(4.5)
    v_prop_blades   = sv(2)
    v_prop_maxrpm   = sv(0)
    v_prop_maxthr   = sv(0)
    v_prop_table    = sv("")
    v_prop_tconst   = sv("")
    v_prop_pconst   = sv("")
    v_prop_wt       = sv("20")

    # Environment / Mission
    v_mission       = sv("")
    v_altitude      = sv(0)
    v_temp          = sv("")
    v_pressure      = sv("")
    v_wind          = sv(0)
    v_max_v_plot    = sv(40)
    v_periph_cur    = sv(0.5)

    config_vars = dict(
        weight=v_weight, num_motors=v_num_motors,
        wing_span=v_wing_span, wing_area=v_wing_area,
        CD0=v_CD0, CL_max=v_CL_max, oswald=v_oswald,
        mu_roll=v_mu_roll, CL_takeoff=v_CL_takeoff,
        prop_eff=v_prop_eff, cruise_speed=v_cruise_speed,
        batt_chem=v_batt_chem, batt_vmin=v_batt_vmin,
        batt_vnom=v_batt_vnom, batt_vmax=v_batt_vmax,
        batt_cell_cap=v_batt_cell_cap, batt_pack_cap=v_batt_pack_cap,
        batt_cell_wt=v_batt_cell_wt, batt_pack_wt=v_batt_pack_wt,
        batt_dens=v_batt_dens, batt_chg=v_batt_chg,
        batt_a_cont=v_batt_a_cont, batt_a_max=v_batt_a_max,
        batt_c_cont=v_batt_c_cont, batt_c_max=v_batt_c_max,
        batt_dischg_pct=v_batt_dischg_pct, batt_r=v_batt_r,
        batt_series=v_batt_series, batt_parallel=v_batt_parallel,
        batt_cells_s=v_batt_cells_s, batt_cells_p=v_batt_cells_p,
        batt_unit_mode=v_batt_unit_mode,
        motor_kv=v_motor_kv, motor_i0=v_motor_i0, motor_v0=v_motor_v0,
        motor_rated_v=v_motor_rated_v, motor_r=v_motor_r,
        motor_imax=v_motor_imax, motor_pmax=v_motor_pmax,
        motor_poles=v_motor_poles, motor_wt=v_motor_wt,
        motor_size=v_motor_size,
        esc_vrating=v_esc_vrating, esc_cont=v_esc_cont,
        esc_max=v_esc_max, esc_idle=v_esc_idle,
        esc_r=v_esc_r, esc_wt=v_esc_wt,
        avionics_str=v_avionics_str,
        prop_d=v_prop_d, prop_pitch=v_prop_pitch, prop_blades=v_prop_blades,
        prop_maxrpm=v_prop_maxrpm, prop_maxthr=v_prop_maxthr,
        prop_table=v_prop_table, prop_tconst=v_prop_tconst,
        prop_pconst=v_prop_pconst, prop_wt=v_prop_wt,
        altitude=v_altitude, temp=v_temp, pressure=v_pressure,
        mission=v_mission, wind=v_wind, max_v_plot=v_max_v_plot, periph_cur=v_periph_cur,
    )

    # ---- row helper ----
    def add_row(parent, row, label, var, **kwargs):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        e = ttk.Entry(parent, textvariable=var, width=14, **kwargs)
        e.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        return e

    # ===== AIRFRAME TAB =====
    r = 0
    add_row(tab_airframe, r, "Aircraft Weight (g)",         v_weight);       r += 1
    add_row(tab_airframe, r, "Number of Motors",            v_num_motors);   r += 1
    add_row(tab_airframe, r, "Cruise Speed (m/s)",          v_cruise_speed); r += 1
    add_row(tab_airframe, r, "Peripheral Current (A)",      v_periph_cur);   r += 1
    ttk.Separator(tab_airframe, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    ttk.Label(tab_airframe, text="── Wing Geometry ──",
              foreground="gray").grid(row=r, column=0, columnspan=2, sticky="w", padx=6); r += 1
    add_row(tab_airframe, r, "Wing Span (m)",               v_wing_span);    r += 1
    add_row(tab_airframe, r, "Wing Area (m²)",              v_wing_area);    r += 1
    # chord is derived; shown in metrics
    ttk.Separator(tab_airframe, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    ttk.Label(tab_airframe, text="── Aerodynamics ──",
              foreground="gray").grid(row=r, column=0, columnspan=2, sticky="w", padx=6); r += 1
    add_row(tab_airframe, r, "CD0 (zero-lift drag coeff)", v_CD0);           r += 1
    add_row(tab_airframe, r, "CL_max",                     v_CL_max);        r += 1
    add_row(tab_airframe, r, "Oswald Efficiency (e)",      v_oswald);        r += 1
    ttk.Separator(tab_airframe, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    ttk.Label(tab_airframe, text="── Takeoff / Ground Roll ──",
              foreground="gray").grid(row=r, column=0, columnspan=2, sticky="w", padx=6); r += 1
    add_row(tab_airframe, r, "Rolling Friction μ",         v_mu_roll);       r += 1
    add_row(tab_airframe, r, "CL at Takeoff Rotation",     v_CL_takeoff);    r += 1
    ttk.Separator(tab_airframe, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    ttk.Label(tab_airframe, text="── Propulsion ──",
              foreground="gray").grid(row=r, column=0, columnspan=2, sticky="w", padx=6); r += 1
    add_row(tab_airframe, r, "Prop Efficiency η",          v_prop_eff);      r += 1

    # ===== BATTERY TAB =====
    ttk.Label(tab_batt, text="Unit mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    unit_mode_cb = ttk.Combobox(tab_batt, textvariable=v_batt_unit_mode,
                                values=["cell","pack"], state="readonly", width=10)
    unit_mode_cb.grid(row=0, column=1, sticky="w", padx=6, pady=4)
    r = 1
    add_row(tab_batt, r, "Vmin/cell (V)",            v_batt_vmin);      r += 1
    add_row(tab_batt, r, "Vnom/cell (V)",            v_batt_vnom);      r += 1
    add_row(tab_batt, r, "Vmax/cell (V)",            v_batt_vmax);      r += 1
    cell_cap_e = add_row(tab_batt, r, "Cell Capacity (mAh)",  v_batt_cell_cap); r += 1
    pack_cap_e = add_row(tab_batt, r, "Pack Capacity (mAh)",  v_batt_pack_cap); r += 1
    cell_wt_e  = add_row(tab_batt, r, "Cell Weight (g)",      v_batt_cell_wt);  r += 1
    pack_wt_e  = add_row(tab_batt, r, "Pack Weight (g)",      v_batt_pack_wt);  r += 1
    add_row(tab_batt, r, "Energy Density (Wh/kg)",   v_batt_dens);      r += 1
    add_row(tab_batt, r, "Max Charge Current (A)",   v_batt_chg);       r += 1
    add_row(tab_batt, r, "Cont Discharge Current (A)", v_batt_a_cont);  r += 1
    add_row(tab_batt, r, "Max Discharge Current (A)", v_batt_a_max);    r += 1
    add_row(tab_batt, r, "Cont C-rate",              v_batt_c_cont);    r += 1
    add_row(tab_batt, r, "Max C-rate",               v_batt_c_max);     r += 1
    add_row(tab_batt, r, "Usable Discharge (%)",     v_batt_dischg_pct); r += 1
    add_row(tab_batt, r, "Rcell (mΩ)",               v_batt_r);         r += 1
    add_row(tab_batt, r, "Series Cells/Packs",       v_batt_series);    r += 1
    add_row(tab_batt, r, "Parallel Cells/Packs",     v_batt_parallel);  r += 1
    cells_s_e = add_row(tab_batt, r, "Cells in series/pack",  v_batt_cells_s); r += 1
    cells_p_e = add_row(tab_batt, r, "Cells in parallel/pack",v_batt_cells_p); r += 1
    add_row(tab_batt, r, "Chemistry",                v_batt_chem);      r += 1

    def on_unit_mode(event=None):
        mode = v_batt_unit_mode.get()
        if mode == "cell":
            cell_cap_e.configure(state="normal"); cell_wt_e.configure(state="normal")
            pack_cap_e.configure(state="disabled"); pack_wt_e.configure(state="disabled")
            cells_s_e.configure(state="disabled"); cells_p_e.configure(state="disabled")
        else:
            cell_cap_e.configure(state="disabled"); cell_wt_e.configure(state="disabled")
            pack_cap_e.configure(state="normal"); pack_wt_e.configure(state="normal")
            cells_s_e.configure(state="normal"); cells_p_e.configure(state="normal")
    unit_mode_cb.bind("<<ComboboxSelected>>", on_unit_mode)
    on_unit_mode()

    # ===== MOTOR TAB =====
    r = 0
    add_row(tab_motor, r, "Kv (RPM/V)",        v_motor_kv);    r += 1
    add_row(tab_motor, r, "Idle Current I0 (A)",v_motor_i0);   r += 1
    add_row(tab_motor, r, "Idle Voltage V0 (V)",v_motor_v0);   r += 1
    add_row(tab_motor, r, "Rated Voltage (V)",  v_motor_rated_v); r += 1
    add_row(tab_motor, r, "Resistance Rm (Ω)",  v_motor_r);    r += 1
    add_row(tab_motor, r, "Max Current (A)",    v_motor_imax); r += 1
    add_row(tab_motor, r, "Max Power (W)",      v_motor_pmax); r += 1
    add_row(tab_motor, r, "Pole Count",         v_motor_poles); r += 1
    add_row(tab_motor, r, "Weight (g)",         v_motor_wt);   r += 1
    add_row(tab_motor, r, "Size (e.g. 2826)",   v_motor_size); r += 1

    # ===== ESC TAB =====
    r = 0
    add_row(tab_esc, r, "Voltage Rating (V)",    v_esc_vrating); r += 1
    add_row(tab_esc, r, "Continuous Current (A)", v_esc_cont);   r += 1
    add_row(tab_esc, r, "Max Current (A)",        v_esc_max);    r += 1
    add_row(tab_esc, r, "Idle Current (A)",       v_esc_idle);   r += 1
    add_row(tab_esc, r, "Resistance (Ω)",         v_esc_r);      r += 1
    add_row(tab_esc, r, "Weight (g)",             v_esc_wt);     r += 1

    # ===== AVIONICS TAB =====
    # Each row = one BEC-regulated rail: voltage (V), current (A), efficiency (0–1].
    # The table is the single source of truth; v_avionics_str is kept in sync
    # so save/load and build_config can share one code path.

    tab_avionics.columnconfigure(0, weight=1)
    tab_avionics.rowconfigure(1, weight=1)   # treeview row expands

    # ---- header label ----
    ttk.Label(
        tab_avionics,
        text="BEC / Avionics voltage rails  —  one row per regulated output bus.\n"
             "Double-click any cell to edit it in-place.",
        wraplength=340, justify="left", foreground="#555555",
    ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))

    # ---- Treeview ----
    av_tree_frame = ttk.Frame(tab_avionics)
    av_tree_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0, 4))
    av_tree_frame.columnconfigure(0, weight=1)
    av_tree_frame.rowconfigure(0, weight=1)

    _AV_COLS = ("voltage", "current", "efficiency")
    av_tree = ttk.Treeview(
        av_tree_frame,
        columns=_AV_COLS,
        show="headings",
        height=8,
        selectmode="browse",
    )
    for col, heading, width in [
        ("voltage",    "Rail Voltage (V)",      130),
        ("current",    "Rail Current (A)",      130),
        ("efficiency", "BEC Efficiency (0–1]",  150),
    ]:
        av_tree.heading(col, text=heading)
        av_tree.column(col, width=width, anchor="center", stretch=True)

    av_tree.grid(row=0, column=0, sticky="nsew")
    av_sb = ttk.Scrollbar(av_tree_frame, orient="vertical", command=av_tree.yview)
    av_sb.grid(row=0, column=1, sticky="ns")
    av_tree.configure(yscrollcommand=av_sb.set)

    # ---- helper: read all rows from treeview → voltage_tree dict ----
    def _av_tree_to_dict() -> dict:
        out = {}
        for iid in av_tree.get_children():
            vals = av_tree.item(iid, "values")
            try:
                v, i, e = float(vals[0]), float(vals[1]), float(vals[2])
                if v <= 0 or i < 0 or not (0 < e <= 1):
                    raise ValueError
                out[v] = (i, e)
            except Exception:
                raise ValueError(
                    f"Invalid avionics row {vals!r}.\n"
                    "Voltage > 0, Current ≥ 0, Efficiency in (0, 1]."
                )
        return out

    # ---- helper: serialise treeview → canonical string (kept in sync) ----
    def _sync_av_str():
        try:
            d = _av_tree_to_dict()
            v_avionics_str.set(
                ", ".join(f"{v}:({i},{e})" for v, (i, e) in sorted(d.items()))
            )
        except Exception:
            pass   # leave string var unchanged if table has a bad row mid-edit

    # ---- helper: load a list of {"voltage", "current", "eff"} dicts into tree ----
    def _av_load_rows(rows: list):
        av_tree.delete(*av_tree.get_children())
        for r in rows:
            try:
                av_tree.insert("", "end", values=(
                    str(r.get("voltage", "")),
                    str(r.get("current", "")),
                    str(r.get("eff", "")),
                ))
            except Exception:
                pass
        _sync_av_str()

    # ---- Double-click in-place cell editor ----
    def _av_begin_edit(event):
        region = av_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = av_tree.identify_row(event.y)
        col_id = av_tree.identify_column(event.x)   # "#1", "#2", "#3"
        if not row_id or not col_id:
            return
        col_idx = int(col_id.replace("#", "")) - 1
        bbox    = av_tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox
        old_vals = list(av_tree.item(row_id, "values"))

        ed = tk.Entry(av_tree, justify="center")
        ed.insert(0, old_vals[col_idx])
        ed.select_range(0, tk.END)
        ed.focus_set()
        ed.place(x=x, y=y, width=w, height=h)

        def _commit(_evt=None):
            new_val = ed.get().strip()
            old_vals[col_idx] = new_val
            av_tree.item(row_id, values=tuple(old_vals))
            ed.destroy()
            _sync_av_str()

        def _cancel(_evt=None):
            ed.destroy()

        ed.bind("<Return>",   _commit)
        ed.bind("<Tab>",      _commit)
        ed.bind("<FocusOut>", _commit)
        ed.bind("<Escape>",   _cancel)

    av_tree.bind("<Double-1>", _av_begin_edit)

    # ---- Add / edit row controls ----
    av_entry_frame = ttk.Frame(tab_avionics)
    av_entry_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 2))
    for c in range(6):
        av_entry_frame.columnconfigure(c, weight=1)

    ttk.Label(av_entry_frame, text="Voltage (V):").grid(row=0, column=0, sticky="e", padx=(0, 2))
    _av_v_var = tk.StringVar()
    ttk.Entry(av_entry_frame, textvariable=_av_v_var, width=7).grid(row=0, column=1, sticky="ew", padx=(0, 6))

    ttk.Label(av_entry_frame, text="Current (A):").grid(row=0, column=2, sticky="e", padx=(0, 2))
    _av_i_var = tk.StringVar()
    ttk.Entry(av_entry_frame, textvariable=_av_i_var, width=7).grid(row=0, column=3, sticky="ew", padx=(0, 6))

    ttk.Label(av_entry_frame, text="Efficiency:").grid(row=0, column=4, sticky="e", padx=(0, 2))
    _av_e_var = tk.StringVar(value="0.90")
    ttk.Entry(av_entry_frame, textvariable=_av_e_var, width=7).grid(row=0, column=5, sticky="ew")

    # ---- Button row ----
    av_btn_frame = ttk.Frame(tab_avionics)
    av_btn_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))

    def _av_add_or_update():
        try:
            v = float(_av_v_var.get())
            i = float(_av_i_var.get())
            e = float(_av_e_var.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                "Please enter numeric values for Voltage, Current, and Efficiency.")
            return
        if v <= 0:
            messagebox.showerror("Invalid input", "Rail voltage must be > 0 V.")
            return
        if i < 0:
            messagebox.showerror("Invalid input", "Rail current must be ≥ 0 A.")
            return
        if not (0 < e <= 1.0):
            messagebox.showerror("Invalid input", "BEC efficiency must be in (0, 1].")
            return

        # If a row with this voltage already exists, update it in-place
        for iid in av_tree.get_children():
            existing = av_tree.item(iid, "values")
            try:
                if abs(float(existing[0]) - v) < 1e-9:
                    av_tree.item(iid, values=(f"{v:g}", f"{i:g}", f"{e:g}"))
                    _sync_av_str()
                    _av_v_var.set(""); _av_i_var.set(""); _av_e_var.set("0.90")
                    return
            except Exception:
                pass

        # Otherwise append a new row
        av_tree.insert("", "end", values=(f"{v:g}", f"{i:g}", f"{e:g}"))
        _sync_av_str()
        _av_v_var.set(""); _av_i_var.set(""); _av_e_var.set("0.90")

    def _av_remove_selected():
        sel = av_tree.selection()
        for iid in sel:
            av_tree.delete(iid)
        _sync_av_str()

    def _av_clear_all():
        av_tree.delete(*av_tree.get_children())
        _sync_av_str()

    # Clicking a row populates the entry fields for quick editing
    def _av_on_select(event):
        sel = av_tree.selection()
        if not sel:
            return
        vals = av_tree.item(sel[0], "values")
        try:
            _av_v_var.set(vals[0])
            _av_i_var.set(vals[1])
            _av_e_var.set(vals[2])
        except Exception:
            pass

    av_tree.bind("<<TreeviewSelect>>", _av_on_select)

    ttk.Button(av_btn_frame, text="➕  Add / Update Rail",
               command=_av_add_or_update).grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Button(av_btn_frame, text="🗑  Remove Selected",
               command=_av_remove_selected).grid(row=0, column=1, sticky="w", padx=(0, 6))
    ttk.Button(av_btn_frame, text="✖  Clear All",
               command=_av_clear_all).grid(row=0, column=2, sticky="w")

    # ---- Seed the table from v_avionics_str if it has a value ----
    try:
        _initial = parse_voltage_tree(v_avionics_str.get().strip() or None)
        _av_load_rows([{"voltage": v, "current": ci[0], "eff": ci[1]}
                       for v, ci in sorted(_initial.items())])
    except Exception:
        pass  # start with an empty table

    # ===== PROPELLER TAB =====
    r = 0
    add_row(tab_prop, r, "Diameter (in)",    v_prop_d);      r += 1
    add_row(tab_prop, r, "Pitch (in)",       v_prop_pitch);  r += 1
    add_row(tab_prop, r, "Blades",           v_prop_blades); r += 1
    add_row(tab_prop, r, "Max RPM (0=auto)", v_prop_maxrpm); r += 1
    add_row(tab_prop, r, "Max Thrust (g)",   v_prop_maxthr); r += 1
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
    add_row(tab_prop, r, "TConst (optional)", v_prop_tconst); r += 1
    add_row(tab_prop, r, "PConst (optional)", v_prop_pconst); r += 1
    add_row(tab_prop, r, "Weight (g)",        v_prop_wt);     r += 1

    # ===== ENVIRONMENT / MISSION TAB =====
    r = 0
    ttk.Label(tab_env, text="Mission JSON (optional)").grid(
        row=r, column=0, sticky="w", padx=6, pady=2)
    mrow = ttk.Frame(tab_env); mrow.grid(row=r, column=1, sticky="ew")
    mrow.columnconfigure(0, weight=1)
    ttk.Entry(mrow, textvariable=v_mission).grid(row=0, column=0, sticky="ew", padx=(6,4))
    ttk.Button(mrow, text="Browse…",
               command=lambda: choose_file(v_mission, [("JSON","*.json"),("All","*.*")])).grid(
        row=0, column=1, padx=(0,6)); r += 1
    ttk.Separator(tab_env, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
    add_row(tab_env, r, "Altitude (m)",            v_altitude);    r += 1
    add_row(tab_env, r, "Temperature (°C, optional)", v_temp);     r += 1
    add_row(tab_env, r, "Pressure (Pa, optional)", v_pressure);    r += 1
    add_row(tab_env, r, "Wind (m/s, + = headwind)", v_wind);       r += 1
    add_row(tab_env, r, "Max speed for plot (m/s)", v_max_v_plot); r += 1

    # ===== RIGHT: output panels =====
    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew", padx=(8,0))
    right.columnconfigure(0, weight=1)
    right.rowconfigure(0, weight=2)
    right.rowconfigure(1, weight=1)

    display_nb = ttk.Notebook(right)
    display_nb.grid(row=0, column=0, sticky="nsew")

    tab_plots        = ttk.Frame(display_nb, padding=0)
    tab_status       = ttk.Frame(display_nb, padding=0)
    tab_metrics      = ttk.Frame(display_nb, padding=0)
    tab_mission_plots= ttk.Frame(display_nb, padding=0)
    for t in (tab_plots, tab_status, tab_metrics, tab_mission_plots):
        t.columnconfigure(0, weight=1); t.rowconfigure(0, weight=1)
    display_nb.add(tab_plots,         text="Plots")
    display_nb.add(tab_status,        text="Status")
    display_nb.add(tab_metrics,       text="Metrics")
    display_nb.add(tab_mission_plots, text="Mission Plots")
    tab_weight_budget = ttk.Frame(display_nb, padding=0)
    tab_weight_budget.columnconfigure(0, weight=1)
    tab_weight_budget.rowconfigure(0, weight=1)
    display_nb.add(tab_weight_budget, text="Weight Budget")


    # ---- Plots panel ----
    plot_frame = ttk.LabelFrame(tab_plots, text="Performance Plots", padding=4)
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

    def show_figure(fig_or_figs):
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

    # ---- Mission Plots panel (matches multicopter layout) ----
    # Series data stored as a one-element list so closures capture the reference.
    last_mission_series = [None]

    # Variables: (series_key, display_label, y-axis_unit)
    # Variables that share the same unit string are plotted on the same y-axis.
    MISSION_VARS = [
        ("airspeed_mps",      "Airspeed",                 "m/s"),
        ("groundspeed_mps",   "Groundspeed",              "m/s"),
        ("altitude_m",        "Altitude",                 "m"),
        ("battery_voltage_V", "Battery voltage (loaded)", "V"),
        ("battery_current_A", "Battery current",          "A"),
        ("battery_energy_Wh", "Battery energy remaining", "Wh"),
        ("total_power_W",     "Total power",              "W"),
        ("motor_power_W",     "Motor power",              "W"),
        ("drag_N",            "Drag force",               "N"),
        ("thrust_avail_N",    "Thrust available",         "N"),
        ("rate_of_climb_mps", "Rate of climb",            "m/s"),
        ("lift_drag_ratio",   "L/D ratio",                "—"),
        ("cl_cruise",         "CL at cruise speed",       "—"),
    ]

    # Two-column layout: left controls, right canvas  (same as multicopter)
    mission_container = ttk.Frame(tab_mission_plots, padding=4)
    mission_container.grid(row=0, column=0, sticky="nsew")
    mission_container.columnconfigure(0, weight=0)
    mission_container.columnconfigure(1, weight=1)
    mission_container.rowconfigure(0, weight=1)

    mission_controls   = ttk.LabelFrame(mission_container, text="Y-axis variables", padding=4)
    mission_controls.grid(row=0, column=0, sticky="ns", padx=(0, 8))
    mission_plot_frame = ttk.LabelFrame(mission_container, text="Mission plot", padding=4)
    mission_plot_frame.grid(row=0, column=1, sticky="nsew")
    mission_plot_frame.columnconfigure(0, weight=1)
    mission_plot_frame.rowconfigure(0, weight=1)

    ttk.Label(mission_controls,
              text="Select variables to plot vs mission time.").grid(
        row=0, column=0, sticky="w")

    mission_var_list = tk.Listbox(mission_controls, selectmode="extended",
                                  height=16, exportselection=False)
    mission_var_list.grid(row=1, column=0, sticky="nsew", pady=(4, 4))
    mission_controls.rowconfigure(1, weight=1)
    mission_controls.columnconfigure(0, weight=1)

    ml_sb = ttk.Scrollbar(mission_controls, orient="vertical",
                           command=mission_var_list.yview)
    ml_sb.grid(row=1, column=1, sticky="ns", pady=(4, 4))
    mission_var_list.configure(yscrollcommand=ml_sb.set)

    _mission_items = []
    for _k, _lbl, _unit in MISSION_VARS:
        mission_var_list.insert(tk.END, f"{_lbl} ({_unit})")
        _mission_items.append((_k, _lbl, _unit))

    mission_canvas_ref = [None]

    def _clear_mission_plot():
        for w in mission_plot_frame.winfo_children():
            w.destroy()
        mission_canvas_ref[0] = None

    def _update_mission_plot():
        ms = last_mission_series[0]
        if ms is None:
            messagebox.showinfo("Mission plot", "Run a mission first.")
            return
        sel = list(mission_var_list.curselection())
        if not sel:
            messagebox.showinfo("Mission plot", "Select at least one variable.")
            return

        t_min = [x / 60.0 for x in ms.get("t_s", [])]
        if not t_min:
            return

        # Group selected variables by unit so same-unit series share a y-axis
        selected = [_mission_items[i] for i in sel]
        by_unit: dict = {}
        for k, lbl, unit in selected:
            by_unit.setdefault(unit, []).append((k, lbl))

        import matplotlib.pyplot as _plt
        fig = _plt.Figure(figsize=(7.5, 4.5), dpi=100)
        ax0 = fig.add_subplot(111)
        unit_list = list(by_unit.keys())

        axes = [(ax0, unit_list[0])]
        ax0.set_ylabel(unit_list[0])
        for ui, unit in enumerate(unit_list[1:], 1):
            axn = ax0.twinx()
            axn.spines["right"].set_position(("outward", 55 * (ui - 1)))
            axn.set_ylabel(unit)
            axes.append((axn, unit))

        lines, labels = [], []
        for ax, unit in axes:
            for key, lbl in by_unit.get(unit, []):
                y = ms.get(key, [])
                if not y:
                    continue
                yy = []
                for v in y:
                    try:
                        yy.append(float("nan") if isinstance(v, float) and v != v
                                  else float(v))
                    except Exception:
                        yy.append(float("nan"))
                ln, = ax.plot(t_min, yy, label=lbl)
                lines.append(ln)
                labels.append(lbl)

        ax0.set_xlabel("Mission time (min)")
        ax0.grid(True)
        fig.suptitle("Mission variables vs time")
        if lines:
            ax0.legend(lines, labels, loc="best")

        _clear_mission_plot()
        mc = FigureCanvasTkAgg(fig, master=mission_plot_frame)
        mc.draw()
        mc.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        mission_canvas_ref[0] = mc

    mission_btns = ttk.Frame(mission_controls)
    mission_btns.grid(row=2, column=0, columnspan=2, sticky="ew")
    ttk.Button(mission_btns, text="Plot selected",
               command=_update_mission_plot).grid(row=0, column=0, padx=(0, 4))
    ttk.Button(mission_btns, text="Clear",
               command=_clear_mission_plot).grid(row=0, column=1)

    def _on_list_wheel(evt):
        if evt.delta:
            mission_var_list.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        return "break"
    mission_var_list.bind("<MouseWheel>", _on_list_wheel)

    # ---- log helper (appends to output text instead of replacing) ----
    def log(msg: str):
        out_text.configure(state="normal")
        out_text.insert("end", msg + "\n")
        out_text.see("end")
        out_text.configure(state="disabled")

    def clear_log():
        out_text.configure(state="normal")
        out_text.delete("1.0", "end")
        out_text.configure(state="disabled")


    # ---- Weight Budget panel ----
    _wb_canvas_ref = [None]

    def _draw_weight_chart(rows):
        import matplotlib.pyplot as _plt
        data_rows = [r for r in rows if r[0] != "TOTAL"]
        if not data_rows:
            return
        labels = [r[0] for r in data_rows]
        totals = [r[3] for r in data_rows]
        grand  = sum(totals)
        COLORS = ["#2E75B6","#ED7D31","#A9D18E","#FFC000","#5B9BD5","#FF7F7F"]
        fig, axes = _plt.subplots(1, 2, figsize=(7, max(3, len(labels)*0.6+1)))
        fig.patch.set_facecolor("white")
        ax = axes[0]
        left_ = 0.0
        pcts = [t/grand*100 if grand>0 else 0 for t in totals]
        for i, (lbl, pct) in enumerate(zip(labels, pcts)):
            ax.barh(0, pct, left=left_, color=COLORS[i % len(COLORS)],
                    label=lbl, edgecolor="white", linewidth=0.5)
            if pct > 5:
                ax.text(left_+pct/2, 0, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=7.5, color="white")
            left_ += pct
        ax.set_xlim(0, 100); ax.set_yticks([])
        ax.set_xlabel("% of total weight"); ax.set_title("Weight Distribution")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.18),
                  ncol=2, fontsize=7, frameon=False)
        ax.grid(axis="x", alpha=0.3)
        ax2 = axes[1]
        _, _, ats = ax2.pie(totals, labels=None, autopct="%1.0f%%",
            colors=COLORS[:len(labels)], startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=0.8))
        for at in ats: at.set_fontsize(7)
        ax2.set_title(f"Total: {grand:.0f} g")
        ax2.legend(labels, loc="lower center", bbox_to_anchor=(0.5,-0.22),
                   ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        if _wb_canvas_ref[0]:
            try: _wb_canvas_ref[0].get_tk_widget().destroy()
            except Exception: pass
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fc = FigureCanvasTkAgg(fig, master=wb_right)
        fc.draw(); fc.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        _wb_canvas_ref[0] = fc; _plt.close(fig)

    wb_outer = ttk.Frame(tab_weight_budget, padding=4)
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
        ("component","Component",160), ("unit_w","Unit (g)",70),
        ("count","Qty",40), ("total_w","Total (g)",75), ("pct","% of Total",70)]:
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
            wb_tv.insert("", "end", values=(label, f"{unit_w:.1f}",
                         str(count), f"{total_w:.1f}", pct))
        if rows:
            label, unit_w, count, total_w = rows[-1]
            wb_tv.insert("", "end", values=(label, "", "",
                         f"{total_w:.1f}", "100%"), tags=("total",))
        _draw_weight_chart(rows)

    # ---- Status panel ----
    style_tv = ttk.Style()
    try: style_tv.theme_use(style_tv.theme_use())
    except Exception: pass

    def _color_tag(value, limit, kind):
        try:
            v = float(value); L = float(limit)
        except Exception: return "na"
        if kind == "max": return "bad" if v > L else ("warn" if v > 0.9*L else "ok")
        else:             return "bad" if v < L else ("warn" if v < 1.1*L else "ok")

    def _make_status_tv(parent, title):
        lf = ttk.LabelFrame(parent, text=title, padding=4)
        lf.pack(fill="both", expand=True, padx=4, pady=4)
        cols = ("metric", "value", "limit", "note")
        tv = ttk.Treeview(lf, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (200, 120, 120, 220)):
            tv.heading(c, text=c.capitalize())
            tv.column(c, width=w, anchor="w" if c in ("metric","note") else "center")
        tv.pack(fill="both", expand=True)
        tv.tag_configure("ok",   background="#d9f2d9")
        tv.tag_configure("warn", background="#fff2cc")
        tv.tag_configure("bad",  background="#f8d7da")
        tv.tag_configure("na",   background="#efefef")
        return tv

    status_scroll = ttk.Frame(tab_status)
    status_scroll.grid(row=0, column=0, sticky="nsew")
    status_scroll.columnconfigure(0, weight=1)

    batt_tv  = _make_status_tv(status_scroll, "Battery Status")
    motor_tv = _make_status_tv(status_scroll, "Motor / ESC Status")
    aero_tv  = _make_status_tv(status_scroll, "Aerodynamic Status")

    def _ins_row(tv, metric, val_str, lim_str, tag, note=""):
        tv.insert("", "end", values=(metric, val_str, lim_str, note), tags=(tag,))

    def _clear_status():
        for tv in (batt_tv, motor_tv, aero_tv):
            for iid in tv.get_children(): tv.delete(iid)

    def update_status(cfg: FixedWingConfig, m: dict):
        _clear_status()
        batt = cfg.battery
        # Battery checks
        Ipack = m.get("pack_current_A", 0.0)
        Vload = m.get("v_load_V", 0.0)
        Ptot  = m.get("total_power_W", 0.0)

        _ins_row(batt_tv, "Pack voltage (loaded)",
                 f"{Vload:.2f} V", f">= {batt.vmin_pack:.2f} V",
                 _color_tag(Vload, batt.vmin_pack, "min"))
        if math.isfinite(batt.discharge_cont_A):
            _ins_row(batt_tv, "Pack current vs cont limit",
                     f"{Ipack:.1f} A", f"<= {batt.discharge_cont_A:.1f} A",
                     _color_tag(Ipack, batt.discharge_cont_A, "max"))
        _ins_row(batt_tv, "Total electrical power",
                 f"{Ptot:.1f} W", "—", "na")
        _ins_row(batt_tv, "Usable energy",
                 f"{batt.usable_Wh:.1f} Wh", "—", "na")

        # Motor checks
        Pmotor  = m.get("motor_power_W", 0.0)
        rpm     = m.get("rpm_est", 0.0)
        T_avail = m.get("thrust_available_N", 0.0)
        T_req   = m.get("thrust_required_N", 0.0)
        pmax    = m.get("max_prop_power_W", cfg.motor.max_power * cfg.num_motors)
        _ins_row(motor_tv, "Motor power vs max",
                 f"{Pmotor:.1f} W", f"<= {pmax:.1f} W",
                 _color_tag(Pmotor, pmax, "max"))
        _ins_row(motor_tv, "Thrust Available",
                 f"{T_avail:.2f} N", "—", "na")
        _ins_row(motor_tv, "Thrust Required",
                 f"{T_req:.2f} N", f"<= {T_avail:.2f} N",
                 _color_tag(T_req, T_avail, "max"),
                 "Must be < T_avail for level flight")
        _ins_row(motor_tv, "Estimated RPM",
                 f"{rpm:.0f}", "—", "na")
        V_tip = m.get("tip_speed_mps", 0.0)
        _ins_row(motor_tv, "Prop Tip Speed",
                 f"{V_tip:.1f} m/s", "<= 200 m/s",
                 _color_tag(V_tip, 200, "max"),
                 "Keep below Mach 0.6 (~200 m/s)")

        # Aerodynamic checks
        V_stall = m.get("stall_speed_mps", 0.0)
        V_cruise= m.get("airspeed_mps", 0.0)
        CL      = m.get("CL", 0.0)
        LD      = m.get("LD_ratio", 0.0)
        RC      = m.get("rate_of_climb_mps", 0.0)
        _ins_row(aero_tv, "Cruise vs Stall speed",
                 f"{V_cruise:.1f} m/s", f"> {V_stall:.1f} m/s",
                 _color_tag(V_cruise, V_stall * 1.1, "min"),
                 "Recommend > 1.1 × V_stall")
        _ins_row(aero_tv, "CL at cruise",
                 f"{CL:.3f}", f"< {cfg.airframe.CL_max:.3f}",
                 _color_tag(CL, cfg.airframe.CL_max, "max"))
        _ins_row(aero_tv, "L/D ratio",    f"{LD:.2f}", "—", "na")
        _ins_row(aero_tv, "Rate of Climb",f"{RC*60:.1f} m/min", "> 0", "na")

    # ---- Metrics panel ----
    metrics_frame = ttk.Frame(tab_metrics, padding=4)
    metrics_frame.grid(row=0, column=0, sticky="nsew")
    metrics_frame.columnconfigure(0, weight=1)
    metrics_frame.rowconfigure(0, weight=1)

    metrics_tv = ttk.Treeview(metrics_frame,
                               columns=("metric","value"), show="headings", height=30)
    metrics_tv.heading("metric", text="Metric")
    metrics_tv.heading("value",  text="Value")
    metrics_tv.column("metric", width=300, anchor="w")
    metrics_tv.column("value",  width=280, anchor="w")
    metrics_sb = ttk.Scrollbar(metrics_frame, orient="vertical", command=metrics_tv.yview)
    metrics_tv.configure(yscrollcommand=metrics_sb.set)
    metrics_tv.grid(row=0, column=0, sticky="nsew")
    metrics_sb.grid(row=0, column=1, sticky="ns")

    try:
        ttk.Style().configure("Metrics.Treeview", rowheight=24)
        metrics_tv.configure(style="Metrics.Treeview")
        metrics_tv.tag_configure("section", font=("TkDefaultFont", 10, "bold"))
    except Exception:
        pass

    def _clear_metrics():
        for iid in metrics_tv.get_children(): metrics_tv.delete(iid)

    def _ins_metric(label, val_str):
        metrics_tv.insert("", "end", values=(label, val_str))

    def _sep_metric(label=""):
        metrics_tv.insert("", "end", values=(f"── {label} ──", ""), tags=("section",))

    def update_metrics(cfg: FixedWingConfig, m: dict):
        _clear_metrics()
        af  = cfg.airframe
        batt= cfg.battery
        prop= cfg.propeller

        _sep_metric("Aircraft")
        _ins_metric("Total Weight",              f"{cfg.aircraft_weight_g:.0f} g  ({cfg.weight_N:.2f} N)")
        _ins_metric("Wing Span",                 f"{af.wing_span_m:.3f} m")
        _ins_metric("Wing Area",                 f"{af.wing_area_m2:.4f} m²")
        _ins_metric("Mean Chord",                f"{af.chord_m:.4f} m")
        _ins_metric("Aspect Ratio",              f"{af.aspect_ratio:.2f}")
        _ins_metric("Induced Drag Factor k",     f"{af.k:.5f}")
        _ins_metric("Lift-curve Slope a",        f"{af.lift_curve_slope():.3f} /rad")

        _sep_metric("Aerodynamics at Cruise")
        V  = m.get("airspeed_mps", cfg.cruise_speed_mps)
        _ins_metric("Cruise Airspeed",           f"{V:.2f} m/s  ({V*3.6:.1f} km/h)")
        _ins_metric("Stall Speed",               f"{m.get('stall_speed_mps',0):.2f} m/s  ({m.get('stall_speed_mps',0)*3.6:.1f} km/h)")
        _ins_metric("CL at Cruise",              f"{m.get('CL',0):.4f}")
        _ins_metric("CD at Cruise",              f"{m.get('CD',0):.5f}")
        _ins_metric("L/D Ratio",                 f"{m.get('LD_ratio',0):.2f}")
        _ins_metric("Angle of Attack",           f"{m.get('aoa_deg',0):.2f} °")
        _ins_metric("Reynolds Number",           f"{m.get('reynolds_number',0):,.0f}")

        _sep_metric("Thrust & Power")
        _ins_metric("Thrust Required",           f"{m.get('thrust_required_N',0):.2f} N")
        _ins_metric("Thrust Available",          f"{m.get('thrust_available_N',0):.2f} N")
        _ins_metric("Specific Thrust (T/W)",     f"{m.get('specific_thrust',0):.3f}")
        _ins_metric("Power Required (propulsive)",f"{m.get('power_required_W',0):.1f} W")
        _ins_metric("Motor Electrical Power",    f"{m.get('motor_power_W',0):.1f} W")
        _ins_metric("ESC Losses",                f"{m.get('esc_loss_W',0):.1f} W")
        _ins_metric("Avionics Power",            f"{m.get('avionics_power_W',0):.1f} W")
        _ins_metric("Total Electrical Power",    f"{m.get('total_power_W',0):.1f} W")
        _ins_metric("Max Propeller Power",       f"{m.get('max_prop_power_W',0):.1f} W")

        _sep_metric("Battery")
        _ins_metric("Pack Voltage (no load)",    f"{batt.vmax_pack:.2f} V  ({cfg.num_motors}S equiv)")
        _ins_metric("Pack Voltage (under load)", f"{m.get('v_load_V',0):.2f} V")
        _ins_metric("Pack Current",              f"{m.get('pack_current_A',0):.2f} A")
        _ins_metric("Pack Capacity",             f"{batt.capacity_mAh:.0f} mAh")
        _ins_metric("Pack Energy",               f"{batt.capacity_Wh:.2f} Wh")
        _ins_metric("Usable Energy",             f"{batt.usable_Wh:.2f} Wh")

        _sep_metric("Propeller")
        _ins_metric("Diameter",                  f"{prop.diameter_in:.1f} in  ({prop.diameter_m*100:.1f} cm)")
        _ins_metric("Pitch",                     f"{prop.pitch_in:.1f} in  ({prop.pitch_m*100:.1f} cm)")
        _ins_metric("Estimated RPM",             f"{m.get('rpm_est',0):.0f}")
        _ins_metric("Tip Speed",                 f"{m.get('tip_speed_mps',0):.1f} m/s  ({m.get('tip_speed_mps',0)*3.6:.1f} km/h)")
        _ins_metric("Pitch Speed",               f"{m.get('pitch_speed_mps',0):.1f} m/s  ({m.get('pitch_speed_mps',0)*3.6:.1f} km/h)")

        _sep_metric("Climb Performance")
        _ins_metric("Rate of Climb @ Cruise",    f"{m.get('rate_of_climb_mps',0)*60:.1f} m/min  ({m.get('rate_of_climb_mps',0):.2f} m/s)")
        _ins_metric("Max Rate of Climb",         f"{m.get('max_rc_mps',0)*60:.1f} m/min  @ {m.get('v_max_rc_mps',0):.1f} m/s")
        _ins_metric("Max Angle of Climb",        f"{m.get('max_aoc_deg',0):.1f} °  @ {m.get('v_max_aoc_mps',0):.1f} m/s")
        _ins_metric("Takeoff Ground Roll",       f"{m.get('takeoff_dist_m', float('inf')):.1f} m"
                    if math.isfinite(m.get('takeoff_dist_m', float('inf')))
                    else "∞ (T < rolling friction)")

        _sep_metric("Optimal Speeds")
        _ins_metric("Best Endurance Speed",      f"{m.get('best_endurance_speed_mps',0):.1f} m/s  ({m.get('best_endurance_speed_mps',0)*3.6:.1f} km/h)")
        _ins_metric("Best Range Speed",          f"{m.get('best_range_speed_mps',0):.1f} m/s  ({m.get('best_range_speed_mps',0)*3.6:.1f} km/h)")
        _ins_metric("Max L/D Ratio",             f"{m.get('best_ld_ratio',0):.2f}")

        _sep_metric("Endurance & Range @ Cruise")
        _ins_metric("Flight Time",               f"{m.get('flight_time_min',0):.1f} min")
        _ins_metric("Flight Range",              f"{m.get('flight_range_km',0):.2f} km")

    # ---- Output text ----
    out_frame = ttk.LabelFrame(right, text="Output", padding=4)
    out_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
    out_frame.columnconfigure(0, weight=1)
    out_frame.rowconfigure(0, weight=1)

    out_text = tk.Text(out_frame, height=8, wrap="word", state="disabled")
    out_text.grid(row=0, column=0, sticky="nsew")
    out_sb   = ttk.Scrollbar(out_frame, orient="vertical", command=out_text.yview)
    out_sb.grid(row=0, column=1, sticky="ns")
    out_text.configure(yscrollcommand=out_sb.set)

    def out_print(msg: str):
        out_text.configure(state="normal")
        out_text.delete("1.0", "end")
        out_text.insert("end", msg)
        out_text.configure(state="disabled")

    # ---- Build config from GUI ----
    def build_config() -> FixedWingConfig:
        alt  = safe_float(v_altitude.get(), 0.0)
        temp = parse_float("Temp", v_temp.get())
        pres = parse_float("Pressure", v_pressure.get())
        rho  = isa_density(alt, temp, pres)

        af = AirframeConfig(
            wing_span_m     = parse_float("Wing span",   v_wing_span.get()),
            wing_area_m2    = parse_float("Wing area",   v_wing_area.get()),
            CD0             = safe_float(v_CD0.get(), 0.030),
            CL_max          = safe_float(v_CL_max.get(), 1.30),
            oswald          = safe_float(v_oswald.get(), 0.82),
            mu_roll         = safe_float(v_mu_roll.get(), 0.04),
            CL_takeoff      = safe_float(v_CL_takeoff.get(), 0.80),
            prop_efficiency = safe_float(v_prop_eff.get(), 0.75),
            num_motors      = parse_int("Num motors", v_num_motors.get()) or 1,
        )

        batt = BatteryConfig(
            chemistry               = v_batt_chem.get().strip() or None,
            operating_voltage_min   = safe_float(v_batt_vmin.get(), 3.5),
            operating_voltage_nominal = safe_float(v_batt_vnom.get(), 3.8),
            operating_voltage_max   = safe_float(v_batt_vmax.get(), 4.2),
            unit_mode               = v_batt_unit_mode.get().strip().lower() or "cell",
            series_units            = parse_int("Series", v_batt_series.get()) or 1,
            parallel_units          = parse_int("Parallel", v_batt_parallel.get()) or 1,
            cells_series_per_unit   = parse_int("Cells series", v_batt_cells_s.get()) or 1,
            cells_parallel_per_unit = parse_int("Cells parallel", v_batt_cells_p.get()) or 1,
            cell_capacity_mAh       = parse_float("Cell cap", v_batt_cell_cap.get()),
            pack_capacity_mAh       = parse_float("Pack cap", v_batt_pack_cap.get()),
            cell_weight_g           = parse_float("Cell wt", v_batt_cell_wt.get()),
            pack_weight_g           = parse_float("Pack wt", v_batt_pack_wt.get()),
            unit_energy_density     = parse_float("Dens", v_batt_dens.get()),
            charge_current_max      = parse_float("Chg", v_batt_chg.get()),
            discharge_cont_A        = parse_float("A_cont", v_batt_a_cont.get()),
            discharge_max_A         = parse_float("A_max", v_batt_a_max.get()),
            discharge_c_cont        = parse_float("C_cont", v_batt_c_cont.get()),
            discharge_c_max         = parse_float("C_max", v_batt_c_max.get()),
            discharge_percent       = safe_float(v_batt_dischg_pct.get(), 80.0),
            resistance_cell_mOhm    = safe_float(v_batt_r.get(), 0.0),
        )

        motor = MotorConfig(
            kv            = parse_float("Kv", v_motor_kv.get()),
            idle_current  = safe_float(v_motor_i0.get(), 0.5),
            idle_voltage  = safe_float(v_motor_v0.get(), 7.0),
            rated_voltage = parse_int("Rated V", v_motor_rated_v.get()) or 16,
            resistance    = safe_float(v_motor_r.get(), 0.05),
            max_current   = safe_float(v_motor_imax.get(), 40),
            max_power     = safe_float(v_motor_pmax.get(), 500),
            pole_count    = parse_int("Poles", v_motor_poles.get()),
            weight_g      = parse_float("Motor wt", v_motor_wt.get()),
            size_mm       = v_motor_size.get().strip() or None,
        )

        esc_fields = [v_esc_vrating.get().strip(), v_esc_cont.get().strip(),
                      v_esc_max.get().strip(),    v_esc_idle.get().strip(),
                      v_esc_r.get().strip(),       v_esc_wt.get().strip()]
        esc = None
        if any(esc_fields):
            esc = ESCConfig(
                voltage_rating      = parse_int("ESC V", v_esc_vrating.get()) or 30,
                continuous_current_A= safe_float(v_esc_cont.get(), 30),
                max_current_A       = safe_float(v_esc_max.get(), 40),
                idle_current_A      = safe_float(v_esc_idle.get(), 0.1),
                resistance          = safe_float(v_esc_r.get(), 0.01),
                weight_g            = parse_float("ESC wt", v_esc_wt.get()),
            )

        avionics = AvionicsConfig(
            voltage_tree = _av_tree_to_dict())

        prop_table_path = v_prop_table.get().strip() or None
        prop = PropellerConfig(
            diameter_in  = safe_float(v_prop_d.get(), 10),
            pitch_in     = safe_float(v_prop_pitch.get(), 4.5),
            blades       = parse_int("Blades", v_prop_blades.get()) or 2,
            max_rpm      = safe_float(v_prop_maxrpm.get(), 0),
            max_thrust_g = safe_float(v_prop_maxthr.get(), 0),
            table_csv    = prop_table_path,
            TConst       = parse_float("TConst", v_prop_tconst.get()),
            PConst       = parse_float("PConst", v_prop_pconst.get()),
            weight_g     = parse_float("Prop wt", v_prop_wt.get()),
        )

        return FixedWingConfig(
            airframe            = af,
            battery             = batt,
            motor               = motor,
            propeller           = prop,
            aircraft_weight_g   = safe_float(v_weight.get(), 2500),
            cruise_speed_mps    = safe_float(v_cruise_speed.get(), 18.0),
            periph_current_A    = safe_float(v_periph_cur.get(), 0.0),
            esc                 = esc,
            avionics            = avionics,
            air_density         = rho,
        )

    # ---- Run: single-point ----
    def run_single_point():
        clear_log()
        try:
            cfg = build_config()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return
        try:
            V_cruise = cfg.cruise_speed_mps
            m = compute_metrics(cfg, V_cruise)

            # Wind correction: headwind reduces groundspeed → shorter range
            wind = safe_float(v_wind.get(), 0.0)
            if wind != 0.0:
                V_gs = max(V_cruise - wind, 0.1)
                if m["flight_time_min"] > 0:
                    m["flight_range_km"] = V_gs * (m["flight_time_min"] * 60.0) / 1000.0

            V_stall = m["stall_speed_mps"]
            V_be    = m["best_endurance_speed_mps"]
            V_br    = m["best_range_speed_mps"]

            update_status(cfg, m)
            update_metrics(cfg, m)

            max_v = safe_float(v_max_v_plot.get(), 40.0)

            # Cache for View-menu re-render
            _last_run["cfg"]     = cfg
            _last_run["max_v"]   = max_v
            _last_run["metrics"] = m
            _last_run["wind"]    = wind

            fig = make_performance_figure(
                cfg,
                max_speed=max_v,
                figsize=(_view["plot_w"], _view["plot_h"]),
            )
            # Generate motor operating point figure if available
            motor_fig = None
            if cfg.propeller.table is not None and m:
                try:
                    motor_fig = make_motor_operating_point_figure(cfg, m, figsize=(_view["plot_w"], 6))
                except Exception:
                    pass
            # Display both figures
            if motor_fig:
                show_figure([fig, motor_fig])
            else:
                show_figure(fig)
            display_nb.select(tab_plots)
            # Capture sweep data for export
            _sp_vs = [stall_speed(cfg)+0.1 + (max_v-stall_speed(cfg)-0.1)*i/300
                       for i in range(301)]
            _sp_vs = [max(v, 0.1) for v in _sp_vs]
            _last_run_sweep.clear()
            _last_run_sweep.update({
                "Speed (m/s)":           _sp_vs,
                "Flight Time (min)":     [flight_time_min(cfg, v) for v in _sp_vs],
                "Range (km)":            [flight_range_km(cfg, v) for v in _sp_vs],
                "Power Required (W)":    [power_required_W(cfg, v) for v in _sp_vs],
                "Drag (N)":              [drag_N(cfg, v) for v in _sp_vs],
                "Rate of Climb (m/s)":   [rate_of_climb_mps(cfg, v) for v in _sp_vs],
                "L/D Ratio":             [cfg.airframe.ld_ratio(
                    cfg.airframe.cl_at_speed(cfg.weight_N, v, cfg.air_density))
                    for v in _sp_vs],
            })
            _last_run_cfg[0] = cfg
            update_weight_budget(cfg)

            log(f"=== Fixed-Wing Single-Point @ {V_cruise:.1f} m/s ({V_cruise*3.6:.1f} km/h) ===")
            log(f"Stall Speed   : {V_stall:.1f} m/s")
            log(f"L/D Ratio     : {m['LD_ratio']:.2f}")
            log(f"Thrust Req    : {m['thrust_required_N']:.2f} N  |  Avail: {m['thrust_available_N']:.2f} N")
            log(f"Total Power   : {m['total_power_W']:.1f} W")
            log(f"Flight Time   : {m['flight_time_min']:.1f} min")
            log(f"Flight Range  : {m['flight_range_km']:.2f} km")
            log(f"Rate of Climb : {m['rate_of_climb_mps']*60:.0f} m/min")
            log(f"Max RC        : {m['max_rc_mps']*60:.0f} m/min  @ {m['v_max_rc_mps']:.1f} m/s")
            log(f"Best Endurance: {V_be:.1f} m/s  |  Best Range: {V_br:.1f} m/s")
            log(f"Takeoff Roll  : {m['takeoff_dist_m']:.1f} m")
        except Exception as e:
            import traceback
            messagebox.showerror("Simulation error", traceback.format_exc())

    # ---- Run: mission JSON ----
    def run_mission():
        clear_log()
        mission_path = v_mission.get().strip()
        if not mission_path:
            messagebox.showerror("No mission file",
                "Select a mission JSON file in the Environment tab first.")
            return
        try:
            cfg = build_config()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return
        try:
            mission = MissionProfile.from_json(mission_path)
            wind    = safe_float(v_wind.get(), 0.0)
            temp    = v_temp.get().strip()
            pres    = v_pressure.get().strip()

            results, worst_m, series = simulate_fw_mission(
                cfg, mission,
                wind_mps      = wind,
                temperature_C = float(temp) if temp else None,
                pressure_Pa   = float(pres) if pres else None,
            )

            # Stash series for Mission Plots tab
            last_mission_series[0] = series

            # Populate status with worst-case metrics
            if worst_m:
                update_status(cfg, worst_m)

            # Rebuild metrics from worst-case point
            if worst_m:
                update_metrics(cfg, worst_m)

            max_v = safe_float(v_max_v_plot.get(), 40.0)
            _last_run["cfg"]   = cfg
            _last_run["max_v"] = max_v
            _last_run["metrics"] = worst_m

            _last_run_cfg[0] = cfg
            update_weight_budget(cfg)
            fig = make_performance_figure(
                cfg, max_speed=max_v,
                figsize=(_view["plot_w"], _view["plot_h"]))
            # Generate motor operating point figure if available
            motor_fig = None
            if cfg.propeller.table is not None and worst_m:
                try:
                    motor_fig = make_motor_operating_point_figure(cfg, worst_m, figsize=(_view["plot_w"], 6))
                except Exception:
                    pass
            # Display both figures
            if motor_fig:
                show_figure([fig, motor_fig])
            else:
                show_figure(fig)
            display_nb.select(tab_plots)

            import os as _os
            log(f"=== Fixed-Wing Mission: {_os.path.basename(mission_path)} ===")
            log(f"Wind: {wind:+.1f} m/s  |  Phases: {len(results)}")
            log("")
            total_t, total_d = 0.0, 0.0
            for name, t_m, d_k, status in results:
                total_t += t_m; total_d += d_k
                log(f"  {name:<18} {t_m:6.1f} min  {d_k:6.2f} km  — {status}")
            log("")
            log(f"TOTAL:             {total_t:6.1f} min  {total_d:6.2f} km")
            log("")
            log("Switch to Mission Plots tab and select variables to visualise.")
        except Exception as e:
            import traceback
            messagebox.showerror("Mission error", traceback.format_exc())


    # ---- Status TV pairs (for PDF report) ----
    _status_tv_pairs = [
        (batt_tv,  "Battery Status"),
        (motor_tv, "Motor / ESC Status"),
        (aero_tv,  "Aerodynamic Status"),
    ]
    _report_title = "Fixed-Wing Power Simulator — Performance Analysis"
    _last_run_sweep: dict = {}
    _last_run_cfg   = [None]

    def _get_metrics_rows() -> list:
        rows = []
        for iid in metrics_tv.get_children():
            vals = metrics_tv.item(iid, "values")
            if vals and len(vals) >= 2:
                rows.append((str(vals[0]), str(vals[1])))
        return rows

    def _get_status_sections() -> list:
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
            messagebox.showinfo("No data", "Run a simulation first.")
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
            messagebox.showinfo("No data", "Run a simulation first.")
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
            cfg  = _last_run_cfg[0]
            figs = []
            # Add all current plot figures from the scrollable area
            figs.extend(_current_plot_figs)
            if mission_canvas_ref[0] is not None:
                figs.append(mission_canvas_ref[0].figure)
            for num in plt.get_fignums():
                fig = plt.figure(num)
                if fig not in figs:
                    figs.append(fig)
            wb   = _extract_weight_budget(cfg) if cfg else []
            _generate_pdf_report(
                path            = path,
                report_title    = _report_title,
                inputs_rows     = _get_inputs_rows(),
                metrics_rows    = _get_metrics_rows(),
                status_sections = _get_status_sections(),
                log_text        = _get_log_text(),
                figures         = figs,
                weight_budget   = wb,
            )
            messagebox.showinfo("Report generated", f"PDF report saved to:\n{path}")
        except Exception as e:
            import traceback
            messagebox.showerror("Report error", traceback.format_exc())

    # ---- Buttons ----
    btn_frame = ttk.Frame(main)
    btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    btn_frame.columnconfigure(2, weight=1)  # spacer

    ttk.Button(btn_frame, text="▶  Run Single-Point",
               command=run_single_point).grid(row=0, column=0, padx=(0, 6), pady=4)
    ttk.Button(btn_frame, text="🗺  Run Mission (JSON)",
               command=run_mission).grid(row=0, column=1, padx=(0, 6), pady=4)

    # ---- Save / Load config ----
    def save_cfg():
        path = filedialog.asksaveasfilename(
            title="Save Config", defaultextension=".json",
            filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        # Persist avionics as a list of row dicts so it round-trips cleanly
        av_rows = []
        for iid in av_tree.get_children():
            vals = av_tree.item(iid, "values")
            try:
                av_rows.append({
                    "voltage":    float(vals[0]),
                    "current":    float(vals[1]),
                    "eff":        float(vals[2]),
                })
            except Exception:
                pass
        data = {
            "schema":        "fixedwing_power_sim_v1",
            "vars":          {k: v.get() for k, v in config_vars.items()},
            "avionics_rows": av_rows,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Saved", f"Configuration saved to:\n{path}")

    def load_cfg():
        path = filedialog.askopenfilename(
            title="Load Config",
            filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        with open(path) as f:
            data = json.load(f)
        for k, val in data.get("vars", {}).items():
            if k in config_vars:
                try: config_vars[k].set("" if val is None else str(val))
                except Exception: pass
        on_unit_mode()
        # Restore avionics table — prefer structured rows, fall back to string var
        av_rows = data.get("avionics_rows", None)
        if isinstance(av_rows, list) and av_rows:
            _av_load_rows(av_rows)
        else:
            # Legacy: parse from the string var that was just restored
            try:
                d = parse_voltage_tree(v_avionics_str.get().strip() or None)
                _av_load_rows([{"voltage": v, "current": ci[0], "eff": ci[1]}
                               for v, ci in sorted(d.items())])
            except Exception:
                pass
        messagebox.showinfo("Loaded", f"Configuration loaded from:\n{path}")

    file_menu.add_command(label="Load Config…", command=load_cfg)
    file_menu.add_command(label="Save Config…", command=save_cfg)
    
    # Wire export items into File menu
    file_menu.add_separator()
    file_menu.add_command(label="Export CSV…",          command=_do_export_csv)
    file_menu.add_command(label="Export Excel…",        command=_do_export_excel)
    file_menu.add_command(label="Generate PDF Report…", command=_do_generate_report)

    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=exit_app)

    ttk.Button(btn_frame, text="💾  Save Config", command=save_cfg).grid(row=0, column=3, padx=4)
    ttk.Button(btn_frame, text="📂  Load Config", command=load_cfg).grid(row=0, column=4, padx=4)
    ttk.Button(btn_frame, text="📊  Export CSV",
               command=_do_export_csv).grid(row=0, column=5, padx=4)
    ttk.Button(btn_frame, text="📗  Export Excel",
               command=_do_export_excel).grid(row=0, column=6, padx=4)
    ttk.Button(btn_frame, text="📄  Generate Report",
               command=_do_generate_report).grid(row=0, column=7, padx=4)

    root.protocol("WM_DELETE_WINDOW", exit_app)
    root.mainloop()


# ============================================================
# CLI
# ============================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Fixed-Wing UAV Power Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gui", action="store_true", help="Launch Tkinter GUI")

    # Airframe
    p.add_argument("--weight",         type=float, help="Total aircraft weight (g)")
    p.add_argument("--num_motors",     type=int,   default=1)
    p.add_argument("--wing_span",      type=float, help="Wing span (m)")
    p.add_argument("--wing_area",      type=float, help="Wing area (m²)")
    p.add_argument("--CD0",            type=float, default=0.028)
    p.add_argument("--CL_max",         type=float, default=1.30)
    p.add_argument("--oswald",         type=float, default=0.82)
    p.add_argument("--mu_roll",        type=float, default=0.04)
    p.add_argument("--CL_takeoff",     type=float, default=0.80)
    p.add_argument("--prop_efficiency",type=float, default=0.75)
    p.add_argument("--cruise_speed",   type=float, default=18.0)
    p.add_argument("--periph_current", type=float, default=0.0)

    # Battery
    p.add_argument("--battery_operating_voltage_min",      type=float, default=3.5)
    p.add_argument("--battery_operating_voltage_nominal",  type=float, default=3.8)
    p.add_argument("--battery_operating_voltage_max",      type=float, default=4.2)
    p.add_argument("--battery_series_units",               type=int,   default=4)
    p.add_argument("--battery_parallel_units",             type=int,   default=1)
    p.add_argument("--battery_cells_series_per_unit",      type=int,   default=1)
    p.add_argument("--battery_cells_parallel_per_unit",    type=int,   default=1)
    p.add_argument("--battery_cell_capacity",              type=float, help="Cell capacity (mAh)")
    p.add_argument("--battery_pack_capacity",              type=float, help="Pack capacity (mAh)")
    p.add_argument("--battery_cell_weight_g",              type=float)
    p.add_argument("--battery_pack_weight_g",              type=float)
    p.add_argument("--battery_unit_mode",                  choices=["cell","pack"], default="cell")
    p.add_argument("--battery_energy_density",             type=float)
    p.add_argument("--battery_charge_current_max",         type=float)
    p.add_argument("--battery_discharge_cont_A",           type=float)
    p.add_argument("--battery_discharge_max_A",            type=float)
    p.add_argument("--battery_discharge_c_cont",           type=float)
    p.add_argument("--battery_discharge_c_max",            type=float)
    p.add_argument("--battery_discharge_percent",          type=float, default=80.0)
    p.add_argument("--battery_resistance_cell",            type=float, default=5.0, help="mΩ")
    p.add_argument("--battery_chemistry",                  type=str,   default="LiPo")

    # Motor
    p.add_argument("--motor_kv",           type=float)
    p.add_argument("--motor_idle_current",  type=float, default=0.5)
    p.add_argument("--motor_idle_voltage",  type=float, default=7.0)
    p.add_argument("--motor_rated_voltage", type=int,   default=16)
    p.add_argument("--motor_resistance",    type=float, default=0.06)
    p.add_argument("--motor_max_current",   type=float, default=40)
    p.add_argument("--motor_max_power",     type=float, default=500)
    p.add_argument("--motor_pole_count",    type=int)
    p.add_argument("--motor_weight",        type=float)

    # Propeller
    p.add_argument("--prop_diameter", type=float, default=10)
    p.add_argument("--prop_pitch",    type=float, default=4.5)
    p.add_argument("--prop_blades",   type=int,   default=2)
    p.add_argument("--prop_table",    type=str,   default=None)

    # Environment
    p.add_argument("--altitude",    type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--wind",        type=float, default=0.0)
    p.add_argument("--plot",        action="store_true")

    return p


def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    # ---- CLI mode ----
    rho = isa_density(args.altitude, args.temperature)
    print(f"Air density: {rho:.4f} kg/m³  at altitude {args.altitude:.0f} m")

    af = AirframeConfig(
        wing_span_m     = args.wing_span,
        wing_area_m2    = args.wing_area,
        CD0             = args.CD0,
        CL_max          = args.CL_max,
        oswald          = args.oswald,
        mu_roll         = args.mu_roll,
        CL_takeoff      = args.CL_takeoff,
        prop_efficiency = args.prop_efficiency,
        num_motors      = args.num_motors,
    )

    batt = BatteryConfig(
        chemistry               = args.battery_chemistry,
        operating_voltage_min   = args.battery_operating_voltage_min,
        operating_voltage_nominal = args.battery_operating_voltage_nominal,
        operating_voltage_max   = args.battery_operating_voltage_max,
        unit_mode               = args.battery_unit_mode,
        series_units            = args.battery_series_units,
        parallel_units          = args.battery_parallel_units,
        cells_series_per_unit   = args.battery_cells_series_per_unit,
        cells_parallel_per_unit = args.battery_cells_parallel_per_unit,
        cell_capacity_mAh       = args.battery_cell_capacity,
        pack_capacity_mAh       = args.battery_pack_capacity,
        cell_weight_g           = args.battery_cell_weight_g,
        pack_weight_g           = args.battery_pack_weight_g,
        unit_energy_density     = args.battery_energy_density,
        charge_current_max      = args.battery_charge_current_max,
        discharge_cont_A        = args.battery_discharge_cont_A,
        discharge_max_A         = args.battery_discharge_max_A,
        discharge_c_cont        = args.battery_discharge_c_cont,
        discharge_c_max         = args.battery_discharge_c_max,
        discharge_percent       = args.battery_discharge_percent,
        resistance_cell_mOhm    = args.battery_resistance_cell,
    )

    motor = MotorConfig(
        kv            = args.motor_kv,
        idle_current  = args.motor_idle_current,
        idle_voltage  = args.motor_idle_voltage,
        rated_voltage = args.motor_rated_voltage,
        resistance    = args.motor_resistance,
        max_current   = args.motor_max_current,
        max_power     = args.motor_max_power,
        pole_count    = args.motor_pole_count,
        weight_g      = args.motor_weight,
    )

    prop = PropellerConfig(
        diameter_in = args.prop_diameter,
        pitch_in    = args.prop_pitch,
        blades      = args.prop_blades,
        table_csv   = args.prop_table,
    )

    cfg = FixedWingConfig(
        airframe          = af,
        battery           = batt,
        motor             = motor,
        propeller         = prop,
        aircraft_weight_g = args.weight,
        cruise_speed_mps  = args.cruise_speed,
        periph_current_A  = args.periph_current,
        air_density       = rho,
    )

    V_cruise = args.cruise_speed
    m = compute_metrics(cfg, V_cruise)

    print(f"\n{'='*55}")
    print(f"  Fixed-Wing UAV Performance @ {V_cruise:.1f} m/s ({V_cruise*3.6:.1f} km/h)")
    print(f"{'='*55}")
    print(f"  Wing span / area      : {af.wing_span_m:.2f} m / {af.wing_area_m2:.3f} m²")
    print(f"  Chord / AR            : {af.chord_m:.3f} m / {af.aspect_ratio:.2f}")
    print(f"  Reynolds Number       : {m['reynolds_number']:,.0f}")
    print(f"  Stall Speed           : {m['stall_speed_mps']:.2f} m/s ({m['stall_speed_mps']*3.6:.1f} km/h)")
    print(f"  CL / CD               : {m['CL']:.4f} / {m['CD']:.5f}")
    print(f"  L/D Ratio             : {m['LD_ratio']:.2f}")
    print(f"  Angle of Attack       : {m['aoa_deg']:.2f} °")
    print(f"  Drag (thrust req)     : {m['drag_N']:.2f} N")
    print(f"  Thrust available      : {m['thrust_available_N']:.2f} N")
    print(f"  Specific Thrust (T/W) : {m['specific_thrust']:.3f}")
    print(f"  Tip Speed             : {m['tip_speed_mps']:.1f} m/s")
    print(f"  Pitch Speed           : {m['pitch_speed_mps']:.1f} m/s")
    print(f"  Motor Power           : {m['motor_power_W']:.1f} W")
    print(f"  Total Elec. Power     : {m['total_power_W']:.1f} W")
    print(f"  Max Prop Power        : {m['max_prop_power_W']:.1f} W")
    print(f"  Rate of Climb         : {m['rate_of_climb_mps']*60:.1f} m/min")
    print(f"  Max Rate of Climb     : {m['max_rc_mps']*60:.1f} m/min @ {m['v_max_rc_mps']:.1f} m/s")
    print(f"  Max Angle of Climb    : {m['max_aoc_deg']:.1f} °  @ {m['v_max_aoc_mps']:.1f} m/s")
    print(f"  Takeoff Ground Roll   : {m['takeoff_dist_m']:.1f} m")
    print(f"  Best Endurance Speed  : {m['best_endurance_speed_mps']:.1f} m/s ({m['best_endurance_speed_mps']*3.6:.1f} km/h)")
    print(f"  Best Range Speed      : {m['best_range_speed_mps']:.1f} m/s ({m['best_range_speed_mps']*3.6:.1f} km/h)")
    print(f"  Flight Time           : {m['flight_time_min']:.1f} min")
    print(f"  Flight Range          : {m['flight_range_km']:.2f} km")
    print(f"{'='*55}\n")

    V_be, t_best, V_br, d_best = find_optimal_speeds(cfg)
    print(f"  Best endurance: {V_be:.1f} m/s → {t_best:.1f} min")
    print(f"  Best range    : {V_br:.1f} m/s → {d_best:.2f} km")

    if args.plot:
        fig = make_performance_figure(cfg)
        plt.show()


if __name__ == "__main__":
    main()
