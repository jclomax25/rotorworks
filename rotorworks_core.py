#!/usr/bin/env python3
"""
rotorworks_core.py
==================
Code shared by the RotorWorks multicopter and fixed-wing simulators.

Why this file exists
--------------------
The two simulators grew separately and accumulated a large amount of
duplicated, domain-neutral code: the battery state-of-charge model, the
standard atmosphere, GUI tooltips, and assorted parsing helpers. Keeping two
copies cost real bugs — a tooltip fix that had to be applied twice and was
nearly missed the second time, two atmosphere implementations that silently
disagreed about pressure overrides, and an ESC feature that existed in one
CLI and not the other for the project's whole history.

Everything here is aircraft-agnostic. Anything that reads a rotor count, a
wing area, or a tilt angle stays in the simulator that owns it.

Deliberately NOT extracted
--------------------------
* ``BatteryConfig`` — the two constructors take genuinely different
  parameters (the multicopter carries temperature limits and an energy-density
  override the fixed-wing does not). The SoC machinery they both rely on IS
  here; merging the container classes needs its own careful pass.
* ``ESCConfig`` / ``AvionicsConfig`` — small, and structured differently
  enough that merging would mean changing one simulator's field names.
* Export, plotting, and GUI-construction code — all of it reaches into
  simulator-specific config attributes.

Importing
---------
Both simulators expect this file to sit beside them. tkinter is imported
lazily inside the tooltip so that headless CLI use works on machines without
tk installed, matching how the simulators themselves behave.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    # atmosphere
    "G0", "RHO0", "T0_K", "P0_PA", "LAPSE_K_PER_M", "R_AIR",
    "air_density",
    # interpolation and parsing
    "interp_linear_clamped", "eval_poly", "parse_float_list",
    "parse_soc_breakpoints",
    # state of charge
    "SOC_PRESETS", "SOC_PRESET_ALIASES", "battery_preset_key",
    "normalize_soc_curves", "load_soc_curve_csv",
    "configure_battery_soc_model", "soc_model_short_label",
    "pack_ocv_from_soc", "pack_resistance_from_soc",
    "pack_voltage_under_load", "soc_after_energy_draw",
    # wind
    "wind_components_mps", "groundspeed_along_track_mps",
    # transient flight
    "kinetic_power_term_W", "ramp_speed",
    # thermal
    "thermal_step",
    # propeller table fitting
    "fit_propeller_curve",
    # sensitivity and comparison
    "sensitivity_sweep", "compare_metric_sets", "format_delta",
    # airframe diagram geometry
    "regular_polygon_vertices", "rotor_ring_layout", "wing_rotor_positions",
    # GUI
    "Tooltip",
]


# ============================================================
# PHYSICAL CONSTANTS  (International Standard Atmosphere)
# ============================================================
G0 = 9.80665            # m/s^2   standard gravity
RHO0 = 1.225            # kg/m^3  sea-level density
T0_K = 288.15           # K       sea-level temperature
P0_PA = 101325.0        # Pa      sea-level pressure
LAPSE_K_PER_M = 0.0065  # K/m     tropospheric lapse rate
R_AIR = 287.05          # J/kg/K  specific gas constant, dry air


def air_density(altitude_m: float,
                temperature_C: Optional[float] = None,
                pressure_Pa: Optional[float] = None) -> float:
    """
    Air density from the International Standard Atmosphere.

        rho = P / (R * T)

    Temperature and pressure are handled INDEPENDENTLY: supplying either one
    alone is valid and the other falls back to its ISA value. An earlier
    fixed-wing implementation only honoured a pressure override when a
    temperature was also given, so ``--pressure`` alone was silently ignored
    and the two simulators disagreed. Having one implementation makes that
    class of divergence impossible.
    """
    h = max(float(altitude_m), 0.0)

    # Temperature: ISA lapse rate unless overridden.
    T_K = (T0_K - LAPSE_K_PER_M * h) if temperature_C is None \
        else (float(temperature_C) + 273.15)
    T_K = max(T_K, 1.0)

    # Pressure: ISA barometric profile unless overridden.
    if pressure_Pa is None:
        T_isa = T0_K - LAPSE_K_PER_M * h
        P = P0_PA * (T_isa / T0_K) ** (G0 / (R_AIR * LAPSE_K_PER_M))
    else:
        P = float(pressure_Pa)

    return P / (R_AIR * T_K)


# ============================================================
# INTERPOLATION AND PARSING HELPERS
# ============================================================

def interp_linear_clamped(x: float, xp: List[float], fp: List[float]) -> float:
    """
    Linear interpolation that clamps rather than extrapolating.

    Values below xp[0] return fp[0]; values above xp[-1] return fp[-1]. This
    matters for SoC curves, where extrapolating past 0% or 100% would produce
    nonsense voltages.
    """
    if not xp or not fp or len(xp) != len(fp):
        raise ValueError("Interpolation vectors must be same non-zero length.")
    if len(xp) == 1:
        return float(fp[0])
    if x <= float(xp[0]):
        return float(fp[0])
    if x >= float(xp[-1]):
        return float(fp[-1])
    for i in range(1, len(xp)):
        x0, x1 = float(xp[i - 1]), float(xp[i])
        if x <= x1:
            y0, y1 = float(fp[i - 1]), float(fp[i])
            if abs(x1 - x0) < 1e-12:
                return y1
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return float(fp[-1])


def eval_poly(coeffs: Optional[List[float]], x: float) -> Optional[float]:
    """Evaluate a polynomial given highest-order-first coefficients."""
    if not coeffs:
        return None
    total = 0.0
    for c in coeffs:
        total = total * float(x) + float(c)
    return float(total)


def parse_float_list(spec: Optional[object]) -> Optional[List[float]]:
    """Parse comma-separated floats into a list; empty input returns None."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple, np.ndarray)):
        vals = [float(x) for x in spec]
        return vals if vals else None
    text = str(spec).strip()
    if not text:
        return None
    vals: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    return vals if vals else None


def parse_soc_breakpoints(spec: Optional[object]) -> Optional[List[float]]:
    """
    Parse state-of-charge breakpoints, accepting fractions or percentages.

    Anything above 1.0 is read as a percentage and divided by 100, so both
    "0,0.5,1.0" and "0,50,100" mean the same thing.
    """
    vals = parse_float_list(spec)
    if vals is None:
        return None
    out = []
    for v in vals:
        v = float(v)
        if v > 1.0:
            v /= 100.0
        out.append(min(max(v, 0.0), 1.0))
    return out or None


# ============================================================
# BATTERY STATE-OF-CHARGE MODEL
# ============================================================
# A real pack's open-circuit voltage sags non-linearly as it empties, and its
# internal resistance climbs steeply at low SoC. Modelling that matters for
# endurance: a linear fallback anchors pack voltage at full charge, which
# flatters current draw late in a flight.
#
# These are deliberately conservative approximations, not cell datasheets.

SOC_PRESETS = {
    "lipo": {
        "soc_bp":      [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "ocv_cell_bp": [3.00, 3.30, 3.50, 3.65, 3.72, 3.76, 3.79, 3.82, 3.86, 3.92, 4.02, 4.20],
        "r_scale_bp":  [2.60, 2.10, 1.70, 1.35, 1.18, 1.08, 1.00, 0.98, 1.00, 1.08, 1.25, 1.50],
    },
    "liion": {
        "soc_bp":      [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "ocv_cell_bp": [2.90, 3.20, 3.35, 3.50, 3.60, 3.67, 3.72, 3.77, 3.82, 3.89, 4.00, 4.20],
        "r_scale_bp":  [2.80, 2.20, 1.80, 1.45, 1.22, 1.10, 1.00, 0.98, 1.00, 1.10, 1.30, 1.60],
    },
    "lifepo4": {
        "soc_bp":      [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "ocv_cell_bp": [2.80, 3.00, 3.15, 3.22, 3.26, 3.29, 3.31, 3.32, 3.33, 3.35, 3.42, 3.60],
        "r_scale_bp":  [2.20, 1.90, 1.55, 1.30, 1.15, 1.06, 1.00, 0.98, 1.00, 1.08, 1.18, 1.35],
    },
}

SOC_PRESET_ALIASES = {
    "lipo": "lipo", "li-po": "lipo",
    "liion": "liion", "li-ion": "liion", "lion": "liion", "nmc": "liion",
    "lifepo4": "lifepo4", "lfp": "lifepo4", "li-fepo4": "lifepo4",
}


def battery_preset_key(chemistry: Optional[str]) -> Optional[str]:
    """Map a chemistry label onto a SoC preset key, tolerating punctuation."""
    if not chemistry:
        return None
    key = str(chemistry).strip().lower().replace(" ", "").replace("_", "")
    return SOC_PRESET_ALIASES.get(key)


def normalize_soc_curves(soc_bp: List[float],
                         ocv_cell_bp: List[float],
                         r_scale_bp: List[float]
                         ) -> Tuple[List[float], List[float], List[float]]:
    """Sort by SoC, clamp to sane ranges, and collapse duplicate breakpoints."""
    if len(soc_bp) != len(ocv_cell_bp) or len(soc_bp) != len(r_scale_bp):
        raise ValueError("SoC curve columns must have same length.")
    if len(soc_bp) < 2:
        raise ValueError("SoC curve requires at least 2 points.")

    rows = sorted((float(s), float(v), float(r))
                  for s, v, r in zip(soc_bp, ocv_cell_bp, r_scale_bp))
    out_s: List[float] = []
    out_v: List[float] = []
    out_r: List[float] = []
    for s, v, r in rows:
        s = min(max(s, 0.0), 1.0)
        v = max(v, 0.0)
        r = max(r, 0.05)
        if out_s and abs(s - out_s[-1]) < 1e-9:
            out_v[-1], out_r[-1] = v, r
        else:
            out_s.append(s)
            out_v.append(v)
            out_r.append(r)
    if len(out_s) < 2:
        raise ValueError("SoC curve must contain at least 2 unique breakpoints.")
    return out_s, out_v, out_r


def load_soc_curve_csv(path: str) -> Tuple[List[float], List[float], List[float]]:
    """Load a measured discharge curve. Columns: soc, ocv_cell, r_scale."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in df.columns}

    soc_col = next((lower[k] for k in ("soc", "soc_frac", "soc_fraction") if k in lower), None)
    ocv_col = next((lower[k] for k in ("ocv_cell", "v_oc_cell", "voltage_cell") if k in lower), None)
    r_col = next((lower[k] for k in ("r_scale", "resistance_scale", "r_rel",
                                     "r_multiplier") if k in lower), None)
    if soc_col is None or ocv_col is None or r_col is None:
        raise ValueError("SoC CSV must include columns: soc, ocv_cell, r_scale")

    soc_vals = pd.to_numeric(df[soc_col], errors="coerce")
    ocv_vals = pd.to_numeric(df[ocv_col], errors="coerce")
    r_vals = pd.to_numeric(df[r_col], errors="coerce")
    mask = ~(soc_vals.isna() | ocv_vals.isna() | r_vals.isna())
    return normalize_soc_curves(soc_vals[mask].tolist(),
                                ocv_vals[mask].tolist(),
                                r_vals[mask].tolist())


def configure_battery_soc_model(battery,
                                model: str,
                                curve_csv: Optional[str],
                                soc_bp: Optional[List[float]],
                                ocv_cell_bp: Optional[List[float]],
                                r_scale_bp: Optional[List[float]]) -> None:
    """
    Resolve which SoC curve a pack should use and attach it to `battery`.

    Priority order:
      1. explicit breakpoint arrays
      2. a CSV curve file
      3. a chemistry preset (or auto-detected from the chemistry label)
      4. linear fallback, anchoring voltage at full charge

    Sets: soc_nonlinear_enabled, soc_model_source, soc_bp, ocv_cell_bp,
    r_scale_bp. Works on either simulator's BatteryConfig by duck typing.
    """
    mode = str(model or "auto").strip().lower()

    if mode in ("linear", "off", "disabled"):
        battery.soc_nonlinear_enabled = False
        battery.soc_model_source = "linear-selected"
        return

    try:
        if soc_bp and ocv_cell_bp and r_scale_bp:
            s, v, r = normalize_soc_curves(list(soc_bp), list(ocv_cell_bp), list(r_scale_bp))
            battery.soc_bp, battery.ocv_cell_bp, battery.r_scale_bp = s, v, r
            battery.soc_nonlinear_enabled = True
            battery.soc_model_source = "custom-arrays"
            return
        if curve_csv:
            s, v, r = load_soc_curve_csv(curve_csv)
            battery.soc_bp, battery.ocv_cell_bp, battery.r_scale_bp = s, v, r
            battery.soc_nonlinear_enabled = True
            battery.soc_model_source = f"csv:{curve_csv}"
            return
    except Exception:
        pass          # fall through to presets, then to the linear fallback

    preset = (battery_preset_key(getattr(battery, "chemistry", None))
              if mode in ("auto", "", "preset") else battery_preset_key(mode))
    if preset and preset in SOC_PRESETS:
        table = SOC_PRESETS[preset]
        battery.soc_bp = list(table["soc_bp"])
        battery.ocv_cell_bp = list(table["ocv_cell_bp"])
        battery.r_scale_bp = list(table["r_scale_bp"])
        battery.soc_nonlinear_enabled = True
        battery.soc_model_source = f"preset:{preset}"
        return

    battery.soc_nonlinear_enabled = False
    battery.soc_model_source = "linear-fallback"


def soc_model_short_label(source: Optional[str]) -> str:
    """Condense a model-source string for display."""
    s = str(source or "").strip().lower()
    if not s:
        return "linear-fallback"
    if s.startswith("preset:"):
        return s.replace("preset:", "preset-", 1)
    if s.startswith("csv:"):
        return "csv"
    return s


def pack_ocv_from_soc(battery, soc: float) -> float:
    """Pack open-circuit voltage at a state of charge in [0, 1]."""
    if bool(getattr(battery, "soc_nonlinear_enabled", False)) and getattr(battery, "soc_bp", None):
        ocv_cell = interp_linear_clamped(
            min(max(float(soc), 0.0), 1.0),
            list(battery.soc_bp), list(battery.ocv_cell_bp))
        return max(float(ocv_cell) * float(battery.series_cells),
                   float(battery.vmin_pack))
    return float(battery.vmax_pack)          # linear fallback: full charge


def pack_resistance_from_soc(battery, soc: float) -> float:
    """Pack internal resistance at a state of charge in [0, 1]."""
    base_r = max(float(getattr(battery, "pack_resistance", 0.0)), 0.0)
    if bool(getattr(battery, "soc_nonlinear_enabled", False)) and getattr(battery, "soc_bp", None):
        scale = interp_linear_clamped(
            min(max(float(soc), 0.0), 1.0),
            list(battery.soc_bp), list(battery.r_scale_bp))
        return base_r * max(float(scale), 0.05)
    return base_r


def pack_voltage_under_load(battery, current_A: float,
                            soc: Optional[float] = None) -> float:
    """
    Loaded pack voltage: V = OCV(soc) - I * R(soc), clamped at V_min.

    `soc` defaults to 1.0 (fully charged), reproducing the behaviour of the
    original single-argument signature exactly.
    """
    soc_eval = 1.0 if soc is None else min(max(float(soc), 0.0), 1.0)
    ocv = pack_ocv_from_soc(battery, soc_eval)
    r = pack_resistance_from_soc(battery, soc_eval)
    return max(float(ocv - float(current_A) * r), float(battery.vmin_pack))


def soc_after_energy_draw(battery, soc_now: float, energy_draw_Wh: float) -> float:
    """Advance state of charge after drawing a given amount of energy."""
    usable = max(float(getattr(battery, "usable_Wh", 0.0)), 1e-9)
    drawn = max(float(energy_draw_Wh), 0.0)
    return min(max(float(soc_now) - drawn / usable, 0.0), 1.0)


# ============================================================
# WIND
# ============================================================

def wind_components_mps(wind_speed_mps: float,
                        wind_direction_deg: float,
                        course_deg: float) -> Tuple[float, float]:
    """
    Resolve wind into along-track headwind (+) and crosswind components.

    Meteorological convention: `wind_direction_deg` is the direction the wind
    is coming FROM, so a north wind (0 deg) is a headwind when flying north.

    Returns (headwind, crosswind) in m/s; headwind is positive when it opposes
    the aircraft.

    The parameter names match what both simulators used before this was
    shared, because several call sites pass them by keyword.
    """
    w = max(float(wind_speed_mps), 0.0)
    rel = math.radians(float(wind_direction_deg) - float(course_deg))
    return w * math.cos(rel), w * math.sin(rel)


def groundspeed_along_track_mps(airspeed_mps: float,
                                headwind_mps: float,
                                crosswind_mps: float = 0.0) -> float:
    """
    Along-track groundspeed when holding a course, allowing for crab.

    Part of the airspeed vector is spent crabbing into the crosswind and does
    not contribute to progress along the track:

        v_along_air = sqrt(V_air^2 - V_cross^2)
        V_ground    = v_along_air - V_head

    `crosswind_mps` defaults to 0, in which case this reduces exactly to
    ``max(V_air - V_head, 0)`` — the simpler form the fixed-wing simulator
    used before this was shared. If the crosswind meets or exceeds airspeed
    the aircraft cannot hold the course at all, so groundspeed is zero.
    """
    v_air = max(float(airspeed_mps), 0.0)
    crosswind = abs(float(crosswind_mps))
    if crosswind >= v_air:
        return 0.0
    along_air = math.sqrt(max(v_air * v_air - crosswind * crosswind, 0.0))
    return max(along_air - float(headwind_mps), 0.0)


# ============================================================
# TRANSIENT FLIGHT (acceleration and deceleration)
# ============================================================

def kinetic_power_term_W(mass_g: float,
                         v_now_mps: float,
                         v_next_mps: float,
                         dt_s: float,
                         regen_eff: float = 0.0) -> float:
    """
    Power absorbed or released by a change of speed.

        P = d(0.5 * m * v^2) / dt

    Accelerating costs power on top of steady-flight drag; decelerating
    releases it. Propellers are poor regenerators, so recovered power is
    scaled by `regen_eff` (default 0 = none recovered, the honest default for
    a fixed-pitch prop).
    """
    if dt_s <= 1e-9:
        return 0.0
    m_kg = max(float(mass_g), 0.0) / 1000.0
    e_now = 0.5 * m_kg * (max(float(v_now_mps), 0.0) ** 2)
    e_next = 0.5 * m_kg * (max(float(v_next_mps), 0.0) ** 2)
    power = (e_next - e_now) / float(dt_s)
    if power >= 0.0:
        return power
    return power * min(max(float(regen_eff), 0.0), 1.0)


def ramp_speed(current_mps: float,
               target_mps: float,
               dt_s: float,
               max_accel_mps2: float,
               max_decel_mps2: float) -> Tuple[float, float]:
    """
    Advance speed one step toward a target, respecting accel/decel limits.

    Returns (next_speed, acceleration). Speed never goes negative, and the
    step never overshoots the target — so a phase settles onto its commanded
    speed and then holds it.
    """
    dv_wanted = float(target_mps) - float(current_mps)
    if dv_wanted >= 0.0:
        dv = min(dv_wanted, max(float(max_accel_mps2), 1e-9) * float(dt_s))
    else:
        dv = max(dv_wanted, -max(float(max_decel_mps2), 1e-9) * float(dt_s))
    v_next = max(0.0, float(current_mps) + dv)
    accel = dv / float(dt_s) if dt_s > 1e-9 else 0.0
    return v_next, accel


# ============================================================
# THERMAL
# ============================================================

def thermal_step(temp_C: float, ambient_C: float, power_loss_W: float,
                 thermal_resistance_C_per_W: float, thermal_mass_J_per_C: float,
                 dt_s: float) -> float:
    """
    Advance a lumped first-order thermal model by one time step.

        dT/dt = (P_loss - (T - T_ambient) / R_th) / C_th

    A single capacity with one resistance to ambient: crude, but adequate for
    flagging a motor that is heading for trouble.
    """
    r_th = max(float(thermal_resistance_C_per_W), 1e-9)
    c_th = max(float(thermal_mass_J_per_C), 1e-9)
    dissipated = (float(temp_C) - float(ambient_C)) / r_th
    return float(temp_C) + (float(power_loss_W) - dissipated) / c_th * float(dt_s)


# ============================================================
# PROPELLER TABLE FITTING
# ============================================================

def fit_propeller_curve(x_vals, y_vals, degree: int = 2
                        ) -> Optional[Tuple[List[float], float, float]]:
    """
    Least-squares polynomial fit over a measured propeller table.

    Returns ``(coeffs, x_min, x_max)`` with coefficients highest-order first,
    or None when there are too few finite points to fit the requested degree.

    The x-range is part of the contract: callers use it to decide whether a
    query point is an interpolation or an extrapolation. An earlier version of
    this function returned only the coefficient list, which silently broke the
    caller's ``coeffs, _, _ = fit_result`` unpacking — it bound `coeffs` to a
    single float and raised "'float' object is not iterable" the moment a
    measured prop table was loaded.
    """
    try:
        xs = np.asarray(x_vals, dtype=float)
        ys = np.asarray(y_vals, dtype=float)
    except (TypeError, ValueError):
        return None

    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < degree + 1:
        return None
    try:
        coeffs = [float(c) for c in np.polyfit(xs, ys, degree)]
    except Exception:
        return None
    return coeffs, float(xs.min()), float(xs.max())


# ============================================================
# AIRFRAME DIAGRAM GEOMETRY
# ============================================================
# Plan-view layout maths, kept out of the drawing code so it can be tested
# without a display and shared between the two simulators.

def regular_polygon_vertices(n_sides: int, circumradius_m: float,
                             rotation_rad: float = 0.0) -> List[Tuple[float, float]]:
    """
    Vertices of a regular (equilateral) polygon, centred on the origin.

    Vertex 0 sits at `rotation_rad` measured counter-clockwise from +X. Used
    for the multicopter body, where each vertex carries one arm.
    """
    n = max(int(n_sides), 3)
    r = max(float(circumradius_m), 0.0)
    return [(r * math.cos(rotation_rad + 2.0 * math.pi * i / n),
             r * math.sin(rotation_rad + 2.0 * math.pi * i / n))
            for i in range(n)]


def rotor_ring_layout(num_positions: int,
                      body_circumradius_m: float,
                      arm_length_m: float,
                      prop_diameter_m: float,
                      rotation_rad: float = 0.0) -> dict:
    """
    Lay out rotors evenly around a body polygon and report clearances.

    Each arm runs from a body vertex radially outward, so a rotor centre sits
    at ``body_circumradius + arm_length`` from the middle.

    Returned keys:
        vertices          body polygon corners
        rotors            rotor centres, one per position
        rotor_radius_m    rotor centre distance from the middle
        prop_radius_m     half the propeller diameter
        adjacent_gap_m    tip-to-tip gap between neighbouring discs; NEGATIVE
                          means the discs overlap
        motor_spacing_m   centre-to-centre distance between neighbours
        overlaps          True when neighbouring discs intersect
        span_m            overall width across opposite rotor tips

    Neighbour spacing on a ring of N points at radius R is
    ``2*R*sin(pi/N)``, so the tip gap is that minus one full prop diameter.
    """
    n = max(int(num_positions), 1)
    r_rotor = max(float(body_circumradius_m), 0.0) + max(float(arm_length_m), 0.0)
    r_prop = max(float(prop_diameter_m), 0.0) / 2.0

    vertices = regular_polygon_vertices(n, body_circumradius_m, rotation_rad)         if n >= 3 else [(body_circumradius_m, 0.0), (-body_circumradius_m, 0.0)][:n]
    rotors = [(r_rotor * math.cos(rotation_rad + 2.0 * math.pi * i / n),
               r_rotor * math.sin(rotation_rad + 2.0 * math.pi * i / n))
              for i in range(n)]

    if n >= 2:
        spacing = 2.0 * r_rotor * math.sin(math.pi / n)
    else:
        spacing = float("inf")
    gap = spacing - 2.0 * r_prop

    return {
        "vertices": vertices,
        "rotors": rotors,
        "rotor_radius_m": r_rotor,
        "prop_radius_m": r_prop,
        "motor_spacing_m": spacing,
        "adjacent_gap_m": gap,
        "overlaps": bool(n >= 2 and gap < 0.0),
        "span_m": 2.0 * (r_rotor + r_prop),
    }


def wing_rotor_positions(num_motors: int, wing_span_m: float,
                         prop_diameter_m: float) -> dict:
    """
    Place propellers across a wing and report tip clearances.

    One motor sits on the centreline (a nose tractor). Two or more are spread
    symmetrically about the centreline, inset by one prop radius plus a small
    margin so the discs stay inboard of the tips.

    `adjacent_gap_m` is negative when neighbouring discs overlap.
    """
    n = max(int(num_motors), 1)
    half_span = max(float(wing_span_m), 0.0) / 2.0
    r_prop = max(float(prop_diameter_m), 0.0) / 2.0

    if n == 1:
        positions = [0.0]
    else:
        usable = max(half_span - r_prop * 1.1, r_prop)
        if n % 2 == 0:
            # Even count: symmetric pairs, none on the centreline.
            step = usable / max(n / 2.0, 1.0)
            half = [step * (i + 0.5) for i in range(n // 2)]
            positions = sorted([-x for x in half] + half)
        else:
            step = usable / max((n - 1) / 2.0, 1.0)
            half = [step * (i + 1) for i in range((n - 1) // 2)]
            positions = sorted([-x for x in half] + [0.0] + half)

    if len(positions) >= 2:
        spacing = min(b - a for a, b in zip(positions, positions[1:]))
    else:
        spacing = float("inf")

    return {
        "positions_y_m": positions,
        "prop_radius_m": r_prop,
        "motor_spacing_m": spacing,
        "adjacent_gap_m": spacing - 2.0 * r_prop,
        "overlaps": bool(len(positions) >= 2 and spacing - 2.0 * r_prop < 0.0),
        "tip_overhang_m": (max(abs(p) for p in positions) + r_prop) - half_span,
    }


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_sweep(levers: List[Tuple[str, object]],
                      evaluate,
                      factors=(0.8, 0.9, 1.1, 1.2)) -> List[dict]:
    """
    Measure how an output responds to scaling each input in turn.

    Parameters
    ----------
    levers
        ``(display_name, mutator)`` pairs. Each mutator takes
        ``(config_copy, factor)`` and scales one input in place.
    evaluate
        Takes a mutated config and returns the scalar output of interest,
        or None/NaN when that configuration cannot fly.
    factors
        Multipliers applied to each lever. 1.0 is added automatically as the
        baseline and never passed to a mutator.

    Returns one row per lever, sorted by influence (widest span first), which
    is the ordering a tornado chart wants:

        {name, baseline, results: {factor: value}, low, high, span, span_pct}

    A lever that yields no finite result at all is dropped. One that yields
    identical results at every factor is KEPT with a span of zero — that is a
    real finding ("this input does not move the answer"), not an error.
    """
    import copy as _copy

    baseline = evaluate(None)
    if baseline is None or not math.isfinite(float(baseline)):
        return []
    baseline = float(baseline)

    rows: List[dict] = []
    for name, mutator in levers:
        results = {}
        for factor in factors:
            trial = None
            try:
                cfg_copy = _copy.deepcopy(evaluate.base_config)
                mutator(cfg_copy, float(factor))
                trial = evaluate(cfg_copy)
            except Exception:
                trial = None
            if trial is not None and math.isfinite(float(trial)):
                results[float(factor)] = float(trial)

        if not results:
            continue          # this lever does nothing the model can measure

        values = list(results.values())
        low, high = min(values), max(values)
        rows.append({
            "name": name,
            "baseline": baseline,
            "results": results,
            "low": low,
            "high": high,
            "span": high - low,
            "span_pct": (high - low) / abs(baseline) * 100.0 if baseline else 0.0,
        })

    rows.sort(key=lambda r: r["span"], reverse=True)
    return rows


# ============================================================
# CONFIGURATION COMPARISON
# ============================================================

def compare_metric_sets(baseline: dict, current: dict,
                        keys: List[Tuple[str, str, int]]) -> List[dict]:
    """
    Build a row-per-metric comparison between two result dictionaries.

    `keys` is a list of ``(metric_key, display_label, decimals)``.

    Each row carries the two values, the absolute change and the percentage
    change. A metric missing from either side is reported with `comparable`
    False rather than silently skipped, so a config that stopped producing a
    number is visible instead of just absent.
    """
    rows: List[dict] = []
    for key, label, decimals in keys:
        a = baseline.get(key)
        b = current.get(key)

        def _num(v):
            try:
                f = float(v)
                return f if math.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        a_num, b_num = _num(a), _num(b)
        if a_num is None or b_num is None:
            rows.append({"key": key, "label": label, "decimals": decimals,
                         "baseline": a_num, "current": b_num,
                         "delta": None, "delta_pct": None,
                         "comparable": False, "direction": 0})
            continue

        delta = b_num - a_num
        pct = (delta / abs(a_num) * 100.0) if a_num else None
        rows.append({
            "key": key, "label": label, "decimals": decimals,
            "baseline": a_num, "current": b_num,
            "delta": delta, "delta_pct": pct, "comparable": True,
            "direction": (1 if delta > 0 else (-1 if delta < 0 else 0)),
        })
    return rows


def format_delta(value: Optional[float], decimals: int = 2) -> str:
    """Signed number for a comparison table; an em dash when not comparable."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+.{decimals}f}"


# ============================================================
# GUI TOOLTIP
# ============================================================

class Tooltip:
    """
    Hover tooltip for any Tk widget.

    Tk has no built-in tooltip, so this creates a borderless Toplevel next to
    the cursor on <Enter> and destroys it on <Leave>.

    tkinter is imported INSIDE _show rather than at module scope, because the
    simulators import tk lazily so that headless CLI runs work on machines
    without it. A previous version of this class referenced a module-level
    `tk` that did not exist at hover time: the markers rendered fine and every
    one of them raised NameError the moment a user pointed at it.
    """

    WRAP_PX = 340
    OFFSET_X = 18
    OFFSET_Y = 14

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self._tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        import tkinter as tk

        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + self.OFFSET_X
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + self.OFFSET_Y
            self._tip = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)           # no title bar or border
            tw.wm_geometry(f"+{x}+{y}")
            try:
                tw.attributes("-topmost", True)
            except Exception:
                pass                               # not supported on every WM
            tk.Label(
                tw, text=self.text, justify="left",
                background="#FFFFE0", foreground="#111111",
                relief="solid", borderwidth=1,
                wraplength=self.WRAP_PX,
                font=("TkDefaultFont", 9), padx=8, pady=6,
            ).pack()
        except Exception:
            # A tooltip is a convenience; never let one break the application.
            self._hide()

    def _hide(self, _event=None):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None
