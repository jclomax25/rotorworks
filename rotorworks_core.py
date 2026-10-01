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
import re
import os
import sys
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
    # rotor inflow
    "induced_velocity_forward_flight",
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
    # propeller coefficients
    "estimate_prop_thrust_coefficient", "estimate_prop_power_coefficient",
    "rpm_from_thrust", "derive_prop_coefficients_from_table",
    "load_prop_table", "table_power_for_thrust", "measured_static_efficiency",
    # mission ground track
    "mission_ground_track", "make_mission_diagram_figure",
    # turning flight
    "turn_bank_deg", "turn_load_factor", "turn_thrust_N",
    # translation geometry
    "translation_drag_area", "pitch_roll_from_tilt", "tilt_from_pitch_roll",
    # per-rotor load sharing
    "rotor_thrust_distribution", "rotor_load_spread",
    # wiring and connectors
    "AWG_OHM_PER_M", "CONNECTOR_RATINGS", "wire_resistance_ohm",
    "wire_loss_W", "wire_voltage_drop_V", "connector_defaults",
    "CONNECTOR_VOLTAGE_V", "connector_voltage_default", "wire_ohm_per_m",
    "wire_temperature_C", "wiring_summary", "WIRE_TEMP_LIMIT_C",
    "WiringConfig", "CONNECTOR_POSITIONS", "connector_currents_A",
    "wiring_from_fields", "WIRING_FIELD_TO_CLI", "wiring_status_rows",
    "add_wiring_arguments", "wiring_from_args",
    # exports
    "export_csv", "export_excel",
    # power budget
    "build_power_budget",
    # figures
    "make_figure",
    # GUI
    "Tooltip",
    # console
    "make_console_safe",
]


# ============================================================
# CONSOLE
# ============================================================

def make_console_safe() -> None:
    """
    Never let printing a symbol crash a run.

    The CLIs print units such as mu, Omega, degrees and arrows. On Windows,
    when output goes to a pipe or a file rather than a terminal, Python
    encodes it with the ANSI code page (cp1252), which has no mu — so a
    perfectly good run died with UnicodeEncodeError halfway through its
    report, and every subprocess test and batch run on Windows failed with
    it. Linux and macOS use UTF-8 and never saw it.

    Keeping the stream's own encoding and replacing only what it cannot
    represent means a UTF-8 console is untouched and a cp1252 one shows "?"
    for the odd symbol instead of a traceback.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass          # not a text stream we can reconfigure; leave it


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
# ROTOR INFLOW
# ============================================================

def induced_velocity_forward_flight(v_hover: float,
                                    airspeed_mps: float,
                                    disk_incidence_rad: float = 0.0,
                                    iters: int = 40) -> float:
    """
    Rotor induced velocity in forward flight (Glauert's momentum theory).

    A hovering rotor must accelerate still air, so it works hard:
        v_h = sqrt( T / (2 * rho * A) )

    Once the vehicle is moving, the rotor meets air that is ALREADY moving,
    so far less velocity has to be added and induced power falls sharply.
    Glauert's relation for a rotor at incidence alpha to the freestream is:

        vi = v_h^2 / sqrt( (V*cos a)^2 + (V*sin a + vi)^2 )

    Special cases this reduces to:
      * V = 0            ->  vi = v_h                     (hover)
      * a = 90 deg       ->  vi = -V/2 + sqrt((V/2)^2 + v_h^2)
                                                          (axial / propeller)
      * a = 0 deg        ->  vi = v_h^2 / sqrt(V^2 + vi^2)
                                                          (edgewise / multirotor)

    A multicopter in forward flight sits near the edgewise case: its rotors
    stay nearly horizontal and only tilt by the angle needed to balance drag,
    so the freestream is almost in the disk plane.

    Why this matters
    ----------------
    Using the hover value at all speeds overstates induced power by roughly
    2x at 10 m/s and 4x at 20 m/s for a typical quad.  That removes the
    "power bucket" — the 10-25% dip below hover power that real multirotors
    show around 8-14 m/s — and leaves a monotonically rising power curve.
    With no minimum in the curve, best-endurance and best-range speed
    searches simply pin to the ends of their search range.

    This returns vi only.  Shaft power is then
        P = T * (V*sin(a) + vi)
    where the first term is the propulsive power overcoming airframe drag.

    Solved by damped fixed-point iteration, which converges quickly because
    the right-hand side is a contraction for all physical inputs.
    """
    vh = max(float(v_hover), 0.0)
    V = max(float(airspeed_mps), 0.0)
    if vh <= 0.0:
        return 0.0
    if V <= 1e-9:
        return vh                       # hover

    a = float(disk_incidence_rad)
    v_par = V * math.cos(a)             # in-plane component
    v_perp = V * math.sin(a)            # through-disk component

    vi = vh                             # start from the hover value
    for _ in range(max(int(iters), 1)):
        denom = math.sqrt(v_par ** 2 + (v_perp + vi) ** 2)
        vi_new = (vh ** 2) / max(denom, 1e-9)
        # Damping keeps the iteration stable near the vortex-ring region.
        vi_next = 0.5 * (vi + vi_new)
        if abs(vi_next - vi) < 1e-9:
            vi = vi_next
            break
        vi = vi_next
    return max(vi, 0.0)

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
# PROPELLER COEFFICIENTS
# ============================================================

def estimate_prop_thrust_coefficient(diameter_in: float,
                                     pitch_in: float,
                                     blades: int = 2) -> float:
    """
    Estimate a propeller's static thrust coefficient C_T.

        T = C_T * rho * n^2 * D^4      (n in rev/s, D in metres)

    Why an estimate is needed at all
    --------------------------------
    Thrust alone does NOT determine RPM. Two propellers of the same diameter
    producing identical thrust turn at different speeds depending on their
    pitch and blade area. Without either a measured table or a thrust
    coefficient, RPM — and everything derived from it: back-EMF, mechanical
    power, motor efficiency, throttle — is genuinely underdetermined.

    This fit gives a usable default so those figures read as numbers rather
    than NaN. It is a rough approximation of typical APC-style UAV
    propellers: C_T rises with pitch/diameter, and with blade count through
    solidity. Expect **+/-30%**, which propagates directly into RPM as
    +/-15% (RPM goes as 1/sqrt(C_T)).

    Supply `TConst` explicitly, or load a measured table, whenever accuracy
    matters. Both take priority over this estimate.
    """
    d_in = max(float(diameter_in), 1e-6)
    p_over_d = max(float(pitch_in), 0.0) / d_in

    # Calibrated against the two measured tables shipped with the tests:
    #   APC-style 22x6.6 (p/D 0.33)  ->  C_T 0.062
    #   APC-style 18x8   (p/D 0.44)  ->  C_T 0.079
    # which give slope 0.129 and intercept 0.022.
    #
    # The earlier fit (0.10 + 0.10*p/D) was invented rather than measured and
    # came out roughly TWICE too high against both propellers. Because RPM
    # goes as 1/sqrt(C_T), that error understated propeller speed by about
    # 40%. It survived an earlier sanity check only because the resulting tip
    # speeds landed inside a plausible band — a reminder that "looks
    # reasonable" is not evidence.
    #
    # Two propellers is still a thin basis. Treat this as an order-of-
    # magnitude default and supply TConst or a table whenever it matters.
    c_t = 0.022 + 0.129 * min(max(p_over_d, 0.2), 0.9)

    # Blade count enters through solidity. Thrust per blade falls slightly as
    # blades are added, so the total scales less than linearly.
    n_blades = max(int(blades or 2), 1)
    c_t *= (n_blades / 2.0) ** 0.8

    return min(max(c_t, 0.05), 0.30)


def estimate_prop_power_coefficient(c_t: float,
                                    figure_of_merit: float = 0.65) -> float:
    """
    Power coefficient consistent with a given thrust coefficient.

        P = C_P * rho * n^3 * D^5

    Derived from momentum theory rather than fitted separately, so the two
    coefficients cannot drift into disagreement:

        C_P_ideal = C_T^1.5 / sqrt(2)      (per unit disc area)

    divided by a figure of merit for real losses. FM 0.65 is typical of a
    decent UAV propeller in hover; 0.75+ is a very good one.
    """
    ct = max(float(c_t), 1e-6)
    fm = min(max(float(figure_of_merit), 0.2), 0.95)
    return (ct ** 1.5) / math.sqrt(2.0) / fm


def load_prop_table(path: str):
    """
    Load a motor/propeller bench table.

    Accepts the two layouts these tools meet in practice: a plain CSV whose
    header row is the first line, and an eCalc-style export where the header
    sits a few rows down under a title block. Column names are matched loosely
    so "Thrust (g)", "thrust_g" and "Thrust" all land on `Thrust_g`.

    Returns a DataFrame sorted by thrust with at least `Thrust_g` and
    `Power_W`, or raises ValueError saying which column is missing. Blank
    cells become NaN and are dropped rather than silently read as zero — a
    zero-power row would make the propeller look infinitely efficient.

    The multicopter and fixed-wing each carry an older loader of their own.
    This one exists so the VTOL does not become a third copy.
    """
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ALIASES = {
        "thrust_g": "Thrust_g", "thrust (g)": "Thrust_g", "thrust": "Thrust_g",
        "thrust_grams": "Thrust_g",
        "power_w": "Power_W", "power (w)": "Power_W", "power": "Power_W",
        "electrical power (w)": "Power_W",
        "rpm": "RPM", "prop rpm": "RPM", "motor rpm": "RPM",
        "current_a": "Current_A", "current (a)": "Current_A", "current": "Current_A",
        "voltage_v": "Voltage_V", "voltage (v)": "Voltage_V", "volts": "Voltage_V",
        "throttle": "Throttle", "throttle (%)": "Throttle",
    }

    def normalise(frame):
        renamed = {}
        for column in frame.columns:
            key = str(column).strip().lower()
            if key in ALIASES:
                renamed[column] = ALIASES[key]
        return frame.rename(columns=renamed)

    frame = normalise(pd.read_csv(path))
    if "Thrust_g" not in frame.columns:
        # eCalc-style: find the row that actually looks like a header.
        raw = pd.read_csv(path, header=None, dtype=str)
        for i in range(min(len(raw), 25)):
            candidate = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
            if any(c in ALIASES and ALIASES[c] == "Thrust_g" for c in candidate):
                frame = normalise(pd.read_csv(path, header=i))
                break

    for required in ("Thrust_g", "Power_W"):
        if required not in frame.columns:
            raise ValueError(
                f"{os.path.basename(path)} has no {required} column. "
                f"Found: {', '.join(str(c) for c in frame.columns)}")

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Thrust_g", "Power_W"])
    frame = frame[(frame["Thrust_g"] > 0) & (frame["Power_W"] > 0)]
    if frame.empty:
        raise ValueError(f"{os.path.basename(path)} has no usable rows")
    return frame.sort_values("Thrust_g").reset_index(drop=True)


def table_power_for_thrust(df, thrust_N: float) -> Optional[float]:
    """
    Electrical power the table reports for a given thrust, interpolated.

    Returns None outside the measured range rather than extrapolating. A
    bench table says nothing about thrusts it never produced, and inventing
    a value there is how a measurement turns into a guess wearing its
    clothes.
    """
    if df is None or thrust_N <= 0:
        return None
    grams = float(thrust_N) * 1000.0 / G0
    thrusts = df["Thrust_g"].tolist()
    powers = df["Power_W"].tolist()
    if grams < thrusts[0] or grams > thrusts[-1]:
        return None
    for i in range(len(thrusts) - 1):
        lo, hi = thrusts[i], thrusts[i + 1]
        if lo <= grams <= hi:
            if hi - lo < 1e-12:
                return float(powers[i])
            frac = (grams - lo) / (hi - lo)
            return float(powers[i] + frac * (powers[i + 1] - powers[i]))
    return float(powers[-1])


def measured_static_efficiency(df, thrust_N: float, rho: float,
                               disc_area_m2: float) -> Optional[float]:
    """
    Combined motor-and-propeller efficiency measured by the table, at this
    thrust:

        eta = P_ideal_static / P_table   where  P_ideal_static = T * sqrt(T / 2*rho*A)

    This is the one thing a static bench table gives reliably. The power
    itself is only valid at zero airspeed, but the EFFICIENCY can be carried
    into forward flight and applied to the correct ideal power there — a
    measured number instead of a guessed one.

    Returns None outside the measured range, and clamps to 15-90%: a real
    motor-and-propeller combination sits inside that, and a value outside it
    means the table or the disc area is wrong rather than the propeller being
    remarkable.
    """
    power = table_power_for_thrust(df, thrust_N)
    if power is None or power <= 0:
        return None
    v_hover = math.sqrt(max(float(thrust_N), 0.0) /
                        max(2.0 * rho * disc_area_m2, 1e-9))
    ideal = float(thrust_N) * v_hover
    return min(max(ideal / power, 0.15), 0.90)


def derive_prop_coefficients_from_table(df, diameter_in: float,
                                        rho: float = 1.225) -> Optional[dict]:
    """
    Fit C_T and C_P from a measured static propeller table.

        T = C_T * rho * n^2 * D^4
        P = C_P * rho * n^3 * D^5      (n in rev/s, D in metres)

    Bench data is normally taken at or near sea level, so `rho` defaults to
    1.225 kg/m3. If your table was recorded somewhere high, pass the density
    it was actually measured at — the coefficients scale directly with it and
    a 15% density error becomes a 15% coefficient error.

    Measured coefficients beat any estimate: they capture the real blade, not
    a fit against pitch and diameter. Returns None when the table lacks the
    RPM column needed, since without a rotational speed there is no way to
    non-dimensionalise.

    Returned keys:
        c_t, c_p        mean coefficients across the table
        c_t_spread      max/min ratio of the per-row C_T values; a well
                        behaved propeller stays near 1.0, and a large spread
                        means the fit is not describing a single regime
        points          rows used
    """
    if df is None:
        return None
    for column in ("Thrust_g", "RPM"):
        if column not in df:
            return None

    d_m = float(diameter_in) * 0.0254
    if d_m <= 0 or rho <= 0:
        return None

    have_power = "Power_W" in df
    ct_values, cp_values = [], []
    for _, row in df.iterrows():
        try:
            rpm = float(row["RPM"])
            thrust_N = float(row["Thrust_g"]) * G0 / 1000.0
        except (TypeError, ValueError):
            continue
        n = rpm / 60.0
        if n <= 0 or thrust_N <= 0:
            continue
        ct_values.append(thrust_N / (rho * n ** 2 * d_m ** 4))
        if have_power:
            try:
                power_W = float(row["Power_W"])
            except (TypeError, ValueError):
                continue
            if power_W > 0:
                cp_values.append(power_W / (rho * n ** 3 * d_m ** 5))

    if not ct_values:
        return None

    c_t = sum(ct_values) / len(ct_values)
    return {
        "c_t": c_t,
        "c_p": (sum(cp_values) / len(cp_values)) if cp_values else None,
        "c_t_spread": (max(ct_values) / max(min(ct_values), 1e-9)),
        "points": len(ct_values),
    }


def rpm_from_thrust(thrust_N: float, diameter_m: float, rho: float,
                    c_t: float) -> Optional[float]:
    """
    Propeller speed implied by a thrust, from  T = C_T * rho * n^2 * D^4.

    Returns RPM, or None when the inputs cannot give an answer.
    """
    d = float(diameter_m)
    if thrust_N is None or d <= 0 or rho <= 0 or c_t <= 0:
        return None
    t = max(float(thrust_N), 0.0)
    if t <= 0:
        return 0.0
    n_rev_s = math.sqrt(t / (float(c_t) * float(rho) * d ** 4))
    return n_rev_s * 60.0


# ============================================================
# TRANSLATION GEOMETRY
# ============================================================
# A multirotor does not have a single "forward". It can translate in any
# direction without yawing, and which way it goes changes both the silhouette
# it presents to the airflow and how the required tilt splits between pitch
# and roll. Treating every translation as nose-first hid both effects.

def translation_drag_area(frontal_area_m2: float,
                          side_area_m2: float,
                          azimuth_deg: float) -> float:
    """
    Reference area presented when translating at `azimuth_deg` off the nose.

    0 deg is straight ahead (frontal silhouette), 90 deg is straight right
    (side silhouette). For a broadly box-shaped airframe the projected area
    between those extremes follows

        A(psi) = A_front * cos^2(psi) + A_side * sin^2(psi)

    which is exact for the projected area of a rectangular prism and a decent
    approximation for a real airframe. Note it interpolates the AREA, not the
    drag: drag also depends on how cleanly the shape sheds flow at that angle,
    which this does not attempt to capture.
    """
    psi = math.radians(float(azimuth_deg))
    a_front = max(float(frontal_area_m2), 0.0)
    a_side = max(float(side_area_m2), 0.0)
    return a_front * math.cos(psi) ** 2 + a_side * math.sin(psi) ** 2


def pitch_roll_from_tilt(tilt_deg: float, azimuth_deg: float) -> Tuple[float, float]:
    """
    Split a total tilt into pitch and roll for a given translation direction.

    The thrust vector tilts by `tilt_deg` toward `azimuth_deg`. Resolving that
    tilt onto the body axes:

        tan(pitch) = tan(tilt) * cos(azimuth)
        tan(roll)  = tan(tilt) * sin(azimuth)

    Translating straight ahead is pure pitch; straight sideways is pure roll;
    a diagonal splits between them. This matters because airframes rarely have
    the same authority in both axes — a long-armed cinelifter can have far
    less roll authority than pitch, and a single "tilt limit" cannot express
    that.

    Returns (pitch_deg, roll_deg), both signed.
    """
    tilt = math.radians(float(tilt_deg))
    psi = math.radians(float(azimuth_deg))
    tan_tilt = math.tan(tilt)
    pitch = math.atan(tan_tilt * math.cos(psi))
    roll = math.atan(tan_tilt * math.sin(psi))
    return math.degrees(pitch), math.degrees(roll)


def tilt_from_pitch_roll(pitch_deg: float, roll_deg: float) -> float:
    """
    Total tilt from its pitch and roll components — the inverse of
    `pitch_roll_from_tilt`.

        tan(tilt)^2 = tan(pitch)^2 + tan(roll)^2
    """
    tp = math.tan(math.radians(float(pitch_deg)))
    tr = math.tan(math.radians(float(roll_deg)))
    return math.degrees(math.atan(math.sqrt(tp * tp + tr * tr)))


# ============================================================
# TURNING FLIGHT
# ============================================================

def turn_bank_deg(speed_mps: float, radius_m: float) -> float:
    """
    Bank angle for a coordinated turn.

        tan(bank) = V^2 / (R * g)

    Identical for a multirotor and a fixed-wing: both must tilt their lift
    vector sideways to supply the centripetal force. The multirotor does it
    by tilting the whole rotor disc, the aeroplane by banking the wing, but
    the trigonometry does not care.
    """
    v = max(float(speed_mps), 0.0)
    r = float(radius_m)
    if r <= 0 or v <= 0:
        return 0.0
    return math.degrees(math.atan((v * v) / (r * G0)))


def turn_load_factor(speed_mps: float, radius_m: float) -> float:
    """
    Load factor in a coordinated turn: n = 1 / cos(bank), and never below 1.

    A turning aircraft must generate more than its own weight, because the
    lift vector is now doing two jobs. For a multirotor that means more
    thrust, more induced velocity and more power — which is why a mission of
    tight turns costs more than the straight-line distance suggests.
    """
    bank = math.radians(turn_bank_deg(speed_mps, radius_m))
    return 1.0 / max(math.cos(bank), 1e-6)


def turn_thrust_N(weight_N: float, drag_N: float,
                  speed_mps: float, radius_m: float) -> Tuple[float, float, float]:
    """
    Total thrust for a multirotor holding a turn, and how its tilt resolves.

    Three forces act, and they are mutually perpendicular:
        vertical    W          holding the aircraft up
        along-track D          beating drag
        lateral     m V^2 / R  turning the corner

    so the thrust magnitude is the 3D vector sum

        T = sqrt(W^2 + D^2 + F_c^2)

    Returns (thrust_N, along_track_tilt_deg, lateral_tilt_deg). The lateral
    tilt IS the bank angle; the along-track tilt is the usual drag tilt. Both
    are needed because they load different body axes once the direction of
    travel is taken into account.
    """
    w = max(float(weight_N), 1e-9)
    d = max(float(drag_N), 0.0)
    v = max(float(speed_mps), 0.0)
    r = float(radius_m)

    centripetal = (w / G0) * v * v / r if r > 0 else 0.0
    thrust = math.sqrt(w * w + d * d + centripetal * centripetal)
    return (thrust,
            math.degrees(math.atan2(d, w)),
            math.degrees(math.atan2(centripetal, w)))


# ============================================================
# PER-ROTOR LOAD SHARING
# ============================================================

def rotor_thrust_distribution(rotor_positions: List[Tuple[float, float]],
                              total_thrust_N: float,
                              drag_N: float = 0.0,
                              translation_azimuth_deg: float = 0.0,
                              drag_height_above_cg_m: float = 0.0) -> List[float]:
    """
    Thrust each rotor must produce, once the drag moment is accounted for.

    In steady translating flight the rotors do NOT share the load equally.
    Drag acts at some height above (or below) the centre of gravity, and that
    offset produces a pitching moment:

        M = D * h

    Attitude is only held if the rotors counter it with differential thrust,
    so the trailing rotors work harder than the leading ones. Which rotor
    saturates first — and therefore what actually limits the aircraft — is
    decided by this imbalance, not by the average.

    The moment axis is perpendicular to the direction of travel. Each rotor's
    share is proportional to its distance along that axis, which is the
    minimum-effort solution a real mixer converges on:

        dT_i = M * r_i / sum(r_j^2)

    With `drag_height_above_cg_m` at 0 the moment vanishes and every rotor
    carries `total_thrust_N / n`, exactly as before.

    `rotor_positions` are (x, y) in metres with +x forward and +y right.
    Returns one thrust per rotor, in the same order. Values are floored at
    zero: a rotor cannot pull down.
    """
    n = len(rotor_positions)
    if n == 0:
        return []
    even = max(float(total_thrust_N), 0.0) / n
    moment = float(drag_N) * float(drag_height_above_cg_m)
    if abs(moment) < 1e-12:
        return [even] * n

    # Travel direction, and the axis the drag moment acts about (90 deg to it).
    psi = math.radians(float(translation_azimuth_deg))
    axis_x, axis_y = math.cos(psi), math.sin(psi)

    # Distance of each rotor along the travel direction. A rotor ahead of the
    # CG gets a positive arm, one behind gets a negative one.
    arms = [x * axis_x + y * axis_y for x, y in rotor_positions]
    denom = sum(a * a for a in arms)
    if denom < 1e-12:
        return [even] * n

    # Drag above the CG pitches the nose down, so the LEADING rotors unload
    # and the trailing ones take up the difference.
    return [max(even - moment * a / denom, 0.0) for a in arms]


def rotor_load_spread(thrusts: List[float]) -> dict:
    """
    Summarise how unevenly the rotors are loaded.

    `spread` is the highest thrust divided by the lowest: 1.0 is perfectly
    even, and a large value means one rotor is close to its limit while
    another idles. `imbalance_pct` is how far the hardest-working rotor sits
    above the average, which is the number to compare against motor headroom.
    """
    if not thrusts:
        return {"max_N": 0.0, "min_N": 0.0, "mean_N": 0.0,
                "spread": 1.0, "imbalance_pct": 0.0}
    hi, lo = max(thrusts), min(thrusts)
    mean = sum(thrusts) / len(thrusts)
    return {
        "max_N": hi,
        "min_N": lo,
        "mean_N": mean,
        "spread": hi / lo if lo > 1e-9 else float("inf"),
        "imbalance_pct": (hi / mean - 1.0) * 100.0 if mean > 1e-9 else 0.0,
    }


# ============================================================
# MISSION GROUND TRACK
# ============================================================

def mission_ground_track(phases: List[dict]) -> List[dict]:
    """
    Dead-reckon a mission into waypoints on the ground.

    Missions are written as legs — a heading, a distance, an altitude — not as
    coordinates, so the shape of the route is never actually stated anywhere.
    Integrating the legs recovers it, which is the only way to see whether a
    pattern closes, overlaps itself, or drifts.

    Compass convention: `course_deg` is 0 for north and 90 for east, so a leg
    advances north by `cos(course)` and east by `sin(course)`. Plotting east
    on x and north on y then puts north up the page, as a map should be.

    Each phase dict may carry `course_deg`, `distance` (m), `altitude` (m),
    `name`, and either `translation_direction_deg` (multirotor, direction of
    travel measured from the nose) or `bank_deg` (fixed-wing). Yaw is the
    heading the AIRFRAME points, which for a multirotor translating sideways
    is not the direction it is moving:

        yaw = course - translation_direction

    Returns one dict per waypoint: x_m (east), y_m (north), altitude_m,
    yaw_deg, name, index. The first entry is the start point.
    """
    track: List[dict] = []
    x = y = 0.0
    alt = float(phases[0].get("altitude", 0.0)) if phases else 0.0

    track.append({"x_m": 0.0, "y_m": 0.0, "altitude_m": 0.0,
                  "yaw_deg": float(phases[0].get("course_deg", 0.0)) if phases else 0.0,
                  "name": "Takeoff", "index": 0})

    for i, phase in enumerate(phases, start=1):
        course = float(phase.get("course_deg", 0.0) or 0.0)
        dist = float(phase.get("distance", 0.0) or 0.0)
        alt = float(phase.get("altitude", alt) or 0.0)

        # A phase with no distance (a hover, a yaw pause, a climb) holds
        # position, so it still earns a waypoint — the altitude changes even
        # though the ground position does not.
        if dist > 0:
            x += dist * math.sin(math.radians(course))
            y += dist * math.cos(math.radians(course))

        psi = phase.get("translation_direction_deg")
        yaw = course - float(psi) if psi is not None else course

        track.append({"x_m": x, "y_m": y, "altitude_m": alt,
                      "yaw_deg": yaw % 360.0,
                      "name": str(phase.get("name", f"Phase {i}")),
                      "index": i})
    return track


def make_mission_diagram_figure(phases: List[dict], figsize=(12, 5.5)):
    """
    Two views of a mission: the ground track from above, and the altitude
    profile. Both number the waypoints in flight order.

    The plan view carries an arrow at each waypoint showing where the AIRFRAME
    is pointing, which for a multirotor is not always where it is going. A
    square flown with the nose fixed and one flown by yawing at each corner
    trace the same path; only the arrows tell them apart.
    """
    track = mission_ground_track(phases)
    fig, (ax_plan, ax_alt) = make_figure(1, 2, figsize=figsize)

    xs = [w["x_m"] for w in track]
    ys = [w["y_m"] for w in track]

    # ---- plan view ----------------------------------------------------
    ax_plan.plot(xs, ys, color="#1565C0", linewidth=1.6, zorder=2)
    ax_plan.scatter(xs, ys, s=26, color="#1565C0", zorder=3)

    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    arrow = span * 0.06
    for w in track:
        # Compass yaw: 0 is north (up), 90 is east (right).
        th = math.radians(w["yaw_deg"])
        ax_plan.annotate(
            "", xytext=(w["x_m"], w["y_m"]),
            xy=(w["x_m"] + arrow * math.sin(th), w["y_m"] + arrow * math.cos(th)),
            arrowprops=dict(arrowstyle="-|>", color="#EF6C00", lw=1.4), zorder=4)
        ax_plan.annotate(str(w["index"]), (w["x_m"], w["y_m"]),
                         textcoords="offset points", xytext=(6, 6),
                         fontsize=8, fontweight="bold", color="#37474F", zorder=5)

    ax_plan.scatter([xs[0]], [ys[0]], s=150, marker="^", color="#2E7D32",
                    zorder=6, label="Takeoff")
    ax_plan.scatter([xs[-1]], [ys[-1]], s=150, marker="v", color="#C62828",
                    zorder=6, label="Landing")
    ax_plan.set_xlabel("East (m)")
    ax_plan.set_ylabel("North (m)")
    ax_plan.set_title("Ground track — arrows show airframe yaw")
    ax_plan.set_aspect("equal", adjustable="datalim")
    ax_plan.grid(True, alpha=0.4)
    ax_plan.legend(fontsize=8, loc="best")

    # ---- altitude profile ---------------------------------------------
    idx = [w["index"] for w in track]
    alts = [w["altitude_m"] for w in track]
    ax_alt.plot(idx, alts, color="#2E7D32", linewidth=1.8, marker="o",
                markersize=5)
    for w in track:
        ax_alt.annotate(str(w["index"]), (w["index"], w["altitude_m"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, fontweight="bold",
                        color="#37474F")
    ax_alt.set_xlabel("Waypoint")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude profile")
    ax_alt.grid(True, alpha=0.4)
    ax_alt.set_xticks(idx)

    fig.tight_layout()
    return fig


# ============================================================
# WIRING AND CONNECTORS
# ============================================================

# Resistance of solid copper at 20 C, ohms per metre of ONE conductor.
# Standard AWG figures; silicone-insulated stranded wire runs a few percent
# higher because the strands are not perfectly parallel.
AWG_OHM_PER_M: Dict[int, float] = {
    8: 0.002061, 10: 0.003277, 12: 0.005211, 14: 0.008286,
    16: 0.013172, 18: 0.020950, 20: 0.033292, 22: 0.052939,
    24: 0.084197, 26: 0.133900, 28: 0.212900,
}

# Typical manufacturer ratings, amps (continuous, burst).
#
# These are DEFAULTS to start from, not specifications. Burst figures in
# particular vary widely between makers and with how long "burst" means, so
# every one of them is editable and the status check uses whatever the user
# entered rather than these.
CONNECTOR_RATINGS: Dict[str, Tuple[float, float]] = {
    "XT30":          (30.0, 45.0),
    "XT60":          (60.0, 90.0),
    "XT90":          (90.0, 135.0),
    "AS150":         (150.0, 225.0),
    "EC3":           (60.0, 90.0),
    "EC5":           (120.0, 180.0),
    "Deans/T-plug":  (60.0, 90.0),
    "Bullet 3.5 mm": (50.0, 75.0),
    "Bullet 4 mm":   (70.0, 105.0),
    "Bullet 5.5 mm": (120.0, 180.0),
    "Bullet 6 mm":   (140.0, 210.0),
    "Bullet 8 mm":   (200.0, 300.0),
}


def wire_resistance_ohm(length_m: float,
                        awg: Optional[int] = None,
                        ohm_per_m: Optional[float] = None,
                        both_conductors: bool = True) -> float:
    """
    Resistance of a wire run.

    `length_m` is the ONE-WAY run — the distance from source to load. Current
    has to come back, so by default the resistance counts both conductors and
    the figure returned is for `2 * length_m` of wire. That doubling is the
    single easiest thing to forget, and forgetting it halves every loss the
    model reports.

    Give either an `awg` (looked up for copper at 20 C) or an explicit
    `ohm_per_m`. An explicit value wins, because it is the one a user can read
    off their own spool.

    Returns 0.0 when there is nothing to compute, so an unfilled input costs
    nothing rather than raising.
    """
    length = max(float(length_m or 0.0), 0.0)
    if length <= 0:
        return 0.0

    if ohm_per_m is not None and float(ohm_per_m) > 0:
        per_m = float(ohm_per_m)
    elif awg is not None and int(awg) in AWG_OHM_PER_M:
        per_m = AWG_OHM_PER_M[int(awg)]
    else:
        return 0.0

    conductors = 2.0 if both_conductors else 1.0
    return per_m * length * conductors


def wire_loss_W(current_A: float, resistance_ohm: float) -> float:
    """Ohmic loss in a wire run: I^2 * R, dissipated as heat."""
    return max(float(current_A), 0.0) ** 2 * max(float(resistance_ohm), 0.0)


def wire_voltage_drop_V(current_A: float, resistance_ohm: float) -> float:
    """
    Voltage lost along the run: I * R.

    This matters beyond the wasted watts. The ESC and motor see the pack
    voltage MINUS this drop, so a long thin battery lead lowers the voltage
    actually available for thrust — and on a low-cell-count pack that can be a
    measurable fraction of the headroom.
    """
    return max(float(current_A), 0.0) * max(float(resistance_ohm), 0.0)


# Rated voltage, volts DC, only where the maker publishes one: Amass rates
# its XT and AS series at 500 V DC. EC3/EC5, Deans and bullet connectors are
# sold without a voltage rating, so they are left out rather than guessed —
# Status then says "Not Specified" and the user can enter a figure.
CONNECTOR_VOLTAGE_V: Dict[str, float] = {
    "XT30": 500.0, "XT60": 500.0, "XT90": 500.0, "AS150": 500.0,
}

COPPER_RESISTIVITY_OHM_M = 1.724e-8      # at 20 C
COPPER_ALPHA_PER_C = 0.00393             # resistance rise per degree
WIRE_TEMP_LIMIT_C = 150.0                # default: margin below silicone's 200 C


def connector_voltage_default(name: Optional[str]) -> Optional[float]:
    """Published rated voltage for a named connector, or None."""
    if not name:
        return None
    folded = str(name).strip().lower().replace(" ", "")
    for known, volts in CONNECTOR_VOLTAGE_V.items():
        if known.lower().replace(" ", "") == folded:
            return volts
    return None


def wire_ohm_per_m(awg: Optional[int] = None,
                   ohm_per_m: Optional[float] = None) -> Optional[float]:
    """Resistance per metre of ONE conductor, or None if neither is given."""
    if ohm_per_m is not None and float(ohm_per_m) > 0:
        return float(ohm_per_m)
    if awg is not None and int(awg) in AWG_OHM_PER_M:
        return AWG_OHM_PER_M[int(awg)]
    return None


def wire_temperature_C(current_A: float, ohm_per_m: Optional[float],
                       ambient_C: float, insulation_mm: float = 0.5,
                       h_W_m2K: float = 12.0) -> float:
    """
    Steady temperature of a wire carrying `current_A`, in still air.

    Per metre, the wire makes I^2 R' of heat and sheds h * pi * D * (T - Ta)
    from its surface. The conductor diameter follows from R' and copper's
    resistivity, and the insulation adds `insulation_mm` a side (about right
    for silicone hook-up wire, 10-20 AWG). Copper's resistance rises with
    temperature, R'(T) = R'20 (1 + alpha (T - 20)), which solved together
    with the heat balance gives

        T - Ta = I^2 R'20 (1 + alpha (Ta - 20)) / (h pi D - I^2 R'20 alpha)

    When the denominator reaches zero the wire cannot shed the extra heat
    its own rising resistance makes — thermal runaway — and this returns
    infinity. `h` of 12 W/m^2K is natural convection plus radiation; a wire
    in the propeller wash runs cooler, so this is the conservative case.
    Returns ambient when there is no wire or no current.
    """
    if not ohm_per_m or ohm_per_m <= 0 or current_A <= 0:
        return float(ambient_C)
    r20 = float(ohm_per_m)
    d = math.sqrt(4.0 * COPPER_RESISTIVITY_OHM_M / (math.pi * r20))
    outer = d + 2.0 * max(float(insulation_mm), 0.0) / 1000.0
    i2r = float(current_A) ** 2 * r20
    shed = h_W_m2K * math.pi * outer - i2r * COPPER_ALPHA_PER_C
    if shed <= 0:
        return float("inf")
    return float(ambient_C) + i2r * (1.0 + COPPER_ALPHA_PER_C * (float(ambient_C) - 20.0)) / shed


def wiring_summary(current_A: float, length_m: float, awg: Optional[int] = None,
                   ohm_per_m: Optional[float] = None,
                   ambient_C: float = 20.0) -> Dict[str, float]:
    """
    Everything Status and Metrics report about the main wire run, at one
    current: resistance of the round trip, loss, voltage drop, and the
    steady wire temperature. All zero (temperature ambient) with no run.
    """
    per_m = wire_ohm_per_m(awg, ohm_per_m)
    r = wire_resistance_ohm(length_m, awg=awg, ohm_per_m=ohm_per_m)
    have = r > 0 and per_m is not None
    return {
        "resistance_ohm": r,
        "ohm_per_m": per_m or 0.0,
        "loss_W": wire_loss_W(current_A, r),
        "drop_V": wire_voltage_drop_V(current_A, r),
        "temp_C": wire_temperature_C(current_A, per_m, ambient_C) if have else float(ambient_C),
        "present": have,
    }


# Where the three connectors sit, and so which current each one carries.
CONNECTOR_POSITIONS = ("Battery", "ESC", "Motor")


class WiringConfig:
    """
    The main battery lead and the three connector ratings, shared by the
    multicopter and fixed-wing (the VTOL keeps the same inputs on its own
    config). Attached to an aircraft as `.wiring`; absent or empty, every
    result is exactly what it was without it.

    `connectors` maps "Battery" / "ESC" / "Motor" to (continuous A, burst A,
    rated V), any of which may be None when not entered.
    """

    def __init__(self, length_m: float = 0.0, awg: Optional[int] = None,
                 ohm_per_m: Optional[float] = None,
                 temp_limit_C: Optional[float] = None,
                 connectors: Optional[Dict[str, Tuple]] = None):
        self.length_m = max(float(length_m or 0.0), 0.0)
        self.awg = int(awg) if awg not in (None, "") else None
        self.ohm_per_m = (float(ohm_per_m) if ohm_per_m not in (None, "")
                          and float(ohm_per_m) > 0 else None)
        self.temp_limit_C = float(temp_limit_C) if temp_limit_C else WIRE_TEMP_LIMIT_C
        self.connectors = {k: tuple(v) for k, v in (connectors or {}).items()
                           if any(x for x in v)}

    @property
    def resistance_ohm(self) -> float:
        """Round-trip resistance of the run, both conductors."""
        return wire_resistance_ohm(self.length_m, awg=self.awg, ohm_per_m=self.ohm_per_m)

    def summary(self, current_A: float, ambient_C: float) -> Dict[str, float]:
        return wiring_summary(current_A, self.length_m, self.awg, self.ohm_per_m, ambient_C)


# The Wiring tab's field keys (the same on all three simulators) and the CLI
# argument each maps to.
WIRING_FIELD_TO_CLI = {
    "wire_len": "wire_length", "wire_awg": "wire_awg",
    "wire_ohm_m": "wire_ohm_per_m", "wire_temp_limit": "wire_temp_limit",
    "conn_batt_cont": "connector_batt_cont", "conn_batt_max": "connector_batt_max",
    "conn_batt_volt": "connector_batt_volt",
    "conn_esc_cont": "connector_esc_cont", "conn_esc_max": "connector_esc_max",
    "conn_esc_volt": "connector_esc_volt",
    "conn_motor_cont": "connector_motor_cont", "conn_motor_max": "connector_motor_max",
    "conn_motor_volt": "connector_motor_volt",
}


def add_wiring_arguments(parser) -> None:
    """The Wiring tab's CLI flags, identical on every simulator."""
    group = parser.add_argument_group("wiring and connectors")
    group.add_argument("--wire_length", type=float, default=None,
                       help="Main battery lead, one-way length (m); both conductors are counted.")
    group.add_argument("--wire_awg", type=int, default=None,
                       help="Main lead gauge (AWG). Higher is thinner.")
    group.add_argument("--wire_ohm_per_m", type=float, default=None,
                       help="Measured resistance of one conductor (ohm/m); overrides the gauge.")
    group.add_argument("--wire_temp_limit", type=float, default=None,
                       help=f"Insulation temperature limit (C); default {WIRE_TEMP_LIMIT_C:g}.")
    for name in ("batt", "esc", "motor"):
        group.add_argument(f"--connector_{name}_cont", type=float, default=None,
                           help=f"{name} connector continuous rating (A).")
        group.add_argument(f"--connector_{name}_max", type=float, default=None,
                           help=f"{name} connector burst rating (A).")
        group.add_argument(f"--connector_{name}_volt", type=float, default=None,
                           help=f"{name} connector rated voltage (V).")


def wiring_from_args(args) -> Optional["WiringConfig"]:
    """The CLI's wiring flags, through the same builder as the GUI."""
    return wiring_from_fields({key: getattr(args, dest, None)
                               for key, dest in WIRING_FIELD_TO_CLI.items()})


def wiring_from_fields(values: dict) -> Optional["WiringConfig"]:
    """
    Build a WiringConfig from Wiring-tab values keyed as above — the GUI's
    field text, or the CLI's arguments mapped onto the same keys, so the two
    cannot disagree about what a wire run costs. None when nothing is set.
    """
    def num(key):
        raw = values.get(key)
        s_ = "" if raw is None else str(raw).strip()
        if s_ == "":
            return None
        try:
            return float(s_)
        except ValueError:
            raise ValueError(f"'{s_}' is not a number (field: {key})")

    connectors = {}
    for name, prefix in (("Battery", "conn_batt"), ("ESC", "conn_esc"),
                         ("Motor", "conn_motor")):
        rating = (num(f"{prefix}_cont"), num(f"{prefix}_max"), num(f"{prefix}_volt"))
        if any(x for x in rating):
            connectors[name] = rating
    awg = num("wire_awg")
    wiring = WiringConfig(length_m=num("wire_len") or 0.0,
                          awg=int(awg) if awg else None,
                          ohm_per_m=num("wire_ohm_m"),
                          temp_limit_C=num("wire_temp_limit"),
                          connectors=connectors)
    if wiring.resistance_ohm <= 0 and not wiring.connectors:
        return None
    return wiring


def connector_currents_A(pack_current_A: float, per_esc_current_A: float) -> Dict[str, float]:
    """
    The current through each connector, which is not the same number in
    each position: the battery connector carries the whole pack current,
    an ESC connector one motor's share, and a motor connector the phase
    current, about 1.15x the ESC's DC input.
    """
    return {"Battery": max(float(pack_current_A), 0.0),
            "ESC": max(float(per_esc_current_A), 0.0),
            "Motor": max(float(per_esc_current_A), 0.0) * 1.15}


def _dual_limit(value: float, cont: Optional[float], mx: Optional[float],
                unit: str) -> Tuple[str, str, str]:
    """(limit text, tag, note) for a value with a continuous and a burst limit."""
    if not cont and not mx:
        return ("Not Specified", "na",
                "No rating entered, so there is nothing to check against.")
    limit = " / ".join(x for x in (f"cont {cont:.0f} {unit}" if cont else "",
                                   f"max {mx:.0f} {unit}" if mx else "") if x)
    if mx and value > mx:
        return limit, "bad", f"Above the burst rating of {mx:.0f} {unit}."
    if cont and value > cont:
        return limit, "warn", (f"Above the continuous rating of {cont:.0f} {unit}; "
                               f"acceptable only briefly.")
    ref = cont or mx
    if value > 0.95 * ref:
        return limit, "edge", f"Within 5% of the {ref:.0f} {unit} rating — no margin left."
    return limit, "ok", "Within the continuous rating."


def wiring_status_rows(wiring: Optional["WiringConfig"], pack_current_A: float,
                       per_esc_current_A: float, pack_voltage_full_V: float,
                       ambient_C: float, where: str = "") -> List[Tuple]:
    """
    The Status rows for the main wire run and the three connectors, for any
    of the three simulators: (group, metric, value, limit, tag, note), where
    group is "battery" (the wire and the battery connector) or "motor" (the
    ESC and motor connectors) and tag is ok / edge / warn / bad / na.

    `where` names the operating point ("hover", "cruise") in the labels.

    Connector voltage is checked against the FULL-charge pack voltage, the
    highest the connector will ever see. The wire temperature is the steady
    still-air figure from wire_temperature_C, so it is conservative.
    """
    if wiring is None:
        return []
    rows: List[Tuple] = []
    at = f" ({where})" if where else ""
    w = wiring.summary(pack_current_A, ambient_C)
    if w["present"]:
        frac = w["drop_V"] / max(float(pack_voltage_full_V), 1e-9)
        tag = "ok" if frac <= 0.03 else ("warn" if frac <= 0.05 else "bad")
        rows.append(("battery", f"Main wire voltage drop{at}",
                     f"{w['drop_V']:.2f} V ({frac * 100:.1f}%)", "<= 3% of pack", tag,
                     f"{w['loss_W']:.1f} W lost as heat in the {w['resistance_ohm'] * 1000:.1f} mΩ "
                     f"round trip. The ESCs see the pack voltage minus this drop, so it "
                     f"costs thrust headroom as well as watts. Above about 5% use a "
                     f"thicker or shorter lead."))
        temp, limit = w["temp_C"], wiring.temp_limit_C
        shown = "above 400 °C" if (not math.isfinite(temp) or temp > 400) else f"{temp:.0f} °C"
        tag = "bad" if temp > limit else ("warn" if temp > limit - 20.0 else "ok")
        rows.append(("battery", f"Wire temperature (est){at}", shown, f"<= {limit:.0f} °C", tag,
                     f"Steady temperature of the main lead at {pack_current_A:.1f} A in still "
                     f"air — conservative, since a lead in the propeller wash runs cooler. "
                     f"Silicone insulation is rated about 200 °C, PVC about 105 °C."))
    currents = connector_currents_A(pack_current_A, per_esc_current_A)
    for name in CONNECTOR_POSITIONS:
        if name not in wiring.connectors:
            continue
        cont, mx, volts = (list(wiring.connectors[name]) + [None, None, None])[:3]
        group = "battery" if name == "Battery" else "motor"
        amps = currents[name]
        limit, tag, note = _dual_limit(amps, cont, mx, "A")
        what = {"Battery": "the whole pack current",
                "ESC": "one ESC's share of the current",
                "Motor": "the motor phase current, about 1.15x the ESC's DC input"}[name]
        rows.append((group, f"{name} connector current{at}", f"{amps:.1f} A", limit, tag,
                     f"Carries {what}. {note}"))
        if volts:
            ok = pack_voltage_full_V <= volts
            rows.append((group, f"{name} connector voltage", f"{pack_voltage_full_V:.1f} V",
                         f"<= {volts:.0f} V", "ok" if ok else "bad",
                         "Full-charge pack voltage against the connector's rating."
                         if ok else "The pack exceeds the connector's rated voltage."))
        else:
            rows.append((group, f"{name} connector voltage", f"{pack_voltage_full_V:.1f} V",
                         "Not Specified", "na",
                         "No voltage rating entered. Amass rates its XT and AS series at "
                         "500 V DC; most others publish none."))
    return rows


def connector_defaults(name: Optional[str]) -> Optional[Tuple[float, float]]:
    """Typical (continuous, burst) amps for a named connector, or None."""
    if not name:
        return None
    key = str(name).strip()
    if key in CONNECTOR_RATINGS:
        return CONNECTOR_RATINGS[key]
    # Tolerate case and spacing differences in saved configs.
    folded = key.lower().replace(" ", "")
    for known, ratings in CONNECTOR_RATINGS.items():
        if known.lower().replace(" ", "") == folded:
            return ratings
    return None


# ============================================================
# POWER BUDGET
# ============================================================

def build_power_budget(total_in_W: float,
                       motor_shaft_W: float,
                       motor_copper_W: float,
                       battery_i2r_W: float,
                       esc_loss_W: float,
                       wire_loss_W: float = 0.0,
                       peripheral_W: float = 0.0,
                       peripheral_V: Optional[float] = None,
                       peripheral_A: Optional[float] = None,
                       rails: Optional[List[dict]] = None,
                       motor_iron_W: float = 0.0) -> List[dict]:
    """
    Break the pack's electrical output into where every watt ends up.

    The weight budget answers "what is this aircraft made of"; this answers
    "what is the battery actually paying for". Both are needed, because a
    design can be light and still waste a third of its energy as heat.

    Every row is either DELIVERED (it did useful work) or LOST (it became
    heat), and the two categories must sum to the total — which is what makes
    the table checkable rather than merely informative.

    The total is taken at the CELLS, not the pack terminals: terminal power is
    already measured at the sagged voltage, so counting the pack's own I2R
    loss inside it would charge that loss twice.

    `rails` is a list of avionics rails, each a dict with `name`, `voltage_V`,
    `current_A` and `efficiency`. A rail's input power splits into the power
    delivered at rail voltage and the conversion loss in its regulator.

    Rows carry a voltage and a current. Anything drawing straight from the
    pack reports "Battery" for voltage, because its voltage is whatever the
    pack happens to be at that moment rather than a designed value.

    Returns a list of dicts: name, watts, pct, kind ("delivered"/"lost"/
    "subtotal"/"total"), voltage, current.
    """
    # The pack's terminal power is measured at the LOADED voltage, so the
    # cells' internal I2R loss is already expressed as voltage sag rather
    # than as extra current. Listing it as a loss inside the terminal power
    # therefore double-counts it — which showed up as a negative
    # "Unaccounted" row exactly equal to the I2R.
    #
    # The budget is therefore taken from the CELLS, not the terminals:
    #     P_cells = P_terminals + I2R
    # so every joule the chemistry gives up is accounted for exactly once.
    total = max(float(total_in_W) + float(battery_i2r_W), 1e-9)
    rows: List[dict] = []

    def add(name, watts, kind, voltage="Battery", current=None):
        rows.append({
            "name": name,
            "watts": float(watts),
            "pct": float(watts) / total * 100.0,
            "kind": kind,
            "voltage": voltage,
            "current": current,
        })

    # ---- delivered ----------------------------------------------------
    add("Motor shaft power (to the air)", motor_shaft_W, "delivered")

    rail_list = rails or []
    for rail in rail_list:
        name = str(rail.get("name", "rail"))
        volts = float(rail.get("voltage_V", 0.0) or 0.0)
        amps = float(rail.get("current_A", 0.0) or 0.0)
        eff = min(max(float(rail.get("efficiency", 0.9) or 0.9), 0.05), 1.0)
        delivered = volts * amps
        add(f"Avionics rail {name} delivered", delivered, "delivered",
            f"{volts:.1f} V", amps)
        add(f"Avionics rail {name} regulator loss",
            delivered * (1.0 / eff - 1.0), "lost", f"{volts:.1f} V", amps)

    # Regulated rails and direct-from-pack peripherals are independent loads
    # and both appear. An earlier version hid this row whenever rails existed,
    # to suppress a negative "Unaccounted" — but that residual was really the
    # model ignoring peripheral current, not the table over-reporting it.
    if peripheral_W > 0:
        add("Peripheral devices (direct from pack)", peripheral_W, "delivered",
            "Battery" if peripheral_V is None else f"{peripheral_V:.1f} V",
            peripheral_A)

    # ---- lost ---------------------------------------------------------
    add("Motor copper loss (I2Rm)", motor_copper_W, "lost")
    if motor_iron_W > 0:
        # Only the VTOL passes this: the no-load current times back-EMF, the
        # iron and bearing loss a motor pays even when lightly loaded.
        add("Motor no-load loss (I0 x V)", motor_iron_W, "lost")
    add("ESC losses", esc_loss_W, "lost")
    if wire_loss_W > 0:
        # Its own row rather than folded into "ESC losses": wiring is the one
        # loss a user can halve with a screwdriver and a thicker cable, so it
        # is worth seeing separately.
        add("Main wire run (I2R)", wire_loss_W, "lost")
    add("Battery internal loss (I2R)", battery_i2r_W, "lost")

    delivered_W = sum(r["watts"] for r in rows if r["kind"] == "delivered")
    lost_W = sum(r["watts"] for r in rows if r["kind"] == "lost")

    # Anything the itemised rows do not account for. Showing it keeps the
    # table honest: a large residual means the breakdown is incomplete, and
    # hiding it would make the percentages quietly wrong.
    residual = total - delivered_W - lost_W
    if abs(residual) > max(total * 0.005, 0.5):
        add("Unaccounted", residual, "lost")
        lost_W += residual

    rows.append({"name": "Total delivered", "watts": delivered_W,
                 "pct": delivered_W / total * 100.0, "kind": "subtotal",
                 "voltage": "", "current": None})
    rows.append({"name": "Total losses", "watts": lost_W,
                 "pct": lost_W / total * 100.0, "kind": "subtotal",
                 "voltage": "", "current": None})
    rows.append({"name": "TOTAL from cells", "watts": total,
                 "pct": 100.0, "kind": "total",
                 "voltage": "Battery", "current": None})
    return rows


# ============================================================
# EXPORTS
# ============================================================

def export_csv(path: str, sections: List[Tuple[str, List[str], List[list]]]) -> None:
    """
    Write several titled tables into one CSV.

    A spreadsheet holds one grid, so multiple tables are stacked with a
    bracketed heading and a blank line between them — the same shape the
    multicopter and fixed-wing have always produced, so downstream scripts
    that already read those files can read these too.

    `sections` is (title, headers, rows).
    """
    import csv
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for i, (title, headers, rows) in enumerate(sections):
            if i:
                writer.writerow([])
            writer.writerow([f"[{title}]"])
            if headers:
                writer.writerow(headers)
            for row in rows:
                writer.writerow(list(row))


def export_excel(path: str, sections: List[Tuple[str, List[str], List[list]]]) -> None:
    """
    Write each table to its own worksheet, headers bolded.

    Unlike the CSV, a workbook can hold real tables side by side, so nothing
    is stacked and nothing has to be parsed back apart.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, headers, rows in sections:
        # Excel sheet names cannot exceed 31 characters or contain []:*?/\
        safe = re.sub(r"[\[\]:*?/\\]", "-", str(title))[:31] or "Sheet"
        sheet = workbook.create_sheet(safe)
        if headers:
            sheet.append(list(headers))
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F3864")
        for row in rows:
            sheet.append(list(row))
        for column in sheet.columns:
            width = max((len(str(c.value)) for c in column if c.value is not None),
                        default=8)
            sheet.column_dimensions[column[0].column_letter].width = min(width + 2, 48)
    workbook.save(path)


# ============================================================
# FIGURES
# ============================================================

def make_figure(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    """
    Create a Figure and its axes WITHOUT entering pyplot's global registry.

    `pyplot.subplots()` keeps every figure it makes alive in a module-level
    registry until something explicitly closes it. In a long-lived GUI that
    redraws on each run, nothing ever does — so figures accumulate for the
    life of the session and matplotlib eventually warns:

        More than 20 figures have been opened...

    Building the Figure directly avoids the registry entirely, so an embedded
    figure is freed as soon as its canvas is destroyed, which is exactly the
    lifetime a Tk-embedded plot should have.

    The return signature matches pyplot.subplots(), including its squeeze
    behaviour, so call sites need only swap the constructor.

    Interactive CLI plotting still uses pyplot: there the registry is doing
    its job, because plt.show() needs to find the figures.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, **kwargs)
    if nrows == 1 and ncols == 1:
        return fig, fig.subplots()
    return fig, fig.subplots(nrows, ncols)


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
