"""
fixed_wing_performance.py

A self-contained (no external web data) fixed-wing performance simulator inspired by
eCalc's Fixed Wing Calculator *style of inputs + outputs + plots*.

It supports TWO propulsion calculation modes:
  (A) "ecalc" motor model: KV + Rm + I0 + battery + ESC + prop using coefficient model
  (B) "motor_table" mode: user-provided motor/prop test table (e.g., T-Motor bench data)
      with interpolation to predict thrust/current/power vs throttle, then mapped into flight.

IMPORTANT REALITY CHECK:
- eCalc uses proprietary prop databases + detailed motor/ESC models + empirical corrections.
- This script gives you the same *benchmarks/metrics/graphs* categories, but absolute accuracy
  depends heavily on (1) your prop coefficients or (2) quality/coverage of your test table.
- If you want it to match your real system, CALIBRATE the prop coefficients (Ct, Cp) and/or
  thrust-vs-airspeed scaling using at least a few measured points.

Requires:
  pip install numpy pandas matplotlib

Run examples:
  # eCalc-style motor model
  python fixed_wing_performance.py --config config_example_ecalc.yaml

  # Using motor test table (CSV)
  python fixed_wing_performance.py --config config_example_table.yaml

Outputs:
  - Prints a summary of key performance metrics
  - Saves plots into ./outputs/
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import math
import os
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml  # optional; if not installed you can use JSON-like dict in code
except ImportError:
    yaml = None


# -----------------------------
# Utilities
# -----------------------------

G0 = 9.80665  # m/s^2


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def interp1(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Safe 1D linear interpolation with endpoint clamping."""
    if len(xp) < 2:
        raise ValueError("Need at least 2 points for interpolation.")
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    return float(np.interp(x, xp, fp))


def isa_density(alt_m: float, temp_offset_C: float = 0.0) -> float:
    """
    Very simple ISA troposphere density model (0-11km) with an optional temperature offset.
    Good enough for performance sweeps.

    Steps:
      T = T0 - L*h + offset
      p = p0*(T/T0)^(g/(R*L))
      rho = p/(R*T)
    """
    T0 = 288.15  # K
    p0 = 101325.0  # Pa
    L = 0.0065  # K/m
    R = 287.05287  # J/kg/K

    h = max(0.0, alt_m)
    T = T0 - L * h + temp_offset_C
    T = max(150.0, T)  # guard
    expo = G0 / (R * L)
    p = p0 * (T / T0) ** expo
    rho = p / (R * T)
    return float(rho)


# -----------------------------
# Config models (inputs)
# -----------------------------

@dc.dataclass
class Aircraft:
    mass_kg: float
    wing_area_m2: float
    wingspan_m: float
    cd0: float                      # zero-lift drag coefficient (clean)
    cl_max: float                   # max lift coefficient (clean)
    oswald_e: float = 0.85          # Oswald efficiency (0.7-0.9 typical)
    rolling_mu: float = 0.04        # rolling friction coefficient (takeoff run)
    # Optional user overrides:
    induced_k: Optional[float] = None  # if set, uses Cd_i = k*Cl^2

    def aspect_ratio(self) -> float:
        return self.wingspan_m**2 / self.wing_area_m2

    def induced_factor_k(self) -> float:
        # Cd_i = k*Cl^2, where k = 1/(pi*e*AR)
        if self.induced_k is not None:
            return self.induced_k
        return 1.0 / (math.pi * self.oswald_e * self.aspect_ratio())


@dc.dataclass
class Environment:
    altitude_m: float = 0.0
    temp_offset_C: float = 0.0
    g_mps2: float = G0

    def rho(self) -> float:
        return isa_density(self.altitude_m, self.temp_offset_C)


@dc.dataclass
class Battery:
    cells_s: int
    capacity_ah: float
    # electrical model: V = V_oc - I*R
    v_oc_per_cell: float = 4.2          # open-circuit at start of flight (charged)
    r_internal_ohm: float = 0.015       # pack resistance (Ohm) rough; tune!
    usable_fraction: float = 0.85       # like eCalc "max. discharge"

    def v_oc(self) -> float:
        return self.cells_s * self.v_oc_per_cell

    def usable_wh(self) -> float:
        return self.cells_s * self.v_oc_per_cell * self.capacity_ah * self.usable_fraction


@dc.dataclass
class ESC:
    efficiency: float = 0.98  # lumped PWM+FET losses


@dc.dataclass
class Propeller:
    diameter_in: float
    pitch_in: float
    blades: int = 2

    # Coefficient model (if using ecalc-mode)
    # Static/low-speed coefficient baselines. Typical small props:
    # Ct ~ 0.08..0.14, Cp ~ 0.035..0.08. You MUST tune for your prop family.
    ct0: float = 0.11
    cp0: float = 0.045

    # Advance-ratio shaping:
    # Ct(J) = ct0 * (1 - ct_J1*J - ct_J2*J^2)
    # Cp(J) = cp0 * (1 + cp_J1*J + cp_J2*J^2)
    ct_J1: float = 1.6
    ct_J2: float = 0.3
    cp_J1: float = 0.4
    cp_J2: float = 0.2

    # Tip Mach guard (very approximate)
    tip_mach_limit: float = 0.75

    def D_m(self) -> float:
        return self.diameter_in * 0.0254

    def pitch_m(self) -> float:
        return self.pitch_in * 0.0254

    def pitch_speed_mps(self, rpm: float) -> float:
        # Ideal pitch speed (no slip):
        # V_pitch = pitch * rev_per_sec
        return self.pitch_m() * (rpm / 60.0)

    def J(self, V_mps: float, n_rps: float) -> float:
        # Advance ratio: J = V/(n*D)
        if n_rps <= 1e-6:
            return 0.0
        return V_mps / (n_rps * self.D_m())

    def Ct(self, J: float) -> float:
        ct = self.ct0 * (1.0 - self.ct_J1 * J - self.ct_J2 * J * J)
        return max(0.0, ct)

    def Cp(self, J: float) -> float:
        cp = self.cp0 * (1.0 + self.cp_J1 * J + self.cp_J2 * J * J)
        return max(0.0, cp)


@dc.dataclass
class MotorECALC:
    """
    eCalc-ish DC motor model parameters.

    KV: rpm/Volt
    Rm: Ohms (phase-to-phase equivalent; treat as effective)
    I0: no-load current at nominal voltage (A)
    """
    kv_rpm_per_v: float
    rm_ohm: float
    i0_A: float
    max_current_A: float
    gearbox_ratio: float = 1.0  # motor_rpm / prop_rpm

    def k_e(self) -> float:
        # Back-emf constant in V/(rad/s)
        # KV [rpm/V] => omega = KV*(2pi/60)*V  => Ke = 1/(KV*(2pi/60))
        return 1.0 / (self.kv_rpm_per_v * (2.0 * math.pi / 60.0))

    def k_t(self) -> float:
        # Torque constant Kt = Ke (SI units) in N*m/A
        return self.k_e()


@dc.dataclass
class MotorTable:
    """
    Motor test table interface.

    Expect a CSV with columns (case-insensitive, flexible):
      throttle (0..1 or 0..100), voltage_V, current_A, rpm, thrust_g, power_W (optional)

    If power_W not present, computed as V*I.
    """
    csv_path: str
    throttle_col: str = "throttle"
    voltage_col: str = "voltage"
    current_col: str = "current"
    rpm_col: str = "rpm"
    thrust_col: str = "thrust"
    power_col: Optional[str] = "power"

    # If your table is at a fixed voltage but your sim varies voltage,
    # you can choose a scaling exponent. Start with 1.0..2.0 and tune.
    thrust_voltage_exp: float = 2.0
    rpm_voltage_exp: float = 1.0
    current_voltage_exp: float = 1.0


@dc.dataclass
class SimulationConfig:
    mode: Literal["ecalc", "motor_table"]  # propulsion mode

    aircraft: Aircraft
    env: Environment
    battery: Battery
    esc: ESC
    prop: Propeller

    motor_ecalc: Optional[MotorECALC] = None
    motor_table: Optional[MotorTable] = None

    # Sweep / output controls
    vmin_mps: float = 5.0
    vmax_mps: float = 45.0
    n_points: int = 80

    # Climb target (like eCalc "time to height")
    climb_height_m: float = 500.0

    # For "Angle of climb" we assume steady climb at given airspeed:
    # sin(gamma) = ROC / V
    # ROC = (Pavail - Preq)/W
    # Optionally cap gamma to prevent numerical weirdness.
    gamma_cap_deg: float = 45.0

    outputs_dir: str = "outputs"


# -----------------------------
# Aerodynamics & performance equations
# -----------------------------

def lift_coefficient(W_N: float, rho: float, V: float, S: float) -> float:
    """
    Level-flight lift requirement:
      L = W = 0.5*rho*V^2*S*Cl  => Cl = 2W/(rho V^2 S)
    """
    return (2.0 * W_N) / (rho * V * V * S)


def drag_force(rho: float, V: float, S: float, cd0: float, k: float, cl: float) -> float:
    """
    Drag model:
      Cd = Cd0 + k*Cl^2
      D  = 0.5*rho*V^2*S*Cd
    """
    Cd = cd0 + k * cl * cl
    return 0.5 * rho * V * V * S * Cd


def power_required_level(D_N: float, V: float) -> float:
    """
    Power required for level flight (mechanical at prop disk / airframe):
      P_req = D * V
    """
    return D_N * V


def stall_speed(W_N: float, rho: float, S: float, cl_max: float) -> float:
    """
    Stall speed (clean):
      V_stall = sqrt( 2W/(rho*S*Cl_max) )
    """
    return math.sqrt((2.0 * W_N) / (rho * S * cl_max))


def best_range_speed_carson(V_md: float) -> float:
    """
    Carson speed approximation (common eCalc-like benchmark):
      V_Carson ≈ 3^(1/4) * V_md
    where V_md is minimum drag (minimum power? depends).
    For prop aircraft often: best range near ~1.316 * V_md.
    """
    return (3.0 ** 0.25) * V_md


def min_drag_speed(W_N: float, rho: float, S: float, cd0: float, k: float) -> float:
    """
    Minimum drag speed occurs when parasite drag = induced drag:
      Cd0 = k*Cl^2  => Cl = sqrt(Cd0/k)
      Using Cl = 2W/(rho V^2 S) => solve for V.
    """
    cl_md = math.sqrt(cd0 / k)
    return math.sqrt((2.0 * W_N) / (rho * S * cl_md))


# -----------------------------
# Propulsion models
# -----------------------------

def prop_thrust_power_coeff(
    rho: float, V: float, prop: Propeller, rpm: float
) -> Tuple[float, float, float]:
    """
    Coefficient-based prop model:
      n = rpm/60 [rev/s]
      J = V/(nD)
      T = Ct(J)*rho*n^2*D^4
      P = Cp(J)*rho*n^3*D^5

    Returns: (T_N, P_W, eta_prop)
      eta_prop = (T*V)/P  (guarded)
    """
    D = prop.D_m()
    n = rpm / 60.0
    if n <= 1e-6:
        return 0.0, 0.0, 0.0

    J = prop.J(V, n)
    Ct = prop.Ct(J)
    Cp = prop.Cp(J)

    T = Ct * rho * (n**2) * (D**4)
    P = Cp * rho * (n**3) * (D**5)

    eta = 0.0
    if P > 1e-6:
        eta = clamp((T * V) / P, 0.0, 0.95)
    return T, P, eta


def tip_mach(prop: Propeller, rpm: float, a_mps: float = 343.0) -> float:
    """
    Tip speed ~ pi*D*n. Mach = V_tip / a
    """
    D = prop.D_m()
    n = rpm / 60.0
    v_tip = math.pi * D * n
    return v_tip / a_mps


def solve_ecalc_motor_prop_at_airspeed(
    cfg: SimulationConfig,
    V: float,
    throttle: float,
) -> Dict[str, float]:
    """
    Given airspeed V and throttle (0..1), compute equilibrium prop RPM, thrust, current, etc.

    Model structure (simplified eCalc-ish):
      Battery: V_pack = V_oc - I*R
      ESC: electrical to motor ~ V_motor = V_pack * esc_eff * throttle
      Motor electrical:
        V_motor = Ke*omega + I*Rm
        Torque = Kt*(I - I0)
      Prop load:
        P_prop = Cp(J)*rho*n^3*D^5
        Q_prop = P_prop / omega_prop
      Gearbox:
        omega_motor = omega_prop*gear_ratio
        torque_motor = torque_prop/gear_ratio

    We solve by iterating on omega_prop until motor torque matches prop torque.
    """
    assert cfg.motor_ecalc is not None
    motor = cfg.motor_ecalc
    rho = cfg.env.rho()
    prop = cfg.prop
    batt = cfg.battery
    esc = cfg.esc

    throttle = clamp(throttle, 0.0, 1.0)

    # Initial guess for prop rpm near KV*V * throttle / gear
    V_guess = batt.v_oc() * esc.efficiency * throttle
    rpm_prop = max(500.0, (motor.kv_rpm_per_v * V_guess) / motor.gearbox_ratio)

    # Iterate to match torque
    for _ in range(50):
        omega_prop = rpm_prop * (2.0 * math.pi / 60.0)
        omega_motor = omega_prop * motor.gearbox_ratio

        # Prop power/torque at this rpm
        T_N, P_prop_W, eta_prop = prop_thrust_power_coeff(rho, V, prop, rpm_prop)
        Q_prop = 0.0 if omega_prop < 1e-6 else (P_prop_W / omega_prop)

        # Motor torque required = prop torque / gear_ratio
        Q_motor_req = Q_prop / motor.gearbox_ratio

        # Motor voltage equation: V = Ke*omega + I*Rm
        Ke = motor.k_e()
        Kt = motor.k_t()

        # We don't know pack voltage because it depends on current.
        # We'll solve current by combining:
        #   V_motor = V_pack*esc_eff*throttle
        #   V_pack  = V_oc - I*R
        # and motor back-emf:
        #   V_motor = Ke*omega_motor + I*Rm
        # => V_oc - I*R all scaled by esc_eff*throttle equals Ke*omega + I*Rm
        # Let A = esc_eff*throttle
        A = max(1e-6, esc.efficiency * throttle)
        V_oc = batt.v_oc()

        # A*(V_oc - I*R) = Ke*omega + I*Rm
        # A*V_oc - A*I*R = Ke*omega + I*Rm
        # => I*(Rm + A*R) = A*V_oc - Ke*omega
        denom = motor.rm_ohm + A * batt.r_internal_ohm
        I = (A * V_oc - Ke * omega_motor) / max(1e-9, denom)
        I = max(0.0, I)

        # Motor torque available:
        #   Q = Kt*(I - I0)  (no-load current subtract)
        Q_motor = Kt * max(0.0, I - motor.i0_A)

        # Enforce max current (simple clip)
        if I > motor.max_current_A:
            I = motor.max_current_A
            Q_motor = Kt * max(0.0, I - motor.i0_A)

        # Error between available and required torque
        err = Q_motor - Q_motor_req

        # Update rpm_prop: if motor torque > required, rpm can increase; else decrease
        # Use a damped proportional step.
        rpm_prop = max(0.0, rpm_prop * (1.0 + 0.25 * clamp(err / max(1e-6, Q_motor_req + 1e-6), -0.5, 0.5)))

        # Converged?
        if abs(err) < 1e-4:
            break

    # Recompute final values
    omega_prop = rpm_prop * (2.0 * math.pi / 60.0)
    omega_motor = omega_prop * motor.gearbox_ratio
    T_N, P_prop_W, eta_prop = prop_thrust_power_coeff(rho, V, prop, rpm_prop)

    Ke = motor.k_e()
    A = max(1e-6, esc.efficiency * throttle)
    V_oc = batt.v_oc()
    denom = motor.rm_ohm + A * batt.r_internal_ohm
    I = (A * V_oc - Ke * omega_motor) / max(1e-9, denom)
    I = clamp(I, 0.0, motor.max_current_A)

    V_pack = V_oc - I * batt.r_internal_ohm
    V_motor = V_pack * esc.efficiency * throttle
    P_elec = V_pack * I
    # "Motor efficiency" (rough): prop power / electrical power (bounded)
    eta_total = 0.0 if P_elec < 1e-6 else clamp(P_prop_W / P_elec, 0.0, 1.0)

    # Tip mach guard
    M_tip = tip_mach(prop, rpm_prop)
    tip_ok = 1.0
    if M_tip > prop.tip_mach_limit:
        # Soft penalty (reduce thrust/power) if tip too fast
        tip_ok = prop.tip_mach_limit / M_tip
        T_N *= tip_ok
        P_prop_W *= tip_ok

    return {
        "throttle": throttle,
        "V_mps": V,
        "rpm": rpm_prop,
        "thrust_N": T_N,
        "prop_power_W": P_prop_W,
        "pack_voltage_V": V_pack,
        "motor_voltage_V": V_motor,
        "current_A": I,
        "elec_power_W": P_elec,
        "eta_prop": eta_prop,
        "eta_total": eta_total,
        "tip_mach": M_tip,
    }


def load_motor_table(mt: MotorTable) -> pd.DataFrame:
    df = pd.read_csv(mt.csv_path)
    cols = {c.lower(): c for c in df.columns}

    def pick(name: str) -> str:
        # allow exact, case-insensitive
        key = name.lower()
        if key in cols:
            return cols[key]
        # allow partial matches
        for k, v in cols.items():
            if key in k:
                return v
        raise ValueError(f"Could not find column '{name}' in {df.columns.tolist()}")

    tcol = pick(mt.throttle_col)
    vcol = pick(mt.voltage_col)
    icol = pick(mt.current_col)
    rcol = pick(mt.rpm_col)
    thcol = pick(mt.thrust_col)
    pcol = None
    if mt.power_col is not None:
        try:
            pcol = pick(mt.power_col)
        except Exception:
            pcol = None

    df = df.rename(columns={tcol: "throttle", vcol: "voltage_V", icol: "current_A", rcol: "rpm", thcol: "thrust_g"})
    if pcol is not None:
        df = df.rename(columns={pcol: "power_W"})
    else:
        df["power_W"] = df["voltage_V"] * df["current_A"]

    # Normalize throttle to 0..1
    if df["throttle"].max() > 1.5:
        df["throttle"] = df["throttle"] / 100.0

    df = df.sort_values("throttle").reset_index(drop=True)
    return df


def table_predict_static_at_voltage(
    df: pd.DataFrame,
    V_target: float,
    throttle: float,
    mt: MotorTable,
) -> Dict[str, float]:
    """
    Interpolate bench data vs throttle, then scale to target voltage.

    Many motor tables are collected near-constant voltage.
    We:
      1) Interpolate at throttle to get baseline at V_ref (the table's interpolated voltage)
      2) Scale thrust/rpm/current using exponents (tunable)

    This is a pragmatic approach; for best results, use a table that spans voltage.
    """
    throttle = clamp(throttle, 0.0, 1.0)
    xp = df["throttle"].to_numpy()
    v_ref = interp1(throttle, xp, df["voltage_V"].to_numpy())
    i_ref = interp1(throttle, xp, df["current_A"].to_numpy())
    rpm_ref = interp1(throttle, xp, df["rpm"].to_numpy())
    thrust_g_ref = interp1(throttle, xp, df["thrust_g"].to_numpy())
    power_ref = interp1(throttle, xp, df["power_W"].to_numpy())

    # Scale ratios
    if v_ref <= 1e-6:
        v_ref = V_target

    r = V_target / v_ref
    rpm = rpm_ref * (r ** mt.rpm_voltage_exp)
    thrust_g = thrust_g_ref * (r ** mt.thrust_voltage_exp)
    current_A = i_ref * (r ** mt.current_voltage_exp)
    power_W = V_target * current_A

    return {
        "throttle": throttle,
        "voltage_V": V_target,
        "current_A": current_A,
        "rpm": rpm,
        "thrust_N": (thrust_g * 0.001) * G0,
        "prop_power_W": power_W,  # treat as "shaft-ish" unless you have separate mech P
        "elec_power_W": power_W,
    }


def scale_static_to_dynamic_thrust(
    prop: Propeller, rpm: float, V: float, T_static_N: float
) -> float:
    """
    Convert static thrust estimate to a crude dynamic-thrust estimate in forward flight.

    Common “first-order” idea:
      - as airspeed approaches pitch speed, thrust drops toward ~0.
      - T_dynamic ≈ T_static * max(0, 1 - V/V_pitch)^p

    This is NOT a substitute for real prop Ct(J) data, but it gives the right *shape*
    for the power diagram.

    Tune exponent p (~1..2) if needed.
    """
    V_pitch = max(1e-6, prop.pitch_speed_mps(rpm))
    x = clamp(1.0 - (V / V_pitch), 0.0, 1.0)
    p = 1.5
    return T_static_N * (x ** p)


def solve_table_mode_at_airspeed(
    cfg: SimulationConfig,
    V: float,
    throttle: float,
    table_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Use the motor bench table to estimate static thrust/current at throttle, then map into forward flight.

    Steps:
      - Compute pack voltage under load (battery sag) using I*R
      - Predict static thrust/rpm/current at that voltage (scaling)
      - Convert static thrust to dynamic thrust with a pitch-speed rolloff
    """
    assert cfg.motor_table is not None
    mt = cfg.motor_table
    batt = cfg.battery
    esc = cfg.esc
    prop = cfg.prop

    throttle = clamp(throttle, 0.0, 1.0)

    # First pass: assume voltage near open-circuit
    V_oc = batt.v_oc()
    pred0 = table_predict_static_at_voltage(table_df, V_oc, throttle, mt)
    I0 = pred0["current_A"]

    # Battery sag with ESC efficiency folded in loosely:
    V_pack = V_oc - I0 * batt.r_internal_ohm
    V_pack = max(0.0, V_pack)
    V_eff = V_pack * esc.efficiency

    pred = table_predict_static_at_voltage(table_df, V_eff, throttle, mt)
    T_static = pred["thrust_N"]
    rpm = pred["rpm"]
    I = pred["current_A"]
    P_elec = V_pack * I

    # Forward-flight correction
    T_dyn = scale_static_to_dynamic_thrust(prop, rpm, V, T_static)

    # Propulsive efficiency estimate
    eta_prop = 0.0 if pred["prop_power_W"] < 1e-6 else clamp((T_dyn * V) / pred["prop_power_W"], 0.0, 0.95)

    return {
        "throttle": throttle,
        "V_mps": V,
        "rpm": rpm,
        "thrust_N": T_dyn,
        "prop_power_W": pred["prop_power_W"],
        "pack_voltage_V": V_pack,
        "current_A": I,
        "elec_power_W": P_elec,
        "eta_prop": eta_prop,
        "eta_total": 0.0 if P_elec < 1e-6 else clamp(pred["prop_power_W"] / P_elec, 0.0, 1.0),
        "tip_mach": tip_mach(cfg.prop, rpm),
    }


# -----------------------------
# Performance sweep + metrics
# -----------------------------

def compute_level_flight_sweep(cfg: SimulationConfig) -> pd.DataFrame:
    """
    Sweep airspeed and compute:
      - required power (level flight)
      - available thrust/power at "full throttle" (or a set throttle schedule)
      - rate of climb, climb angle, max speed, best range speed, etc.
    """
    os.makedirs(cfg.outputs_dir, exist_ok=True)

    rho = cfg.env.rho()
    W = cfg.aircraft.mass_kg * cfg.env.g_mps2
    S = cfg.aircraft.wing_area_m2
    cd0 = cfg.aircraft.cd0
    k = cfg.aircraft.induced_factor_k()

    # Prepare propulsion source
    table_df = None
    if cfg.mode == "motor_table":
        assert cfg.motor_table is not None
        table_df = load_motor_table(cfg.motor_table)

    V = np.linspace(cfg.vmin_mps, cfg.vmax_mps, cfg.n_points)
    rows = []

    for v in V:
        cl = lift_coefficient(W, rho, v, S)
        D = drag_force(rho, v, S, cd0, k, cl)
        P_req = power_required_level(D, v)

        # "Available" at full throttle (1.0)
        if cfg.mode == "ecalc":
            avail = solve_ecalc_motor_prop_at_airspeed(cfg, v, throttle=1.0)
        else:
            avail = solve_table_mode_at_airspeed(cfg, v, throttle=1.0, table_df=table_df)

        T_av = avail["thrust_N"]
        P_prop = avail["prop_power_W"]
        P_elec = avail["elec_power_W"]

        # Excess power -> climb rate:
        #   ROC = (P_avail_mech - P_req) / W
        # Here we treat prop_power_W as "power imparted by prop" (mechanical/air power proxy).
        roc = (P_prop - P_req) / W
        roc = max(-50.0, min(50.0, roc))

        # climb angle gamma: sin(gamma) = ROC/V
        sin_g = 0.0 if v < 1e-6 else clamp(roc / v, -1.0, 1.0)
        gamma = math.asin(sin_g)
        gamma_deg = clamp(math.degrees(gamma), -cfg.gamma_cap_deg, cfg.gamma_cap_deg)

        # "3D capability" gauge-ish metric: thrust-to-weight at static-ish (use v~0 estimate)
        rows.append({
            "V_mps": v,
            "V_kmh": v * 3.6,
            "Cl": cl,
            "Drag_N": D,
            "P_req_W": P_req,
            "T_avail_N": T_av,
            "P_prop_W": P_prop,
            "P_elec_W": P_elec,
            "I_A": avail["current_A"],
            "V_pack": avail.get("pack_voltage_V", np.nan),
            "RPM": avail["rpm"],
            "ROC_mps": roc,
            "gamma_deg": gamma_deg,
            "eta_prop": avail.get("eta_prop", np.nan),
            "eta_total": avail.get("eta_total", np.nan),
            "tip_mach": avail.get("tip_mach", np.nan),
        })

    df = pd.DataFrame(rows)

    # Derive min-drag speed and Carson speed (benchmarks)
    V_stall = stall_speed(W, rho, S, cfg.aircraft.cl_max)
    V_md = min_drag_speed(W, rho, S, cd0, k)
    V_carson = best_range_speed_carson(V_md)

    # Find max level speed (where P_avail >= P_req) or thrust >= drag
    feasible = df["P_prop_W"] >= df["P_req_W"]
    if feasible.any():
        vmax = df.loc[feasible, "V_mps"].max()
    else:
        vmax = float("nan")

    # Best ROC & corresponding angle
    idx_best_roc = df["ROC_mps"].idxmax()
    best_roc = float(df.loc[idx_best_roc, "ROC_mps"])
    best_roc_speed = float(df.loc[idx_best_roc, "V_mps"])
    best_gamma = float(df.loc[idx_best_roc, "gamma_deg"])

    # Time to climb target height at best ROC (if positive)
    t_to_h = float("inf") if best_roc <= 1e-6 else (cfg.climb_height_m / best_roc)

    # Estimate endurance/range at Carson speed (simple)
    # energy usable / electrical power at that speed
    P_elec_at_carson = interp1(V_carson, df["V_mps"].to_numpy(), df["P_elec_W"].to_numpy())
    t_endurance_s = float("inf") if P_elec_at_carson <= 1e-6 else (cfg.battery.usable_wh() * 3600.0 / P_elec_at_carson)
    range_m = V_carson * t_endurance_s

    # Attach summary attributes
    df.attrs["summary"] = {
        "rho_kgm3": rho,
        "W_N": W,
        "stall_speed_mps": V_stall,
        "stall_speed_kmh": V_stall * 3.6,
        "V_md_mps": V_md,
        "V_md_kmh": V_md * 3.6,
        "V_carson_mps": V_carson,
        "V_carson_kmh": V_carson * 3.6,
        "max_speed_mps": vmax,
        "max_speed_kmh": vmax * 3.6 if np.isfinite(vmax) else float("nan"),
        "best_ROC_mps": best_roc,
        "best_ROC_speed_mps": best_roc_speed,
        "best_ROC_speed_kmh": best_roc_speed * 3.6,
        "best_gamma_deg": best_gamma,
        "time_to_height_s": t_to_h,
        "range_m_est": range_m,
        "range_km_est": range_m / 1000.0,
        "endurance_min_est": t_endurance_s / 60.0,
    }
    return df


def estimate_takeoff_distance(cfg: SimulationConfig, thrust_static_N: float) -> float:
    """
    Extremely simplified takeoff run (ground roll) estimate to show the same category as eCalc.

    We assume:
      - accelerate from 0 to 1.2*Vstall
      - net force = T - D - mu*(W - L)
      - during ground roll, approximate L grows with V^2, so average (W-L) ~ 0.5*W
      - drag during roll uses Cd0 only (rough)

    This is meant as a *benchmark* not a certification calc.
    """
    rho = cfg.env.rho()
    ac = cfg.aircraft
    W = ac.mass_kg * cfg.env.g_mps2
    S = ac.wing_area_m2

    Vstall = stall_speed(W, rho, S, ac.cl_max)
    V_to = 1.2 * Vstall

    # Average lift fraction during ground roll (very rough)
    normal_force_avg = 0.5 * W

    # Average drag during takeoff run using Cd0
    V_avg = 0.7 * V_to
    D_avg = 0.5 * rho * V_avg * V_avg * S * ac.cd0

    F_net = thrust_static_N - D_avg - ac.rolling_mu * normal_force_avg
    if F_net <= 1e-6:
        return float("inf")

    a = F_net / ac.mass_kg
    s = (V_to * V_to) / (2.0 * a)
    return float(s)


# -----------------------------
# Plotting (eCalc-like outputs)
# -----------------------------

def plot_power_diagram(df: pd.DataFrame, cfg: SimulationConfig) -> str:
    """
    eCalc-like "Power Diagram in Level Flight":
      - min power required for level flight
      - available prop power (dynamic)
    """
    out = os.path.join(cfg.outputs_dir, "power_diagram_level_flight.png")
    plt.figure(figsize=(10, 6))
    plt.plot(df["V_kmh"], df["P_req_W"], label="min. Power for Level Flight [W]")
    plt.plot(df["V_kmh"], df["P_prop_W"], label="dynamic Propeller Power [W]")
    plt.xlabel("Air Speed [km/h]")
    plt.ylabel("Power [W]")
    plt.title("Power Diagram in Level Flight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_rate_of_climb(df: pd.DataFrame, cfg: SimulationConfig) -> str:
    out = os.path.join(cfg.outputs_dir, "rate_of_climb.png")
    plt.figure(figsize=(10, 6))
    plt.plot(df["V_kmh"], df["ROC_mps"], label="Rate of Climb [m/s]")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Air Speed [km/h]")
    plt.ylabel("Rate of Climb [m/s]")
    plt.title("Rate of Climb vs Air Speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_climb_angle(df: pd.DataFrame, cfg: SimulationConfig) -> str:
    out = os.path.join(cfg.outputs_dir, "climb_angle.png")
    plt.figure(figsize=(10, 6))
    plt.plot(df["V_kmh"], df["gamma_deg"], label="Angle of Climb [deg]")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Air Speed [km/h]")
    plt.ylabel("Angle [deg]")
    plt.title("Angle of Climb vs Air Speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_static_vertical_climb_indicator(cfg: SimulationConfig, thrust_static_N: float) -> str:
    """
    eCalc-like "Static Vertical Climb / insufficient static thrust" indicator.

    For a fixed-wing, pure vertical climb is usually not meaningful, but eCalc shows a 3D/vertical capability gauge.
    We'll compute:
      thrust_to_weight = T_static / W
      a_vertical = (T - W)/m
    """
    out = os.path.join(cfg.outputs_dir, "static_vertical_climb_indicator.png")
    W = cfg.aircraft.mass_kg * cfg.env.g_mps2
    tw = thrust_static_N / W

    plt.figure(figsize=(8, 4))
    plt.axis("off")
    txt = f"Static Thrust-to-Weight: {tw:.2f}\n"
    if tw < 1.0:
        txt += "Result: insufficient static thrust for vertical climb (T/W < 1)\n"
    else:
        a = (thrust_static_N - W) / cfg.aircraft.mass_kg
        txt += f"Estimated vertical acceleration: {a:.2f} m/s² (ignoring aero)\n"
    plt.text(0.02, 0.6, txt, fontsize=14)
    plt.title("Static Vertical Climb (3D pull-up performance)")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


# -----------------------------
# Config loading
# -----------------------------

def load_config(path: str) -> SimulationConfig:
    """
    YAML config loader for convenience.
    """
    if yaml is None:
        raise RuntimeError("pyyaml not installed. Run: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    aircraft = Aircraft(**d["aircraft"])
    env = Environment(**d.get("environment", {}))
    battery = Battery(**d["battery"])
    esc = ESC(**d.get("esc", {}))
    prop = Propeller(**d["propeller"])

    mode = d["mode"]
    motor_ecalc = MotorECALC(**d["motor_ecalc"]) if "motor_ecalc" in d else None
    motor_table = MotorTable(**d["motor_table"]) if "motor_table" in d else None

    sim = d.get("simulation", {})
    cfg = SimulationConfig(
        mode=mode,
        aircraft=aircraft,
        env=env,
        battery=battery,
        esc=esc,
        prop=prop,
        motor_ecalc=motor_ecalc,
        motor_table=motor_table,
        vmin_mps=sim.get("vmin_mps", 5.0),
        vmax_mps=sim.get("vmax_mps", 45.0),
        n_points=sim.get("n_points", 80),
        climb_height_m=sim.get("climb_height_m", 500.0),
        gamma_cap_deg=sim.get("gamma_cap_deg", 45.0),
        outputs_dir=sim.get("outputs_dir", "outputs"),
    )
    return cfg


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Compute sweep
    df = compute_level_flight_sweep(cfg)
    summary = df.attrs["summary"]

    # Estimate static thrust at V ~ 0 for "3D capability" and takeoff
    if cfg.mode == "ecalc":
        static = solve_ecalc_motor_prop_at_airspeed(cfg, V=0.1, throttle=1.0)
    else:
        table_df = load_motor_table(cfg.motor_table)  # type: ignore[arg-type]
        static = solve_table_mode_at_airspeed(cfg, V=0.1, throttle=1.0, table_df=table_df)
    T_static = static["thrust_N"]

    takeoff_m = estimate_takeoff_distance(cfg, T_static)

    # Print eCalc-like metric block
    print("\n================ Aircraft Performance (eCalc-style categories) ================\n")
    print(f"Air density rho:                 {summary['rho_kgm3']:.3f} kg/m^3")
    print(f"All-up mass:                     {cfg.aircraft.mass_kg:.3f} kg")
    print(f"Wing area:                       {cfg.aircraft.wing_area_m2:.4f} m^2")
    print(f"Aspect ratio:                    {cfg.aircraft.aspect_ratio():.2f}")
    print(f"Cd0 (clean):                     {cfg.aircraft.cd0:.4f}")
    print(f"Induced k:                       {cfg.aircraft.induced_factor_k():.5f}")
    print()
    print(f"Stall speed (1g, clean):         {summary['stall_speed_kmh']:.1f} km/h")
    print(f"Best range speed (Carson):       {summary['V_carson_kmh']:.1f} km/h")
    print(f"Max speed (horizontal):          {summary['max_speed_kmh']:.1f} km/h")
    print(f"Max rate of climb:               {summary['best_ROC_mps']:.2f} m/s @ {summary['best_ROC_speed_kmh']:.1f} km/h")
    print(f"Max angle of climb (approx):     {summary['best_gamma_deg']:.1f} deg")
    print(f"Time to {cfg.climb_height_m:.0f} m (best ROC):      {summary['time_to_height_s']:.1f} s")
    print(f"Static thrust-to-weight:         {(T_static/(cfg.aircraft.mass_kg*cfg.env.g_mps2)):.2f}")
    print(f"Estimated takeoff distance:      {takeoff_m:.1f} m")
    print()
    print(f"Estimated endurance @ Carson:    {summary['endurance_min_est']:.1f} min")
    print(f"Estimated range @ Carson:        {summary['range_km_est']:.2f} km")
    print("\n===============================================================================\n")

    # Save CSV like eCalc "Download.csv"
    csv_out = os.path.join(cfg.outputs_dir, "performance_sweep.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved sweep CSV: {csv_out}")

    # Plots (eCalc-like)
    p1 = plot_power_diagram(df, cfg)
    p2 = plot_rate_of_climb(df, cfg)
    p3 = plot_climb_angle(df, cfg)
    p4 = plot_static_vertical_climb_indicator(cfg, T_static)

    print(f"Saved plots:\n  {p1}\n  {p2}\n  {p3}\n  {p4}")


if __name__ == "__main__":
    main()


"""
-----------------------------
Example YAML configs
-----------------------------

1) config_example_ecalc.yaml  (eCalc-ish motor model)
-----------------------------------------------------
mode: ecalc

aircraft:
  mass_kg: 0.850
  wing_area_m2: 0.50         # 50 dm^2
  wingspan_m: 1.270
  cd0: 0.060
  cl_max: 1.2
  oswald_e: 0.85
  rolling_mu: 0.04

environment:
  altitude_m: 500
  temp_offset_C: 0.0

battery:
  cells_s: 3
  capacity_ah: 3.0
  v_oc_per_cell: 4.2
  r_internal_ohm: 0.020
  usable_fraction: 0.85

esc:
  efficiency: 0.98

propeller:
  diameter_in: 10
  pitch_in: 4.7
  blades: 2
  # Start points; tune these to your prop family / test data:
  ct0: 0.11
  cp0: 0.045
  ct_J1: 1.6
  ct_J2: 0.3
  cp_J1: 0.4
  cp_J2: 0.2
  tip_mach_limit: 0.75

motor_ecalc:
  kv_rpm_per_v: 900
  rm_ohm: 0.05
  i0_A: 1.2
  max_current_A: 45
  gearbox_ratio: 1.0

simulation:
  vmin_mps: 6.0
  vmax_mps: 45.0
  n_points: 90
  climb_height_m: 500
  outputs_dir: outputs


2) config_example_table.yaml  (motor test results table)
--------------------------------------------------------
mode: motor_table

aircraft:
  mass_kg: 0.850
  wing_area_m2: 0.50
  wingspan_m: 1.270
  cd0: 0.060
  cl_max: 1.2
  oswald_e: 0.85
  rolling_mu: 0.04

environment:
  altitude_m: 500
  temp_offset_C: 0.0

battery:
  cells_s: 12
  capacity_ah: 10.0
  v_oc_per_cell: 4.2
  r_internal_ohm: 0.012
  usable_fraction: 0.85

esc:
  efficiency: 0.98

propeller:
  diameter_in: 21
  pitch_in: 6.3
  blades: 2
  tip_mach_limit: 0.75

motor_table:
  csv_path: tmotor_test_table.csv
  throttle_col: throttle
  voltage_col: voltage
  current_col: current
  rpm_col: rpm
  thrust_col: thrust
  power_col: power
  thrust_voltage_exp: 2.0
  rpm_voltage_exp: 1.0
  current_voltage_exp: 1.0

simulation:
  vmin_mps: 8.0
  vmax_mps: 60.0
  n_points: 100
  climb_height_m: 500
  outputs_dir: outputs

-----------------------------
Notes on making your T-Motor table CSV:
- Your screenshot shows columns like:
  Throttle, Voltage (V), Thrust (g), Torque (N*m), Current (A), RPM, Power (W), Efficiency (g/W)
- Save a CSV with at least: throttle, voltage, current, rpm, thrust, power (optional)
- throttle can be 40..100 or 0.4..1.0 (script normalizes automatically if > 1.5)
"""