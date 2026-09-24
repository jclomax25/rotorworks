#!/usr/bin/env python3
"""
vtol-power-sim-gui.py
=====================
VTOL UAV power and endurance simulator.

A VTOL is not a new physics problem so much as two existing ones joined by a
transition. This reuses the rotor model from the multicopter simulator and the
wing model from the fixed-wing simulator, both via `rotorworks_core`, and adds
the part neither has: the region where the wing and the rotors share the lift.

Configurations
--------------
Only **lift+cruise** is implemented. The others are present in the dropdown so
the input set is defined and a saved configuration written today stays
readable when they land:

  * ``lift+cruise``  - separate lift rotors and a cruise propeller. The lift
    rotors stop in cruise and are carried as drag. IMPLEMENTED.
  * ``tiltrotor``    - the lift rotors tilt forward to become cruise thrust.
  * ``tiltwing``     - the whole wing tilts, so the rotors stay aligned with
    the wing throughout.
  * ``tailsitter``   - the airframe itself rotates; reference areas change
    continuously through the transition.

The three unimplemented modes are rejected with a clear message rather than
silently approximated as lift+cruise, because their transition physics differs
substantially and a wrong answer that looks plausible is worse than a refusal.

Where the energy goes
---------------------
Hover is expensive and cruise is cheap, so a VTOL's endurance is dominated by
how long it spends in each and how much the transition costs. That is why the
mission model, not the single-point model, is the useful part of this tool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.patches import Circle as _Circle, Rectangle as _Rectangle

# ------------------------------------------------------------------
# Shared core, same as the other two simulators. Must sit beside this file.
# ------------------------------------------------------------------
try:
    import rotorworks_core as core
except ImportError as _exc:      # pragma: no cover - install/deploy problem
    raise SystemExit(
        "rotorworks_core.py could not be imported. It must sit in the same "
        f"folder as this script.\nOriginal error: {_exc}"
    )

SIM_VERSION = "0.9.0"
SIM_BUILD_NOTE = "VTOL simulator - lift+cruise, tiltrotor, tiltwing, tailsitter"

G0 = core.G0

CONFIG_TYPES = ["lift+cruise", "tiltrotor", "tiltwing", "tailsitter"]
IMPLEMENTED_CONFIG_TYPES = {"lift+cruise", "tiltrotor", "tiltwing", "tailsitter"}


# ============================================================
# CONFIGURATION
# ============================================================

class VTOLBattery:
    """
    Pack model. Deliberately thin: capacity, voltage and resistance, with the
    state-of-charge behaviour delegated to the shared core so all three
    simulators agree on what a battery does.
    """

    def __init__(self,
                 chemistry: str = "LiPo",
                 cell_capacity_mAh: float = 5000.0,
                 series_cells: int = 6,
                 parallel_cells: int = 1,
                 cell_weight_g: float = 120.0,
                 voltage_min: float = 3.3,
                 voltage_nominal: float = 3.7,
                 voltage_max: float = 4.2,
                 resistance_cell_mOhm: float = 4.0,
                 usable_percent: float = 80.0,
                 soc_model: str = "auto",
                 discharge_c_cont: Optional[float] = None,
                 discharge_c_max: Optional[float] = None,
                 soc_curve_csv: Optional[str] = None):
        self.chemistry = chemistry
        # Optional C-rate ratings. Blank means "not specified", and Status
        # says so rather than inventing a limit to check against.
        self.discharge_c_cont = (float(discharge_c_cont)
                                 if discharge_c_cont not in (None, "", 0) else None)
        self.discharge_c_max = (float(discharge_c_max)
                                if discharge_c_max not in (None, "", 0) else None)
        self.series_cells = max(int(series_cells), 1)
        self.parallel_cells = max(int(parallel_cells), 1)

        # Series raises voltage, parallel raises capacity. Never both.
        self.capacity_mAh = float(cell_capacity_mAh) * self.parallel_cells
        self.capacity_Ah = self.capacity_mAh / 1000.0
        self.weight_g = float(cell_weight_g) * self.series_cells * self.parallel_cells

        self.vmin_pack = float(voltage_min) * self.series_cells
        self.vnom_pack = float(voltage_nominal) * self.series_cells
        self.vmax_pack = float(voltage_max) * self.series_cells

        self.resistance_cell = float(resistance_cell_mOhm) / 1000.0
        self.usable_fraction = min(max(float(usable_percent) / 100.0, 0.0), 1.0)

        self.soc_nonlinear_enabled = False
        self.soc_model_source = "linear-fallback"
        self.soc_bp: List[float] = []
        self.ocv_cell_bp: List[float] = []
        self.r_scale_bp: List[float] = []
        # A measured discharge curve, when one is given, outranks the
        # chemistry preset — the core resolver applies the priority order.
        # This needs no time-stepping: the curve maps state of charge to
        # open-circuit voltage and resistance, and the mission already tracks
        # state of charge phase by phase.
        self.soc_curve_csv = (str(soc_curve_csv).strip() if soc_curve_csv else None)
        # The core resolver falls back to the chemistry preset when a curve
        # cannot be read, which is the wrong behaviour here: a curve you
        # think is loaded but is not gives preset numbers wearing measured
        # clothes. Fail loudly instead, as the propeller tables do.
        if self.soc_curve_csv and not os.path.exists(self.soc_curve_csv):
            raise FileNotFoundError(
                f"SoC curve not found: {self.soc_curve_csv}")
        core.configure_battery_soc_model(self, soc_model, self.soc_curve_csv,
                                         None, None, None)

    @property
    def discharge_cont_A(self) -> Optional[float]:
        """Continuous current limit from the C-rating, or None if unrated."""
        return (self.discharge_c_cont * self.capacity_Ah
                if self.discharge_c_cont else None)

    @property
    def discharge_max_A(self) -> Optional[float]:
        """Burst current limit from the C-rating, or None if unrated."""
        return (self.discharge_c_max * self.capacity_Ah
                if self.discharge_c_max else None)

    @property
    def pack_resistance(self) -> float:
        return self.resistance_cell * self.series_cells / max(self.parallel_cells, 1)

    @property
    def capacity_Wh(self) -> float:
        return self.capacity_Ah * self.vnom_pack

    @property
    def usable_Wh(self) -> float:
        return self.capacity_Wh * self.usable_fraction

    def ocv_at_soc(self, soc: float) -> float:
        return core.pack_ocv_from_soc(self, soc)

    def resistance_at_soc(self, soc: float) -> float:
        return core.pack_resistance_from_soc(self, soc)

    def voltage_under_load(self, current_A: float, soc: Optional[float] = None) -> float:
        return core.pack_voltage_under_load(self, current_A, soc)


class VTOLConfig:
    """
    A lift+cruise VTOL: a wing, a set of lift rotors, and a cruise propeller.

    Weights are in grams and lengths in metres, matching the other two
    simulators so a user moving between them is not caught out.
    """

    def __init__(self,
                 config_type: str = "lift+cruise",
                 aircraft_weight_g: float = 6000.0,
                 payload_mass_g: float = 0.0,
                 # wing
                 wing_span_m: float = 2.4,
                 wing_area_m2: float = 0.60,
                 CD0: float = 0.035,
                 oswald: float = 0.80,
                 CL_max: float = 1.20,
                 CL_cruise_max: float = 0.90,
                 # lift rotors
                 num_lift_rotors: int = 4,
                 lift_prop_diameter_in: float = 18.0,
                 lift_prop_pitch_in: float = 6.0,
                 lift_motor_kv: float = 300.0,
                 lift_motor_resistance: float = 0.08,
                 lift_motor_weight_g: float = 200.0,
                 lift_figure_of_merit: float = 0.65,
                 # cruise propulsion
                 num_cruise_motors: int = 1,
                 cruise_prop_diameter_in: float = 14.0,
                 cruise_prop_pitch_in: float = 8.0,
                 cruise_motor_kv: float = 500.0,
                 cruise_motor_resistance: float = 0.06,
                 cruise_motor_weight_g: float = 180.0,
                 cruise_prop_efficiency: float = 0.75,
                 # drag of the stopped lift rotors in cruise
                 stopped_rotor_drag_area_m2: Optional[float] = None,
                 # systems
                 battery: Optional[VTOLBattery] = None,
                 avionics_power_W: float = 15.0,
                 esc_efficiency: float = 0.96,
                 # environment
                 air_density: float = 1.225,
                 cruise_speed_mps: float = 22.0,
                 reference_altitude_m: float = 0.0,
                 wire_resistance_ohm: float = 0.0,
                 connectors: Optional[dict] = None,
                 hover_download_fraction: Optional[float] = None,
                 lift_motor_max_power_W: Optional[float] = None,
                 cruise_motor_max_power_W: Optional[float] = None,
                 lift_prop_weight_g: float = 0.0,
                 cruise_prop_weight_g: float = 0.0,
                 avionics_mass_g: float = 0.0,
                 lift_prop_table_csv: Optional[str] = None,
                 cruise_prop_table_csv: Optional[str] = None):
        self.config_type = str(config_type).strip().lower()
        self.aircraft_weight_g = float(aircraft_weight_g)
        self.payload_mass_g = max(float(payload_mass_g), 0.0)

        self.wing_span_m = float(wing_span_m)
        self.wing_area_m2 = float(wing_area_m2)
        self.CD0 = float(CD0)
        self.oswald = float(oswald)
        self.CL_max = float(CL_max)
        self.CL_cruise_max = float(CL_cruise_max)

        self.num_lift_rotors = max(int(num_lift_rotors), 1)
        self.lift_prop_diameter_in = float(lift_prop_diameter_in)
        self.lift_prop_pitch_in = float(lift_prop_pitch_in)
        self.lift_motor_kv = float(lift_motor_kv)
        self.lift_motor_resistance = float(lift_motor_resistance)
        self.lift_motor_weight_g = float(lift_motor_weight_g)
        self.lift_figure_of_merit = min(max(float(lift_figure_of_merit), 0.2), 0.9)

        self.num_cruise_motors = max(int(num_cruise_motors), 1)
        self.cruise_prop_diameter_in = float(cruise_prop_diameter_in)
        self.cruise_prop_pitch_in = float(cruise_prop_pitch_in)
        self.cruise_motor_kv = float(cruise_motor_kv)
        self.cruise_motor_resistance = float(cruise_motor_resistance)
        self.cruise_motor_weight_g = float(cruise_motor_weight_g)
        self.cruise_prop_efficiency = min(max(float(cruise_prop_efficiency), 0.1), 0.95)

        self._stopped_rotor_drag_area_m2 = stopped_rotor_drag_area_m2

        self.battery = battery if battery is not None else VTOLBattery()
        self.avionics_power_W = max(float(avionics_power_W), 0.0)
        self.esc_efficiency = min(max(float(esc_efficiency), 0.5), 1.0)

        self.air_density = float(air_density)
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.reference_altitude_m = max(float(reference_altitude_m), 0.0)
        # ---- optional detail, all blank by default ----------------------
        # Left blank, the model behaves exactly as it did before these
        # inputs existed.
        self.wire_resistance_ohm = max(float(wire_resistance_ohm or 0.0), 0.0)
        self.connectors = dict(connectors or {})
        self.hover_download_fraction = (None if hover_download_fraction in (None, "")
                                        else max(float(hover_download_fraction), 0.0))
        self.lift_motor_max_power_W = (float(lift_motor_max_power_W)
                                       if lift_motor_max_power_W else None)
        self.cruise_motor_max_power_W = (float(cruise_motor_max_power_W)
                                         if cruise_motor_max_power_W else None)
        self.lift_prop_weight_g = max(float(lift_prop_weight_g or 0.0), 0.0)
        self.cruise_prop_weight_g = max(float(cruise_prop_weight_g or 0.0), 0.0)
        self.avionics_mass_g = max(float(avionics_mass_g or 0.0), 0.0)

        # Measured bench tables. Loaded once here rather than per call, and a
        # bad path raises now rather than silently leaving the estimate in
        # place — a table you think is loaded but is not is worse than none.
        self.lift_prop_table_csv = lift_prop_table_csv or None
        self.cruise_prop_table_csv = cruise_prop_table_csv or None
        self.lift_prop_table = (core.load_prop_table(self.lift_prop_table_csv)
                                if self.lift_prop_table_csv else None)
        self.cruise_prop_table = (core.load_prop_table(self.cruise_prop_table_csv)
                                  if self.cruise_prop_table_csv else None)

    # ---- derived ----------------------------------------------------

    @property
    def all_up_weight_g(self) -> float:
        return self.aircraft_weight_g + self.payload_mass_g

    @property
    def weight_N(self) -> float:
        return self.all_up_weight_g * G0 / 1000.0

    @property
    def aspect_ratio(self) -> float:
        return (self.wing_span_m ** 2) / max(self.wing_area_m2, 1e-9)

    @property
    def induced_drag_factor(self) -> float:
        return 1.0 / (math.pi * max(self.aspect_ratio, 1e-9) * max(self.oswald, 1e-9))

    @property
    def lift_disc_area_m2(self) -> float:
        d = self.lift_prop_diameter_in * 0.0254
        return math.pi / 4.0 * d * d * self.num_lift_rotors

    @property
    def cruise_disc_area_m2(self) -> float:
        d = self.cruise_prop_diameter_in * 0.0254
        return math.pi / 4.0 * d * d

    @property
    def disc_loading_N_m2(self) -> float:
        return self.weight_N / max(self.lift_disc_area_m2, 1e-9)

    @property
    def wing_loading_N_m2(self) -> float:
        return self.weight_N / max(self.wing_area_m2, 1e-9)

    @property
    def stopped_rotor_drag_area_m2(self) -> float:
        """
        Equivalent flat-plate area of the stopped lift rotors in cruise.

        This is the price a lift+cruise pays for its simplicity, and it is not
        negligible — four stopped props and their booms are a real drag item.

        The default is 1.5% of disc area per rotor. A two-blade prop has a
        solidity near 0.10, but a stopped blade is normally parked aligned
        with the airflow (many aircraft do this deliberately), so only a
        fraction of the blade planform is presented. 1.5% corresponds to a
        blade roughly edge-on with an effective Cd near 1.

        Measure or estimate this properly if you can: on the reference
        airframe it is still around a quarter of total cruise drag, so it
        directly sets whether the cruise leg pays for itself. Values from 1%
        (folding props) to 4% (large flat blades parked across the flow) are
        all realistic.
        """
        if self._stopped_rotor_drag_area_m2 is not None:
            return max(float(self._stopped_rotor_drag_area_m2), 0.0)
        d = self.lift_prop_diameter_in * 0.0254
        disc = math.pi / 4.0 * d * d
        return 0.015 * disc * self.num_lift_rotors


# ============================================================
# AERODYNAMICS
# ============================================================

def stall_speed_mps(cfg: VTOLConfig) -> float:
    """Minimum speed at which the wing alone can carry the weight."""
    return math.sqrt(2.0 * cfg.weight_N /
                     (cfg.air_density * cfg.wing_area_m2 * max(cfg.CL_max, 1e-9)))


def wing_lift_N(cfg: VTOLConfig, airspeed_mps: float, cl_cap: Optional[float] = None) -> float:
    """
    Lift the wing produces at a given airspeed.

    Capped at CL_max (or a lower cap during transition, where flying at the
    stall boundary would be reckless). Never more than the aircraft weight —
    surplus lift is not useful here, the rotors simply unload.
    """
    cl_limit = cfg.CL_max if cl_cap is None else min(cl_cap, cfg.CL_max)
    v = max(float(airspeed_mps), 0.0)
    lift = 0.5 * cfg.air_density * v * v * cfg.wing_area_m2 * cl_limit
    return min(lift, cfg.weight_N)


def wing_drag_N(cfg: VTOLConfig, airspeed_mps: float, lift_N: float) -> float:
    """Wing drag at the CL needed to produce `lift_N`, plus parasite drag."""
    v = max(float(airspeed_mps), 0.0)
    if v < 1e-6:
        return 0.0
    q = 0.5 * cfg.air_density * v * v
    cl = lift_N / max(q * cfg.wing_area_m2, 1e-9)
    cd = cfg.CD0 + cfg.induced_drag_factor * cl * cl
    return q * cfg.wing_area_m2 * cd


def stopped_rotor_drag_N(cfg: VTOLConfig, airspeed_mps: float) -> float:
    """Drag of the lift rotors once they have stopped for cruise."""
    v = max(float(airspeed_mps), 0.0)
    return 0.5 * cfg.air_density * v * v * cfg.stopped_rotor_drag_area_m2


# ============================================================
# ROTOR AND PROPELLER POWER
# ============================================================

def measured_lift_efficiency(cfg: VTOLConfig, thrust_per_rotor_N: float,
                             disc_area_m2: float) -> Optional[float]:
    """
    Figure of merit measured by the lift-rotor bench table, or None.

    The table's power is ELECTRICAL, while the figure of merit describes the
    shaft. Dividing out the ESC efficiency converts between them, so that
    running the chain forward — ideal / FoM, then / esc_efficiency — lands
    back on the power the bench actually recorded.

    Returns None outside the measured thrust range, where the estimate is
    used instead. That is deliberate: a table says nothing about thrusts it
    never produced.
    """
    if cfg.lift_prop_table is None:
        return None
    eta = core.measured_static_efficiency(cfg.lift_prop_table, thrust_per_rotor_N,
                                          cfg.air_density, disc_area_m2)
    if eta is None:
        return None
    return min(max(eta / max(cfg.esc_efficiency, 1e-9), 0.2), 0.9)


def measured_cruise_efficiency(cfg: VTOLConfig, thrust_per_motor_N: float,
                               disc_area_m2: float) -> Optional[float]:
    """Combined cruise motor-and-propeller efficiency from its table, or None."""
    if cfg.cruise_prop_table is None:
        return None
    eta = core.measured_static_efficiency(cfg.cruise_prop_table, thrust_per_motor_N,
                                          cfg.air_density, disc_area_m2)
    if eta is None:
        return None
    return min(max(eta / max(cfg.esc_efficiency, 1e-9), 0.2), 0.95)


def rotor_power_W(cfg: VTOLConfig, thrust_N: float, airspeed_mps: float = 0.0) -> float:
    """
    Shaft power for the lift rotors to make `thrust_N` in total.

    Momentum theory with a figure of merit for real losses, using the shared
    forward-flight inflow solver so a rotor climbing away or translating in
    the transition is not charged its hover induced power.
    """
    if thrust_N <= 0:
        return 0.0
    n = cfg.num_lift_rotors
    d = cfg.lift_prop_diameter_in * 0.0254
    area = math.pi / 4.0 * d * d
    t_per = float(thrust_N) / n

    v_hover = math.sqrt(t_per / max(2.0 * cfg.air_density * area, 1e-9))
    # Lift rotors stay level, so the freestream is edgewise: incidence 0.
    vi = core.induced_velocity_forward_flight(v_hover, airspeed_mps, 0.0)
    ideal_per = t_per * vi
    fom = measured_lift_efficiency(cfg, t_per, area) or cfg.lift_figure_of_merit
    return ideal_per / fom * n


def cruise_prop_power_W(cfg: VTOLConfig, thrust_N: float, airspeed_mps: float) -> float:
    """
    Shaft power for the cruise propeller to make `thrust_N` at `airspeed_mps`.

    Forward-flight momentum theory: P = T * (V + vi). The V term is the
    propulsive work against drag and dominates in cruise.
    """
    if thrust_N <= 0:
        return 0.0
    n = cfg.num_cruise_motors
    area = cfg.cruise_disc_area_m2
    t_per = float(thrust_N) / n
    v = max(float(airspeed_mps), 0.0)

    # vi from T = 2 rho A vi (V + vi), positive root.
    vi = -v / 2.0 + math.sqrt((v / 2.0) ** 2 +
                              t_per / max(2.0 * cfg.air_density * area, 1e-9))
    eff = measured_cruise_efficiency(cfg, t_per, area) or cfg.cruise_prop_efficiency
    return t_per * (v + vi) / eff * n


def wire_loss_W(cfg: VTOLConfig, power_before_wire_W: float) -> float:
    """
    I^2 R loss in the main wire run for a given load.

    Solved with the current rather than added afterwards: the loss draws
    extra current, which costs extra loss. A few passes converge, since the
    loss is a small fraction of the total.
    """
    r = float(getattr(cfg, "wire_resistance_ohm", 0.0) or 0.0)
    if r <= 0 or power_before_wire_W <= 0:
        return 0.0
    v = max(cfg.battery.vnom_pack, 1e-9)
    loss = 0.0
    for _ in range(4):
        current = (power_before_wire_W + loss) / v
        loss = current * current * r
    return loss


def electrical_power_W(cfg: VTOLConfig, shaft_power_W: float) -> float:
    """
    Shaft power to pack power: through the ESC, plus the avionics load, plus
    the main wire run if one was entered.
    """
    base = shaft_power_W / cfg.esc_efficiency + cfg.avionics_power_W
    return base + wire_loss_W(cfg, base)


# ============================================================
# FLIGHT REGIMES
# ============================================================

def hover_power_W(cfg: VTOLConfig, climb_rate_mps: float = 0.0) -> Dict[str, float]:
    """Power to hover, or to climb vertically on the lift rotors."""
    if uses_vectored_thrust(cfg):
        return vectored_power_W(cfg, 0.0, climb_rate_mps=climb_rate_mps)
    # In hover the rotors carry the whole weight, so they pay the whole
    # download. The transition branch scales the same fraction by the share
    # of weight the rotors still hold, which makes the two meet exactly at
    # zero airspeed instead of stepping.
    thrust_N = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
    shaft = rotor_power_W(cfg, thrust_N, airspeed_mps=0.0)
    # Vertical climb adds potential power directly.
    shaft += thrust_N * max(float(climb_rate_mps), 0.0)
    total = electrical_power_W(cfg, shaft)
    return {
        "regime": "hover",
        "airspeed_mps": 0.0,
        "rotor_thrust_N": thrust_N,
        "wing_lift_N": 0.0,
        "cruise_thrust_N": 0.0,
        "rotor_shaft_W": shaft,
        "cruise_shaft_W": 0.0,
        "shaft_power_W": shaft,
        "total_power_W": total,
    }


def cruise_power_W(cfg: VTOLConfig, airspeed_mps: float) -> Dict[str, float]:
    """
    Power in wing-borne cruise, lift rotors stopped.

    Valid only above the stall speed; below it the wing cannot carry the
    aircraft and the transition model applies instead.
    """
    v = max(float(airspeed_mps), 1e-6)
    lift_N = cfg.weight_N
    drag_N = wing_drag_N(cfg, v, lift_N) + stopped_rotor_drag_N(cfg, v)
    shaft = cruise_prop_power_W(cfg, drag_N, v)
    total = electrical_power_W(cfg, shaft)
    return {
        "regime": "cruise",
        "airspeed_mps": v,
        "rotor_thrust_N": 0.0,
        "wing_lift_N": lift_N,
        "cruise_thrust_N": drag_N,
        "drag_N": drag_N,
        "rotor_shaft_W": 0.0,
        "cruise_shaft_W": shaft,
        "shaft_power_W": shaft,
        "total_power_W": total,
    }


def transition_power_W(cfg: VTOLConfig, airspeed_mps: float,
                       cl_cap: Optional[float] = None) -> Dict[str, float]:
    """
    Power during transition, where the wing and the rotors share the lift.

    This is the part a VTOL model lives or dies on. The split is set by what
    the wing can actually carry at this airspeed:

        L_wing  = min( q * S * CL_cap , W )
        T_rotor = W - L_wing

    The rotors make up the shortfall while the cruise propeller pushes against
    the drag of a wing flying at high CL. Both draw at once, which is why the
    transition is the most power-hungry part of the flight and why a slow
    transition is expensive.

    `cl_cap` defaults to CL_cruise_max rather than CL_max: transitioning at
    the stall boundary leaves no margin for a gust, and no sane controller
    would do it.
    """
    v = max(float(airspeed_mps), 0.0)
    cap = cfg.CL_cruise_max if cl_cap is None else cl_cap

    lift_N = wing_lift_N(cfg, v, cl_cap=cap)
    # Download is a hover effect that fades as the wing takes over: it exists
    # because the rotor wash strikes structure below, and there is less wash
    # to strike with the less the rotors are lifting. Scaling the fraction by
    # the rotors' share of the weight makes hover (share 1, full download)
    # and wing-borne cruise (share 0, none) the two ends of one continuous
    # curve rather than two branches that disagree where they meet.
    rotor_share = min(max(1.0 - lift_N / max(cfg.weight_N, 1e-9), 0.0), 1.0)
    download_N = cfg.weight_N * hover_download_fraction(cfg) * rotor_share
    rotor_thrust_N = max(cfg.weight_N + download_N - lift_N, 0.0)

    # Drag: the wing at whatever CL it is holding, plus a share of the
    # stopped-rotor drag.
    #
    # The rotors do not stop instantly. As the wing takes over they unload
    # and spin down, turning progressively from thrust producers into drag.
    # Scaling their flat-plate drag by the wing's lift share captures that,
    # and — importantly — makes the transition and cruise models agree at the
    # boundary. Without it, power jumped 105 -> 131 W the instant the regime
    # switched, which is the same flight condition described two ways.
    rotor_drag_N = stopped_rotor_drag_N(cfg, v) * min(max(
        lift_N / max(cfg.weight_N, 1e-9), 0.0), 1.0)
    drag_N = wing_drag_N(cfg, v, lift_N) + rotor_drag_N
    cruise_thrust_N = drag_N

    rotor_shaft = rotor_power_W(cfg, rotor_thrust_N, airspeed_mps=v)
    cruise_shaft = cruise_prop_power_W(cfg, cruise_thrust_N, v)
    shaft = rotor_shaft + cruise_shaft
    total = electrical_power_W(cfg, shaft)

    return {
        "regime": "transition",
        "airspeed_mps": v,
        "rotor_thrust_N": rotor_thrust_N,
        "wing_lift_N": lift_N,
        "cruise_thrust_N": cruise_thrust_N,
        "drag_N": drag_N,
        "lift_share_wing": lift_N / max(cfg.weight_N, 1e-9),
        "rotor_shaft_W": rotor_shaft,
        "cruise_shaft_W": cruise_shaft,
        "shaft_power_W": shaft,
        "total_power_W": total,
    }


# ============================================================
# VECTORED-THRUST VTOL (tiltrotor, tiltwing, tailsitter)
# ============================================================
#
# Lift+cruise carries two propulsion systems: rotors that only lift and a
# propeller that only pushes. The other three types carry ONE set of rotors
# that does both, by pointing their thrust somewhere between straight up and
# straight ahead. That single fact is what separates them physically:
#
#   * no dead rotors in cruise — nothing stopped, nothing dragging;
#   * but the rotors are sized for HOVER, so in cruise they are large,
#     lightly loaded propellers and give up some propulsive efficiency;
#   * and the thrust has to be resolved into a lifting part and a pushing
#     part, which the wing shares as it comes up to speed.
#
# What distinguishes the three from EACH OTHER is mostly where the rotor wash
# goes in hover. A tiltrotor's rotors blow down onto a wing lying flat below
# them, and that "download" has to be lifted as well — around a tenth of the
# weight on real aircraft. A tiltwing turns the wing with the rotors, and a
# tailsitter stands the whole wing on end, so in both the wing is edge-on to
# the wash and the penalty nearly vanishes.

# WHY TILTWING AND TAILSITTER COME OUT THE SAME HERE
# In still air, at steady state, they are genuinely very close: both turn the
# wing edge-on to the rotor wash in hover, and both use the same rotors as
# propellers in cruise. What really separates them is DYNAMICS, which a power
# model does not see:
#   * a tailsitter hovers with its whole wing standing vertical, so any
#     crosswind strikes that wing broadside and it weathervanes hard;
#   * a tiltwing keeps its fuselage level and only swings the wing;
#   * a tiltwing's wing sits in the rotor slipstream through transition,
#     which buys lift at low airspeed — but quantifying that needs a
#     slipstream-coverage factor this model has no data to set.
# Rather than invent a coefficient to force them apart, both default to the
# same download and either can be overridden with measured data.

# Extra hover thrust needed because rotor wash strikes the airframe beneath,
# as a fraction of weight. Typical published values, NOT measurements of any
# particular aircraft: treat these as starting points and override them.
HOVER_DOWNLOAD_FRACTION = {
    "lift+cruise": 0.04,   # booms outboard of the wing; partial blockage
    "tiltrotor":   0.10,   # rotors over a flat wing — the V-22 runs ~10-12%
    "tiltwing":    0.02,   # wing turns edge-on to the wash
    "tailsitter":  0.02,   # whole wing stands edge-on
}


def hover_download_fraction(cfg: VTOLConfig) -> float:
    """Download for this aircraft: an explicit override, else the type default."""
    override = getattr(cfg, "hover_download_fraction", None)
    if override is not None:
        return max(float(override), 0.0)
    return HOVER_DOWNLOAD_FRACTION.get(cfg.config_type, 0.04)


def uses_vectored_thrust(cfg: VTOLConfig) -> bool:
    """True for the types whose rotors both lift and push."""
    return cfg.config_type in ("tiltrotor", "tiltwing", "tailsitter")


def vectored_rotor_power_W(cfg: VTOLConfig, thrust_N: float,
                           tilt_deg: float, airspeed_mps: float) -> float:
    """
    Shaft power for the rotors to make `thrust_N` pointed `tilt_deg` from
    vertical (0 = straight up, hover; 90 = straight ahead, cruise).

    The freestream meets the disc at the tilt angle, so the inflow solver is
    given that incidence: edgewise in hover, axial in cruise, and everything
    between during the transition. Power is then T * (V_axial + vi).

    Efficiency blends from the hover figure of merit to the propeller
    efficiency with the tilt, so the rotor is judged as a rotor when lifting
    and as a propeller when pushing, and the two meet smoothly rather than
    switching at some arbitrary angle.
    """
    if thrust_N <= 0:
        return 0.0
    n = cfg.num_lift_rotors
    d = cfg.lift_prop_diameter_in * 0.0254
    area = math.pi / 4.0 * d * d
    t_per = float(thrust_N) / n
    v = max(float(airspeed_mps), 0.0)
    tilt = min(max(float(tilt_deg), 0.0), 90.0)

    v_hover = math.sqrt(t_per / max(2.0 * cfg.air_density * area, 1e-9))
    vi = core.induced_velocity_forward_flight(v_hover, v, math.radians(tilt))
    v_axial = v * math.sin(math.radians(tilt))

    # A vectored rotor is the same hardware in both roles, so one table
    # describes it throughout: the hover end of the blend uses the measured
    # figure of merit where the table covers the thrust.
    hover_eff = measured_lift_efficiency(cfg, t_per, area) or cfg.lift_figure_of_merit
    blend = tilt / 90.0
    efficiency = (1.0 - blend) * hover_eff + blend * cfg.cruise_prop_efficiency
    return t_per * (v_axial + vi) / max(efficiency, 0.05) * n


def vectored_power_W(cfg: VTOLConfig, airspeed_mps: float,
                     cl_cap: Optional[float] = None,
                     climb_rate_mps: float = 0.0) -> Dict[str, float]:
    """
    Power for a vectored-thrust VTOL at any airspeed, hover through cruise.

    Two force balances fix both the thrust and where it points:

        vertical:     T*cos(tilt) + L_wing = W * (1 + download)
        horizontal:   T*sin(tilt)          = D

    so the rotors deliver the vector sum

        T = sqrt( D^2 + (W*(1+download) - L_wing)^2 )

    pointed at  tilt = atan2(D, W*(1+download) - L_wing).

    The download only applies while the rotors are still doing lifting work:
    it scales with the share of weight they carry, falling to nothing once
    the wing has taken the whole load and the wash is streaming aft. That
    keeps hover and cruise continuous — the same aircraft described at two
    speeds, not two different aircraft.
    """
    v = max(float(airspeed_mps), 0.0)
    cap = cfg.CL_cruise_max if cl_cap is None else cl_cap
    weight = cfg.weight_N

    lift_N = wing_lift_N(cfg, v, cl_cap=cap) if v > 1e-6 else 0.0
    rotor_share = min(max(1.0 - lift_N / max(weight, 1e-9), 0.0), 1.0)
    download = hover_download_fraction(cfg) * rotor_share

    vertical_N = max(weight * (1.0 + download) - lift_N, 0.0)
    drag_N = wing_drag_N(cfg, v, lift_N) if v > 1e-6 else 0.0

    thrust_N = math.hypot(vertical_N, drag_N)
    tilt_deg = math.degrees(math.atan2(drag_N, max(vertical_N, 1e-12)))

    shaft = vectored_rotor_power_W(cfg, thrust_N, tilt_deg, v)
    shaft += thrust_N * max(float(climb_rate_mps), 0.0) * math.cos(
        math.radians(tilt_deg))
    total = electrical_power_W(cfg, shaft)

    if v < 1e-6:
        regime = "hover"
    elif lift_N >= weight - 1e-9:
        regime = "cruise"
    else:
        regime = "transition"

    return {
        "regime": regime,
        "airspeed_mps": v,
        "rotor_thrust_N": thrust_N,
        "tilt_deg": tilt_deg,
        "wing_lift_N": lift_N,
        "cruise_thrust_N": drag_N,
        "drag_N": drag_N,
        "download_N": weight * download,
        "lift_share_wing": lift_N / max(weight, 1e-9),
        "rotor_shaft_W": shaft,
        "cruise_shaft_W": 0.0,
        "shaft_power_W": shaft,
        "total_power_W": total,
    }


def power_at_airspeed(cfg: VTOLConfig, airspeed_mps: float) -> Dict[str, float]:
    """
    Power at any airspeed, choosing the regime automatically.

    Below the speed at which the wing can carry the whole aircraft, the
    rotors are still contributing and the transition model applies. Above it,
    the aircraft is wing-borne and the rotors can stop.
    """
    v = max(float(airspeed_mps), 0.0)
    if uses_vectored_thrust(cfg):
        return vectored_power_W(cfg, v)
    if v < 1e-6:
        return hover_power_W(cfg)

    lift_available = wing_lift_N(cfg, v, cl_cap=cfg.CL_cruise_max)
    if lift_available >= cfg.weight_N - 1e-9:
        return cruise_power_W(cfg, v)
    return transition_power_W(cfg, v)


def transition_speed_mps(cfg: VTOLConfig) -> float:
    """
    Lowest speed at which the wing alone carries the aircraft, at the
    transition CL cap. Above this the lift rotors can be shut down.
    """
    return math.sqrt(2.0 * cfg.weight_N /
                     (cfg.air_density * cfg.wing_area_m2 *
                      max(cfg.CL_cruise_max, 1e-9)))


# ============================================================
# METRICS
# ============================================================

# Fields shown in Simple mode. Everything else is hidden until the user
# switches to Advanced.
#
# The rule used to pick them: someone sizing a first VTOL needs the weight,
# the wing, the pack, the rotor and propeller sizes, and the environment.
# They do NOT need figure-of-merit tuning, stopped-rotor drag area, bench
# tables, connector ratings or a download override — every one of those has
# a sensible default and exists to be refined later, not chosen up front.
#
# This is a view setting only. Hidden fields keep their values, so switching
# modes never changes a result.
VTOL_SIMPLE_FIELDS = {
    "weight", "payload", "span", "area", "cd0", "clmax",
    "n_lift", "lift_d", "lift_p", "lift_kv", "lift_wt",
    "n_cruise", "cruise_d", "cruise_p", "cruise_kv", "cruise_wt",
    "chem", "cell_cap", "series", "parallel", "cell_wt",
    "cruise_v", "alt", "temp", "wind", "wind_dir", "mission", "avionics",
}


def make_airframe_diagram_figure(cfg: VTOLConfig, figsize=(9, 7.5)):
    """
    Plan view of the aircraft, to scale, drawn from the numbers entered.
    Nose up, starboard right, as a plan view is normally read.

    The layout differs by configuration because the configurations differ in
    where the lifting rotors live: lift+cruise puts them on booms fore and
    aft of the wing and carries a separate propeller at the nose, while the
    vectored types mount them along the wing, since the same rotors cruise.

    Only span, wing area and propeller diameters are real inputs. Boom
    positions are a reasonable arrangement, not a claim about a particular
    airframe — so this is for checking proportions and clearances, not a
    layout to build from.
    """
    span = max(float(cfg.wing_span_m), 1e-3)
    chord = max(float(cfg.wing_area_m2) / span, 1e-3)
    lift_d = float(cfg.lift_prop_diameter_in) * 0.0254
    lift_r = lift_d / 2.0
    n = max(int(cfg.num_lift_rotors), 1)

    fig, ax = core.make_figure(figsize=figsize)
    ax.add_patch(_Rectangle((-span / 2.0, -chord / 2.0), span, chord,
                            facecolor="#cfd8dc", edgecolor="#37474F",
                            linewidth=1.4, zorder=2))

    vectored = uses_vectored_thrust(cfg)
    positions = []
    per_side = max(n // 2, 1)
    if vectored:
        for side in (-1, 1):
            for k in range(per_side):
                frac = (k + 1) / (per_side + 1)
                positions.append((side * frac * span / 2.0, chord / 2.0 + lift_r * 0.15))
    else:
        boom_x = span * 0.30
        for side in (-1, 1):
            for k in range(per_side):
                fore = 1 if k % 2 == 0 else -1
                offset = (k // 2 + 1) * (chord / 2.0 + lift_r * 1.15)
                positions.append((side * boom_x, fore * offset))
        for x, y in positions:
            ax.plot([x, x], [0, y], color="#37474F", linewidth=3.0, zorder=1)
        ax.plot([-boom_x, boom_x], [0, 0], color="#37474F", linewidth=3.0, zorder=1)

    for idx, (x, y) in enumerate(positions[:n], start=1):
        ax.add_patch(_Circle((x, y), lift_r, fill=False, edgecolor="#2E7D32",
                             linewidth=1.3, zorder=3))
        ax.plot([x], [y], marker="o", markersize=4, color="#2E7D32", zorder=4)
        ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, fontweight="bold", color="#2E7D32")

    if not vectored and cfg.num_cruise_motors > 0:
        cruise_r = float(cfg.cruise_prop_diameter_in) * 0.0254 / 2.0
        nose = chord / 2.0 + cruise_r * 1.1
        ax.add_patch(_Circle((0.0, nose), cruise_r, fill=False,
                             edgecolor="#EF6C00", linewidth=1.3, zorder=3))
        ax.annotate("cruise", (0.0, nose), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#EF6C00")

    # Gap between adjacent discs on the same side. Negative means they
    # overlap, which is exactly what a plan view is for catching.
    gap_note = ""
    same_side = [pt for pt in positions if pt[0] > 0]
    if len(same_side) >= 2:
        gap = math.dist(same_side[0], same_side[1]) - lift_d
        gap_note = f"\nTip-to-tip gap   {gap * 1000:+.0f} mm"
        if gap < 0:
            ax.text(0, -chord / 2.0 - lift_d, "ROTOR DISCS OVERLAP", ha="center",
                    color="#B71C1C", fontsize=10, fontweight="bold")

    ax.text(0.02, 0.02,
            f"Span             {span * 1000:.0f} mm\n"
            f"Chord (area/span) {chord * 1000:.0f} mm\n"
            f"Lift prop        {lift_d * 1000:.0f} mm x {n}" + gap_note,
            transform=ax.transAxes, fontsize=8, family="monospace", va="bottom",
            bbox=dict(boxstyle="round", facecolor="#FFFDE7", edgecolor="#BDBDBD"))

    reach = max(span / 2.0,
                max((abs(y) for _x, y in positions), default=0.0) + lift_r) * 1.15
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("metres (starboard +)")
    ax.set_ylabel("metres (nose +)")
    ax.set_title(f"{cfg.config_type} — plan view, to scale")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def compute_metrics(cfg: VTOLConfig, airspeed_mps: Optional[float] = None) -> Dict[str, object]:
    """Single-point metrics at a cruise airspeed, plus the hover comparison."""
    _require_implemented(cfg)

    v = cfg.cruise_speed_mps if airspeed_mps is None else float(airspeed_mps)

    hover = hover_power_W(cfg)
    point = power_at_airspeed(cfg, v)
    usable_Wh = cfg.battery.usable_Wh

    hover_min = usable_Wh / max(hover["total_power_W"], 1e-9) * 60.0
    cruise_min = usable_Wh / max(point["total_power_W"], 1e-9) * 60.0

    v_stall = stall_speed_mps(cfg)
    v_trans = transition_speed_mps(cfg)

    pack_I = point["total_power_W"] / max(cfg.battery.vnom_pack, 1e-9)
    v_load = cfg.battery.voltage_under_load(pack_I)

    metrics: Dict[str, object] = {
        "config_type": cfg.config_type,
        "airspeed_mps": v,
        "regime": point["regime"],
        "all_up_weight_g": cfg.all_up_weight_g,
        "weight_N": cfg.weight_N,
        "wing_loading_N_m2": cfg.wing_loading_N_m2,
        "disc_loading_N_m2": cfg.disc_loading_N_m2,
        "aspect_ratio": cfg.aspect_ratio,
        "stall_speed_mps": v_stall,
        "transition_speed_mps": v_trans,

        "hover_power_W": hover["total_power_W"],
        "hover_endurance_min": hover_min,
        "hover_disc_loading_N_m2": cfg.disc_loading_N_m2,

        "total_power_W": point["total_power_W"],
        "shaft_power_W": point["shaft_power_W"],
        "rotor_shaft_W": point["rotor_shaft_W"],
        "cruise_shaft_W": point["cruise_shaft_W"],
        "rotor_thrust_N": point["rotor_thrust_N"],
        "wing_lift_N": point["wing_lift_N"],
        "cruise_thrust_N": point["cruise_thrust_N"],
        "drag_N": point.get("drag_N", 0.0),

        "cruise_endurance_min": cruise_min,
        "cruise_range_km": cruise_min * 60.0 * v / 1000.0,
        "hover_to_cruise_power_ratio": hover["total_power_W"] /
                                       max(point["total_power_W"], 1e-9),

        "pack_current_A": pack_I,
        "v_load_V": v_load,
        "usable_Wh": usable_Wh,
        "battery_weight_g": cfg.battery.weight_g,
        "soc_model": core.soc_model_short_label(cfg.battery.soc_model_source),

        "avionics_power_W": cfg.avionics_power_W,
        "stopped_rotor_drag_area_m2": cfg.stopped_rotor_drag_area_m2,
    }
    metrics.update(_detail_metrics(cfg, point, hover, pack_I))
    return metrics


def _detail_metrics(cfg: VTOLConfig, point: dict, hover: dict,
                    pack_I: float) -> Dict[str, object]:
    """
    Figures the Status, Power Budget and Weight Budget tabs need that the
    headline metrics do not carry: where the losses go, what each motor
    carries, and what each component weighs.
    """
    shaft = float(point.get("shaft_power_W", 0.0))
    esc_loss = shaft / max(cfg.esc_efficiency, 1e-9) - shaft
    before_wire = shaft / max(cfg.esc_efficiency, 1e-9) + cfg.avionics_power_W
    hover_shaft = float(hover.get("shaft_power_W", 0.0))
    hover_total = float(hover.get("total_power_W", 0.0))

    # In a lift+cruise the lift rotors and the cruise motor are separate, so
    # each is loaded by its own job. In a vectored type the same rotors do
    # both, so they carry the whole shaft power in both regimes.
    if uses_vectored_thrust(cfg):
        cruise_per_motor = shaft / max(cfg.num_lift_rotors, 1)
    else:
        cruise_per_motor = (float(point.get("cruise_shaft_W", 0.0))
                            / max(cfg.num_cruise_motors, 1))

    return {
        "tilt_deg": float(point.get("tilt_deg", 90.0 if point.get("regime") == "cruise"
                                    else 0.0)),
        "download_N": float(point.get("download_N", 0.0)),
        "lift_share_wing": float(point.get("lift_share_wing",
                                           1.0 if point.get("regime") == "cruise" else 0.0)),
        "esc_loss_W": esc_loss,
        "wire_loss_W": wire_loss_W(cfg, before_wire),
        "wire_drop_V": pack_I * float(getattr(cfg, "wire_resistance_ohm", 0.0) or 0.0),
        "c_rate": pack_I / max(cfg.battery.capacity_Ah, 1e-9),
        "hover_pack_current_A": hover_total / max(cfg.battery.vnom_pack, 1e-9),
        "hover_power_per_lift_motor_W": hover_shaft / max(cfg.num_lift_rotors, 1),
        "cruise_power_per_motor_W": cruise_per_motor,
        "lift_motor_mass_g": cfg.lift_motor_weight_g * cfg.num_lift_rotors,
        "cruise_motor_mass_g": (0.0 if uses_vectored_thrust(cfg)
                                else cfg.cruise_motor_weight_g * cfg.num_cruise_motors),
        "lift_prop_mass_g": cfg.lift_prop_weight_g * cfg.num_lift_rotors,
        "cruise_prop_mass_g": (0.0 if uses_vectored_thrust(cfg)
                               else cfg.cruise_prop_weight_g * cfg.num_cruise_motors),
        "avionics_mass_g": cfg.avionics_mass_g,
        "payload_mass_g": cfg.payload_mass_g,
    }


def _require_implemented(cfg: VTOLConfig) -> None:
    """
    Refuse unimplemented configurations rather than approximating them.

    A tiltrotor is not a lift+cruise with different labels: its rotors carry
    thrust through the whole transition and its disc loading in cruise is
    completely different. Silently treating one as the other would produce a
    plausible-looking answer that is simply wrong.
    """
    if cfg.config_type not in IMPLEMENTED_CONFIG_TYPES:
        raise NotImplementedError(
            f"Configuration '{cfg.config_type}' is not implemented yet. "
            f"Currently available: {', '.join(sorted(IMPLEMENTED_CONFIG_TYPES))}.\n"
            "The input set is defined so saved configurations stay readable, "
            "but the transition physics differs enough that approximating it "
            "as lift+cruise would give a confidently wrong answer."
        )


# ============================================================
# MISSION
# ============================================================

@dataclass
class VTOLPhase:
    name: str
    kind: str                     # hover | climb | transition | cruise | descend
    duration_s: Optional[float] = None
    distance_m: Optional[float] = None
    airspeed_mps: float = 0.0
    climb_rate_mps: float = 0.0
    altitude_m: float = 0.0
    # Heading, compass degrees. Only used to draw the route; a VTOL's power
    # does not depend on which way it is pointed in still air.
    course_deg: float = 0.0


@dataclass
class VTOLMission:
    phases: List[VTOLPhase] = field(default_factory=list)
    reserve_percent: float = 20.0
    transition_time_s: float = 12.0

    @staticmethod
    def from_json(path: str) -> "VTOLMission":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        phases = []
        for entry in data.get("phases", []):
            phases.append(VTOLPhase(
                name=str(entry.get("name", "phase")),
                kind=str(entry.get("kind", "cruise")).strip().lower(),
                duration_s=(float(entry["duration"]) if "duration" in entry else None),
                distance_m=(float(entry["distance"]) if "distance" in entry else None),
                airspeed_mps=float(entry.get("speed", 0.0)),
                climb_rate_mps=float(entry.get("climb_rate_mps", 0.0)),
                altitude_m=float(entry.get("altitude", 0.0)),
                course_deg=float(entry.get("course_deg", 0.0)),
            ))
        return VTOLMission(
            phases=phases,
            reserve_percent=float(data.get("reserve_percent", 20.0)),
            transition_time_s=float(data.get("transition_time_s", 12.0)),
        )


def simulate_mission(cfg: VTOLConfig, mission: VTOLMission,
                     wind_mps: float = 0.0, wind_direction_deg: float = 0.0,
                     max_accel_mps2: float = 0.0, max_decel_mps2: float = 0.0,
                     regen_eff: float = 0.0, transient_dt_s: float = 0.25) -> Tuple[List[tuple], Dict[str, float]]:
    """
    Fly the mission phase by phase, draining the pack.

    Transitions are integrated across their speed range rather than charged at
    a single point, because power varies steeply through them — that is the
    whole reason the transition matters to a VTOL's energy budget.
    """
    _require_implemented(cfg)

    usable_Wh = cfg.battery.usable_Wh
    reserve_Wh = usable_Wh * mission.reserve_percent / 100.0
    remaining_Wh = usable_Wh

    results: List[tuple] = []
    totals = {"time_s": 0.0, "distance_m": 0.0, "energy_Wh": 0.0,
              "hover_Wh": 0.0, "transition_Wh": 0.0, "cruise_Wh": 0.0}

    # A time series for Mission Plots and the altitude trace, plus the worst
    # value every check reaches (for Status) and the last instant flown (for
    # Metrics). Kept inside `totals` so the (results, totals) return shape
    # every existing caller relies on does not change.
    vnom = max(cfg.battery.vnom_pack, 1e-9)
    # Wind changes GROUND progress, not the air the aircraft flies through.
    wind_mps = max(float(wind_mps or 0.0), 0.0)
    series: Dict[str, list] = {k: [] for k in (
        "t_s", "phase", "kind", "airspeed_mps", "altitude_m", "distance_km",
        "total_power_W", "pack_current_A", "c_rate", "energy_remaining_Wh",
        "tilt_deg", "lift_share_wing")}
    worst = {"total_power_W": 0.0, "pack_current_A": 0.0, "c_rate": 0.0,
             "remaining_Wh": usable_Wh, "peak_phase": ""}
    state = {"t": 0.0, "alt": 0.0, "dist": 0.0, "v": 0.0}

    def _record(name, kind, v, alt, power, detail=None):
        current = power / vnom
        series["t_s"].append(state["t"])
        series["phase"].append(name)
        series["kind"].append(kind)
        series["airspeed_mps"].append(v)
        series["altitude_m"].append(alt)
        series["distance_km"].append(state["dist"] / 1000.0)
        series["total_power_W"].append(power)
        series["pack_current_A"].append(current)
        series["c_rate"].append(current / max(cfg.battery.capacity_Ah, 1e-9))
        series["energy_remaining_Wh"].append(remaining_Wh)
        series["tilt_deg"].append(float((detail or {}).get("tilt_deg", 0.0)))
        series["lift_share_wing"].append(float((detail or {}).get("lift_share_wing", 0.0)))
        if power > worst["total_power_W"]:
            worst["total_power_W"] = power
            worst["peak_phase"] = name
        worst["pack_current_A"] = max(worst["pack_current_A"], current)
        worst["c_rate"] = max(worst["c_rate"],
                              current / max(cfg.battery.capacity_Ah, 1e-9))
        worst["remaining_Wh"] = min(worst["remaining_Wh"], remaining_Wh)

    for phase in mission.phases:
        alt_start = state["alt"]
        # Reset per phase: only cruise legs run a transient lead-in, but the
        # status line below is common to every phase.
        overshoot_m = 0.0
        kind = phase.kind
        if kind in ("hover", "climb", "descend"):
            climb = phase.climb_rate_mps if kind == "climb" else 0.0
            # Holding station in wind is not free: the aircraft must fly at
            # the wind speed through the air to stay over one spot, so it
            # carries that drag the whole time.
            point = (power_at_airspeed(cfg, wind_mps) if wind_mps > 0.1
                     else hover_power_W(cfg, climb_rate_mps=climb))
            if wind_mps > 0.1 and climb > 0:
                point = dict(point)
                point["total_power_W"] += (cfg.weight_N * climb
                                           / max(cfg.esc_efficiency, 1e-9))
            duration = float(phase.duration_s or 0.0)
            distance = 0.0
            bucket = "hover_Wh"

        elif kind == "transition":
            # Integrate from hover to the transition speed (or the reverse),
            # which is where the power peak lives.
            duration = float(phase.duration_s or mission.transition_time_s)
            v_end = phase.airspeed_mps or transition_speed_mps(cfg)
            steps = 20
            energy_Ws = 0.0
            distance = 0.0
            # Integrate from the speed the aircraft is ACTUALLY at, not from
            # zero. Both transitions used to sweep 0 -> v_end, so the landing
            # transition was modelled as another acceleration and the kinetic
            # cost was charged twice per round trip instead of once out and
            # released on the way back.
            v_start = state["v"]
            for i in range(steps):
                frac = (i + 0.5) / steps
                v = v_start + (v_end - v_start) * frac
                sub = power_at_airspeed(cfg, v)
                dt = duration / steps

                # The transition is not a quasi-static sweep through speeds:
                # the aircraft is ACCELERATING from hover to flying speed,
                # and that kinetic energy has to come from the pack. It was
                # missing here while the cruise legs already paid it.
                #
                # It is not small. A 6 kg VTOL reaching 13.3 m/s needs 534 J,
                # which over an 8 s transition is 67 W against a transition
                # power around 300 W — a fifth of the bill.
                #
                # Decelerating back to hover releases it, scaled by regen_eff
                # (zero by default, the honest figure for a fixed-pitch prop),
                # which kinetic_power_term_W handles by sign.
                v_prev = v_start + (v_end - v_start) * (i / steps)
                v_next = v_start + (v_end - v_start) * ((i + 1) / steps)
                # kinetic_power_term_W returns MECHANICAL power, so it pays
                # the ESC like any other shaft power on its way from the pack.
                kinetic = core.kinetic_power_term_W(
                    cfg.all_up_weight_g, v_prev, v_next, dt, regen_eff=regen_eff)
                p = max(sub["total_power_W"]
                        + kinetic / max(cfg.esc_efficiency, 1e-9), 0.0)
                energy_Ws += p * dt
                distance += v * dt
                state["t"] += dt
                state["dist"] += v * dt
                remaining_Wh -= p * dt / 3600.0
                _record(phase.name, kind, v,
                        alt_start + (phase.altitude_m - alt_start) * frac, p, sub)
            # Undo the running totals: the common bookkeeping below applies
            # the phase as a whole, and must not count it twice.
            state["t"] -= duration
            state["dist"] -= distance
            remaining_Wh += energy_Ws / 3600.0
            state["v"] = v_end
            point = {"total_power_W": energy_Ws / max(duration, 1e-9)}
            bucket = "transition_Wh"

        else:                                   # cruise
            v = phase.airspeed_mps or cfg.cruise_speed_mps
            transient_m = 0.0

            # Transient lead-in: the aircraft does not step from the previous
            # leg's speed to this one. Accelerating costs power on top of
            # steady drag, and that energy is real — a survey flown as short
            # legs with a speed change at each end pays it many times over.
            # Left at zero (the default) the leg behaves exactly as before.
            if max_accel_mps2 > 0 and abs(v - state["v"]) > 1e-6:
                # Decelerating is limited the same way unless told otherwise;
                # a VTOL can pitch up harder than it can accelerate, but
                # assuming so without data would flatter the model.
                max_decel = max_decel_mps2 or max_accel_mps2
                v_now = state["v"]
                # Ground covered while getting up to speed counts TOWARD the
                # leg, not on top of it. A 400 m leg that spends 120 m
                # accelerating has 280 m left to fly, and forgetting that
                # inflates both the distance and the energy.
                transient_m = 0.0
                guard = 0
                while abs(v - v_now) > 1e-6 and guard < 2000:
                    guard += 1
                    # ramp_speed(current, target, dt, max_accel, max_decel)
                    # and it returns (next_speed, acceleration).
                    v_next, _accel = core.ramp_speed(
                        v_now, v, transient_dt_s, max_accel_mps2, max_decel)
                    sub = power_at_airspeed(cfg, max(v_next, 0.0))
                    kinetic = core.kinetic_power_term_W(
                        cfg.all_up_weight_g, v_now, v_next, transient_dt_s,
                        regen_eff=regen_eff)
                    p_now = max(sub["total_power_W"] + kinetic, 0.0)
                    energy_Wh_transient = p_now * transient_dt_s / 3600.0
                    remaining_Wh -= energy_Wh_transient
                    totals["energy_Wh"] += energy_Wh_transient
                    totals["cruise_Wh"] += energy_Wh_transient
                    totals["time_s"] += transient_dt_s
                    step_m = 0.5 * (v_now + v_next) * transient_dt_s
                    transient_m += step_m
                    totals["distance_m"] += step_m
                    state["t"] += transient_dt_s
                    state["dist"] += step_m
                    _record(phase.name, "transient", v_next,
                            alt_start, p_now, sub)
                    v_now = v_next
                state["v"] = v_now

            point = power_at_airspeed(cfg, v)
            # Power follows AIRSPEED; progress follows GROUNDSPEED. A leg
            # measured over the ground therefore takes longer into a headwind
            # and costs more energy for exactly the same track — which is the
            # whole reason wind matters to a mission.
            head, cross = core.wind_components_mps(
                wind_mps, wind_direction_deg, phase.course_deg)
            ground = max(core.groundspeed_along_track_mps(v, head, cross), 0.1)
            point["headwind_mps"] = head
            point["crosswind_mps"] = cross
            point["groundspeed_mps"] = ground
            if phase.distance_m is not None:
                # Whatever the lead-in already covered comes off the leg. If
                # it covered MORE than the leg is long, the aircraft could
                # not reach its commanded speed inside that leg and has
                # overshot — a real limit worth seeing rather than clamping
                # away, so it is flagged in the phase status.
                if transient_m > float(phase.distance_m) + 1e-9:
                    overshoot_m = transient_m - float(phase.distance_m)
                distance = max(float(phase.distance_m) - transient_m, 0.0)
                duration = distance / ground
            else:
                duration = float(phase.duration_s or 0.0)
                distance = ground * duration
            bucket = "cruise_Wh"

        if kind != "transition":
            _record(phase.name, kind, float(point.get("airspeed_mps", 0.0)),
                    alt_start, point["total_power_W"], point)

        energy_Wh = point["total_power_W"] * duration / 3600.0
        remaining_Wh -= energy_Wh
        state["t"] += duration
        state["dist"] += distance
        state["alt"] = float(phase.altitude_m)
        state["v"] = float(point.get("airspeed_mps", 0.0))
        _record(phase.name, kind, state["v"], state["alt"],
                point["total_power_W"], point)
        totals["time_s"] += duration
        totals["distance_m"] += distance
        totals["energy_Wh"] += energy_Wh
        totals[bucket] += energy_Wh

        status = "OK"
        if remaining_Wh < 0:
            status = "BATTERY DEPLETED"
        elif remaining_Wh < reserve_Wh:
            status = "RESERVE VIOLATION"

        if overshoot_m > 0:

            status = f"{status} — could not reach {v:.0f} m/s within this leg; overshot by {overshoot_m:.0f} m"

        results.append((phase.name, duration / 60.0, distance / 1000.0,
                        point["total_power_W"], energy_Wh, status))

        if remaining_Wh < 0:
            break

    totals["remaining_Wh"] = remaining_Wh
    totals["reserve_Wh"] = reserve_Wh
    totals["series"] = series
    worst["reserve_margin_Wh"] = worst["remaining_Wh"] - reserve_Wh
    totals["worst"] = worst
    # The last instant flown, as full metrics, so the Metrics tab can show a
    # real operating point rather than the worst-case composite.
    try:
        totals["last"] = compute_metrics(cfg, state["v"])
    except Exception:
        totals["last"] = {}
    return results, totals


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VTOL UAV power and endurance simulator (lift+cruise).")
    p.add_argument("--gui", action="store_true", help="Open the graphical interface.")
    p.add_argument("--config_type", type=str, default="lift+cruise",
                   choices=CONFIG_TYPES,
                   help="Airframe configuration. Only lift+cruise is implemented.")

    p.add_argument("--weight", type=float, default=6000.0, help="Airframe weight (g).")
    p.add_argument("--payload_mass_g", type=float, default=0.0)

    p.add_argument("--wing_span", type=float, default=2.4)
    p.add_argument("--wing_area", type=float, default=0.60)
    p.add_argument("--CD0", type=float, default=0.035)
    p.add_argument("--oswald", type=float, default=0.80)
    p.add_argument("--CL_max", type=float, default=1.20)
    p.add_argument("--CL_cruise_max", type=float, default=0.90,
                   help="CL cap used during transition; below CL_max for margin.")

    p.add_argument("--num_lift_rotors", type=int, default=4)
    p.add_argument("--lift_prop_diameter", type=float, default=18.0)
    p.add_argument("--lift_prop_pitch", type=float, default=6.0)
    p.add_argument("--lift_motor_kv", type=float, default=300.0)
    p.add_argument("--lift_motor_resistance", type=float, default=0.08)
    p.add_argument("--lift_motor_weight", type=float, default=200.0)
    p.add_argument("--lift_figure_of_merit", type=float, default=0.65)

    p.add_argument("--num_cruise_motors", type=int, default=1)
    p.add_argument("--cruise_prop_diameter", type=float, default=14.0)
    p.add_argument("--cruise_prop_pitch", type=float, default=8.0)
    p.add_argument("--cruise_motor_kv", type=float, default=500.0)
    p.add_argument("--cruise_motor_resistance", type=float, default=0.06)
    p.add_argument("--cruise_motor_weight", type=float, default=180.0)
    p.add_argument("--cruise_prop_efficiency", type=float, default=0.75)
    p.add_argument("--stopped_rotor_drag_area", type=float, default=None)

    p.add_argument("--battery_chemistry", type=str, default="LiPo")
    p.add_argument("--battery_cell_capacity", type=float, default=5000.0)
    p.add_argument("--battery_series_cells", type=int, default=6)
    p.add_argument("--battery_parallel_cells", type=int, default=2)
    p.add_argument("--battery_cell_weight_g", type=float, default=120.0)
    p.add_argument("--battery_voltage_min", type=float, default=3.3)
    p.add_argument("--battery_voltage_nominal", type=float, default=3.7)
    p.add_argument("--battery_voltage_max", type=float, default=4.2)
    p.add_argument("--battery_resistance_cell", type=float, default=4.0)
    p.add_argument("--battery_usable_percent", type=float, default=80.0)
    p.add_argument("--battery_soc_model", type=str, default="auto")

    p.add_argument("--avionics_power", type=float, default=15.0)
    p.add_argument("--esc_efficiency", type=float, default=0.96)

    p.add_argument("--cruise_speed", type=float, default=22.0)
    p.add_argument("--altitude", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--mission", type=str, default=None)
    p.add_argument("--wind", type=float, default=0.0,
                   help="Steady wind speed (m/s).")
    p.add_argument("--wind_direction", type=float, default=0.0,
                   help="Direction the wind comes FROM, compass degrees.")
    p.add_argument("--lift_prop_table", type=str, default=None,
                   help="Measured lift-rotor thrust/power CSV.")
    p.add_argument("--cruise_prop_table", type=str, default=None,
                   help="Measured cruise-prop thrust/power CSV.")
    # Wiring, connectors and component ratings — the optional detail that was
    # GUI-only until now. A config saved from the GUI carries these, so the
    # CLI has to be able to express them too or the two disagree.
    p.add_argument("--wire_length", type=float, default=0.0,
                   help="One-way battery lead length (m); both conductors counted.")
    p.add_argument("--wire_awg", type=int, default=None,
                   help="Wire gauge (AWG). Higher is thinner.")
    p.add_argument("--wire_ohm_per_m", type=float, default=None,
                   help="Measured wire resistance (ohm/m); overrides the gauge.")
    p.add_argument("--battery_c_cont", type=float, default=None,
                   help="Pack continuous discharge C-rating.")
    p.add_argument("--battery_c_max", type=float, default=None,
                   help="Pack burst discharge C-rating.")
    p.add_argument("--soc_curve", type=str, default=None,
                   help="Measured pack discharge curve CSV (SoC, OCV per cell).")
    p.add_argument("--max_accel", type=float, default=0.0,
                   help="Acceleration limit (m/s^2). 0 ignores transients.")
    p.add_argument("--max_decel", type=float, default=0.0,
                   help="Deceleration limit (m/s^2). Defaults to --max_accel.")
    p.add_argument("--regen_eff", type=float, default=0.0,
                   help="Fraction of braking energy recovered (0-1).")
    p.add_argument("--lift_motor_max_power", type=float, default=None,
                   help="Rated power per lift motor (W).")
    p.add_argument("--cruise_motor_max_power", type=float, default=None,
                   help="Rated power per cruise motor (W).")
    p.add_argument("--hover_download", type=float, default=None,
                   help="Hover download as a fraction of weight; overrides the "
                        "per-configuration default. Applies to tiltrotor, "
                        "tiltwing and tailsitter only.")
    p.add_argument("--lift_prop_weight", type=float, default=0.0,
                   help="Mass of one lift propeller (g), for the weight budget.")
    p.add_argument("--cruise_prop_weight", type=float, default=0.0,
                   help="Mass of one cruise propeller (g).")
    p.add_argument("--avionics_mass", type=float, default=0.0,
                   help="Avionics mass (g), for the weight budget.")
    for _name in ("batt", "esc", "motor"):
        p.add_argument(f"--connector_{_name}_cont", type=float, default=None,
                       help=f"{_name} connector continuous rating (A).")
        p.add_argument(f"--connector_{_name}_max", type=float, default=None,
                       help=f"{_name} connector burst rating (A).")
    return p


def config_from_args(args) -> VTOLConfig:
    battery = VTOLBattery(
        chemistry=args.battery_chemistry,
        cell_capacity_mAh=args.battery_cell_capacity,
        series_cells=args.battery_series_cells,
        parallel_cells=args.battery_parallel_cells,
        cell_weight_g=args.battery_cell_weight_g,
        voltage_min=args.battery_voltage_min,
        voltage_nominal=args.battery_voltage_nominal,
        voltage_max=args.battery_voltage_max,
        resistance_cell_mOhm=args.battery_resistance_cell,
        usable_percent=args.battery_usable_percent,
        discharge_c_cont=args.battery_c_cont,
        discharge_c_max=args.battery_c_max,
        soc_curve_csv=args.soc_curve,
        soc_model=args.battery_soc_model,
    )
    return VTOLConfig(
        config_type=args.config_type,
        aircraft_weight_g=args.weight, payload_mass_g=args.payload_mass_g,
        wing_span_m=args.wing_span, wing_area_m2=args.wing_area,
        CD0=args.CD0, oswald=args.oswald, CL_max=args.CL_max,
        CL_cruise_max=args.CL_cruise_max,
        num_lift_rotors=args.num_lift_rotors,
        lift_prop_diameter_in=args.lift_prop_diameter,
        lift_prop_pitch_in=args.lift_prop_pitch,
        lift_motor_kv=args.lift_motor_kv,
        lift_motor_resistance=args.lift_motor_resistance,
        lift_motor_weight_g=args.lift_motor_weight,
        lift_figure_of_merit=args.lift_figure_of_merit,
        num_cruise_motors=args.num_cruise_motors,
        cruise_prop_diameter_in=args.cruise_prop_diameter,
        cruise_prop_pitch_in=args.cruise_prop_pitch,
        cruise_motor_kv=args.cruise_motor_kv,
        cruise_motor_resistance=args.cruise_motor_resistance,
        cruise_motor_weight_g=args.cruise_motor_weight,
        cruise_prop_efficiency=args.cruise_prop_efficiency,
        stopped_rotor_drag_area_m2=args.stopped_rotor_drag_area,
        battery=battery,
        avionics_power_W=args.avionics_power,
        esc_efficiency=args.esc_efficiency,
        air_density=core.air_density(args.altitude, args.temperature),
        cruise_speed_mps=args.cruise_speed,
        reference_altitude_m=args.altitude,
        lift_prop_table_csv=args.lift_prop_table,
        cruise_prop_table_csv=args.cruise_prop_table,
        wire_resistance_ohm=core.wire_resistance_ohm(
            args.wire_length, awg=args.wire_awg, ohm_per_m=args.wire_ohm_per_m),
        connectors={
            label: (getattr(args, f"connector_{name}_cont") or 0.0,
                    getattr(args, f"connector_{name}_max") or 0.0)
            for label, name in (("Battery", "batt"), ("ESC", "esc"), ("Motor", "motor"))
            if getattr(args, f"connector_{name}_cont")
            or getattr(args, f"connector_{name}_max")
        },
        hover_download_fraction=args.hover_download,
        lift_motor_max_power_W=args.lift_motor_max_power,
        cruise_motor_max_power_W=args.cruise_motor_max_power,
        lift_prop_weight_g=args.lift_prop_weight,
        cruise_prop_weight_g=args.cruise_prop_weight,
        avionics_mass_g=args.avionics_mass,
    )


def _print_single_point(cfg: VTOLConfig) -> None:
    m = compute_metrics(cfg)
    print(f"\n=== VTOL Single-Point ({cfg.config_type}) @ "
          f"{m['airspeed_mps']:.1f} m/s ===")
    print(f"  Air density          : {cfg.air_density:.4f} kg/m³")
    print(f"  All-up weight        : {m['all_up_weight_g']:.0f} g "
          f"({m['weight_N']:.1f} N)")
    print(f"  Wing loading         : {m['wing_loading_N_m2']:.1f} N/m²")
    print(f"  Disc loading (hover) : {m['disc_loading_N_m2']:.1f} N/m²")
    print(f"  Stall speed          : {m['stall_speed_mps']:.1f} m/s")
    print(f"  Transition speed     : {m['transition_speed_mps']:.1f} m/s")
    print(f"  Regime at this speed : {m['regime']}")
    print()
    print(f"  Hover power          : {m['hover_power_W']:.0f} W")
    print(f"  Hover endurance      : {m['hover_endurance_min']:.1f} min")
    print(f"  Cruise power         : {m['total_power_W']:.0f} W")
    print(f"  Cruise endurance     : {m['cruise_endurance_min']:.1f} min")
    print(f"  Cruise range         : {m['cruise_range_km']:.2f} km")
    print(f"  Hover / cruise power : {m['hover_to_cruise_power_ratio']:.2f}x")
    print()
    print(f"  Pack current         : {m['pack_current_A']:.2f} A")
    print(f"  Loaded voltage       : {m['v_load_V']:.2f} V")
    print(f"  Usable energy        : {m['usable_Wh']:.1f} Wh")
    print(f"  SoC model            : {m['soc_model']}")


def _print_mission(cfg: VTOLConfig, path: str,
                   wind_mps: float = 0.0,
                   wind_direction_deg: float = 0.0,
                   max_accel: float = 0.0, max_decel: float = 0.0,
                   regen_eff: float = 0.0) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"Mission file not found: {path}")
    try:
        mission = VTOLMission.from_json(path)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Mission file {path} is not valid JSON: {exc}")

    results, totals = simulate_mission(
        cfg, mission, wind_mps=wind_mps, wind_direction_deg=wind_direction_deg,
        max_accel_mps2=max_accel, max_decel_mps2=max_decel, regen_eff=regen_eff)
    print(f"\n=== VTOL Mission: {os.path.basename(path)} ===")
    print(f"{'Phase':<24}{'Time (min)':>11}{'Dist (km)':>11}"
          f"{'Power (W)':>11}{'Energy (Wh)':>13}  Status")
    for name, minutes, km, power, energy, status in results:
        print(f"{name:<24}{minutes:>11.2f}{km:>11.3f}{power:>11.0f}"
              f"{energy:>13.2f}  {status}")
    print("-" * 82)
    print(f"{'TOTAL':<24}{totals['time_s']/60.0:>11.2f}"
          f"{totals['distance_m']/1000.0:>11.3f}{'':>11}"
          f"{totals['energy_Wh']:>13.2f}")
    print()
    print(f"  Energy split : hover {totals['hover_Wh']:.1f} Wh | "
          f"transition {totals['transition_Wh']:.1f} Wh | "
          f"cruise {totals['cruise_Wh']:.1f} Wh")
    print(f"  Remaining    : {totals['remaining_Wh']:.1f} Wh "
          f"(reserve target {totals['reserve_Wh']:.1f} Wh)")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.gui:
        launch_gui(args)
        return

    cfg = config_from_args(args)
    try:
        if args.mission:
            _print_mission(cfg, args.mission, args.wind, args.wind_direction,
                           args.max_accel, args.max_decel, args.regen_eff)
        else:
            _print_single_point(cfg)
    except NotImplementedError as exc:
        raise SystemExit(str(exc))


# ============================================================
# GUI
# ============================================================

def launch_gui(args=None) -> None:
    """
    Minimal but complete GUI: inputs on the left, results on the right.

    tkinter is imported here, not at module scope, so headless CLI use works
    on a machine with no tk — the same arrangement the other two simulators
    use.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title(f"VTOL Power Simulator  v{SIM_VERSION}")
    root.geometry("1500x900")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    left = ttk.Frame(root, padding=6)
    left.grid(row=0, column=0, sticky="ns")
    right = ttk.Frame(root, padding=6)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(0, weight=1)

    # ---- configuration type -----------------------------------------
    type_bar = ttk.Frame(left)
    type_bar.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    ttk.Label(type_bar, text="Configuration:").pack(side="left")
    v_config_type = tk.StringVar(value="lift+cruise")
    type_box = ttk.Combobox(type_bar, textvariable=v_config_type, width=18,
                            state="readonly", values=CONFIG_TYPES)
    type_box.pack(side="left", padx=(4, 8))
    type_note = ttk.Label(type_bar, text="", foreground="#B71C1C",
                          font=("TkDefaultFont", 8))
    type_note.pack(side="left")

    def _on_type_change(*_a):
        chosen = v_config_type.get()
        if chosen in IMPLEMENTED_CONFIG_TYPES:
            type_note.configure(text="")
        else:
            type_note.configure(
                text=f"{chosen} is not implemented yet — inputs shown for reference")
        if chosen in ("tiltrotor", "tiltwing", "tailsitter"):
            # One set of rotors lifts AND pushes, so the Cruise tab's motor
            # and prop geometry do not apply — only its efficiency does.
            type_note.configure(
                text="vectored thrust: lift rotors also cruise; the Cruise "
                     "tab supplies only the propeller efficiency",
                foreground="#0B6BCB")
        else:
            type_note.configure(foreground="#B71C1C")
    v_config_type.trace_add("write", _on_type_change)

    # ---- Simple / Advanced -------------------------------------------
    # A view setting only: hidden fields keep their values, so switching
    # modes never changes a computed result.
    # Packed INSIDE type_bar: its parent `left` uses grid, and mixing the two
    # geometry managers in one container raises.
    mode_bar = ttk.Frame(type_bar)
    mode_bar.pack(side="left", padx=(16, 0))
    ttk.Label(mode_bar, text="Input detail:").pack(side="left")
    v_ui_mode = tk.StringVar(value="Simple")
    for label in ("Simple", "Advanced"):
        ttk.Radiobutton(mode_bar, text=label, value=label,
                        variable=v_ui_mode).pack(side="left", padx=2)
    mode_note = ttk.Label(mode_bar, foreground="#555555", font=("TkDefaultFont", 8),
                          text="Simple hides advanced tuning inputs.")
    mode_note.pack(side="left", padx=8)

    def _apply_field_mode(*_a):
        simple = (v_ui_mode.get() == "Simple")
        for row in _field_rows:
            show = (not simple) or (row["key"] in VTOL_SIMPLE_FIELDS)
            for widget in row["widgets"]:
                try:
                    widget.grid() if show else widget.grid_remove()
                except Exception:
                    pass
    v_ui_mode.trace_add("write", _apply_field_mode)

    nb = ttk.Notebook(left)
    nb.grid(row=1, column=0, columnspan=3, sticky="nsew")
    left.rowconfigure(1, weight=1)

    def make_tab(title):
        frame = ttk.Frame(nb, padding=6)
        nb.add(frame, text=title)
        return frame

    tab_airframe = make_tab("Airframe")
    tab_lift = make_tab("Lift Rotors")
    tab_cruise = make_tab("Cruise")
    tab_batt = make_tab("Battery")
    tab_env = make_tab("Mission/Env")

    fields: Dict[str, tk.StringVar] = {}

    # Every row records its widgets so Simple mode can hide it. grid_remove()
    # keeps the grid options, so re-showing is a plain grid().
    _field_rows = []

    def add_row(parent, row, label, key, default, help_text=""):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(default))
        fields[key] = var
        entry = ttk.Entry(parent, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 4))
        parent.columnconfigure(1, weight=1)
        widgets = [label_widget, entry]
        if help_text:
            marker = ttk.Label(parent, text="?", foreground="#0B6BCB",
                               cursor="question_arrow")
            marker.grid(row=row, column=2, sticky="w")
            core.Tooltip(marker, help_text)
            widgets.append(marker)
        _field_rows.append({"key": key, "widgets": widgets})
        return row + 1

    r = 0
    r = add_row(tab_airframe, r, "Airframe weight (g)", "weight", 6000,
                "Everything except the payload, including battery and motors.")
    r = add_row(tab_airframe, r, "Payload mass (g)", "payload", 0,
                "Added on top of the airframe weight.")
    r = add_row(tab_airframe, r, "Wing span (m)", "span", 2.4, "Tip to tip.")
    r = add_row(tab_airframe, r, "Wing area (m²)", "area", 0.60,
                "Planform area. With span this sets the aspect ratio.")
    r = add_row(tab_airframe, r, "CD0", "cd0", 0.035,
                "Zero-lift drag. A VTOL is draggier than a clean fixed-wing "
                "because of booms and stopped rotors.\nTypical 0.03-0.05.")
    r = add_row(tab_airframe, r, "Oswald efficiency", "oswald", 0.80,
                "Span efficiency. 0.75-0.85 for a VTOL with booms.")
    r = add_row(tab_airframe, r, "CL_max", "clmax", 1.20,
                "Maximum lift coefficient, which sets the stall speed.")
    r = add_row(tab_airframe, r, "CL cap in transition", "clcruise", 0.90,
                "CL the controller will actually hold while transitioning.\n"
                "Below CL_max so a gust does not stall the wing.")

    r = 0
    r = add_row(tab_lift, r, "Number of lift rotors", "n_lift", 4,
                "Rotors used for hover only. Stopped in cruise.")
    r = add_row(tab_lift, r, "Lift prop diameter (in)", "lift_d", 18,
                "Larger discs hover far more efficiently.")
    r = add_row(tab_lift, r, "Lift prop pitch (in)", "lift_p", 6)
    r = add_row(tab_lift, r, "Lift motor Kv", "lift_kv", 300)
    r = add_row(tab_lift, r, "Lift motor Rm (ohm)", "lift_rm", 0.08)
    r = add_row(tab_lift, r, "Lift motor weight (g)", "lift_wt", 200)
    r = add_row(tab_lift, r, "Figure of merit", "fom", 0.65,
                "Hover efficiency of the rotor against ideal momentum theory.\n"
                "0.6-0.7 typical; 0.75+ is a very good rotor.")
    r = add_row(tab_lift, r, "Stopped rotor drag area (m²)", "stopped_area", "",
                "Flat-plate area of the stopped rotors in cruise.\n"
                "Blank estimates it from blade planform.")

    r = 0
    r = add_row(tab_cruise, r, "Number of cruise motors", "n_cruise", 1)
    r = add_row(tab_cruise, r, "Cruise prop diameter (in)", "cruise_d", 14)
    r = add_row(tab_cruise, r, "Cruise prop pitch (in)", "cruise_p", 8)
    r = add_row(tab_cruise, r, "Cruise motor Kv", "cruise_kv", 500)
    r = add_row(tab_cruise, r, "Cruise motor Rm (ohm)", "cruise_rm", 0.06)
    r = add_row(tab_cruise, r, "Cruise motor weight (g)", "cruise_wt", 180)
    r = add_row(tab_cruise, r, "Cruise prop efficiency", "cruise_eff", 0.75,
                "Combined motor and propeller efficiency in cruise.")

    r = 0
    r = add_row(tab_batt, r, "Chemistry", "chem", "LiPo",
                "Selects the state-of-charge curve. LiPo, Li-ion or LiFePO4.")
    r = add_row(tab_batt, r, "Cell capacity (mAh)", "cell_cap", 5000)
    r = add_row(tab_batt, r, "Series cells", "series", 6, "Sets pack voltage.")
    r = add_row(tab_batt, r, "Parallel cells", "parallel", 2, "Sets pack capacity.")
    r = add_row(tab_batt, r, "Cell weight (g)", "cell_wt", 120)
    r = add_row(tab_batt, r, "Cell V min", "vmin", 3.3)
    r = add_row(tab_batt, r, "Cell V nominal", "vnom", 3.7)
    r = add_row(tab_batt, r, "Cell V max", "vmax", 4.2)
    r = add_row(tab_batt, r, "Cell resistance (mOhm)", "rcell", 4.0)
    r = add_row(tab_batt, r, "Usable percent", "usable", 80,
                "Fraction of pack energy you are willing to use.")

    r = 0
    r = add_row(tab_env, r, "Cruise speed (m/s)", "cruise_v", 22)
    r = add_row(tab_env, r, "Altitude (m)", "alt", 0)
    r = add_row(tab_env, r, "Temperature (°C)", "temp", "",
                "Blank uses the ISA value for the altitude.")
    r = add_row(tab_env, r, "Avionics power (W)", "avionics", 15,
                "Autopilot, radios and payload electronics.")
    r = add_row(tab_env, r, "ESC efficiency", "esc_eff", 0.96)
    ttk.Label(tab_env, text="Mission JSON").grid(row=r, column=0, sticky="w", pady=2)
    v_mission = tk.StringVar(value="")
    fields["mission"] = v_mission
    mission_frame = ttk.Frame(tab_env)
    mission_frame.grid(row=r, column=1, columnspan=2, sticky="ew")
    ttk.Entry(mission_frame, textvariable=v_mission, width=14).pack(side="left")

    def browse_mission():
        path = filedialog.askopenfilename(
            title="Mission JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            v_mission.set(path)
    ttk.Button(mission_frame, text="Browse...", command=browse_mission).pack(side="left")

    # ---- optional detail inputs --------------------------------------
    # Everything below is optional. Blank means "not specified": the model
    # behaves exactly as before, and Status says a check has nothing to test
    # against rather than inventing a limit.
    def _append_row(parent, label, key, default="", help_text=""):
        """add_row at the next free grid row of an already-populated tab."""
        return add_row(parent, parent.grid_size()[1], label, key, default, help_text)

    _append_row(tab_airframe, "Avionics mass (g)", "avionics_mass", "",
                "Autopilot, radios, GPS and payload electronics. Shown as its "
                "own line in the Weight Budget; already inside the airframe "
                "weight, so it does not change the flying weight.")
    _append_row(tab_lift, "Lift prop weight (g, each)", "lift_prop_wt", "",
                "For the Weight Budget.")
    _append_row(tab_lift, "Lift motor max power (W)", "lift_pmax", "",
                "Rated power per motor. Status checks hover — the heaviest "
                "load these motors see — against it.")
    _append_row(tab_lift, "Hover download fraction", "download", "",
                "Extra hover thrust needed because the rotor wash strikes the "
                "airframe below, as a fraction of weight.\n"
                "Blank uses the type default: tiltrotor 0.10, tiltwing 0.02, "
                "tailsitter 0.02. Typical published values, not measurements — "
                "override with test data if you have it.\n\n"
                "Not modelled for lift+cruise: its hover and transition are "
                "computed by separate branches, so a download applied to one "
                "and not the other would put a step at zero airspeed.")
    _append_row(tab_cruise, "Cruise prop weight (g, each)", "cruise_prop_wt", "",
                "For the Weight Budget.")
    _append_row(tab_cruise, "Cruise motor max power (W)", "cruise_pmax", "",
                "Rated power per cruise motor. Ignored for tiltrotor, tiltwing "
                "and tailsitter, whose lift rotors do the cruise work.")

    # ---- measured bench tables ---------------------------------------
    # A thrust/power table replaces the figure-of-merit and propeller
    # efficiency GUESSES with the efficiency the hardware actually achieved.
    # Outside the thrust range the table covers, the estimate is used again —
    # a bench test says nothing about thrusts it never produced.
    def _table_picker(parent, label, key, help_text):
        row = parent.grid_size()[1]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value="")
        fields[key] = var
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky="ew", padx=(8, 4))
        ttk.Entry(holder, textvariable=var, width=14).pack(side="left")

        def browse():
            path = filedialog.askopenfilename(
                title=label, filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if path:
                var.set(path)
        ttk.Button(holder, text="...", width=3, command=browse).pack(side="left")
        marker = ttk.Label(parent, text=" ? ", foreground="#0B6BCB",
                           cursor="question_arrow")
        marker.grid(row=row, column=2, sticky="w")
        core.Tooltip(marker, help_text)

    _table_picker(tab_lift, "Lift rotor table (CSV)", "lift_table",
                  "Measured thrust/power bench data for one lift rotor.\n\n"
                  "Columns: Thrust_g and Power_W, with optional RPM. The "
                  "measured combined efficiency replaces the figure of merit "
                  "wherever the table covers the thrust; outside that range "
                  "the figure of merit is used again.")
    _table_picker(tab_cruise, "Cruise prop table (CSV)", "cruise_table",
                  "Measured thrust/power bench data for one cruise motor.\n\n"
                  "Replaces the propeller efficiency estimate where the table "
                  "covers the thrust. Ignored for the vectored types, whose "
                  "lift rotors do the cruising.")
    _append_row(tab_env, "Max acceleration (m/s²)", "accel", "",
                "Limits how fast the aircraft may change speed between legs.\n\n"
                "Blank ignores transients entirely and each leg starts at its "
                "commanded speed, as before. Given a value, accelerating costs "
                "power on top of steady drag — a survey flown as short legs "
                "with a speed change at each end pays that many times over.\n\n"
                "If a leg is too short to reach its speed, the phase status "
                "says so rather than pretending it fits.")
    _append_row(tab_env, "Max deceleration (m/s²)", "decel", "",
                "Blank uses the acceleration limit. A VTOL can usually slow "
                "harder than it can speed up, but assuming so without data "
                "would flatter the model.")
    _append_row(tab_env, "Regen efficiency (0-1)", "regen", "0",
                "Fraction of braking energy recovered. Fixed-pitch propellers "
                "are poor regenerators, so 0 is the honest default.")
    _append_row(tab_env, "Wind speed (m/s)", "wind", "0",
                "Steady wind. Power follows AIRSPEED, but progress follows "
                "GROUNDSPEED, so a leg measured over the ground takes longer "
                "into a headwind and costs more energy for the same track.\n\n"
                "Hovering in wind is not free either: to hold station the "
                "aircraft must fly at the wind speed through the air.")
    _append_row(tab_env, "Wind direction FROM (deg)", "wind_dir", "0",
                "Meteorological convention — the direction the wind blows "
                "FROM. A north wind (0) is a headwind when flying north.\n\n"
                "Each mission leg's own course_deg decides whether that is a "
                "head, tail or crosswind for that leg.")
    _table_picker(tab_batt, "SoC curve (CSV)", "soc_curve",
                  "Measured discharge curve for this pack.\n\n"
                  "Columns: SoC (0-1 or 0-100) with OCV per cell, and "
                  "optionally a resistance scale. A measured curve outranks "
                  "the chemistry preset, so the sag near the end of the pack "
                  "comes from your cells rather than a generic LiPo shape.")
    _append_row(tab_batt, "Continuous C-rate", "c_cont", "",
                "Pack continuous discharge rating. Status checks the pack "
                "current against it.")
    _append_row(tab_batt, "Max / burst C-rate", "c_max", "",
                "Pack burst rating. Above continuous but below this is amber; "
                "above this is red.")

    # ---- Wiring tab ----------------------------------------------------
    tab_wiring = make_tab("Wiring")
    ttk.Label(tab_wiring, text="Main battery-to-ESC wire run. Blank ignores "
              "wiring losses entirely.", wraplength=300, justify="left",
              foreground="#555555").grid(row=0, column=0, columnspan=3,
                                         sticky="w", pady=(0, 4))
    _append_row(tab_wiring, "Wire length one-way (m)", "wire_len", "",
                "One-way run. Current comes back too, so both conductors are "
                "counted — 2 x this length.")
    _append_row(tab_wiring, "Wire gauge (AWG)", "wire_awg", "",
                "Copper at 20 C. Higher AWG is THINNER wire and loses more.")
    _append_row(tab_wiring, "Wire resistance (ohm/m)", "wire_ohm_m", "",
                "Measured from your own spool. Overrides the gauge if given.")
    ttk.Label(tab_wiring, text="Connector ratings — pick a type to fill "
              "typical figures, then edit to match your parts.",
              wraplength=300, justify="left", foreground="#555555"
              ).grid(row=tab_wiring.grid_size()[1], column=0, columnspan=3,
                     sticky="w", pady=(10, 4))
    _conn_names = ("",) + tuple(sorted(core.CONNECTOR_RATINGS))
    for _label, _prefix in (("Battery connector", "conn_batt"),
                            ("ESC connector", "conn_esc"),
                            ("Motor connector", "conn_motor")):
        _row = tab_wiring.grid_size()[1]
        ttk.Label(tab_wiring, text=_label).grid(row=_row, column=0, sticky="w", pady=2)
        _type_var = tk.StringVar(value="")
        fields[_prefix] = _type_var
        ttk.Combobox(tab_wiring, textvariable=_type_var, state="readonly",
                     width=16, values=_conn_names).grid(
            row=_row, column=1, sticky="w", padx=(8, 4))
        _append_row(tab_wiring, "   continuous (A)", f"{_prefix}_cont", "")
        _append_row(tab_wiring, "   max / burst (A)", f"{_prefix}_max", "")

        def _fill(*_a, _t=_type_var, _p=_prefix):
            # Convenience, not a lock: ratings stay editable, because burst
            # figures vary widely between manufacturers.
            defaults = core.connector_defaults(_t.get())
            if defaults:
                fields[f"{_p}_cont"].set(f"{defaults[0]:g}")
                fields[f"{_p}_max"].set(f"{defaults[1]:g}")
        _type_var.trace_add("write", _fill)

    # ---- output ------------------------------------------------------
    out_nb = ttk.Notebook(right)
    out_nb.grid(row=0, column=0, sticky="nsew")
    tab_metrics = ttk.Frame(out_nb); out_nb.add(tab_metrics, text="Metrics")
    tab_plots = ttk.Frame(out_nb); out_nb.add(tab_plots, text="Fixed Speed Plots")
    for frame in (tab_metrics, tab_plots):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    metrics_tv = ttk.Treeview(tab_metrics, columns=("metric", "value"),
                              show="headings")
    metrics_tv.heading("metric", text="Metric")
    metrics_tv.heading("value", text="Value")
    metrics_tv.column("metric", width=280, anchor="w")
    metrics_tv.column("value", width=260, anchor="w")
    metrics_tv.grid(row=0, column=0, sticky="nsew")

    plot_holder = ttk.Frame(tab_plots)
    plot_holder.grid(row=0, column=0, sticky="nsew")
    plot_holder.columnconfigure(0, weight=1)
    plot_holder.rowconfigure(0, weight=1)
    _canvas = {"widget": None}

    # ==================================================================
    # DISPLAY TABS PORTED FROM THE MULTICOPTER AND FIXED-WING
    # ==================================================================
    # Same behaviour as the other two simulators, so a user moving between
    # them finds the same tabs answering the same questions:
    #   * a fixed speed run fills Status, Metrics, the plots and both budgets;
    #   * a mission run shows the WORST value of each check on Status and the
    #     LAST instant flown on Metrics, and clears anything that only makes
    #     sense for a single operating point;
    #   * any new run clears Sensitivity, whose results belong to the old one.

    def _tab(title):
        frame = ttk.Frame(out_nb, padding=4)
        out_nb.add(frame, text=title)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        return frame

    def _tree(parent, columns, height=16, row=1):
        tv = ttk.Treeview(parent, columns=[c for c, _h, _w in columns],
                          show="headings", height=height)
        for key, heading, width in columns:
            tv.heading(key, text=heading)
            tv.column(key, width=width, stretch=True,
                      anchor="w" if key in ("metric", "item", "name", "label") else "center")
        tv.grid(row=row, column=0, sticky="nsew")
        sb = ttk.Scrollbar(parent, orient="vertical", command=tv.yview)
        sb.grid(row=row, column=1, sticky="ns")
        tv.configure(yscrollcommand=sb.set)
        for tag, colour in (("ok", "#d9f2d9"), ("edge", "#e8f4d9"),
                            ("warn", "#fff2cc"), ("bad", "#f8d7da"),
                            ("na", "#efefef"), ("delivered", "#e8f4d9"),
                            ("lost", "#fdecea")):
            tv.tag_configure(tag, background=colour)
        tv.tag_configure("total", font=("TkDefaultFont", 9, "bold"), background="#e3eefb")
        tv.tag_configure("subtotal", font=("TkDefaultFont", 9, "bold"))
        return tv

    def _scope_label(parent):
        lbl = ttk.Label(parent, text="", foreground="#0B6BCB", wraplength=900,
                        justify="left", font=("TkDefaultFont", 9, "bold"))
        lbl.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        return lbl

    def _placeholder(parent, text):
        lbl = ttk.Label(parent, text=text, foreground="#888888", justify="center",
                        wraplength=460)
        lbl.grid(row=1, column=0, padx=20, pady=40)
        return lbl

    def _clear_tree(tv):
        for iid in tv.get_children():
            tv.delete(iid)

    def _destroy_canvas(holder):
        widget = holder.get("widget")
        if widget is not None:
            try:
                widget.get_tk_widget().destroy()
            except Exception:
                pass
            holder["widget"] = None

    def _show_figure(parent, holder, fig, row=1, col=0):
        _destroy_canvas(holder)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=col, sticky="nsew")
        holder["widget"] = canvas

    # ---- Metrics scope banner (the Metrics tab already exists) ----------
    tab_metrics.rowconfigure(1, weight=1)
    metrics_tv.grid_configure(row=1)
    metrics_scope = _scope_label(tab_metrics)

    # ---- Status --------------------------------------------------------
    tab_status = _tab("Status")
    status_scope = _scope_label(tab_status)
    status_tv = _tree(tab_status, [("metric", "Check", 230), ("value", "Value", 150),
                                   ("limit", "Limit", 190), ("note", "Note", 420)])
    status_detail = ttk.Label(tab_status, text="", wraplength=900, justify="left",
                              foreground="#333333")
    status_detail.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    def _on_status_select(_e=None):
        sel = status_tv.selection()
        if sel:
            vals = status_tv.item(sel[0], "values")
            status_detail.configure(text=f"{vals[0]} — {vals[3]}" if vals[3] else vals[0])
    status_tv.bind("<<TreeviewSelect>>", _on_status_select)

    def _row(metric, value, limit, tag, note=""):
        status_tv.insert("", "end", values=(metric, value, limit, note), tags=(tag,))

    def _classify(value, limit, edge_frac=0.05):
        try:
            v, lim = float(value), float(limit)
        except (TypeError, ValueError):
            return "na"
        if not (math.isfinite(v) and math.isfinite(lim)) or lim <= 0:
            return "na"
        if v > lim:
            return "bad"
        return "edge" if v > lim * (1.0 - edge_frac) else "ok"

    def _dual(metric, value, cont, mx, unit, decimals=1):
        """A quantity with both a continuous and an absolute limit."""
        v = float(value)
        if not cont and not mx:
            _row(metric, f"{v:.{decimals}f} {unit}", "Not Specified", "na",
                 "No rating entered, so there is nothing to check against.")
            return
        limit = " / ".join(x for x in (
            f"cont {cont:.{decimals}f} {unit}" if cont else "",
            f"max {mx:.{decimals}f} {unit}" if mx else "") if x)
        if mx and v > mx:
            tag, note = "bad", f"Above the absolute maximum of {mx:.{decimals}f} {unit}."
        elif cont and v > cont:
            tag, note = "warn", (f"Above the continuous rating of {cont:.{decimals}f} "
                                 f"{unit}; acceptable only briefly.")
        else:
            ref = cont or mx
            tag = _classify(v, ref)
            note = ("Within the continuous rating." if tag == "ok" else
                    f"Within 5% of the {ref:.{decimals}f} {unit} rating — no margin left.")
        _row(metric, f"{v:.{decimals}f} {unit}", limit, tag, note)

    def update_status(cfg, m, worst=None):
        """
        Fixed speed: every row at the cruise speed, plus hover — which for a
        VTOL is the heaviest steady load and so the case that sizes the
        battery, the motors and the connectors.
        Mission: the worst value each check reached anywhere in the flight.
        """
        _clear_tree(status_tv)
        status_detail.configure(text="")
        batt = cfg.battery
        cont_A, max_A = batt.discharge_cont_A, batt.discharge_max_A

        if worst is not None:
            _dual("Peak pack current", worst["pack_current_A"], cont_A, max_A, "A")
            _dual("Peak discharge C-rate", worst["c_rate"],
                  batt.discharge_c_cont, batt.discharge_c_max, "C")
            _row("Peak power", f"{worst['total_power_W']:.0f} W", "—", "na",
                 f"Reached during '{worst.get('peak_phase', '')}'.")
            margin = worst.get("reserve_margin_Wh", 0.0)
            _row("Lowest reserve margin", f"{margin:.1f} Wh", ">= 0 Wh",
                 "ok" if margin >= 0 else "bad",
                 "Energy left above the reserve at the lowest point of the "
                 "mission. Negative means the reserve was eaten into.")
            return

        hover_I = float(m.get("hover_pack_current_A", 0.0))
        point_I = float(m.get("pack_current_A", 0.0))
        _dual("Hover pack current", hover_I, cont_A, max_A, "A")
        _dual("Cruise pack current", point_I, cont_A, max_A, "A")
        _dual("Hover C-rate", hover_I / max(batt.capacity_Ah, 1e-9),
              batt.discharge_c_cont, batt.discharge_c_max, "C")

        if cfg.lift_motor_max_power_W:
            per = float(m.get("hover_power_per_lift_motor_W", 0.0))
            _row("Lift motor power in hover", f"{per:.0f} W",
                 f"<= {cfg.lift_motor_max_power_W:.0f} W",
                 _classify(per, cfg.lift_motor_max_power_W),
                 "Hover is the heaviest load a lift motor carries.")
        else:
            _row("Lift motor power in hover",
                 f"{float(m.get('hover_power_per_lift_motor_W', 0.0)):.0f} W",
                 "Not Specified", "na", "Enter a motor max power to check it.")

        if not uses_vectored_thrust(cfg):
            per = float(m.get("cruise_power_per_motor_W", 0.0))
            if cfg.cruise_motor_max_power_W:
                _row("Cruise motor power", f"{per:.0f} W",
                     f"<= {cfg.cruise_motor_max_power_W:.0f} W",
                     _classify(per, cfg.cruise_motor_max_power_W))
            else:
                _row("Cruise motor power", f"{per:.0f} W", "Not Specified", "na")

        v_stall = float(m.get("stall_speed_mps", 0.0))
        v_trans = float(m.get("transition_speed_mps", 0.0))
        ratio = v_trans / max(v_stall, 1e-9)
        _row("Transition / stall speed", f"{v_trans:.1f} / {v_stall:.1f} m/s",
             ">= 1.10x stall", "ok" if ratio >= 1.10 else ("warn" if ratio >= 1.0 else "bad"),
             f"Transitioning at {ratio:.2f}x stall. Below about 1.1x a gust can "
             f"stall the wing while the rotors are spinning down.")

        share = float(m.get("lift_share_wing", 0.0))
        v_cruise = float(m.get("airspeed_mps", 0.0))
        _row("Wing lift share at cruise speed", f"{share * 100:.0f}%", "100%",
             "ok" if share >= 0.999 else "warn",
             "Fully wing-borne." if share >= 0.999 else
             f"At {v_cruise:.1f} m/s the wing carries only part of the weight, "
             f"so the rotors are still lifting — the cruise speed is below the "
             f"{v_trans:.1f} m/s transition speed.")

        if uses_vectored_thrust(cfg):
            tilt = float(m.get("tilt_deg", 0.0))
            _row("Rotor tilt at cruise speed", f"{tilt:.1f} deg", "90 deg in cruise",
                 "ok" if tilt >= 89.0 else "warn",
                 "Thrust pointed straight ahead." if tilt >= 89.0 else
                 "Rotors not yet fully tilted — still partly lifting.")
        import os as _os
        area = math.pi / 4.0 * (cfg.lift_prop_diameter_in * 0.0254) ** 2
        t_per = cfg.weight_N * (1.0 + hover_download_fraction(cfg)) / max(cfg.num_lift_rotors, 1)
        measured = measured_lift_efficiency(cfg, t_per, area)
        if cfg.lift_prop_table is None:
            _row("Lift rotor efficiency", f"{cfg.lift_figure_of_merit:.3f} FoM",
                 "estimate", "na",
                 "No bench table loaded, so this is the figure of merit you "
                 "entered. Load a measured table on the Lift Rotors tab to "
                 "replace the estimate.")
        elif measured is None:
            _row("Lift rotor efficiency", f"{cfg.lift_figure_of_merit:.3f} FoM",
                 "estimate (outside table)", "warn",
                 f"A table is loaded ({_os.path.basename(cfg.lift_prop_table_csv)}) "
                 f"but {t_per:.1f} N per rotor is outside the thrust range it "
                 f"covers, so the estimate is being used instead.")
        else:
            _row("Lift rotor efficiency", f"{measured:.3f} FoM", "measured", "ok",
                 f"From {_os.path.basename(cfg.lift_prop_table_csv)}, against "
                 f"the {cfg.lift_figure_of_merit:.3f} you entered.")

        _row("Hover download fraction", f"{hover_download_fraction(cfg) * 100:.0f}%",
             "—", "na",
             "Extra thrust to lift the rotor wash striking the airframe. "
             + ("Overridden on the Lift Rotors tab." if cfg.hover_download_fraction is not None
                else f"Type default for {cfg.config_type}."))

        conns = dict(getattr(cfg, "connectors", {}) or {})
        if conns:
            # Checked at hover, the largest steady current a VTOL draws.
            n_esc = max(cfg.num_lift_rotors, 1)
            seen = {"Battery": hover_I, "ESC": hover_I / n_esc,
                    "Motor": hover_I / n_esc * 1.15}
            for name, (c_cont, c_max) in conns.items():
                if name in seen:
                    _dual(f"{name} connector (hover)", seen[name],
                          c_cont or None, c_max or None, "A")

        drop = float(m.get("wire_drop_V", 0.0))
        if drop > 0:
            _row("Main wire voltage drop", f"{drop:.2f} V", "—", "na",
                 f"Lost along the battery lead at cruise; "
                 f"{float(m.get('wire_loss_W', 0.0)):.1f} W as heat.")

    # ---- Mission Plots -------------------------------------------------
    tab_mplots = _tab("Mission Plots")
    mplot_bar = ttk.Frame(tab_mplots)
    mplot_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
    ttk.Label(mplot_bar, text="Variables (up to 4):").pack(side="left")
    _MPLOT_KEYS = [("total_power_W", "Total power (W)"), ("pack_current_A", "Pack current (A)"),
                   ("c_rate", "C-rate"), ("energy_remaining_Wh", "Energy remaining (Wh)"),
                   ("altitude_m", "Altitude (m)"), ("airspeed_mps", "Airspeed (m/s)"),
                   ("tilt_deg", "Rotor tilt (deg)"), ("lift_share_wing", "Wing lift share"),
                   ("distance_km", "Distance (km)")]
    mplot_list = tk.Listbox(mplot_bar, selectmode="multiple", height=3, width=26,
                            exportselection=False)
    for _k, _lbl in _MPLOT_KEYS:
        mplot_list.insert("end", _lbl)
    mplot_list.selection_set(0)
    mplot_list.selection_set(4)
    mplot_list.pack(side="left", padx=4)
    v_mx = tk.StringVar(value="time")
    ttk.Radiobutton(mplot_bar, text="vs time", value="time", variable=v_mx).pack(side="left")
    ttk.Radiobutton(mplot_bar, text="vs distance", value="distance",
                    variable=v_mx).pack(side="left")
    mplot_holder = ttk.Frame(tab_mplots)
    mplot_holder.grid(row=1, column=0, columnspan=2, sticky="nsew")
    mplot_holder.columnconfigure(0, weight=1)
    mplot_holder.rowconfigure(1, weight=1)
    mplot_placeholder = _placeholder(mplot_holder, "Run a mission to plot its history.")
    _mplot_canvas = {"widget": None}
    _mission_state = {"series": None, "phases": None, "results": None}

    def draw_mission_plots():
        series = _mission_state["series"]
        if not series or not series.get("t_s"):
            return
        chosen = [_MPLOT_KEYS[i] for i in mplot_list.curselection()][:4]
        if not chosen:
            return
        if v_mx.get() == "distance":
            xs, xlabel = series["distance_km"], "Distance (km)"
        else:
            xs, xlabel = [t / 60.0 for t in series["t_s"]], "Mission time (min)"
        fig, ax0 = core.make_figure(figsize=(10, 4.6))
        colours = ["#1565C0", "#C62828", "#2E7D32", "#EF6C00"]
        axes = [ax0]
        for i in range(1, len(chosen)):
            ax = ax0.twinx()
            if i >= 2:
                ax.spines["right"].set_position(("outward", 55 * (i - 1)))
            axes.append(ax)
        for ax, (key, label), colour in zip(axes, chosen, colours):
            ax.plot(xs, series[key], color=colour, lw=1.6, label=label)
            ax.set_ylabel(label, color=colour)
            ax.tick_params(axis="y", colors=colour)
        ax0.set_xlabel(xlabel)
        ax0.grid(True, alpha=0.4)
        ax0.set_title("Mission history")
        fig.tight_layout()
        mplot_placeholder.grid_remove()
        _show_figure(mplot_holder, _mplot_canvas, fig)

    ttk.Button(mplot_bar, text="Plot", command=draw_mission_plots).pack(side="left", padx=6)

    def clear_mission_plots():
        _mission_state.update(series=None, phases=None, results=None)
        _destroy_canvas(_mplot_canvas)
        mplot_placeholder.configure(
            text="These plots come from a mission run.\n\nYou have just run a "
                 "fixed speed sweep, so there is no mission history to plot.")
        mplot_placeholder.grid()

    # ---- Weight Budget -------------------------------------------------
    tab_wb = _tab("Weight Budget")
    wb_note = ttk.Label(tab_wb, text="", foreground="#555555", wraplength=900,
                        justify="left")
    wb_note.grid(row=0, column=0, columnspan=2, sticky="ew")
    wb_tv = _tree(tab_wb, [("item", "Component", 240), ("unit", "Each (g)", 90),
                           ("qty", "Qty", 60), ("total", "Total (g)", 100),
                           ("pct", "% of AUW", 90)], height=12)

    def update_weight_budget(cfg, m):
        """
        The airframe weight already includes the battery and motors, so
        structure is the residual once the itemised parts are taken out.
        If the parts exceed the airframe weight the residual is negative —
        an impossible aircraft — and that is shown in red, not hidden.
        """
        _clear_tree(wb_tv)
        auw = cfg.all_up_weight_g
        items = [
            ("Battery", cfg.battery.weight_g, 1),
            ("Lift motors", cfg.lift_motor_weight_g, cfg.num_lift_rotors),
            ("Lift propellers", cfg.lift_prop_weight_g, cfg.num_lift_rotors),
        ]
        if not uses_vectored_thrust(cfg):
            items += [("Cruise motors", cfg.cruise_motor_weight_g, cfg.num_cruise_motors),
                      ("Cruise propellers", cfg.cruise_prop_weight_g, cfg.num_cruise_motors)]
        items.append(("Avionics", cfg.avionics_mass_g, 1))
        parts = 0.0
        for name, each, qty in items:
            total = float(each) * int(qty)
            if total <= 0:
                continue
            parts += total
            wb_tv.insert("", "end", values=(name, f"{float(each):.0f}", qty,
                                            f"{total:.0f}", f"{total / auw * 100:.1f}%"))
        structure = cfg.aircraft_weight_g - parts
        wb_tv.insert("", "end", tags=(("bad",) if structure < 0 else ()),
                     values=("Airframe / structure", "", "", f"{structure:.0f}",
                             f"{structure / auw * 100:.1f}%"))
        if cfg.payload_mass_g > 0:
            wb_tv.insert("", "end", values=("Payload", f"{cfg.payload_mass_g:.0f}", 1,
                                            f"{cfg.payload_mass_g:.0f}",
                                            f"{cfg.payload_mass_g / auw * 100:.1f}%"))
        wb_tv.insert("", "end", tags=("total",),
                     values=("ALL-UP WEIGHT", "", "", f"{auw:.0f}", "100%"))
        wb_note.configure(
            foreground="#B71C1C" if structure < 0 else "#555555",
            text=(f"The itemised parts weigh {parts:.0f} g — more than the "
                  f"{cfg.aircraft_weight_g:.0f} g airframe weight. That leaves a "
                  f"negative structure mass, which is impossible: raise the "
                  f"airframe weight or correct a part."
                  if structure < 0 else
                  "Airframe weight includes the battery and motors; structure is "
                  "what remains after the itemised parts."))

    # ---- Power Budget --------------------------------------------------
    tab_pb = _tab("Power Budget")
    pb_scope = _scope_label(tab_pb)
    pb_tv = _tree(tab_pb, [("item", "Component / loss", 280), ("watts", "Power (W)", 100),
                           ("pct", "% of total", 90)], height=14)

    def update_power_budget(cfg, m):
        _clear_tree(pb_tv)
        pack_I = float(m.get("pack_current_A", 0.0))
        rows = core.build_power_budget(
            total_in_W=float(m.get("total_power_W", 0.0)),
            motor_shaft_W=float(m.get("shaft_power_W", 0.0)),
            motor_copper_W=0.0,
            battery_i2r_W=pack_I ** 2 * cfg.battery.pack_resistance,
            esc_loss_W=float(m.get("esc_loss_W", 0.0)),
            wire_loss_W=float(m.get("wire_loss_W", 0.0)),
            peripheral_W=cfg.avionics_power_W)
        for r in rows:
            pb_tv.insert("", "end", tags=(r["kind"],),
                         values=(r["name"], f"{r['watts']:.1f}", f"{r['pct']:.1f}%"))
        pb_scope.configure(text=(
            f"At the cruise speed of {float(m.get('airspeed_mps', 0.0)):.1f} m/s "
            f"({m.get('regime', '')}). Motor losses are folded into the ESC "
            f"efficiency and rotor losses into the figure of merit, so there is "
            f"no separate motor copper-loss line."))

    def clear_power_budget():
        _clear_tree(pb_tv)
        pb_tv.insert("", "end", tags=("total",),
                     values=("Mission run — no single operating point to break down", "", ""))
        pb_scope.configure(text="Press Run Single-Point to build a power budget.")

    # ---- Mission Diagram ----------------------------------------------
    tab_md = _tab("Mission Diagram")
    md_holder = ttk.Frame(tab_md)
    md_holder.grid(row=1, column=0, columnspan=2, sticky="nsew")
    md_holder.columnconfigure(0, weight=1)
    md_holder.rowconfigure(1, weight=1)
    md_placeholder = _placeholder(md_holder, "Run a mission to map its route.")
    _md_canvas = {"widget": None}

    def draw_mission_diagram(mission, results):
        # Transition and hover phases carry a duration rather than a distance,
        # so take the ground distance each phase actually covered from the
        # simulation rather than from the JSON.
        phases = []
        for i, ph in enumerate(mission.phases):
            covered_km = results[i][2] if i < len(results) else 0.0
            phases.append({"name": ph.name, "course_deg": ph.course_deg,
                           "distance": covered_km * 1000.0, "altitude": ph.altitude_m})
        try:
            fig = core.make_mission_diagram_figure(phases, figsize=(11, 5))
        except Exception:
            return
        md_placeholder.grid_remove()
        _show_figure(md_holder, _md_canvas, fig)

    def clear_mission_diagram():
        _destroy_canvas(_md_canvas)
        md_placeholder.configure(text="A fixed speed run has no route.\n\n"
                                      "Run a mission to map it.")
        md_placeholder.grid()

    # ---- Sensitivity ---------------------------------------------------
    tab_sens = _tab("Sensitivity")
    sens_bar = ttk.Frame(tab_sens)
    sens_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
    ttk.Label(sens_bar, text="Output:").pack(side="left")
    _SENS_POINT = ["Cruise power (W)", "Cruise endurance (min)", "Cruise range (km)",
                   "Hover power (W)", "Hover endurance (min)", "Transition speed (m/s)"]
    _SENS_MISSION = ["Mission energy (Wh)", "Mission time (min)",
                     "Reserve margin (Wh)", "Peak power (W)"]
    v_sens_out = tk.StringVar(value=_SENS_POINT[0])
    sens_box = ttk.Combobox(sens_bar, textvariable=v_sens_out, state="readonly",
                            width=24, values=_SENS_POINT)
    sens_box.pack(side="left", padx=4)
    sens_scope = ttk.Label(sens_bar, text="", foreground="#0B6BCB",
                           font=("TkDefaultFont", 8))
    sens_scope.pack(side="left", padx=8)
    sens_tv = _tree(tab_sens, [("name", "Input (±10%, ±20%)", 220),
                               ("base", "Baseline", 90), ("low", "Low", 90),
                               ("high", "High", 90), ("span", "Swing", 110)], height=12)
    _sens_state = {"cfg": None, "mission": None, "wind": 0.0, "wind_dir": 0.0}

    _LEVERS = [
        ("Airframe weight", lambda c, f: setattr(c, "aircraft_weight_g", c.aircraft_weight_g * f)),
        ("Payload mass", lambda c, f: setattr(c, "payload_mass_g", c.payload_mass_g * f)),
        ("Battery capacity", lambda c, f: setattr(c.battery, "capacity_Ah", c.battery.capacity_Ah * f)),
        ("Wing area", lambda c, f: setattr(c, "wing_area_m2", c.wing_area_m2 * f)),
        ("CD0", lambda c, f: setattr(c, "CD0", c.CD0 * f)),
        ("Oswald efficiency", lambda c, f: setattr(c, "oswald", min(c.oswald * f, 1.0))),
        ("Lift rotor figure of merit", lambda c, f: setattr(
            c, "lift_figure_of_merit", min(c.lift_figure_of_merit * f, 0.9))),
        ("Lift prop diameter", lambda c, f: setattr(
            c, "lift_prop_diameter_in", c.lift_prop_diameter_in * f)),
        ("Cruise prop efficiency", lambda c, f: setattr(
            c, "cruise_prop_efficiency", min(c.cruise_prop_efficiency * f, 0.95))),
        ("Cruise speed", lambda c, f: setattr(c, "cruise_speed_mps", c.cruise_speed_mps * f)),
        ("Avionics power", lambda c, f: setattr(c, "avionics_power_W", c.avionics_power_W * f)),
        ("Hover download", lambda c, f: setattr(
            c, "hover_download_fraction", hover_download_fraction(c) * f)),
    ]

    def _set_sens_mode(from_mission):
        values = _SENS_MISSION if from_mission else _SENS_POINT
        sens_box.configure(values=values)
        if v_sens_out.get() not in values:
            v_sens_out.set(values[0])
        sens_scope.configure(text=(
            "Mission mode — each input is perturbed and the WHOLE mission re-flown."
            if from_mission else
            "Fixed speed mode — each input is perturbed at the cruise speed."))

    def clear_sensitivity(reason):
        _clear_tree(sens_tv)
        sens_tv.insert("", "end", values=(reason, "", "", "", "Press Run Sensitivity"))

    def run_sensitivity():
        base = _sens_state["cfg"]
        if base is None:
            messagebox.showinfo("Sensitivity", "Run a single point or a mission first.")
            return
        choice = v_sens_out.get()
        mission = _sens_state["mission"]

        def evaluate(cfg):
            cfg = base if cfg is None else cfg
            try:
                if mission is not None:
                    _res, tot = simulate_mission(
                        cfg, mission, wind_mps=_sens_state.get("wind", 0.0),
                        wind_direction_deg=_sens_state.get("wind_dir", 0.0),
                        max_accel_mps2=_sens_state.get("accel", 0.0),
                        regen_eff=_sens_state.get("regen", 0.0))
                    return {"Mission energy (Wh)": tot["energy_Wh"],
                            "Mission time (min)": tot["time_s"] / 60.0,
                            "Reserve margin (Wh)": tot["worst"]["reserve_margin_Wh"],
                            "Peak power (W)": tot["worst"]["total_power_W"]}[choice]
                mm = compute_metrics(cfg)
                return {"Cruise power (W)": mm["total_power_W"],
                        "Cruise endurance (min)": mm["cruise_endurance_min"],
                        "Cruise range (km)": mm["cruise_range_km"],
                        "Hover power (W)": mm["hover_power_W"],
                        "Hover endurance (min)": mm["hover_endurance_min"],
                        "Transition speed (m/s)": mm["transition_speed_mps"]}[choice]
            except Exception:
                return None
        evaluate.base_config = base

        rows = core.sensitivity_sweep(_LEVERS, evaluate)
        _clear_tree(sens_tv)
        if not rows:
            sens_tv.insert("", "end", values=("No result — the baseline did not evaluate",
                                              "", "", "", ""))
            return
        for r in rows:
            sens_tv.insert("", "end", values=(
                r["name"], f"{r['baseline']:.2f}", f"{r['low']:.2f}", f"{r['high']:.2f}",
                f"{r['span']:.2f}  ({r['span_pct']:.0f}%)"))

    ttk.Button(sens_bar, text="Run Sensitivity", command=run_sensitivity).pack(
        side="left", padx=6)

    # ---- Compare -------------------------------------------------------
    tab_cmp = _tab("Compare")
    cmp_bar = ttk.Frame(tab_cmp)
    cmp_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
    cmp_label = ttk.Label(cmp_bar, text="No baseline pinned.", foreground="#555555")
    cmp_label.pack(side="left")
    cmp_tv = _tree(tab_cmp, [("metric", "Metric", 230), ("base", "Baseline", 110),
                             ("cur", "Current", 110), ("delta", "Change", 100),
                             ("pct", "Change %", 90)], height=14)
    _cmp = {"baseline": None, "base_mission": False, "current": None, "cur_mission": False}
    _CMP_POINT = [("total_power_W", "Cruise power (W)", 1),
                  ("cruise_endurance_min", "Cruise endurance (min)", 2),
                  ("cruise_range_km", "Cruise range (km)", 2),
                  ("hover_power_W", "Hover power (W)", 1),
                  ("hover_endurance_min", "Hover endurance (min)", 2),
                  ("transition_speed_mps", "Transition speed (m/s)", 2),
                  ("stall_speed_mps", "Stall speed (m/s)", 2),
                  ("pack_current_A", "Cruise pack current (A)", 2),
                  ("hover_pack_current_A", "Hover pack current (A)", 2),
                  ("c_rate", "Cruise C-rate", 2),
                  ("tilt_deg", "Rotor tilt at cruise (deg)", 1),
                  ("lift_share_wing", "Wing lift share", 3),
                  ("esc_loss_W", "ESC loss (W)", 2),
                  ("wire_loss_W", "Wire loss (W)", 2),
                  ("all_up_weight_g", "All-up weight (g)", 0)]
    _CMP_MISSION = [("energy_Wh", "Mission energy (Wh)", 2),
                    ("time_min", "Mission time (min)", 2),
                    ("distance_km", "Distance (km)", 3),
                    ("hover_Wh", "Hover energy (Wh)", 2),
                    ("transition_Wh", "Transition energy (Wh)", 2),
                    ("cruise_Wh", "Cruise energy (Wh)", 2),
                    ("remaining_Wh", "Energy remaining (Wh)", 2),
                    ("peak_power_W", "Peak power (W)", 1),
                    ("peak_current_A", "Peak pack current (A)", 2),
                    ("reserve_margin_Wh", "Lowest reserve margin (Wh)", 2)]

    def refresh_comparison():
        _clear_tree(cmp_tv)
        base, cur = _cmp["baseline"], _cmp["current"]
        if base is None or cur is None:
            return
        if _cmp["base_mission"] != _cmp["cur_mission"]:
            kinds = ("a mission run", "a single-point run")
            cmp_tv.insert("", "end", values=(
                f"Baseline is {kinds[0] if _cmp['base_mission'] else kinds[1]}; "
                f"current is {kinds[0] if _cmp['cur_mission'] else kinds[1]}. "
                "They measure different things and cannot be compared.", "", "", "", ""))
            return
        keys = _CMP_MISSION if _cmp["cur_mission"] else _CMP_POINT
        for r in core.compare_metric_sets(base, cur, keys):
            d = r["decimals"]
            fmt = lambda x: "—" if x is None else f"{x:.{d}f}"
            cmp_tv.insert("", "end", values=(
                r["label"], fmt(r["baseline"]), fmt(r["current"]),
                core.format_delta(r["delta"], d),
                "—" if r["delta_pct"] is None else f"{r['delta_pct']:+.1f}%"))

    def pin_baseline():
        if _cmp["current"] is None:
            messagebox.showinfo("Compare", "Run something first, then pin it.")
            return
        _cmp["baseline"] = dict(_cmp["current"])
        _cmp["base_mission"] = _cmp["cur_mission"]
        cmp_label.configure(text=f"Baseline pinned ({'mission' if _cmp['base_mission'] else 'single point'}).")
        refresh_comparison()

    def clear_baseline():
        _cmp["baseline"] = None
        cmp_label.configure(text="No baseline pinned.")
        _clear_tree(cmp_tv)

    ttk.Button(cmp_bar, text="Clear Baseline", command=clear_baseline).pack(side="right")
    ttk.Button(cmp_bar, text="Pin Current as Baseline", command=pin_baseline).pack(
        side="right", padx=6)

    # ---- Airframe Diagram ---------------------------------------------
    tab_ad = _tab("Airframe Diagram")
    ad_holder = ttk.Frame(tab_ad)
    ad_holder.grid(row=1, column=0, columnspan=2, sticky="nsew")
    ad_holder.columnconfigure(0, weight=1)
    ad_holder.rowconfigure(1, weight=1)
    _placeholder(ad_holder, "Run anything to draw the airframe to scale.")
    _ad_canvas = {"widget": None}

    def draw_airframe_diagram(cfg):
        """
        Always valid: the diagram describes the aircraft, not a flight, so it
        survives both run types rather than being cleared by either.
        """
        try:
            fig = make_airframe_diagram_figure(cfg, figsize=(8, 6.6))
        except Exception:
            return
        for child in ad_holder.winfo_children():
            if isinstance(child, ttk.Label):
                child.grid_remove()
        _show_figure(ad_holder, _ad_canvas, fig)

    # ---- exports --------------------------------------------------------
    # Built by reading the tables on screen, so an export cannot disagree
    # with what the user is looking at — the two cannot drift apart because
    # there is only one source.
    _export_state = {"cfg": None, "from_mission": False, "results": None}

    def _tree_section(title, tree):
        headers = [tree.heading(c)["text"] for c in tree.cget("columns")]
        rows = [list(tree.item(i, "values")) for i in tree.get_children()]
        return (title, headers, rows)

    def _sweep_section(cfg):
        """The speed sweep behind the Fixed Speed Plots, as numbers."""
        speeds, rows = [], []
        v_max = max(cfg.cruise_speed_mps * 1.6, transition_speed_mps(cfg) * 1.6)
        steps = 40
        for i in range(steps + 1):
            v = v_max * i / steps
            point = power_at_airspeed(cfg, v)
            rows.append([round(v, 2), round(point["total_power_W"], 1),
                         point.get("regime", ""),
                         round(float(point.get("tilt_deg", 0.0)), 1),
                         round(float(point.get("lift_share_wing", 0.0)), 3)])
            speeds.append(v)
        return ("Speed Sweep",
                ["Airspeed (m/s)", "Total power (W)", "Regime", "Tilt (deg)",
                 "Wing lift share"], rows)

    def _export_sections():
        cfg = _export_state["cfg"]
        if cfg is None:
            return None
        sections = [_tree_section("Metrics", metrics_tv),
                    _tree_section("Status", status_tv),
                    _tree_section("Weight Budget", wb_tv)]
        if not _export_state["from_mission"]:
            sections.append(_tree_section("Power Budget", pb_tv))
            sections.append(_sweep_section(cfg))
        results = _export_state.get("results")
        if results:
            sections.append((
                "Mission", ["Phase", "Time (min)", "Distance (km)", "Power (W)",
                            "Energy (Wh)", "Status"],
                [list(r) for r in results]))
        return sections

    def export_csv():
        sections = _export_sections()
        if sections is None:
            messagebox.showinfo("Export", "Run something first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="vtol_results.csv")
        if not path:
            return
        try:
            core.export_csv(path, sections)
            messagebox.showinfo("Export", f"Written to {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_excel():
        sections = _export_sections()
        if sections is None:
            messagebox.showinfo("Export", "Run something first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
            initialfile="vtol_results.xlsx")
        if not path:
            return
        try:
            core.export_excel(path, sections)
            messagebox.showinfo("Export", f"Written to {os.path.basename(path)}")
        except ImportError:
            messagebox.showerror(
                "Export failed",
                "Excel export needs openpyxl.\n\npip install openpyxl")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def generate_report():
        """A PDF of the same tables plus the airframe and performance figures."""
        sections = _export_sections()
        cfg = _export_state["cfg"]
        if sections is None:
            messagebox.showinfo("Report", "Run something first.")
            return
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.lib import colors
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                            Table, TableStyle, Image, PageBreak)
        except ImportError:
            messagebox.showerror(
                "Report failed",
                "PDF report needs reportlab.\n\npip install reportlab")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
            initialfile="vtol_report.pdf")
        if not path:
            return

        import tempfile
        styles = getSampleStyleSheet()
        story = [Paragraph("VTOL Power Simulator report", styles["Title"]),
                 Paragraph(f"{cfg.config_type} &mdash; v{SIM_VERSION}", styles["Normal"]),
                 Paragraph(
                     "Mission run: worst-case Status, last-instant Metrics."
                     if _export_state["from_mission"] else
                     "Single-point run at the cruise speed.", styles["Normal"]),
                 Spacer(1, 6 * mm)]
        temp_files = []
        try:
            for title, headers, rows in sections:
                if title == "Speed Sweep":
                    continue                    # a figure says it better
                story.append(Paragraph(title, styles["Heading2"]))
                data = [headers] + [[str(c) for c in r] for r in rows]
                table = Table(data, repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
                story.append(table)
                story.append(Spacer(1, 5 * mm))

            for maker in (lambda: make_airframe_diagram_figure(cfg, figsize=(7, 5.8)),):
                try:
                    fig = maker()
                except Exception:
                    continue
                handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                handle.close()
                fig.savefig(handle.name, dpi=130, bbox_inches="tight")
                temp_files.append(handle.name)
                story.append(PageBreak())
                story.append(Image(handle.name, width=170 * mm, height=130 * mm,
                                   kind="proportional"))

            SimpleDocTemplate(path, pagesize=A4).build(story)
            messagebox.showinfo("Report", f"Written to {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Report failed", str(exc))
        finally:
            for name in temp_files:
                try:
                    os.unlink(name)
                except OSError:
                    pass

    # ---- menu bar -------------------------------------------------------
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load Config…", command=lambda: load_config())
    file_menu.add_command(label="Save Config…", command=lambda: save_config())
    file_menu.add_separator()
    file_menu.add_command(label="Export CSV…", command=export_csv)
    file_menu.add_command(label="Export Excel…", command=export_excel)
    file_menu.add_command(label="Generate PDF Report…", command=generate_report)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)

    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)
    _scale_var = tk.IntVar(value=100)
    scale_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Window Scale", menu=scale_menu)

    def _apply_scale(pct):
        # Tk scaling is in points per pixel; 1.3333 is the 100% baseline.
        try:
            root.tk.call("tk", "scaling", 1.3333 * pct / 100.0)
        except Exception:
            pass
    for pct, label in [(75, "75 % – Compact"), (90, "90 % – Smaller"),
                       (100, "100 % – Default"), (115, "115 % – Slightly Larger"),
                       (125, "125 % – Large"), (150, "150 % – Extra Large"),
                       (175, "175 % – Very Large"), (200, "200 % – Max")]:
        scale_menu.add_radiobutton(label=label, variable=_scale_var, value=pct,
                                   command=lambda p=pct: _apply_scale(p))

    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(
        label="About / Version",
        command=lambda: messagebox.showinfo(
            "About", f"VTOL Power Simulator\nVersion {SIM_VERSION}\n{SIM_BUILD_NOTE}"))
    root.config(menu=menubar)

    def _set_scope(from_mission):
        status_scope.configure(text=(
            "Mission run — every row is the WORST value reached at any point: "
            "the highest current, C-rate and power, and the LOWEST reserve margin."
            if from_mission else
            "Single-point run — rows are evaluated at the cruise speed, with hover "
            "checked alongside because it is the heaviest steady load a VTOL carries."))
        metrics_scope.configure(text=(
            "Mission run — values at the LAST instant flown, not an average and "
            "not the worst case. Worst-case figures are on the Status tab."
            if from_mission else
            "Single-point run — a steady operating point at the cruise speed."))
        _set_sens_mode(from_mission)

    _set_scope(False)
    _apply_field_mode()

    out_text = tk.Text(right, height=12, wrap="none")
    out_text.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    def log(msg):
        out_text.delete("1.0", "end")
        out_text.insert("end", msg)

    def num(key, default=0.0):
        raw = fields[key].get().strip()
        if raw == "":
            return default
        return float(raw)

    def opt(key):
        """An optional numeric field: None when blank, a clear error when not a number."""
        var = fields.get(key)
        if var is None:
            return None
        raw = var.get().strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"'{raw}' is not a number (field: {key})")

    def build_config() -> VTOLConfig:
        battery = VTOLBattery(
            chemistry=fields["chem"].get().strip() or "LiPo",
            cell_capacity_mAh=num("cell_cap", 5000),
            series_cells=int(num("series", 6)),
            parallel_cells=int(num("parallel", 1)),
            cell_weight_g=num("cell_wt", 120),
            voltage_min=num("vmin", 3.3), voltage_nominal=num("vnom", 3.7),
            voltage_max=num("vmax", 4.2),
            resistance_cell_mOhm=num("rcell", 4.0),
            usable_percent=num("usable", 80),
            discharge_c_cont=opt("c_cont"),
            discharge_c_max=opt("c_max"),
            soc_curve_csv=(fields["soc_curve"].get().strip() or None),
        )
        temp = fields["temp"].get().strip()
        return VTOLConfig(
            config_type=v_config_type.get(),
            aircraft_weight_g=num("weight", 6000),
            payload_mass_g=num("payload", 0),
            wing_span_m=num("span", 2.4), wing_area_m2=num("area", 0.6),
            CD0=num("cd0", 0.035), oswald=num("oswald", 0.8),
            CL_max=num("clmax", 1.2), CL_cruise_max=num("clcruise", 0.9),
            num_lift_rotors=int(num("n_lift", 4)),
            lift_prop_diameter_in=num("lift_d", 18),
            lift_prop_pitch_in=num("lift_p", 6),
            lift_motor_kv=num("lift_kv", 300),
            lift_motor_resistance=num("lift_rm", 0.08),
            lift_motor_weight_g=num("lift_wt", 200),
            lift_figure_of_merit=num("fom", 0.65),
            num_cruise_motors=int(num("n_cruise", 1)),
            cruise_prop_diameter_in=num("cruise_d", 14),
            cruise_prop_pitch_in=num("cruise_p", 8),
            cruise_motor_kv=num("cruise_kv", 500),
            cruise_motor_resistance=num("cruise_rm", 0.06),
            cruise_motor_weight_g=num("cruise_wt", 180),
            cruise_prop_efficiency=num("cruise_eff", 0.75),
            stopped_rotor_drag_area_m2=(num("stopped_area")
                                        if fields["stopped_area"].get().strip() else None),
            battery=battery,
            avionics_power_W=num("avionics", 15),
            esc_efficiency=num("esc_eff", 0.96),
            air_density=core.air_density(num("alt", 0),
                                         float(temp) if temp else None),
            cruise_speed_mps=num("cruise_v", 22),
            reference_altitude_m=num("alt", 0),
            wire_resistance_ohm=core.wire_resistance_ohm(
                opt("wire_len") or 0.0,
                awg=int(opt("wire_awg")) if opt("wire_awg") else None,
                ohm_per_m=opt("wire_ohm_m")),
            connectors={
                name: (opt(f"{prefix}_cont") or 0.0, opt(f"{prefix}_max") or 0.0)
                for name, prefix in (("Battery", "conn_batt"), ("ESC", "conn_esc"),
                                     ("Motor", "conn_motor"))
                if (opt(f"{prefix}_cont") or 0.0) > 0 or (opt(f"{prefix}_max") or 0.0) > 0
            },
            hover_download_fraction=opt("download"),
            lift_motor_max_power_W=opt("lift_pmax"),
            cruise_motor_max_power_W=opt("cruise_pmax"),
            lift_prop_table_csv=(fields["lift_table"].get().strip() or None),
            cruise_prop_table_csv=(fields["cruise_table"].get().strip() or None),
            lift_prop_weight_g=opt("lift_prop_wt") or 0.0,
            cruise_prop_weight_g=opt("cruise_prop_wt") or 0.0,
            avionics_mass_g=opt("avionics_mass") or 0.0,
        )

    def show_metrics(m):
        for item in metrics_tv.get_children():
            metrics_tv.delete(item)
        rows = [
            ("Configuration", str(m["config_type"])),
            ("Regime at cruise speed", str(m["regime"])),
            ("All-up weight", f"{m['all_up_weight_g']:.0f} g  ({m['weight_N']:.1f} N)"),
            ("Wing loading", f"{m['wing_loading_N_m2']:.1f} N/m²"),
            ("Disc loading (hover)", f"{m['disc_loading_N_m2']:.1f} N/m²"),
            ("Aspect ratio", f"{m['aspect_ratio']:.2f}"),
            ("Stall speed", f"{m['stall_speed_mps']:.2f} m/s"),
            ("Transition speed", f"{m['transition_speed_mps']:.2f} m/s"),
            ("", ""),
            ("Hover power", f"{m['hover_power_W']:.0f} W"),
            ("Hover endurance", f"{m['hover_endurance_min']:.1f} min"),
            ("Cruise power", f"{m['total_power_W']:.0f} W"),
            ("  rotor share", f"{m['rotor_shaft_W']:.0f} W"),
            ("  cruise prop share", f"{m['cruise_shaft_W']:.0f} W"),
            ("Cruise endurance", f"{m['cruise_endurance_min']:.1f} min"),
            ("Cruise range", f"{m['cruise_range_km']:.2f} km"),
            ("Hover / cruise power", f"{m['hover_to_cruise_power_ratio']:.2f} x"),
            ("", ""),
            ("Pack current", f"{m['pack_current_A']:.2f} A"),
            ("Loaded voltage", f"{m['v_load_V']:.2f} V"),
            ("Usable energy", f"{m['usable_Wh']:.1f} Wh"),
            ("Battery weight", f"{m['battery_weight_g']:.0f} g"),
            ("SoC model", str(m["soc_model"])),
            ("Stopped-rotor drag area", f"{m['stopped_rotor_drag_area_m2']*1e4:.0f} cm²"),
        ]
        for label, value in rows:
            metrics_tv.insert("", "end", values=(label, value))

    def draw_plots(cfg):
        v_stall = stall_speed_mps(cfg)
        v_trans = transition_speed_mps(cfg)
        speeds = [i * 0.5 for i in range(1, int((cfg.cruise_speed_mps * 1.6) / 0.5) + 1)]
        powers, rotor_W, cruise_W = [], [], []
        for v in speeds:
            p = power_at_airspeed(cfg, v)
            powers.append(p["total_power_W"])
            rotor_W.append(p["rotor_shaft_W"])
            cruise_W.append(p["cruise_shaft_W"])

        fig, axes = core.make_figure(1, 2, figsize=(11, 4.5))
        ax1, ax2 = axes
        ax1.plot(speeds, powers, color="#C62828", label="Total")
        ax1.axhline(hover_power_W(cfg)["total_power_W"], color="#1565C0",
                    linestyle="--", label="Hover")
        ax1.axvline(v_trans, color="#6A1B9A", linestyle=":",
                    label=f"transition {v_trans:.1f} m/s")
        ax1.set_xlabel("Airspeed (m/s)"); ax1.set_ylabel("Power (W)")
        ax1.set_title("Power vs Airspeed"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

        ax2.stackplot(speeds, rotor_W, cruise_W,
                      labels=["Lift rotors", "Cruise prop"],
                      colors=["#90CAF9", "#A5D6A7"])
        ax2.axvline(v_trans, color="#6A1B9A", linestyle=":")
        ax2.set_xlabel("Airspeed (m/s)"); ax2.set_ylabel("Shaft power (W)")
        ax2.set_title("Where the power goes"); ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc="upper center")
        fig.tight_layout()

        old = _canvas.get("widget")
        if old is not None:
            try:
                old.get_tk_widget().destroy()
            except Exception:
                pass
        # Remove the mission placeholder if one is showing.
        for child in plot_holder.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_holder)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        _canvas["widget"] = canvas

    def clear_fixed_speed_plots():
        """A mission has no fixed-speed sweep; say so instead of showing a stale one."""
        widget = _canvas.get("widget")
        if widget is not None:
            try:
                widget.get_tk_widget().destroy()
            except Exception:
                pass
            _canvas["widget"] = None
        for child in plot_holder.winfo_children():
            child.destroy()
        ttk.Label(plot_holder, foreground="#888888", justify="center",
                  text="These plots come from a single-point run.\n\nYou have "
                       "just run a mission, so there is nothing to show here.").grid(
            row=0, column=0, padx=20, pady=40)

    def run_single_point():
        try:
            cfg = build_config()
            m = compute_metrics(cfg)
        except NotImplementedError as exc:
            messagebox.showinfo("Not implemented", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        show_metrics(m)
        draw_plots(cfg)
        update_status(cfg, m)
        update_weight_budget(cfg, m)
        draw_airframe_diagram(cfg)
        _export_state.update(cfg=cfg, from_mission=False, results=None)
        update_power_budget(cfg, m)
        clear_mission_plots()
        clear_mission_diagram()
        _set_scope(False)
        _sens_state.update(cfg=cfg, mission=None)
        clear_sensitivity("Single-point re-run — sensitivity is out of date")
        # Store the result BEFORE refreshing Compare. Refreshing first
        # compared the baseline against the PREVIOUS run, so every delta read
        # zero — the fault the fixed-wing Compare tab had.
        _cmp.update(current=dict(m), cur_mission=False)
        refresh_comparison()
        log(f"VTOL Power Simulator  v{SIM_VERSION}\n"
            f"{'=' * 52}\n"
            f"Configuration : {m['config_type']}\n"
            f"Regime        : {m['regime']} at {m['airspeed_mps']:.1f} m/s\n"
            f"Hover power   : {m['hover_power_W']:.0f} W  "
            f"({m['hover_endurance_min']:.1f} min)\n"
            f"Cruise power  : {m['total_power_W']:.0f} W  "
            f"({m['cruise_endurance_min']:.1f} min, "
            f"{m['cruise_range_km']:.1f} km)\n"
            f"Hover costs {m['hover_to_cruise_power_ratio']:.1f}x cruise — "
            f"minimise time in hover.\n")

    def run_mission():
        path = v_mission.get().strip()
        if not path:
            messagebox.showinfo("Mission", "Choose a mission JSON first.")
            return
        try:
            cfg = build_config()
            mission = VTOLMission.from_json(path)
            results, totals = simulate_mission(
            cfg, mission, wind_mps=opt("wind") or 0.0,
            wind_direction_deg=opt("wind_dir") or 0.0,
            max_accel_mps2=opt("accel") or 0.0,
            max_decel_mps2=opt("decel") or 0.0,
            regen_eff=opt("regen") or 0.0)
        except NotImplementedError as exc:
            messagebox.showinfo("Not implemented", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        worst = totals.get("worst") or {}
        last = totals.get("last") or {}
        if last:
            show_metrics(last)
        update_status(cfg, last, worst=worst)
        draw_airframe_diagram(cfg)
        _export_state.update(cfg=cfg, from_mission=True, results=results)
        update_weight_budget(cfg, last)
        clear_power_budget()
        clear_fixed_speed_plots()
        _mission_state.update(series=totals.get("series"), phases=mission.phases,
                              results=results)
        draw_mission_plots()
        draw_mission_diagram(mission, results)
        _set_scope(True)
        _sens_state.update(cfg=cfg, mission=mission,
                           wind=opt("wind") or 0.0, wind_dir=opt("wind_dir") or 0.0,
                           accel=opt("accel") or 0.0, regen=opt("regen") or 0.0)
        clear_sensitivity("Mission re-run — sensitivity is out of date")
        _cmp.update(cur_mission=True, current={
            "energy_Wh": totals["energy_Wh"], "time_min": totals["time_s"] / 60.0,
            "distance_km": totals["distance_m"] / 1000.0,
            "hover_Wh": totals["hover_Wh"], "transition_Wh": totals["transition_Wh"],
            "cruise_Wh": totals["cruise_Wh"], "remaining_Wh": totals["remaining_Wh"],
            "peak_power_W": worst.get("total_power_W"),
            "peak_current_A": worst.get("pack_current_A"),
            "reserve_margin_Wh": worst.get("reserve_margin_Wh")})
        refresh_comparison()

        lines = [f"=== Mission: {os.path.basename(path)} ===",
                 f"{'Phase':<22}{'min':>8}{'km':>9}{'W':>9}{'Wh':>9}  Status"]
        for name, minutes, km, power, energy, status in results:
            lines.append(f"{name:<22}{minutes:>8.2f}{km:>9.3f}"
                         f"{power:>9.0f}{energy:>9.2f}  {status}")
        lines.append("-" * 68)
        lines.append(f"{'TOTAL':<22}{totals['time_s']/60:>8.2f}"
                     f"{totals['distance_m']/1000:>9.3f}{'':>9}"
                     f"{totals['energy_Wh']:>9.2f}")
        lines.append("")
        lines.append(f"Energy split: hover {totals['hover_Wh']:.1f} Wh | "
                     f"transition {totals['transition_Wh']:.1f} Wh | "
                     f"cruise {totals['cruise_Wh']:.1f} Wh")
        lines.append(f"Remaining {totals['remaining_Wh']:.1f} Wh "
                     f"(reserve {totals['reserve_Wh']:.1f} Wh)")
        log("\n".join(lines))

    # ---- configuration save / load -----------------------------------
    # The other two simulators have this; without it a VTOL design could not
    # be kept, shared, or used as a worked example.
    def save_config():
        path = filedialog.asksaveasfilename(
            title="Save VTOL configuration", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        payload = {
            "schema": "vtol_power_sim_v1",
            "config_type": v_config_type.get(),
            "vars": {key: var.get() for key, var in fields.items()},
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        log(f"Configuration saved to {os.path.basename(path)}\n")

    def load_config():
        path = filedialog.askopenfilename(
            title="Load VTOL configuration",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        if payload.get("schema") not in (None, "vtol_power_sim_v1"):
            messagebox.showerror(
                "Wrong file",
                f"That looks like a '{payload.get('schema')}' file, not a VTOL "
                "configuration. Loading it would silently mis-assign fields.")
            return
        for key, value in (payload.get("vars") or {}).items():
            if key in fields:
                fields[key].set(str(value))
        chosen = payload.get("config_type")
        if chosen in CONFIG_TYPES:
            v_config_type.set(chosen)
        log(f"Loaded {os.path.basename(path)}\n"
            f"Press Run Single-Point to evaluate it.\n")

    buttons = ttk.Frame(root, padding=6)
    buttons.grid(row=1, column=0, columnspan=2, sticky="ew")
    ttk.Button(buttons, text="▶  Run Single-Point",
               command=run_single_point).pack(side="left")
    ttk.Button(buttons, text="📋  Run Mission (JSON)",
               command=run_mission).pack(side="left", padx=(6, 0))
    ttk.Button(buttons, text="💾  Save Config",
               command=save_config).pack(side="right")
    ttk.Button(buttons, text="📂  Load Config",
               command=load_config).pack(side="right", padx=(0, 6))

    log(f"VTOL Power Simulator  v{SIM_VERSION}\n"
        f"{'=' * 52}\n"
        "Lift+cruise is implemented. The other configurations appear in the\n"
        "dropdown so the input set is defined, and are refused rather than\n"
        "approximated.\n\n"
        "Press Run Single-Point to begin.\n")

    root.mainloop()


if __name__ == "__main__":
    main()
