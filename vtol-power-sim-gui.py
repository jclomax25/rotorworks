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

SIM_VERSION = "0.1.0"
SIM_BUILD_NOTE = "VTOL simulator - lift+cruise configuration"

G0 = core.G0

CONFIG_TYPES = ["lift+cruise", "tiltrotor", "tiltwing", "tailsitter"]
IMPLEMENTED_CONFIG_TYPES = {"lift+cruise"}


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
                 soc_model: str = "auto"):
        self.chemistry = chemistry
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
        core.configure_battery_soc_model(self, soc_model, None, None, None, None)

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
                 reference_altitude_m: float = 0.0):
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
    return ideal_per / cfg.lift_figure_of_merit * n


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
    return t_per * (v + vi) / cfg.cruise_prop_efficiency * n


def electrical_power_W(cfg: VTOLConfig, shaft_power_W: float) -> float:
    """Shaft power to pack power, through the ESC, plus the avionics load."""
    return shaft_power_W / cfg.esc_efficiency + cfg.avionics_power_W


# ============================================================
# FLIGHT REGIMES
# ============================================================

def hover_power_W(cfg: VTOLConfig, climb_rate_mps: float = 0.0) -> Dict[str, float]:
    """Power to hover, or to climb vertically on the lift rotors."""
    thrust_N = cfg.weight_N
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
    rotor_thrust_N = max(cfg.weight_N - lift_N, 0.0)

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


def power_at_airspeed(cfg: VTOLConfig, airspeed_mps: float) -> Dict[str, float]:
    """
    Power at any airspeed, choosing the regime automatically.

    Below the speed at which the wing can carry the whole aircraft, the
    rotors are still contributing and the transition model applies. Above it,
    the aircraft is wing-borne and the rotors can stop.
    """
    v = max(float(airspeed_mps), 0.0)
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
    return metrics


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
            ))
        return VTOLMission(
            phases=phases,
            reserve_percent=float(data.get("reserve_percent", 20.0)),
            transition_time_s=float(data.get("transition_time_s", 12.0)),
        )


def simulate_mission(cfg: VTOLConfig, mission: VTOLMission) -> Tuple[List[tuple], Dict[str, float]]:
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

    for phase in mission.phases:
        kind = phase.kind
        if kind in ("hover", "climb", "descend"):
            climb = phase.climb_rate_mps if kind == "climb" else 0.0
            point = hover_power_W(cfg, climb_rate_mps=climb)
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
            for i in range(steps):
                frac = (i + 0.5) / steps
                v = v_end * frac
                p = power_at_airspeed(cfg, v)["total_power_W"]
                dt = duration / steps
                energy_Ws += p * dt
                distance += v * dt
            point = {"total_power_W": energy_Ws / max(duration, 1e-9)}
            bucket = "transition_Wh"

        else:                                   # cruise
            v = phase.airspeed_mps or cfg.cruise_speed_mps
            point = power_at_airspeed(cfg, v)
            if phase.distance_m is not None:
                distance = float(phase.distance_m)
                duration = distance / max(v, 1e-9)
            else:
                duration = float(phase.duration_s or 0.0)
                distance = v * duration
            bucket = "cruise_Wh"

        energy_Wh = point["total_power_W"] * duration / 3600.0
        remaining_Wh -= energy_Wh
        totals["time_s"] += duration
        totals["distance_m"] += distance
        totals["energy_Wh"] += energy_Wh
        totals[bucket] += energy_Wh

        status = "OK"
        if remaining_Wh < 0:
            status = "BATTERY DEPLETED"
        elif remaining_Wh < reserve_Wh:
            status = "RESERVE VIOLATION"

        results.append((phase.name, duration / 60.0, distance / 1000.0,
                        point["total_power_W"], energy_Wh, status))

        if remaining_Wh < 0:
            break

    totals["remaining_Wh"] = remaining_Wh
    totals["reserve_Wh"] = reserve_Wh
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


def _print_mission(cfg: VTOLConfig, path: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"Mission file not found: {path}")
    try:
        mission = VTOLMission.from_json(path)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Mission file {path} is not valid JSON: {exc}")

    results, totals = simulate_mission(cfg, mission)
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
            _print_mission(cfg, args.mission)
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
    v_config_type.trace_add("write", _on_type_change)

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

    def add_row(parent, row, label, key, default, help_text=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(default))
        fields[key] = var
        entry = ttk.Entry(parent, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 4))
        parent.columnconfigure(1, weight=1)
        if help_text:
            marker = ttk.Label(parent, text="?", foreground="#0B6BCB",
                               cursor="question_arrow")
            marker.grid(row=row, column=2, sticky="w")
            core.Tooltip(marker, help_text)
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

    # ---- output ------------------------------------------------------
    out_nb = ttk.Notebook(right)
    out_nb.grid(row=0, column=0, sticky="nsew")
    tab_metrics = ttk.Frame(out_nb); out_nb.add(tab_metrics, text="Metrics")
    tab_plots = ttk.Frame(out_nb); out_nb.add(tab_plots, text="Plots")
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
        canvas = FigureCanvasTkAgg(fig, master=plot_holder)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        _canvas["widget"] = canvas

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
            results, totals = simulate_mission(cfg, mission)
        except NotImplementedError as exc:
            messagebox.showinfo("Not implemented", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

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

    buttons = ttk.Frame(root, padding=6)
    buttons.grid(row=1, column=0, columnspan=2, sticky="ew")
    ttk.Button(buttons, text="▶  Run Single-Point",
               command=run_single_point).pack(side="left")
    ttk.Button(buttons, text="📋  Run Mission (JSON)",
               command=run_mission).pack(side="left", padx=(6, 0))

    log(f"VTOL Power Simulator  v{SIM_VERSION}\n"
        f"{'=' * 52}\n"
        "Lift+cruise is implemented. The other configurations appear in the\n"
        "dropdown so the input set is defined, and are refused rather than\n"
        "approximated.\n\n"
        "Press Run Single-Point to begin.\n")

    root.mainloop()


if __name__ == "__main__":
    main()
