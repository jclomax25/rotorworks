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

SIM_VERSION = "1.10.0"
SIM_BUILD_NOTE = "VTOL simulator - lift+cruise, tiltrotor, tiltwing, tailsitter; wind-aware fixed-speed sweeps"

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
                 soc_curve_csv: Optional[str] = None,
                 # --- pack-level entry -------------------------------
                 unit_mode: str = "cell",
                 cells_series_per_unit: int = 1,
                 cells_parallel_per_unit: int = 1,
                 pack_capacity_mAh: Optional[float] = None,
                 pack_weight_g: Optional[float] = None,
                 energy_density_Wh_per_kg: Optional[float] = None,
                 # --- limits in amps, as datasheets quote them -------
                 discharge_cont_A: Optional[float] = None,
                 discharge_max_A: Optional[float] = None,
                 charge_current_max_A: Optional[float] = None,
                 # --- explicit discharge curve -----------------------
                 soc_bp: Optional[List[float]] = None,
                 ocv_cell_bp: Optional[List[float]] = None,
                 r_scale_bp: Optional[List[float]] = None):
        self.chemistry = chemistry

        # "cell": one unit IS one cell, and series/parallel count cells.
        # "pack": one unit is a finished pack of cells_series_per_unit cells
        # in series, and series/parallel count PACKS. Cell mode forces the
        # per-unit counts to 1, which makes it arithmetically identical to
        # the cell-only model this replaced.
        self.unit_mode = (str(unit_mode).strip().lower()
                          if unit_mode else "cell")
        if self.unit_mode not in ("cell", "pack"):
            self.unit_mode = "pack" if pack_weight_g is not None else "cell"

        self.series_units = max(int(series_cells), 1)
        self.parallel_units = max(int(parallel_cells), 1)
        if self.unit_mode == "cell":
            cells_series_per_unit = 1
            cells_parallel_per_unit = 1
        self.cells_series_per_unit = max(int(cells_series_per_unit), 1)
        self.cells_parallel_per_unit = max(int(cells_parallel_per_unit), 1)

        self.series_cells = self.series_units * self.cells_series_per_unit
        self.parallel_cells = self.parallel_units * self.cells_parallel_per_unit
        self.total_cells = self.series_cells * self.parallel_cells

        self.cell_capacity_mAh = float(cell_capacity_mAh or 0.0)
        self.pack_capacity_mAh = (float(pack_capacity_mAh)
                                  if pack_capacity_mAh not in (None, "") else None)
        self.cell_weight_g = float(cell_weight_g or 0.0)
        self.pack_weight_g = (float(pack_weight_g)
                              if pack_weight_g not in (None, "") else None)

        # Series raises voltage, parallel raises capacity. Never both.
        if self.unit_mode == "cell":
            self.capacity_mAh = self.cell_capacity_mAh * self.parallel_cells
            self.weight_g = self.cell_weight_g * self.total_cells
        else:
            # Only packs wired in PARALLEL add capacity; every physical pack
            # is carried, however it is wired, so weight counts them all.
            self.capacity_mAh = (self.pack_capacity_mAh or 0.0) * self.parallel_units
            self.weight_g = ((self.pack_weight_g or 0.0)
                             * self.series_units * self.parallel_units)
        self.capacity_Ah = self.capacity_mAh / 1000.0

        self.vmin_pack = float(voltage_min) * self.series_cells
        self.vnom_pack = float(voltage_nominal) * self.series_cells
        self.vmax_pack = float(voltage_max) * self.series_cells

        self.resistance_cell = float(resistance_cell_mOhm) / 1000.0
        self.usable_fraction = min(max(float(usable_percent) / 100.0, 0.0), 1.0)

        # Energy density is reported, not used to drive anything: it is a
        # sanity check on the pack (roughly 150-200 Wh/kg for LiPo, 200-260
        # for Li-ion). An entered figure is kept as given rather than
        # overwritten by the derived one, so a datasheet value can be
        # compared against what the entered weight and capacity imply.
        wkg = self.weight_g / 1000.0
        self.energy_density_derived_Wh_per_kg = (
            (self.capacity_Ah * self.vnom_pack) / wkg if wkg > 0 else 0.0)
        self.energy_density_Wh_per_kg = (
            float(energy_density_Wh_per_kg)
            if energy_density_Wh_per_kg not in (None, "", 0)
            else self.energy_density_derived_Wh_per_kg)

        # Limits given in amps outrank the C-rate, because that is how a
        # datasheet quotes them. Blank stays blank: Status reports "Not
        # Specified" rather than inventing a limit to check against, which
        # is why these are None and not inf.
        self._discharge_cont_A = (float(discharge_cont_A)
                                  if discharge_cont_A not in (None, "", 0) else None)
        self._discharge_max_A = (float(discharge_max_A)
                                 if discharge_max_A not in (None, "", 0) else None)
        self.charge_current_max_A = (float(charge_current_max_A)
                                     if charge_current_max_A not in (None, "", 0)
                                     else None)

        # Optional C-rate ratings. Blank means "not specified", and Status
        # says so rather than inventing a limit to check against.
        self.discharge_c_cont = (float(discharge_c_cont)
                                 if discharge_c_cont not in (None, "", 0) else None)
        self.discharge_c_max = (float(discharge_c_max)
                                if discharge_c_max not in (None, "", 0) else None)

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
        # Explicit breakpoints outrank a CSV curve, which outranks the
        # chemistry preset. The resolver applies that order; passing the
        # arrays through is what makes the GUI's breakpoint fields reach it —
        # they were being dropped here, so entering a curve by hand did
        # nothing at all.
        core.configure_battery_soc_model(self, soc_model, self.soc_curve_csv,
                                         soc_bp, ocv_cell_bp, r_scale_bp)

    @property
    def discharge_cont_A(self) -> Optional[float]:
        """
        Continuous current limit in amps, or None if unrated.

        An explicit amp figure wins over the C-rating: a datasheet quoting
        both is quoting the amps as the real limit, and deriving C x Ah from
        a rounded C-rating loses that.
        """
        if self._discharge_cont_A is not None:
            return self._discharge_cont_A
        return (self.discharge_c_cont * self.capacity_Ah
                if self.discharge_c_cont else None)

    @property
    def discharge_max_A(self) -> Optional[float]:
        """Burst current limit in amps, or None if unrated."""
        if self._discharge_max_A is not None:
            return self._discharge_max_A
        return (self.discharge_c_max * self.capacity_Ah
                if self.discharge_c_max else None)

    @property
    def charge_time_h(self) -> Optional[float]:
        """
        Hours to refill the pack at its rated charge current.

        Capacity over current — no charge-curve modelling, so this is the
        constant-current part only and a real charge takes longer once it
        tapers. Stated rather than modelled, because the taper depends on
        the charger.
        """
        if not self.charge_current_max_A:
            return None
        return self.capacity_Ah / self.charge_current_max_A

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
                 avionics_rails: Optional[dict] = None,
                 periph_current_A: float = 0.0,
                 esc_resistance_ohm: float = 0.0,
                 esc_max_current_A: Optional[float] = None,
                 esc_efficiency: float = 0.96,
                 esc_weight_g: float = 0.0,
                 # environment
                 air_density: float = 1.225,
                 cruise_speed_mps: float = 22.0,
                 reference_altitude_m: float = 0.0,
                 wire_resistance_ohm: float = 0.0,
                 connectors: Optional[dict] = None,
                 hover_download_fraction: Optional[float] = None,
                 lift_motor_max_power_W: Optional[float] = None,
                 lift_motor_max_current_A: Optional[float] = None,
                 lift_prop_max_thrust_g: float = 0.0,
                 cruise_motor_max_current_A: Optional[float] = None,
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
        # Regulated rails, as {volts: (amps, efficiency)}. When any are given
        # they REPLACE the flat avionics figure, because they describe the
        # same load in more detail: a 5 V 2 A rail behind a 90% BEC costs
        # 11.1 W at the pack, not 10.
        self.avionics_rails = dict(avionics_rails or {})
        # Loads wired straight to the pack rather than through a regulated
        # rail — a heater, a winch, a payload on raw pack voltage. This ADDS
        # to the avionics figure rather than replacing it: unlike the rails,
        # it describes a different load, not the same one in more detail.
        self.periph_current_A = max(float(periph_current_A or 0.0), 0.0)
        self.esc_resistance_ohm = max(float(esc_resistance_ohm or 0.0), 0.0)
        self.esc_max_current_A = (float(esc_max_current_A)
                                  if esc_max_current_A else None)
        self.esc_efficiency = min(max(float(esc_efficiency), 0.5), 1.0)
        # One ESC per driven rotor. Left at zero the Weight Budget simply
        # has no ESC line, which is what it had before — their mass went
        # into the structure residual unlabelled.
        self.esc_weight_g = max(float(esc_weight_g or 0.0), 0.0)

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
        self.lift_motor_max_current_A = (float(lift_motor_max_current_A)
                                         if lift_motor_max_current_A else None)
        # Rated static thrust of ONE lift propeller, for the margin column on
        # the per-rotor table. Blank means the margin is simply not checked.
        self.lift_prop_max_thrust_g = max(float(lift_prop_max_thrust_g or 0.0), 0.0)
        self.cruise_motor_max_current_A = (float(cruise_motor_max_current_A)
                                           if cruise_motor_max_current_A else None)
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


def avionics_input_power_W(cfg: VTOLConfig) -> float:
    """
    Power the avionics draw AT THE PACK.

    With regulated rails entered, each rail's delivered power is divided by
    its converter efficiency, so a 5 V 2 A load behind a 90% BEC costs 11.1 W
    rather than 10 — the regulator loss is real and belongs in the budget.

    Rails REPLACE the flat figure rather than adding to it: they describe the
    same load, and counting both would double it. That is exactly the fault
    the multicopter shipped for two releases before it was caught.
    """
    rails = getattr(cfg, "avionics_rails", None)
    if not rails:
        return cfg.avionics_power_W
    total = 0.0
    for volts, (amps, efficiency) in rails.items():
        total += (float(volts) * float(amps)) / max(float(efficiency), 1e-9)
    return total


def peripheral_power_W(cfg: VTOLConfig) -> float:
    """
    Power drawn by loads wired straight to the pack, at the pack.

    Valued at NOMINAL pack voltage, matching how the rest of the model
    charges the battery. Using the loaded voltage here would disagree with
    the model by the sag, which shows up as an "Unaccounted" row in the
    Power Budget — the fault the multicopter shipped and had to fix.
    """
    return max(float(getattr(cfg, "periph_current_A", 0.0) or 0.0), 0.0) * \
        cfg.battery.vnom_pack


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
    base = (shaft_power_W / cfg.esc_efficiency
            + avionics_input_power_W(cfg)
            + peripheral_power_W(cfg))
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
    discs_overlap = False
    same_side = [pt for pt in positions if pt[0] > 0]
    if len(same_side) >= 2:
        gap = math.dist(same_side[0], same_side[1]) - lift_d
        gap_note = f"\nTip-to-tip gap   {gap * 1000:+.0f} mm"
        # Drawn later, once the dimension bands are known: placed here it
        # landed on top of the span dimension on the vectored layouts.
        discs_overlap = gap < 0

    # ---- dimension annotations ---------------------------------------
    # The multicopter labels its span, arm length and motor pitch directly
    # on the drawing. A corner text box alone makes the reader match numbers
    # to features by eye, which is exactly what a scale drawing should save
    # them from.
    #
    # Every dimension is placed in a band OUTSIDE the aircraft's own extent,
    # and the view is then widened to include those bands. Placing them by
    # eye inside the drawing put the span arrow through the summary box and
    # the rotor-pitch label across a disc.
    y_hi = max((y for _x, y in positions), default=0.0) + lift_r
    y_lo = min((y for _x, y in positions), default=0.0) - lift_r
    y_hi = max(y_hi, chord / 2.0)
    y_lo = min(y_lo, -chord / 2.0)
    x_hi = max(max((x for x, _y in positions), default=0.0) + lift_r, span / 2.0)
    margin = max(span, y_hi - y_lo) * 0.13

    # The rotor-diameter dimension sits just below the lowest disc, so it
    # extends the drawing downward and the span band has to clear it. Every
    # band is therefore fixed BEFORE anything is drawn — computing one after
    # drawing another silently leaves the earlier arrow where it was.
    # Below the lowest disc, or below the wing if the discs sit on it — the
    # vectored types mount their rotors along the wing, so "below the lowest
    # disc" would put the arrow across the wing rectangle.
    y_rotor = (min(min(positions, key=lambda p: p[1])[1] - lift_r * 1.12,
                   -chord / 2.0 - lift_r * 0.25)
               if positions else y_lo)

    # Reserved bands: span below the aircraft, rotor pitch above it, the
    # vertical dimensions off to port.
    y_span = min(y_lo, y_rotor) - margin
    y_pitch = y_hi + margin
    x_left = -x_hi - margin

    def _dim(p0, p1, text, colour, offset=(0, 0), ha="center", va="center"):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=1.1,
                                    shrinkA=0, shrinkB=0), zorder=6)
        mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        ax.annotate(text, mid, textcoords="offset points", xytext=offset,
                    ha=ha, va=va, fontsize=7.5, color=colour,
                    fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85))

    # Span, in the band below everything.
    _dim((-span / 2.0, y_span), (span / 2.0, y_span),
         f"{span * 1000:.0f} mm span", "#6A1B9A", offset=(0, -9), va="top")

    # Lateral spacing between the port and starboard rotor groups, above.
    port = sorted((p for p in positions if p[0] < 0), key=lambda p: p[0])
    stbd = sorted((p for p in positions if p[0] > 0), key=lambda p: p[0])
    if port and stbd:
        _dim((port[-1][0], y_pitch), (stbd[0][0], y_pitch),
             f"{abs(stbd[0][0] - port[-1][0]) * 1000:.0f} mm rotor pitch",
             "#AD1457", offset=(0, 9), va="bottom")

    # Chord, off to port. Labelled ABOVE its arrow rather than beside it:
    # a horizontal label here runs off the left-hand edge of the axes and
    # into the y-axis title.
    _dim((x_left, -chord / 2.0), (x_left, chord / 2.0),
         f"{chord * 1000:.0f} mm\nchord", "#00695C", offset=(0, 8),
         va="bottom")

    # One rotor's diameter, measured just BELOW the lowest disc rather than
    # across it — a diameter line through the centre runs straight over that
    # rotor's index number.
    if positions:
        rx = min(positions, key=lambda p: p[1])[0]
        _dim((rx - lift_r, y_rotor), (rx + lift_r, y_rotor),
             f"{lift_d * 1000:.0f} mm rotor", "#2E7D32",
             offset=(0, -9), va="top")

    # Fore-and-aft boom reach on lift+cruise, where the rotors sit off the
    # wing rather than on it. Drawn on the starboard side, clear of the
    # chord dimension.
    if not vectored and positions and stbd:
        y_max = max(y for _x, y in positions)
        y_min = min(y for _x, y in positions)
        if y_max - y_min > 1e-6:
            x_boom = x_hi + margin * 0.55
            _dim((x_boom, y_min), (x_boom, y_max),
                 f"{(y_max - y_min) * 1000:.0f} mm\nboom spread", "#EF6C00",
                 offset=(6, 0), ha="left")
            x_hi = x_boom

    # The overlap warning goes in its own band under the span dimension,
    # where nothing else is drawn.
    y_warn = y_span
    if discs_overlap:
        y_warn = y_span - margin * 0.85
        ax.text(0, y_warn, "ROTOR DISCS OVERLAP", ha="center", va="top",
                color="#B71C1C", fontsize=10, fontweight="bold", zorder=8)

    # What is left in the box is only what the drawing does NOT annotate,
    # so the two cannot disagree and nothing is said twice.
    ax.text(0.015, 0.985,
            f"Lift props  {lift_d * 1000:.0f} mm x {n}"
            + (gap_note.replace("\nTip-to-tip gap   ", "\nTip-to-tip  ") if gap_note else ""),
            transform=ax.transAxes, fontsize=8, family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#FFFDE7", edgecolor="#BDBDBD"))

    reach_x = max(x_hi, abs(x_left)) + margin * 0.55
    reach_y = max(abs(y_warn), abs(y_span), abs(y_pitch)) + margin * 0.55
    reach = max(reach_x, reach_y)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("metres (starboard +)")
    ax.set_ylabel("metres (nose +)")
    ax.set_title(f"{cfg.config_type} — plan view, to scale")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def compute_metrics(cfg: VTOLConfig, airspeed_mps: Optional[float] = None,
                    wind_mps: float = 0.0,
                    wind_direction_deg: float = 0.0,
                    course_deg: float = 0.0) -> Dict[str, object]:
    """
    Single-point metrics at a cruise airspeed, plus the hover comparison.

    Wind changes what the flight ACHIEVES, not what it costs. The entered
    speed is an AIRSPEED, and power is a function of airspeed alone, so
    endurance is the same in any wind. What moves is distance over the
    ground:

        groundspeed = sqrt(V_air^2 - V_cross^2) - V_head
        range       = groundspeed x endurance

    Part of the airspeed is spent crabbing into a crosswind and never
    becomes progress, which is why the crosswind term subtracts before the
    headwind does.

    Hovering is the exception, and the reason this is not just the
    fixed-wing treatment: holding station in wind means flying at the wind
    speed THROUGH THE AIR, so a VTOL holding position in a breeze is not
    hovering at all — it is flying slowly, and translational lift makes
    that cheaper than still-air hover. That is the model simulate_mission
    already uses for its hover phases, and the two must agree or the same
    aircraft reports two different numbers for the same condition.
    """
    _require_implemented(cfg)

    v = cfg.cruise_speed_mps if airspeed_mps is None else float(airspeed_mps)
    wind = max(float(wind_mps or 0.0), 0.0)
    head, cross = core.wind_components_mps(wind, wind_direction_deg, course_deg)

    hover = hover_power_W(cfg)
    point = power_at_airspeed(cfg, v)
    usable_Wh = cfg.battery.usable_Wh

    hover_min = usable_Wh / max(hover["total_power_W"], 1e-9) * 60.0
    cruise_min = usable_Wh / max(point["total_power_W"], 1e-9) * 60.0

    # Station keeping: zero groundspeed, airspeed equal to the wind. At zero
    # wind power_at_airspeed(cfg, 0) IS hover_power_W(cfg), so every figure
    # below collapses to the still-air one and nothing moves.
    station = power_at_airspeed(cfg, wind)
    station_min = usable_Wh / max(station["total_power_W"], 1e-9) * 60.0

    groundspeed = core.groundspeed_along_track_mps(v, head, cross)

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
        # Ground distance, so a headwind shortens it and a tailwind does not
        # flatter the endurance figure into looking like range.
        "cruise_range_km": cruise_min * 60.0 * groundspeed / 1000.0,
        "cruise_range_still_air_km": cruise_min * 60.0 * v / 1000.0,
        "hover_to_cruise_power_ratio": hover["total_power_W"] /
                                       max(point["total_power_W"], 1e-9),

        "wind_mps": wind,
        "wind_direction_deg": float(wind_direction_deg),
        "course_deg": float(course_deg),
        "wind_head_mps": head,
        "wind_cross_mps": cross,
        "groundspeed_mps": groundspeed,
        # False when the crosswind meets or exceeds the airspeed: the
        # aircraft cannot crab far enough to hold the course at all, and
        # groundspeed along it is zero rather than merely small.
        "course_holdable": abs(cross) < v,

        "station_power_W": station["total_power_W"],
        "station_endurance_min": station_min,
        "station_regime": station["regime"],

        "pack_current_A": pack_I,
        "v_load_V": v_load,
        "usable_Wh": usable_Wh,
        "battery_weight_g": cfg.battery.weight_g,
        "soc_model": core.soc_model_short_label(cfg.battery.soc_model_source),
        "battery_unit_mode": cfg.battery.unit_mode,
        "battery_series_cells": cfg.battery.series_cells,
        "battery_parallel_cells": cfg.battery.parallel_cells,
        "battery_total_cells": cfg.battery.total_cells,
        "battery_capacity_mAh": cfg.battery.capacity_mAh,
        "battery_energy_density_Wh_per_kg": cfg.battery.energy_density_Wh_per_kg,
        "battery_charge_time_h": cfg.battery.charge_time_h,
        "battery_cont_A": cfg.battery.discharge_cont_A,
        "battery_max_A": cfg.battery.discharge_max_A,

        "avionics_power_W": cfg.avionics_power_W,
        "avionics_input_power_W": avionics_input_power_W(cfg),
        "periph_current_A": cfg.periph_current_A,
        "peripheral_power_W": peripheral_power_W(cfg),
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
    before_wire = (shaft / max(cfg.esc_efficiency, 1e-9)
                   + avionics_input_power_W(cfg) + peripheral_power_W(cfg))
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
                     regen_eff: float = 0.0,
                     transient_dt_s: float = 0.25) -> Tuple[List[tuple], Dict[str, float]]:
    """
    Fly the mission by integrating it in time, as the multicopter and
    fixed-wing do.

    Every phase runs through ONE loop that steps `transient_dt_s` at a time
    and, at each step, ramps the speed toward the phase's target within the
    acceleration limit, moves the altitude toward the phase's target at the
    commanded climb or descent rate, charges the power for the speed and
    climb actually flown, and adds the kinetic cost of any speed change.

    The phase-level model this replaces evaluated each phase at a single
    operating point and layered special cases on top — sub-steps for
    transitions, a lead-in for cruise legs, a separate branch for hovering in
    wind. Those cases could and did disagree with each other: a transition
    paid for accelerating while a cruise leg did not, and a leg that could
    not reach its speed overshot its own distance. Stepping everything
    through one loop removes the category rather than adding more cases.

    What this buys beyond tidiness: pack voltage sags WITHIN a leg as the
    state of charge falls, so a long cruise ends drawing more current than it
    began, and a leg terminates exactly when its distance is covered rather
    than when an averaged speed says it should.
    """
    _require_implemented(cfg)

    dt = max(float(transient_dt_s or 0.25), 0.01)
    usable_Wh = cfg.battery.usable_Wh
    reserve_Wh = usable_Wh * mission.reserve_percent / 100.0
    remaining_Wh = usable_Wh

    results: List[tuple] = []
    totals = {"time_s": 0.0, "distance_m": 0.0, "energy_Wh": 0.0,
              "hover_Wh": 0.0, "transition_Wh": 0.0, "cruise_Wh": 0.0}

    vnom = max(cfg.battery.vnom_pack, 1e-9)
    wind_mps = max(float(wind_mps or 0.0), 0.0)
    # Every one of these is a quantity the model already computes at each
    # step. Nothing here is derived by a separate correlation: where a value
    # is arithmetic on the others (energy used, state of charge) that is
    # stated in the label rather than dressed up as a new measurement.
    series: Dict[str, list] = {k: [] for k in (
        "t_s", "phase", "kind", "airspeed_mps", "altitude_m", "distance_km",
        "total_power_W", "pack_current_A", "c_rate", "energy_remaining_Wh",
        "tilt_deg", "lift_share_wing",
        "rotor_shaft_W", "cruise_shaft_W", "shaft_power_W",
        "rotor_thrust_N", "wing_lift_N", "cruise_thrust_N", "drag_N",
        "energy_used_Wh", "soc_pct", "reserve_margin_Wh",
        "capacity_remaining_mAh", "climb_rate_mps", "specific_power_W_per_kg",
        "segment_code")}
    # Segment kind as a number, so it can share an axis with everything else.
    # The multicopter plots its segment type the same way.
    _KIND_CODE = {"hover": 0, "climb": 1, "descend": 2,
                  "transition": 3, "cruise": 4}
    auw_kg = max(cfg.all_up_weight_g / 1000.0, 1e-9)
    worst = {"total_power_W": 0.0, "pack_current_A": 0.0, "c_rate": 0.0,
             "remaining_Wh": usable_Wh, "peak_phase": ""}
    state = {"t": 0.0, "alt": 0.0, "dist": 0.0, "v": 0.0}

    def _record(name, kind, v, alt, power, detail=None):
        current = power / vnom
        d = detail or {}
        prev_t = series["t_s"][-1] if series["t_s"] else 0.0
        prev_alt = series["altitude_m"][-1] if series["altitude_m"] else 0.0
        dt = state["t"] - prev_t
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
        series["tilt_deg"].append(float(d.get("tilt_deg", 0.0)))
        series["lift_share_wing"].append(float(d.get("lift_share_wing", 0.0)))

        series["rotor_shaft_W"].append(float(d.get("rotor_shaft_W", 0.0)))
        series["cruise_shaft_W"].append(float(d.get("cruise_shaft_W", 0.0)))
        series["shaft_power_W"].append(float(d.get("shaft_power_W", 0.0)))
        series["rotor_thrust_N"].append(float(d.get("rotor_thrust_N", 0.0)))
        series["wing_lift_N"].append(float(d.get("wing_lift_N", 0.0)))
        series["cruise_thrust_N"].append(float(d.get("cruise_thrust_N", 0.0)))
        series["drag_N"].append(float(d.get("drag_N", 0.0)))

        series["energy_used_Wh"].append(usable_Wh - remaining_Wh)
        series["soc_pct"].append(remaining_Wh / max(usable_Wh, 1e-9) * 100.0)
        series["reserve_margin_Wh"].append(remaining_Wh - reserve_Wh)
        series["capacity_remaining_mAh"].append(
            remaining_Wh / vnom * 1000.0)
        # Rate of climb over the step just flown. The first sample has no
        # previous step, so it is zero rather than a division by zero.
        series["climb_rate_mps"].append((alt - prev_alt) / dt if dt > 1e-9 else 0.0)
        series["specific_power_W_per_kg"].append(power / auw_kg)
        series["segment_code"].append(float(_KIND_CODE.get(kind, -1)))
        if power > worst["total_power_W"]:
            worst["total_power_W"] = power
            worst["peak_phase"] = name
        worst["pack_current_A"] = max(worst["pack_current_A"], current)
        worst["c_rate"] = max(worst["c_rate"],
                              current / max(cfg.battery.capacity_Ah, 1e-9))
        worst["remaining_Wh"] = min(worst["remaining_Wh"], remaining_Wh)

    depleted = False
    for phase in mission.phases:
        kind = phase.kind
        alt_start = state["alt"]
        alt_target = float(phase.altitude_m)

        # --- what this phase is asking for ----------------------------
        if kind in ("hover", "climb", "descend"):
            # Holding station in wind means flying at the wind speed through
            # the air — which is why a VTOL hovers more cheaply into a breeze.
            target_v = wind_mps
        elif kind == "transition":
            target_v = float(phase.airspeed_mps or transition_speed_mps(cfg))
        else:
            target_v = float(phase.airspeed_mps or cfg.cruise_speed_mps)

        climb_rate = float(phase.climb_rate_mps or 0.0) if kind == "climb" else 0.0
        if kind == "descend" and alt_target < alt_start:
            # Descent rate is implied by the phase duration when not given.
            span = float(phase.duration_s or 0.0)
            climb_rate = -((alt_start - alt_target) / span) if span > 0 else 0.0

        head, cross = core.wind_components_mps(
            wind_mps, wind_direction_deg, phase.course_deg)

        distance_goal = (float(phase.distance_m)
                         if (kind not in ("hover", "climb", "descend")
                             and phase.distance_m is not None) else None)
        if kind == "transition":
            duration_goal = float(phase.duration_s or mission.transition_time_s)
        elif distance_goal is None:
            duration_goal = float(phase.duration_s or 0.0)
        else:
            duration_goal = None

        max_decel = max_decel_mps2 or max_accel_mps2
        # With no limit set, a phase reaches its speed at once — the old
        # behaviour, and still the default.
        accel_limit = max_accel_mps2 if max_accel_mps2 > 0 else 1e9
        decel_limit = max_decel if max_decel > 0 else 1e9

        # --- step it ---------------------------------------------------
        phase_t = 0.0
        phase_m = 0.0
        phase_Wh = 0.0
        steps = 0
        max_steps = 2_000_000
        bucket = ("hover_Wh" if kind in ("hover", "climb", "descend")
                  else "transition_Wh" if kind == "transition" else "cruise_Wh")

        while steps < max_steps:
            steps += 1
            v_prev = state["v"]

            # The last step of a phase is TRUNCATED so the leg ends exactly
            # on its goal. Without this a distance leg overshoots by up to
            # one step's worth of ground — 5001.5 m for a 5000 m leg — which
            # is small per leg and compounds across a survey.
            step_dt = dt
            if duration_goal is not None:
                step_dt = min(step_dt, max(duration_goal - phase_t, 0.0))
            if distance_goal is not None:
                v_peek, _a = core.ramp_speed(v_prev, target_v, step_dt,
                                             accel_limit, decel_limit)
                ground_peek = max(core.groundspeed_along_track_mps(
                    v_peek, head, cross), 0.0)
                if ground_peek > 1e-9:
                    remaining_m = max(distance_goal - phase_m, 0.0)
                    step_dt = min(step_dt, remaining_m / ground_peek)
            if step_dt <= 1e-12:
                break

            v_next, _accel = core.ramp_speed(v_prev, target_v, step_dt,
                                             accel_limit, decel_limit)

            point = power_at_airspeed(cfg, max(v_next, 0.0))
            power = float(point["total_power_W"])

            # Climbing lifts the aircraft AND whatever download the rotors
            # are still pushing onto it; descending gives nothing back,
            # because a propeller is a poor brake and pretending otherwise
            # would flatter the endurance.
            if climb_rate > 0:
                thrust = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
                power += thrust * climb_rate / max(cfg.esc_efficiency, 1e-9)

            # The kinetic cost of the speed change, as mechanical power
            # through the ESC. Decelerating releases it, scaled by regen_eff.
            kinetic = core.kinetic_power_term_W(
                cfg.all_up_weight_g, v_prev, v_next, step_dt, regen_eff=regen_eff)
            power = max(power + kinetic / max(cfg.esc_efficiency, 1e-9), 0.0)

            # Pack loss, charged against the CELLS rather than only shown in
            # the Power Budget. This is what time-stepping buys: the terminal
            # power above is what the aircraft needs, but the cells also have
            # to cover their own I^2 R, and the current that causes it rises
            # as the pack empties and its voltage falls.
            soc = min(max(remaining_Wh / max(usable_Wh, 1e-9), 0.0), 1.0)
            cell_v = (float(np.interp(soc, cfg.battery.soc_bp,
                                      cfg.battery.ocv_cell_bp))
                      if cfg.battery.soc_bp else cfg.battery.vnom_cell)
            pack_v = max(cell_v * cfg.battery.series_cells, 1e-6)
            pack_I = power / pack_v
            r_scale = (float(np.interp(soc, cfg.battery.soc_bp,
                                       cfg.battery.r_scale_bp))
                       if getattr(cfg.battery, "r_scale_bp", None) else 1.0)
            power += pack_I * pack_I * cfg.battery.pack_resistance * r_scale

            ground = max(core.groundspeed_along_track_mps(v_next, head, cross), 0.0)
            step_m = ground * step_dt
            if distance_goal is not None:
                # The truncated step is sized from a PEEK at the speed, and
                # the speed actually reached differs by a hair, so the last
                # step can still run a few millimetres past the mark. Clamp
                # it: a leg must cover the distance it was asked for, not
                # that plus rounding.
                step_m = min(step_m, max(distance_goal - phase_m, 0.0))
            step_Wh = power * step_dt / 3600.0

            # Altitude moves at the commanded rate, stopping at the target.
            if climb_rate > 0:
                state["alt"] = min(state["alt"] + climb_rate * step_dt, alt_target)
            elif climb_rate < 0:
                state["alt"] = max(state["alt"] + climb_rate * step_dt, alt_target)

            state["v"] = v_next
            state["t"] += step_dt
            state["dist"] += step_m
            remaining_Wh -= step_Wh
            phase_t += step_dt
            phase_m += step_m
            phase_Wh += step_Wh
            _record(phase.name, kind, v_next, state["alt"], power, point)

            if remaining_Wh < 0:
                depleted = True
                break
            if distance_goal is not None:
                if phase_m >= distance_goal - 1e-9:
                    break
            elif duration_goal is not None and phase_t >= duration_goal - 1e-9:
                break
            elif duration_goal is None and distance_goal is None:
                break

        # A phase ends on its own terms, so the altitude lands on target even
        # if the commanded rate would not quite have got there.
        state["alt"] = alt_target

        totals["time_s"] += phase_t
        totals["distance_m"] += phase_m
        totals["energy_Wh"] += phase_Wh
        totals[bucket] += phase_Wh

        status = "OK"
        if remaining_Wh < 0:
            status = "BATTERY DEPLETED"
        elif remaining_Wh < reserve_Wh:
            status = "RESERVE VIOLATION"
        # A leg can now END before reaching its commanded speed, rather than
        # overshooting its own distance to get there. Saying so is the point:
        # the pattern is not flyable at that acceleration limit.
        if abs(state["v"] - target_v) > 0.5 and not depleted:
            status = (f"{status} — could not reach {target_v:.0f} m/s within "
                      f"this leg (ended at {state['v']:.0f} m/s)")

        results.append((phase.name, phase_t / 60.0, phase_m / 1000.0,
                        phase_Wh * 3600.0 / max(phase_t, 1e-9), phase_Wh, status))
        if depleted:
            break

    totals["remaining_Wh"] = remaining_Wh
    totals["reserve_Wh"] = reserve_Wh
    totals["series"] = series
    worst["reserve_margin_Wh"] = worst["remaining_Wh"] - reserve_Wh
    totals["worst"] = worst
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
    p.add_argument("--battery_soc_model", type=str, default="auto",
                   help="auto | linear | a chemistry name. 'linear' turns the "
                        "discharge curve off: voltage is held at full charge "
                        "all flight, which is optimistic near the end of the "
                        "pack.")
    p.add_argument("--battery_unit_mode", type=str, default="cell",
                   choices=["cell", "pack"],
                   help="cell: series/parallel count CELLS. pack: they count "
                        "finished PACKS, each of --battery_cells_series_per_pack "
                        "cells in series.")
    p.add_argument("--battery_cells_series_per_pack", type=int, default=1)
    p.add_argument("--battery_cells_parallel_per_pack", type=int, default=1)
    p.add_argument("--battery_pack_capacity", type=float, default=None,
                   help="Capacity of ONE pack (mAh). Pack mode only.")
    p.add_argument("--battery_pack_weight_g", type=float, default=None,
                   help="Weight of ONE pack (g). Pack mode only.")
    p.add_argument("--battery_energy_density", type=float, default=None,
                   help="Wh/kg. Reported only; blank derives it from the "
                        "entered weight and capacity.")
    p.add_argument("--battery_a_cont", type=float, default=None,
                   help="Continuous discharge limit in amps. Outranks the "
                        "C-rating when both are given.")
    p.add_argument("--battery_a_max", type=float, default=None,
                   help="Burst discharge limit in amps.")
    p.add_argument("--battery_charge_current", type=float, default=None,
                   help="Maximum charge current (A), for the charge-time "
                        "estimate. Not a flight limit.")
    p.add_argument("--battery_soc_bp", type=str, default=None,
                   help="State-of-charge breakpoints, 0..1, comma separated. "
                        "With the two arrays below this defines a discharge "
                        "curve by hand and outranks a CSV or a preset.")
    p.add_argument("--battery_ocv_cell_bp", type=str, default=None,
                   help="Open-circuit volts PER CELL at each breakpoint.")
    p.add_argument("--battery_r_scale_bp", type=str, default=None,
                   help="Resistance multiplier at each breakpoint (1.0 = the "
                        "entered cell resistance).")

    p.add_argument("--avionics_power", type=float, default=15.0)
    p.add_argument("--peripheral_current", type=float, default=0.0,
                   help="Current drawn straight from the pack by loads that "
                        "do not sit behind a regulated rail (A). Adds to the "
                        "avionics figure rather than replacing it.")
    p.add_argument("--esc_efficiency", type=float, default=0.96)
    p.add_argument("--esc_weight_g", type=float, default=0.0,
                   help="Mass of ONE ESC (g), for the weight budget. One is "
                        "counted per driven rotor.")

    p.add_argument("--cruise_speed", type=float, default=22.0)
    p.add_argument("--altitude", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--pressure", type=float, default=None,
                   help="Static pressure (Pa). Overrides the value the "
                        "standard atmosphere derives from altitude.")
    p.add_argument("--mission", type=str, default=None)
    p.add_argument("--wind", type=float, default=0.0,
                   help="Steady wind speed (m/s).")
    p.add_argument("--wind_direction", type=float, default=0.0,
                   help="Direction the wind comes FROM, compass degrees.")
    p.add_argument("--course_deg", type=float, default=0.0,
                   help="Heading the fixed-speed run flies, compass degrees. "
                        "With --wind and --wind_direction this sets the head "
                        "and cross components, and so the groundspeed and "
                        "range. Missions ignore it: each leg carries its own "
                        "course.")
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


def _parse_float_list(text) -> Optional[List[float]]:
    """
    "0,0.25,0.5,1" into [0.0, 0.25, 0.5, 1.0], or None when blank.

    Returns None rather than [] for an unparseable list, so a typo falls
    back to the chemistry preset instead of handing the resolver an empty
    curve it would treat as valid.
    """
    if text in (None, ""):
        return None
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            return None
    return out or None


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
        unit_mode=args.battery_unit_mode,
        cells_series_per_unit=args.battery_cells_series_per_pack,
        cells_parallel_per_unit=args.battery_cells_parallel_per_pack,
        pack_capacity_mAh=args.battery_pack_capacity,
        pack_weight_g=args.battery_pack_weight_g,
        energy_density_Wh_per_kg=args.battery_energy_density,
        discharge_cont_A=args.battery_a_cont,
        discharge_max_A=args.battery_a_max,
        charge_current_max_A=args.battery_charge_current,
        soc_bp=_parse_float_list(args.battery_soc_bp),
        ocv_cell_bp=_parse_float_list(args.battery_ocv_cell_bp),
        r_scale_bp=_parse_float_list(args.battery_r_scale_bp),
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
        periph_current_A=args.peripheral_current,
        esc_efficiency=args.esc_efficiency,
        esc_weight_g=args.esc_weight_g,
        air_density=core.air_density(args.altitude, args.temperature,
                                     args.pressure),
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


def _print_single_point(cfg: VTOLConfig, wind_mps: float = 0.0,
                        wind_direction_deg: float = 0.0,
                        course_deg: float = 0.0) -> None:
    m = compute_metrics(cfg, wind_mps=wind_mps,
                        wind_direction_deg=wind_direction_deg,
                        course_deg=course_deg)
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
    if m["wind_mps"] > 0:
        print(f"  Wind                 : {m['wind_mps']:.1f} m/s from "
              f"{m['wind_direction_deg']:.0f} deg, course "
              f"{m['course_deg']:.0f} deg")
        print(f"  Head/cross wind      : {m['wind_head_mps']:+.2f} / "
              f"{m['wind_cross_mps']:+.2f} m/s")
        print(f"  Ground speed         : {m['groundspeed_mps']:.2f} m/s "
              f"(still-air range {m['cruise_range_still_air_km']:.2f} km)")
        if not m["course_holdable"]:
            print("  WARNING: the crosswind meets or exceeds the airspeed — "
                  "this course cannot be held.")
    print(f"  Station keeping      : {m['station_power_W']:.0f} W "
          f"({m['station_regime']}), {m['station_endurance_min']:.1f} min")
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
            _print_single_point(cfg, wind_mps=args.wind,
                                wind_direction_deg=args.wind_direction,
                                course_deg=args.course_deg)
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
    root.rowconfigure(1, weight=1)

    left = ttk.Frame(root, padding=6)
    left.grid(row=1, column=0, sticky="ns")
    left.rowconfigure(0, weight=1)
    right = ttk.Frame(root, padding=6)
    right.grid(row=1, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(0, weight=1)

    # ---- configuration type -----------------------------------------
    # The header spans the whole window rather than sitting inside the input
    # pane. Inside it, its natural width became the pane's width: adding the
    # "Hover any ? for help." hint and the Config filename pushed the input
    # column from ~410 px to over 1100, stretching every entry field across
    # the window. As a full-width row it can be as long as it likes, and a
    # long filename can never squeeze the inputs.
    type_bar = ttk.Frame(root, padding=(6, 4, 6, 0))
    type_bar.grid(row=0, column=0, columnspan=2, sticky="w")
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
    # The second sentence is not decoration: 66 of the 68 input fields carry a
    # tooltip, and without this line nothing on screen says so.
    mode_note = ttk.Label(mode_bar, foreground="#555555", font=("TkDefaultFont", 8),
                          text="Simple hides advanced tuning inputs. "
                               "Hover any ? for help.")
    mode_note.pack(side="left", padx=8)

    # Which config is loaded, in the header where the other two simulators put
    # it. Without this there is nothing on screen naming the loaded file.
    v_loaded_cfg = tk.StringVar(value="(no config loaded)")
    ttk.Label(mode_bar, text="Config:", foreground="#666666",
              font=("TkDefaultFont", 8)).pack(side="left", padx=(12, 2))
    ttk.Label(mode_bar, textvariable=v_loaded_cfg, foreground="#0B6BCB",
              font=("TkDefaultFont", 8, "bold")).pack(side="left")

    def _apply_field_mode(*_a):
        simple = (v_ui_mode.get() == "Simple")
        visible = set()
        for row in _field_rows:
            show = (not simple) or (row["key"] in VTOL_SIMPLE_FIELDS)
            if show:
                visible.add(row["key"])
            for widget in row["widgets"]:
                try:
                    widget.grid() if show else widget.grid_remove()
                except Exception:
                    pass
        # A heading whose every field is hidden has nothing left to head.
        for section in _section_rows:
            show = bool(section["keys"] & visible) if section["keys"] else True
            for widget in section["widgets"]:
                try:
                    widget.grid() if show else widget.grid_remove()
                except Exception:
                    pass
    v_ui_mode.trace_add("write", _apply_field_mode)

    nb = ttk.Notebook(left)
    nb.grid(row=0, column=0, columnspan=3, sticky="nsew")

    # Scrollable input tabs, matching make_scrollable_tab in the other two
    # simulators. A bare Frame made the bottom fields of Airframe and
    # Mission/Env unreachable on a short window — there was no way to get at
    # them at all.
    # Keyed by the tab's own frame rather than held in a list: the tabs are
    # reordered once they all exist, so a positional index would scroll
    # whichever tab happened to be built in that slot, not the one on screen.
    _tab_canvases: dict = {}

    def make_tab(title):
        """
        Wrap a notebook tab in a Canvas + vertical Scrollbar.

        Two details that are easy to get wrong, both learned in the
        multicopter:

          1. Bind <Configure> on the CANVAS, not just the inner frame. The
             canvas one fires when the tab is first shown, which is the only
             moment winfo_width() stops returning the startup dummy value of
             1 px. Without it the inner frame renders one pixel wide and the
             tab looks blank until you switch away and back.
          2. Do NOT call bind_all for the wheel here. One binding per tab
             stacks N handlers onto every widget in the application and
             stalls the event loop. A single notebook-level binding is
             attached once, after all tabs exist.
        """
        outer = ttk.Frame(nb)
        nb.add(outer, text=title)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        cv = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        inner = ttk.Frame(cv, padding=6)
        cv.configure(yscrollcommand=sb.set)
        cv.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        win_id = cv.create_window((0, 0), window=inner, anchor="nw")

        def _resize_inner(width):
            if width > 1:              # ignore the startup 1-px dummy value
                cv.itemconfig(win_id, width=width)

        def _on_inner_configure(_evt):
            cv.configure(scrollregion=cv.bbox("all"))
            _resize_inner(cv.winfo_width())

        def _on_canvas_configure(evt):
            cv.configure(scrollregion=cv.bbox("all"))
            _resize_inner(evt.width)

        inner.bind("<Configure>", _on_inner_configure)
        cv.bind("<Configure>", _on_canvas_configure)
        inner.columnconfigure(1, weight=1)
        _tab_canvases[str(outer)] = cv
        return inner

    tab_airframe = make_tab("Airframe")
    tab_lift = make_tab("Lift Rotors")
    tab_cruise = make_tab("Cruise")
    tab_batt = make_tab("Battery")
    tab_env = make_tab("Mission/Environment")

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

    def _append_row(parent, label, key, default="", help_text=""):
        """add_row at the next free grid row of an already-populated tab."""
        return add_row(parent, parent.grid_size()[1], label, key, default, help_text)

    def add_combo_row(parent, label, key, values, default, help_text=""):
        """
        A row whose input is a fixed choice rather than free text.

        A typed "Pack" or "PACk" is a silent wrong answer — the battery
        falls back to cell mode and the capacity halves with nothing on
        screen to say why. A dropdown makes that unreachable.
        """
        row = parent.grid_size()[1]
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(default))
        fields[key] = var
        box = ttk.Combobox(parent, textvariable=var, state="readonly",
                           values=list(values), width=16)
        box.grid(row=row, column=1, sticky="w", padx=(8, 4))
        widgets = [label_widget, box]
        if help_text:
            marker = ttk.Label(parent, text="?", foreground="#0B6BCB",
                               cursor="question_arrow")
            marker.grid(row=row, column=2, sticky="w")
            core.Tooltip(marker, help_text)
            widgets.append(marker)
        _field_rows.append({"key": key, "widgets": widgets})
        return var

    # Section headings carry the keys they head, so a heading whose every
    # field is hidden in Simple mode hides with them rather than standing
    # over nothing.
    _section_rows = []

    def add_section(parent, title, keys=(), row=None):
        """
        A separator and heading over the rows that follow.

        `row` is explicit where the caller is tracking its own row counter
        and implicit (next free row) where it is not — the two waves of
        fields on each tab are built both ways, and mixing an explicit
        counter with grid_size() silently stacks widgets on top of
        each other.
        """
        if row is None:
            row = parent.grid_size()[1]
        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        lab = ttk.Label(parent, text=f"—— {title} ——", foreground="#777777")
        lab.grid(row=row + 1, column=0, columnspan=3, sticky="w")
        _section_rows.append({"keys": set(keys), "widgets": [sep, lab]})
        return row + 2

    r = 0
    r = add_row(tab_airframe, r, "Airframe weight (g)", "weight", 6000,
                "Everything except the payload, including battery and motors.")
    r = add_row(tab_airframe, r, "Payload mass (g)", "payload", 0,
                "Added on top of the airframe weight.")
    r = add_section(tab_airframe, "Wing Geometry", ("span", "area"), row=r)
    r = add_row(tab_airframe, r, "Wing span (m)", "span", 2.4, "Tip to tip.")
    r = add_row(tab_airframe, r, "Wing area (m²)", "area", 0.60,
                "Planform area. With span this sets the aspect ratio.")
    r = add_section(tab_airframe, "Aerodynamics",
                    ("cd0", "oswald", "clmax", "clcruise"), row=r)
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
    r = add_row(tab_lift, r, "Lift prop pitch (in)", "lift_p", 6,
                    "How far one revolution would advance in a solid. Barely matters "
                    "in hover, where the rotor is working against still air — it "
                    "matters for a rotor that also cruises, which is why the vectored "
                    "types care and lift+cruise does not.")
    r = add_section(tab_lift, "Lift Motor",
                    ("lift_kv", "lift_rm", "lift_wt"), row=r)
    r = add_row(tab_lift, r, "Lift motor Kv", "lift_kv", 300,
                    "RPM per volt, unloaded. Low Kv on a big rotor is the efficient "
                    "combination; high Kv on a small one buys responsiveness at the "
                    "cost of endurance.")
    r = add_row(tab_lift, r, "Lift motor Rm (ohm)", "lift_rm", 0.08,
                    "Winding resistance of one lift motor. Drives the copper loss, "
                    "which grows with the SQUARE of current — so it bites hardest in "
                    "hover.")
    r = add_row(tab_lift, r, "Lift motor weight (g)", "lift_wt", 200,
                    "Mass of one lift motor, for the Weight Budget. On a VTOL the "
                    "lift motors are dead weight in cruise, so this is a real cruise "
                    "penalty too.")
    r = add_section(tab_lift, "Rotor Efficiency",
                    ("fom", "stopped_area"), row=r)
    r = add_row(tab_lift, r, "Figure of merit", "fom", 0.65,
                "Hover efficiency of the rotor against ideal momentum theory.\n"
                "0.6-0.7 typical; 0.75+ is a very good rotor.")
    r = add_row(tab_lift, r, "Stopped rotor drag area (m²)", "stopped_area", "",
                "Flat-plate area of the stopped rotors in cruise.\n"
                "Blank estimates it from blade planform.")

    r = 0
    r = add_row(tab_cruise, r, "Number of cruise motors", "n_cruise", 1,
                    "Motors driving forward flight. Ignored for tiltrotor, tiltwing "
                    "and tailsitter, whose lift rotors tilt and do the cruising "
                    "themselves.")
    r = add_row(tab_cruise, r, "Cruise prop diameter (in)", "cruise_d", 14,
                    "Larger is more efficient at a given thrust, but has to clear the "
                    "airframe and adds drag when stopped. Ignored for the vectored "
                    "types.")
    r = add_row(tab_cruise, r, "Cruise prop pitch (in)", "cruise_p", 8,
                    "Matters far more here than on a lift rotor: a cruise propeller "
                    "spends its life advancing through air, and too little pitch caps "
                    "the speed it can reach.")
    r = add_section(tab_cruise, "Cruise Motor",
                    ("cruise_kv", "cruise_rm", "cruise_wt"), row=r)
    r = add_row(tab_cruise, r, "Cruise motor Kv", "cruise_kv", 500,
                    "RPM per volt for the cruise motor. Usually higher than the lift "
                    "motors', because it turns a smaller propeller faster.")
    r = add_row(tab_cruise, r, "Cruise motor Rm (ohm)", "cruise_rm", 0.06,
                    "Winding resistance of one cruise motor, driving its copper loss.")
    r = add_row(tab_cruise, r, "Cruise motor weight (g)", "cruise_wt", 180,
                    "Mass of one cruise motor, for the Weight Budget.")
    r = add_section(tab_cruise, "Propeller Efficiency",
                    ("cruise_eff",), row=r)
    r = add_row(tab_cruise, r, "Cruise prop efficiency", "cruise_eff", 0.75,
                "Combined motor and propeller efficiency in cruise.")

    r = 0
    # A dropdown, not free text: only three chemistries have a preset curve,
    # and anything else falls through to the linear fallback silently. Typing
    # "LiPO" got you a different discharge model with nothing on screen to
    # say so.
    add_combo_row(tab_batt, "Chemistry", "chem",
                  ("LiPo", "Li-ion", "LiFePO4"), "LiPo",
                  "Selects the state-of-charge curve when the SoC model "
                  "below is set to auto.\n\n"
                  "These three are the chemistries with a stored preset. "
                  "Anything else has no curve to fall back on.")
    r = tab_batt.grid_size()[1]
    r = add_row(tab_batt, r, "Cell capacity (mAh)", "cell_cap", 5000,
                    "Capacity of ONE cell. Total pack capacity is this times the "
                    "parallel count; series cells raise voltage, not capacity.")
    r = add_row(tab_batt, r, "Series units", "series", 6,
                "Sets pack voltage. Counts CELLS in cell mode, finished "
                "PACKS in pack mode.")
    r = add_row(tab_batt, r, "Parallel units", "parallel", 2,
                "Sets pack capacity. Counts CELLS in cell mode, finished "
                "PACKS in pack mode.")
    r = add_row(tab_batt, r, "Cell weight (g)", "cell_wt", 120,
                    "Mass of one cell. Usually the largest single line in the Weight "
                    "Budget — on this class of aircraft the pack is a quarter of the "
                    "all-up weight.")
    r = add_row(tab_batt, r, "Cell V min", "vmin", 3.3,
                    "Lowest voltage per cell you are willing to reach. Sets usable "
                    "energy together with the discharge percentage; going lower buys "
                    "endurance at the cost of cycle life.")
    r = add_row(tab_batt, r, "Cell V nominal", "vnom", 3.7,
                    "Average cell voltage over a discharge, used for pack energy. 3.7 "
                    "V for LiPo, about 3.6-3.7 V for Li-ion.")
    r = add_row(tab_batt, r, "Cell V max", "vmax", 4.2,
                    "Fully charged cell voltage. 4.2 V for standard LiPo and Li-ion; "
                    "4.35 V for the high-voltage chemistries DJI and others use.")
    r = add_row(tab_batt, r, "Cell resistance (mOhm)", "rcell", 4.0,
                    "Internal resistance of ONE cell. Causes voltage sag under load "
                    "and I^2 R heating, and it rises as the pack empties — which is "
                    "why a long cruise ends drawing more than it began.")
    r = add_row(tab_batt, r, "Usable percent", "usable", 80,
                "Fraction of pack energy you are willing to use.")

    # ---- pack-level entry ---------------------------------------------
    # Most real aircraft are built from finished packs, not loose cells, and
    # the datasheet quotes the PACK. Forcing a per-cell figure meant dividing
    # by the cell count by hand before typing anything in.
    add_section(tab_batt, "Pack Entry (optional)",
                ("unit_mode", "cells_s_per_pack", "cells_p_per_pack",
                 "pack_cap", "pack_wt", "energy_density"))
    add_combo_row(tab_batt, "Entry mode", "unit_mode", ("cell", "pack"), "cell",
                  "cell: the fields above describe ONE CELL, and Series / "
                  "Parallel count cells.\n\n"
                  "pack: they describe ONE FINISHED PACK, Series / Parallel "
                  "count packs, and the pack's own cell count comes from "
                  "'Cells in series per pack' below.")
    _append_row(tab_batt, "Cells in series per pack", "cells_s_per_pack", "1",
                "The 'S' number of one pack: a 6S pack is 6. Pack mode only — "
                "this is what turns a pack count into a cell count, and so "
                "into a voltage.")
    _append_row(tab_batt, "Cells in parallel per pack", "cells_p_per_pack", "1",
                "The 'P' number of one pack. Usually 1. Pack mode only.")
    _append_row(tab_batt, "Pack capacity (mAh)", "pack_cap", "",
                "Capacity of ONE pack. Pack mode only. Packs in PARALLEL add "
                "capacity; packs in series raise voltage instead.")
    _append_row(tab_batt, "Pack weight (g)", "pack_wt", "",
                "Weight of ONE pack. Pack mode only. Every pack is carried "
                "however it is wired, so this is multiplied by series x "
                "parallel.")
    _append_row(tab_batt, "Energy density (Wh/kg)", "energy_density", "",
                "Reported only — nothing is computed from it. Blank derives "
                "it from the weight and capacity you entered, so a datasheet "
                "figure typed here can be compared against what your numbers "
                "actually imply.\n\n"
                "Typical: 150-200 for LiPo, 200-260 for Li-ion.")

    # ---- limits in amps ------------------------------------------------
    add_section(tab_batt, "Current Limits (optional)",
                ("a_cont", "a_max", "c_cont", "c_max", "charge_a"))
    _append_row(tab_batt, "Cont discharge (A)", "a_cont", "",
                "Continuous current the pack is rated for, in amps. Outranks "
                "the C-rate below when both are given, because a datasheet "
                "quoting both means the amps.\n\nBlank leaves the check "
                "unrated rather than inventing a limit.")
    _append_row(tab_batt, "Max discharge (A)", "a_max", "",
                "Burst current rating, in amps. Outranks the C-rate below.")
    _append_row(tab_batt, "Max charge current (A)", "charge_a", "",
                "Charge rating, used for the charge-time estimate on the "
                "Metrics tab. Not a flight limit — nothing in a run is "
                "checked against it.")

    r = 0
    r = add_row(tab_env, r, "Cruise speed (m/s)", "cruise_v", 22,
                    "The speed the fixed-speed results are evaluated at. Must be "
                    "above the transition speed, or the rotors are still carrying "
                    "part of the weight and Status will say so.")
    r = add_row(tab_env, r, "Altitude (m)", "alt", 0,
                    "Sets air density. Thinner air means a faster rotor downwash for "
                    "the same thrust, so hover costs more — the reason VTOL "
                    "performance falls off with field elevation.")
    r = add_row(tab_env, r, "Temperature (°C)", "temp", "",
                "Blank uses the ISA value for the altitude.")
    r = add_row(tab_env, r, "Pressure (Pa)", "pressure", "",
                "Static pressure, if you have a barometer reading. Blank "
                "derives it from the altitude using the standard "
                "atmosphere.\n\n"
                "Worth entering on a day well away from standard: the ISA "
                "is an average, and density drives every rotor and wing "
                "number on this aircraft. Sea-level standard is 101325 Pa.")
    r = add_section(tab_env, "Systems (superseded by their own tabs)",
                    ("avionics", "esc_eff"), row=r)
    r = add_row(tab_env, r, "Avionics power (W)", "avionics", 15,
                "Autopilot, radios and payload electronics.")
    r = add_row(tab_env, r, "ESC efficiency", "esc_eff", 0.96,
                    "Combined ESC and motor electrical efficiency. Motor losses are "
                    "folded in here because this model has no separate motor "
                    "electrical model, which is why it sits lower than an ESC "
                    "datasheet figure.")
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

    # Mass entry mode, as the other two have. Structure is normally the
    # residual once the itemised parts come out of the airframe weight; this
    # lets you work the other way when you know the structure and want the
    # all-up weight derived.
    _mass_row = tab_airframe.grid_size()[1]
    ttk.Label(tab_airframe, text="Mass Entry Mode").grid(
        row=_mass_row, column=0, sticky="w", pady=2)
    v_mass_mode = tk.StringVar(value="derive structure")
    fields["mass_mode"] = v_mass_mode
    ttk.Combobox(tab_airframe, textvariable=v_mass_mode, state="readonly",
                 width=18, values=("derive structure", "enter structure")).grid(
        row=_mass_row, column=1, sticky="w", padx=(8, 4))
    _mm = ttk.Label(tab_airframe, text="?", foreground="#0B6BCB",
                    cursor="question_arrow")
    _mm.grid(row=_mass_row, column=2, sticky="w")
    core.Tooltip(_mm,
                 "derive structure: you enter the airframe weight, and the "
                 "Weight Budget shows structure as whatever is left after the "
                 "battery, motors, propellers and avionics.\n\n"
                 "enter structure: you enter the structure mass below, and "
                 "the airframe weight is the sum of that plus the itemised "
                 "parts. Useful when you know the frame but are still "
                 "choosing components.")
    _append_row(tab_airframe, "Structure mass (g)", "structure_mass", "",
                "Bare frame, booms, skins and fasteners — everything that is "
                "not a component listed elsewhere. Used only in "
                "'enter structure' mode.")

    _append_row(tab_airframe, "Avionics mass (g)", "avionics_mass", "",
                "Autopilot, radios, GPS and payload electronics. Shown as its "
                "own line in the Weight Budget; already inside the airframe "
                "weight, so it does not change the flying weight.")
    add_section(tab_lift, "Optional Detail",
                ("lift_prop_wt", "lift_pmax", "lift_max_thrust", "lift_imax",
                 "download", "lift_table"))
    _append_row(tab_lift, "Lift prop weight (g, each)", "lift_prop_wt", "",
                "For the Weight Budget.")
    _append_row(tab_lift, "Lift motor max power (W)", "lift_pmax", "",
                "Rated power per motor. Status checks hover — the heaviest "
                "load these motors see — against it.")
    _append_row(tab_lift, "Lift prop rated thrust (g)", "lift_max_thrust", "",
                "Static thrust ONE lift propeller is rated for, from its "
                "bench data. The Airframe Diagram tab uses it to show how "
                "much margin each rotor has in hover. Blank skips the check.")
    _append_row(tab_lift, "Lift motor max current (A)", "lift_imax", "",
                "Rated continuous current per lift motor. Status checks the "
                "current in HOVER against it — hover is the heaviest steady "
                "load a VTOL's lift motors ever see.")
    _append_row(tab_lift, "Hover download fraction", "download", "",
                "Extra hover thrust needed because the rotor wash strikes the "
                "airframe below, as a fraction of weight.\n"
                "Blank uses the type default: tiltrotor 0.10, tiltwing 0.02, "
                "tailsitter 0.02. Typical published values, not measurements — "
                "override with test data if you have it.\n\n"
                "Not modelled for lift+cruise: its hover and transition are "
                "computed by separate branches, so a download applied to one "
                "and not the other would put a step at zero airspeed.")
    add_section(tab_cruise, "Optional Detail",
                ("cruise_prop_wt", "cruise_pmax", "cruise_imax",
                 "cruise_table"))
    _append_row(tab_cruise, "Cruise prop weight (g, each)", "cruise_prop_wt", "",
                "For the Weight Budget.")
    _append_row(tab_cruise, "Cruise motor max power (W)", "cruise_pmax", "",
                "Rated power per cruise motor. Ignored for tiltrotor, tiltwing "
                "and tailsitter, whose lift rotors do the cruise work.")

    _append_row(tab_cruise, "Cruise motor max current (A)", "cruise_imax", "",
                "Rated continuous current per cruise motor. Ignored for the "
                "vectored types, whose lift rotors do the cruising.")

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
        ttk.Button(holder, text="Browse…", width=9, command=browse).pack(side="left")
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
    add_section(tab_env, "Transients (mission runs)",
                ("accel", "decel", "regen"))
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
    add_section(tab_env, "Wind", ("wind", "wind_dir"))
    _append_row(tab_env, "Wind speed (m/s)", "wind", "0",
                "Steady wind. Power follows AIRSPEED, but progress follows "
                "GROUNDSPEED, so a leg measured over the ground takes longer "
                "into a headwind and costs more energy for the same track.\n\n"
                "Hovering in wind is not free either: to hold station the "
                "aircraft must fly at the wind speed through the air.")
    _append_row(tab_env, "Wind direction FROM (deg)", "wind_dir", "0",
                "Meteorological convention — the direction the wind blows "
                "FROM. A north wind (0) is a headwind when flying north.\n\n"
                "In a mission each leg's own course_deg decides whether that "
                "is a head, tail or crosswind for that leg. For a fixed "
                "speed sweep, the Course heading below does.")
    _append_row(tab_env, "Course heading (deg)", "course_deg", "0",
                "The direction the fixed-speed run is flying, in compass "
                "degrees. With the wind above it decides the head and cross "
                "components.\n\n"
                "It changes the DISTANCE covered, never the power: the speed "
                "you enter is an airspeed, and power depends on airspeed "
                "alone. A headwind leaves endurance untouched and cuts "
                "range.\n\n"
                "Missions ignore this — each leg carries its own course.")
    _append_row(tab_batt, "Continuous C-rate", "c_cont", "",
                "Pack continuous discharge rating. Status checks the pack "
                "current against it. Ignored when Cont discharge (A) above "
                "is filled in.")
    _append_row(tab_batt, "Max / burst C-rate", "c_max", "",
                "Pack burst rating. Above continuous but below this is amber; "
                "above this is red. Ignored when Max discharge (A) above is "
                "filled in.")

    # ---- discharge curve -----------------------------------------------
    # The resolver behind this has always been in the shared core, and the
    # CSV picker reached it — but the model selector and the breakpoint
    # arrays had no field at all, so the only way to choose 'linear' or to
    # type a curve by hand was the CLI.
    add_section(tab_batt, "Discharge Curve (optional)",
                ("soc_model", "soc_curve", "soc_bp", "ocv_cell_bp",
                 "r_scale_bp"))
    add_combo_row(tab_batt, "SoC model", "soc_model",
                  ("auto", "linear", "LiPo", "Li-ion", "LiFePO4"), "auto",
                  "Which discharge curve the pack uses.\n\n"
                  "auto: breakpoints if given, else a CSV curve, else the "
                  "preset for the chemistry above.\n\n"
                  "linear: no curve at all. Open-circuit voltage is held at "
                  "FULL CHARGE for the whole flight, so this is OPTIMISTIC "
                  "near the end of the pack — it ignores the sag that makes "
                  "the last minutes draw more current for the same power.\n\n"
                  "A chemistry name forces that preset.")
    _table_picker(tab_batt, "SoC curve (CSV)", "soc_curve",
                  "Measured discharge curve for this pack.\n\n"
                  "Columns: SoC (0-1 or 0-100) with OCV per cell, and "
                  "optionally a resistance scale. A measured curve outranks "
                  "the chemistry preset, so the sag near the end of the pack "
                  "comes from your cells rather than a generic LiPo shape.")
    _append_row(tab_batt, "SoC breakpoints (0..1)", "soc_bp", "",
                "Comma separated, ascending, e.g. 0, 0.1, 0.5, 0.9, 1.\n\n"
                "Filled in together with the two rows below, these define "
                "the curve by hand and outrank both the CSV and the preset. "
                "All three must have the same number of entries.")
    _append_row(tab_batt, "OCV per cell (V)", "ocv_cell_bp", "",
                "Open-circuit volts of ONE cell at each breakpoint above, "
                "e.g. 3.3, 3.5, 3.75, 4.1, 4.2.\n\n"
                "The pack voltage never reports below Cell V min x series, "
                "so a breakpoint under that value is floored rather than "
                "used — set Cell V min lower if you mean to model it.")
    _append_row(tab_batt, "Resistance scale", "r_scale_bp", "",
                "Multiplier on the cell resistance at each breakpoint, e.g. "
                "1.7, 1.3, 1.0, 1.0, 1.0. A pack's resistance rises as it "
                "empties, which is why a long cruise ends drawing more than "
                "it began.")

    # ---- Plot Settings tab ---------------------------------------------
    # The other two simulators let you set how far the speed sweep runs.
    # Without it the VTOL always plotted to a fixed multiple of cruise speed,
    # which is wrong for an aircraft being pushed past its design point.
    tab_plotset = make_tab("Plot Settings")
    ttk.Label(tab_plotset,
              text="Controls the Fixed Speed Plots only. Nothing here changes "
                   "a computed result.",
              wraplength=300, justify="left", foreground="#555555"
              ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
    _append_row(tab_plotset, "Max speed for plots (m/s)", "plot_vmax", "",
                "How far the speed sweep runs. Blank uses 1.6x the larger of "
                "the cruise and transition speeds, which suits most aircraft "
                "but clips the curve on one being flown well past its design "
                "point.")

    # ---- ESC tab -------------------------------------------------------
    # The VTOL had a single "ESC efficiency" buried on the Mission/Env tab.
    # The other two simulators give the ESC its own tab, because its
    # resistance and current limit are checkable design constraints, not a
    # fudge factor.
    tab_esc = make_tab("ESC")
    ttk.Label(tab_esc, text="Efficiency is the headline number; resistance and "
              "the current limit let Status check the ESC against what "
              "actually flows through it.",
              wraplength=300, justify="left", foreground="#555555"
              ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
    _append_row(tab_esc, "ESC efficiency", "esc_eff_tab", "0.96",
                "Combined switching and conduction efficiency, one ESC.\n\n"
                "Motor losses are folded in here too, because the VTOL model "
                "has no separate motor electrical model — which is why this "
                "sits lower than an ESC datasheet figure would.")
    _append_row(tab_esc, "ESC resistance (Ω)", "esc_r", "",
                "Optional. Series resistance of one ESC, for the loss "
                "breakdown. Blank leaves it out entirely.")
    _append_row(tab_esc, "ESC max current (A)", "esc_imax", "",
                "Optional. Continuous rating of one ESC. Status checks the "
                "current each ESC actually carries in HOVER against it, "
                "because hover is the heaviest steady load a VTOL sees.")
    _append_row(tab_esc, "ESC weight (g, each)", "esc_wt", "",
                "Mass of ONE ESC. Counted once per lift rotor, and once per "
                "cruise motor on a lift+cruise.\n\n"
                "Blank leaves ESCs out of the Weight Budget, which is what "
                "it did before this field existed — their mass was quietly "
                "rolled into the structure residual instead of being "
                "itemised.")

    # ---- Avionics tab --------------------------------------------------
    # Replaces a single flat watts figure with the rail editor the other two
    # simulators have. A rail's converter loss is real: 5 V at 2 A behind a
    # 90% BEC costs 11.1 W at the pack, not 10.
    tab_avionics = make_tab("Avionics")
    ttk.Label(tab_avionics,
              text="Add a row per regulated rail. Any rails entered REPLACE "
                   "the flat figure below — they describe the same load in "
                   "more detail, and counting both would double it.\n"
                   "Double-click any cell to edit it in-place.",
              wraplength=300, justify="left", foreground="#555555"
              ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

    _AV_COLS = ("volts", "amps", "eff")
    av_tree = ttk.Treeview(tab_avionics, columns=_AV_COLS, show="headings",
                           height=6, selectmode="browse")
    for _c, _h, _w in (("volts", "Rail Voltage (V)", 110),
                       ("amps", "Rail Current (A)", 110),
                       ("eff", "BEC Efficiency (0–1]", 130)):
        av_tree.heading(_c, text=_h)
        av_tree.column(_c, width=_w, anchor="center", stretch=True)
    av_tree.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 4))

    # One horizontal row rather than three stacked ones, matching the other
    # two simulators: three labelled boxes that are filled in together read
    # as one action, which is what adding a rail is.
    _av_entry = {}
    _av_form = ttk.Frame(tab_avionics)
    _av_form.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 2))
    for _key, _label, _default, _w in (("av_v", "Voltage (V):", "5.0", 8),
                                       ("av_a", "Current (A):", "2.0", 8),
                                       ("av_e", "Efficiency:", "0.90", 7)):
        ttk.Label(_av_form, text=_label).pack(side="left", padx=(0, 3))
        _v = tk.StringVar(value=_default)
        _av_entry[_key] = _v
        ttk.Entry(_av_form, textvariable=_v, width=_w).pack(side="left",
                                                            padx=(0, 10))
    _av_row = 3

    # The rails themselves live in one hidden field so they save and load
    # with everything else, as "V:A:eff, V:A:eff".
    fields["avionics_rails"] = tk.StringVar(value="")

    def _av_refresh():
        for _iid in av_tree.get_children():
            av_tree.delete(_iid)
        for part in fields["avionics_rails"].get().split(","):
            part = part.strip()
            if not part:
                continue
            bits = part.split(":")
            if len(bits) == 3:
                av_tree.insert("", "end", values=tuple(b.strip() for b in bits))

    def _av_store():
        rows = [":".join(str(x) for x in av_tree.item(i, "values"))
                for i in av_tree.get_children()]
        fields["avionics_rails"].set(", ".join(rows))

    def _av_add():
        try:
            volts = float(_av_entry["av_v"].get())
            amps = float(_av_entry["av_a"].get())
            eff = float(_av_entry["av_e"].get())
        except ValueError:
            messagebox.showerror("Avionics rail",
                                 "Voltage, current and efficiency must all be numbers.")
            return
        if volts <= 0 or amps < 0 or not (0 < eff <= 1):
            messagebox.showerror(
                "Avionics rail",
                "Voltage must be positive, current non-negative, and "
                "efficiency in (0, 1].")
            return
        # One row per voltage: adding 5 V twice replaces it rather than
        # silently drawing the load twice.
        for iid in av_tree.get_children():
            if abs(float(av_tree.item(iid, "values")[0]) - volts) < 1e-9:
                av_tree.delete(iid)
        av_tree.insert("", "end", values=(f"{volts:g}", f"{amps:g}", f"{eff:g}"))
        _av_store()

    def _av_remove():
        for iid in av_tree.selection():
            av_tree.delete(iid)
        _av_store()

    def _av_clear():
        for iid in av_tree.get_children():
            av_tree.delete(iid)
        _av_store()

    def _av_edit_cell(event):
        """
        Edit a rail in place on a double-click.

        Without this the only way to change a 2.0 A rail to 2.5 A was to
        retype all three numbers into the form and add it again — which is
        what the other two simulators' "Double-click any cell" hint exists
        to avoid.

        The entry floats over the cell being edited and writes back on
        Return or focus-out; Escape abandons the edit.
        """
        if av_tree.identify_region(event.x, event.y) != "cell":
            return
        iid = av_tree.identify_row(event.y)
        column = av_tree.identify_column(event.x)
        if not iid or not column:
            return
        index = int(column[1:]) - 1
        if not (0 <= index < len(_AV_COLS)):
            return
        bbox = av_tree.bbox(iid, column)
        if not bbox:
            return
        x, y, width, height = bbox
        var = tk.StringVar(value=str(av_tree.item(iid, "values")[index]))
        entry = ttk.Entry(av_tree, textvariable=var, justify="center")
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()
        entry.select_range(0, "end")
        done = {"closed": False}

        def commit(_e=None):
            if done["closed"]:
                return
            done["closed"] = True
            raw = var.get().strip()
            entry.destroy()
            try:
                value = float(raw)
            except ValueError:
                return                      # leave the old value alone
            # The same rules the Add button enforces: an in-place edit must
            # not be a way past the validation the form applies.
            if index == 0 and value <= 0:
                return
            if index == 1 and value < 0:
                return
            if index == 2 and not (0 < value <= 1):
                messagebox.showerror(
                    "Avionics rail",
                    "BEC efficiency must be greater than 0 and at most 1.")
                return
            # Editing a voltage onto one another row already carries would
            # give two rows for one bus, and the load would be counted twice.
            if index == 0:
                for other in av_tree.get_children():
                    if other != iid and abs(
                            float(av_tree.item(other, "values")[0]) - value) < 1e-9:
                        messagebox.showerror(
                            "Avionics rail",
                            f"A {value:g} V rail already exists. Edit that row "
                            f"instead, or remove it first.")
                        return
            values = list(av_tree.item(iid, "values"))
            values[index] = f"{value:g}"
            av_tree.item(iid, values=tuple(values))
            _av_store()

        def abandon(_e=None):
            done["closed"] = True
            entry.destroy()

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", abandon)

    av_tree.bind("<Double-1>", _av_edit_cell)

    _av_btns = ttk.Frame(tab_avionics)
    _av_btns.grid(row=_av_row, column=0, columnspan=3, sticky="w", pady=(4, 6))
    ttk.Button(_av_btns, text="➕  Add / Update Rail", command=_av_add).pack(side="left")
    ttk.Button(_av_btns, text="🗑  Remove Selected", command=_av_remove).pack(side="left", padx=6)
    ttk.Button(_av_btns, text="✖  Clear All", command=_av_clear).pack(side="left")
    _av_row += 1
    _append_row(tab_avionics, "Flat avionics power (W)", "avionics_flat", "15",
                "Used only when NO rails are entered above. A single figure "
                "for everything drawing from the pack: autopilot, radios, "
                "GPS, payload electronics.")
    _append_row(tab_avionics, "Peripheral current (A)", "periph_current", "0",
                "Current drawn STRAIGHT from the pack, at raw pack voltage, by "
                "anything that does not sit behind a regulated rail — a "
                "heater, a winch, a payload fed from the main bus.\n\n"
                "Unlike the rails this ADDS to the avionics figure rather "
                "than replacing it, because it is a different load, not the "
                "same one described in more detail.\n\n"
                "Typical: 0 on a survey aircraft; 1-5 A when a payload runs "
                "off pack voltage.")

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

    # One notebook-level wheel binding, attached after every input tab exists.
    # It routes the scroll to whichever canvas belongs to the selected tab,
    # so there is exactly one handler rather than one per tab.
    def _on_nb_mousewheel(evt):
        try:
            cv = _tab_canvases.get(str(nb.select()))
            if cv is not None:
                cv.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        except Exception:
            pass
    nb.bind("<MouseWheel>", _on_nb_mousewheel)
    # Also bind each canvas directly, so the wheel works while the pointer is
    # over the tab's contents rather than its edge.
    for _cv in _tab_canvases.values():
        _cv.bind("<MouseWheel>", lambda evt, c=_cv:
                 c.yview_scroll(int(-1 * (evt.delta / 120)), "units"))

    # Input tabs in the other two simulators' order, for the same reason the
    # output tabs are reordered: these are created wherever their contents
    # happen to be built, which left Plot Settings sitting in the middle of
    # the hardware tabs. Lift Rotors and Cruise take the places Motor and
    # Propeller hold there, since they describe the same part of the aircraft.
    _INPUT_TAB_ORDER = ["Airframe", "Battery", "Lift Rotors", "Cruise", "ESC",
                        "Wiring", "Avionics", "Mission/Environment",
                        "Plot Settings"]
    _in_by_title = {nb.tab(t, "text"): t for t in nb.tabs()}
    for _position, _title in enumerate(_INPUT_TAB_ORDER):
        if _title in _in_by_title:
            nb.insert(_position, _in_by_title[_title])
    nb.select(0)

    # ---- output ------------------------------------------------------
    out_nb = ttk.Notebook(right)
    out_nb.grid(row=0, column=0, sticky="nsew")
    tab_metrics = ttk.Frame(out_nb); out_nb.add(tab_metrics, text="Metrics")
    tab_plots = ttk.Frame(out_nb); out_nb.add(tab_plots, text="Fixed Speed Plots")
    for frame in (tab_metrics, tab_plots):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    # `show="tree headings"` rather than `"headings"`: sections are real
    # parent nodes, so the Treeview gives expand/collapse for nothing. Blank
    # spacer rows, which is what this used before, look like separators but
    # cannot be folded and leave the reader to guess where a group ends.
    metrics_tv = ttk.Treeview(tab_metrics, columns=("metric", "value", "note"),
                              show="tree headings")
    metrics_tv.heading("#0", text="")
    metrics_tv.heading("metric", text="Metric")
    metrics_tv.heading("value", text="Value")
    metrics_tv.heading("note", text="What it means")
    metrics_tv.column("#0", width=22, minwidth=22, stretch=False)
    metrics_tv.column("metric", width=230, anchor="w")
    metrics_tv.column("value", width=180, anchor="w")
    metrics_tv.column("note", width=430, anchor="w")
    metrics_tv.tag_configure("section", font=("TkDefaultFont", 9, "bold"),
                             background="#e3eefb")
    metrics_tv.grid(row=0, column=0, sticky="nsew")

    # Scrollable plot pane, as the other two simulators have.
    #
    # Gridding the figure straight into the tab with sticky="nsew" let
    # matplotlib's own <Configure> handler resize the figure down to whatever
    # the container happened to be, which silently discards the requested
    # figure size — a Plot Size setting would tick in the menu and change
    # nothing on screen. Inside a scrolling pane the figure keeps its natural
    # height and the pane scrolls instead.
    plot_frame = ttk.LabelFrame(tab_plots, text="Performance Plots", padding=4)
    plot_frame.grid(row=0, column=0, sticky="nsew")
    plot_frame.columnconfigure(0, weight=1)
    plot_frame.rowconfigure(0, weight=1)

    plot_canvas = tk.Canvas(plot_frame, highlightthickness=0)
    plot_canvas.grid(row=0, column=0, sticky="nsew")
    plot_scroll = ttk.Scrollbar(plot_frame, orient="vertical",
                                command=plot_canvas.yview)
    plot_scroll.grid(row=0, column=1, sticky="ns")
    plot_canvas.configure(yscrollcommand=plot_scroll.set)

    plot_holder = ttk.Frame(plot_canvas)
    _plot_holder_id = plot_canvas.create_window((0, 0), window=plot_holder,
                                                anchor="nw")
    plot_holder.columnconfigure(0, weight=1)

    plot_holder.bind("<Configure>", lambda _e: plot_canvas.configure(
        scrollregion=plot_canvas.bbox("all")))
    plot_canvas.bind("<Configure>", lambda e: plot_canvas.itemconfigure(
        _plot_holder_id, width=e.width))

    def _on_plot_mousewheel(evt):
        plot_canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        return "break"

    for _w in (plot_canvas, plot_holder):
        _w.bind("<Enter>", lambda _e: plot_canvas.bind_all("<MouseWheel>",
                                                           _on_plot_mousewheel))
        _w.bind("<Leave>", lambda _e: plot_canvas.unbind_all("<MouseWheel>"))

    def _refresh_plot_scrollregion():
        plot_holder.update_idletasks()
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

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

    def _tree(parent, columns, height=16, row=1, scrollbar=True):
        """
        A results table.

        `scrollbar=False` is for the Status sub-tables: they already sit
        inside one scrolling pane, and a second scrollbar per table is
        clutter that never moves.
        """
        tv = ttk.Treeview(parent, columns=[c for c, _h, _w in columns],
                          show="headings", height=height)
        for key, heading, width in columns:
            tv.heading(key, text=heading)
            tv.column(key, width=width, stretch=True,
                      anchor="w" if key in ("metric", "item", "name", "label") else "center")
        tv.grid(row=row, column=0, sticky="nsew")
        if scrollbar:
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

    # Shared by both budget tabs. A table of numbers answers "how much?";
    # this answers "what dominates?", which is the question someone opens a
    # budget to ask.
    _BUDGET_COLOURS = ["#2E75B6", "#ED7D31", "#A9D18E", "#FFC000",
                       "#5B9BD5", "#FF7F7F", "#9E7FD4", "#70AD47"]

    def _draw_share_figure(items, total_label, pie=True):
        """
        A horizontal 100% stacked bar, optionally beside a pie.

        `items` is [(label, value), ...] with values in one unit. Entries
        that are zero or negative are dropped rather than drawn: a negative
        structure mass is a real condition the table shows in red, but it
        cannot be a slice of a whole.
        """
        data = [(lbl, float(v)) for lbl, v in items if float(v) > 0]
        if not data:
            return None
        labels = [d[0] for d in data]
        values = [d[1] for d in data]
        grand = sum(values)
        cols = 2 if pie else 1
        # A lone stacked bar is one row tall however many components it has,
        # so its height only has to cover the legend beneath it. The pie
        # beside it, by contrast, wants to stay roughly square.
        height = (max(3.0, len(labels) * 0.5 + 1.6) if pie
                  else 2.2 + 0.22 * len(labels))
        fig, axes = core.make_figure(1, cols,
                                     figsize=(6.5 if pie else 5.0, height))
        axes = list(axes) if cols > 1 else [axes]

        ax = axes[0]
        left = 0.0
        handles = []
        for i, (lbl, val) in enumerate(zip(labels, values)):
            pct = val / grand * 100.0
            handles.append(ax.barh(
                0, pct, left=left, label=lbl,
                color=_BUDGET_COLOURS[i % len(_BUDGET_COLOURS)],
                edgecolor="white", linewidth=0.5))
            if pct > 5:
                ax.text(left + pct / 2.0, 0, f"{pct:.0f}%", ha="center",
                        va="center", fontsize=7.5, color="white")
            left += pct
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("% of total")
        ax.grid(axis="x", alpha=0.3)

        if pie:
            ax2 = axes[1]
            _, _, texts = ax2.pie(
                values, labels=None, autopct="%1.0f%%",
                colors=_BUDGET_COLOURS[:len(labels)], startangle=90,
                pctdistance=0.75,
                wedgeprops=dict(edgecolor="white", linewidth=0.8))
            for t in texts:
                t.set_fontsize(7)
            ax2.set_title(total_label)
        else:
            ax.set_title(total_label)

        # A figure-level legend, with the space for it reserved explicitly.
        # An axes-level legend placed below its axes is clipped by the figure
        # edge, because tight_layout() does not account for artists drawn
        # outside the axes — which is how the component names ended up half
        # cut off.
        ncol = 2 if len(labels) <= 6 else 3
        rows = -(-len(labels) // ncol)          # ceiling division
        bottom = min(0.08 + 0.05 * rows, 0.45)
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=ncol,
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.005))
        return fig

    # ---- Metrics scope banner (the Metrics tab already exists) ----------
    tab_metrics.rowconfigure(1, weight=1)
    # Row 0 held the table before the scope banner was added and still
    # carried its weight, so the banner floated in the middle of a tall
    # empty row with the table pushed down below it.
    tab_metrics.rowconfigure(0, weight=0)
    metrics_tv.grid_configure(row=1)
    metrics_scope = _scope_label(tab_metrics)

    # ---- Status --------------------------------------------------------
    # One flat table forced the reader to work out which subsystem each row
    # belonged to. The other two simulators group theirs into titled
    # sub-tables; a VTOL needs four rather than three, because it has both
    # lifting rotors and a wing and they fail in different ways.
    tab_status = _tab("Status")
    status_scope = _scope_label(tab_status)

    status_outer = ttk.Frame(tab_status)
    status_outer.grid(row=1, column=0, columnspan=2, sticky="nsew")
    status_outer.columnconfigure(0, weight=1)
    status_outer.rowconfigure(0, weight=1)

    status_canvas = tk.Canvas(status_outer, highlightthickness=0)
    status_canvas.grid(row=0, column=0, sticky="nsew")
    status_sb = ttk.Scrollbar(status_outer, orient="vertical",
                              command=status_canvas.yview)
    status_sb.grid(row=0, column=1, sticky="ns")
    status_canvas.configure(yscrollcommand=status_sb.set)
    status_scroll = ttk.Frame(status_canvas)
    _status_win = status_canvas.create_window((0, 0), window=status_scroll,
                                              anchor="nw")
    status_scroll.columnconfigure(0, weight=1)
    status_scroll.bind("<Configure>", lambda _e: status_canvas.configure(
        scrollregion=status_canvas.bbox("all")))
    status_canvas.bind("<Configure>", lambda e: status_canvas.itemconfigure(
        _status_win, width=e.width))

    def _on_status_wheel(evt):
        status_canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        return "break"
    for _w in (status_canvas, status_scroll):
        _w.bind("<Enter>", lambda _e: status_canvas.bind_all(
            "<MouseWheel>", _on_status_wheel))
        _w.bind("<Leave>", lambda _e: status_canvas.unbind_all("<MouseWheel>"))

    status_detail = ttk.Label(tab_status, text="", wraplength=900, justify="left",
                              foreground="#333333")
    status_detail.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    def _make_status_tv(title):
        lf = ttk.LabelFrame(status_scroll, text=title, padding=4)
        lf.grid(sticky="nsew", padx=2, pady=3)
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)
        tv = _tree(lf, [("metric", "Metric", 230), ("value", "Value", 150),
                        ("limit", "Limit", 190), ("note", "Note", 420)],
                   height=4, row=0, scrollbar=False)

        def _on_select(_e=None, _tv=tv):
            sel = _tv.selection()
            if sel:
                vals = _tv.item(sel[0], "values")
                status_detail.configure(
                    text=f"{vals[0]} — {vals[3]}" if vals[3] else vals[0])
        tv.bind("<<TreeviewSelect>>", _on_select)
        return tv

    batt_status_tv = _make_status_tv("Battery Status")
    motor_status_tv = _make_status_tv("Motor / ESC Status")
    rotor_status_tv = _make_status_tv("Rotor / Propeller Status")
    aero_status_tv = _make_status_tv("Aerodynamic Status")
    _STATUS_TABLES = [("Battery Status", batt_status_tv),
                      ("Motor / ESC Status", motor_status_tv),
                      ("Rotor / Propeller Status", rotor_status_tv),
                      ("Aerodynamic Status", aero_status_tv)]

    def _row(tv, metric, value, limit, tag, note=""):
        tv.insert("", "end", values=(metric, value, limit, note), tags=(tag,))

    def _size_status_tables():
        """
        Fit each sub-table to its own contents.

        These have no scrollbar of their own — the whole Status tab scrolls
        instead — so a fixed height would either hide rows from a group that
        grew or leave blank space under one that did not. The battery group
        gains rows when connectors or a wire run are entered, so the count is
        not known when the table is built.
        """
        for _title, _tv in _STATUS_TABLES:
            _tv.configure(height=max(len(_tv.get_children("")), 1))

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

    def _dual(tv, metric, value, cont, mx, unit, decimals=1):
        """A quantity with both a continuous and an absolute limit."""
        v = float(value)
        if not cont and not mx:
            _row(tv, metric, f"{v:.{decimals}f} {unit}", "Not Specified", "na",
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
        _row(tv, metric, f"{v:.{decimals}f} {unit}", limit, tag, note)

    def update_status(cfg, m, worst=None):
        """
        Fixed speed: every row at the cruise speed, plus hover — which for a
        VTOL is the heaviest steady load and so the case that sizes the
        battery, the motors and the connectors.
        Mission: the worst value each check reached anywhere in the flight.
        """
        for _title, _tv in _STATUS_TABLES:
            _clear_tree(_tv)
        status_detail.configure(text="")
        batt = cfg.battery
        cont_A, max_A = batt.discharge_cont_A, batt.discharge_max_A

        if worst is not None:
            _dual(batt_status_tv, "Peak pack current", worst["pack_current_A"],
                  cont_A, max_A, "A")
            _dual(batt_status_tv, "Peak discharge C-rate", worst["c_rate"],
                  batt.discharge_c_cont, batt.discharge_c_max, "C")
            _row(batt_status_tv, "Peak power", f"{worst['total_power_W']:.0f} W",
                 "—", "na", f"Reached during '{worst.get('peak_phase', '')}'.")
            margin = worst.get("reserve_margin_Wh", 0.0)
            _row(batt_status_tv, "Lowest reserve margin", f"{margin:.1f} Wh",
                 ">= 0 Wh", "ok" if margin >= 0 else "bad",
                 "Energy left above the reserve at the lowest point of the "
                 "mission. Negative means the reserve was eaten into.")
            for _title, _tv in _STATUS_TABLES:
                if not _tv.get_children():
                    _row(_tv, "—", "", "",  "na",
                         "These checks need a single operating point. Run a "
                         "fixed speed sweep to evaluate them.")
            _size_status_tables()
            return

        hover_I = float(m.get("hover_pack_current_A", 0.0))
        point_I = float(m.get("pack_current_A", 0.0))
        _dual(batt_status_tv, "Hover pack current", hover_I, cont_A, max_A, "A")
        _dual(batt_status_tv, "Cruise pack current", point_I, cont_A, max_A, "A")
        _dual(batt_status_tv, "Hover C-rate", hover_I / max(batt.capacity_Ah, 1e-9),
              batt.discharge_c_cont, batt.discharge_c_max, "C")

        if cfg.lift_motor_max_power_W:
            per = float(m.get("hover_power_per_lift_motor_W", 0.0))
            _row(motor_status_tv, "Lift motor power in hover", f"{per:.0f} W",
                 f"<= {cfg.lift_motor_max_power_W:.0f} W",
                 _classify(per, cfg.lift_motor_max_power_W),
                 "Hover is the heaviest load a lift motor carries.")
        else:
            _row(motor_status_tv, "Lift motor power in hover",
                 f"{float(m.get('hover_power_per_lift_motor_W', 0.0)):.0f} W",
                 "Not Specified", "na", "Enter a motor max power to check it.")

        if not uses_vectored_thrust(cfg):
            per = float(m.get("cruise_power_per_motor_W", 0.0))
            if cfg.cruise_motor_max_power_W:
                _row(motor_status_tv, "Cruise motor power", f"{per:.0f} W",
                     f"<= {cfg.cruise_motor_max_power_W:.0f} W",
                     _classify(per, cfg.cruise_motor_max_power_W))
            else:
                _row(motor_status_tv, "Cruise motor power", f"{per:.0f} W",
                     "Not Specified", "na")

        v_stall = float(m.get("stall_speed_mps", 0.0))
        v_trans = float(m.get("transition_speed_mps", 0.0))
        ratio = v_trans / max(v_stall, 1e-9)
        _row(aero_status_tv, "Transition / stall speed",
             f"{v_trans:.1f} / {v_stall:.1f} m/s",
             ">= 1.10x stall", "ok" if ratio >= 1.10 else ("warn" if ratio >= 1.0 else "bad"),
             f"Transitioning at {ratio:.2f}x stall. Below about 1.1x a gust can "
             f"stall the wing while the rotors are spinning down.")

        share = float(m.get("lift_share_wing", 0.0))
        v_cruise = float(m.get("airspeed_mps", 0.0))
        _row(aero_status_tv, "Wing lift share at cruise speed",
             f"{share * 100:.0f}%", "100%",
             "ok" if share >= 0.999 else "warn",
             "Fully wing-borne." if share >= 0.999 else
             f"At {v_cruise:.1f} m/s the wing carries only part of the weight, "
             f"so the rotors are still lifting — the cruise speed is below the "
             f"{v_trans:.1f} m/s transition speed.")

        if uses_vectored_thrust(cfg):
            tilt = float(m.get("tilt_deg", 0.0))
            _row(rotor_status_tv, "Rotor tilt at cruise speed",
                 f"{tilt:.1f} deg", "90 deg in cruise",
                 "ok" if tilt >= 89.0 else "warn",
                 "Thrust pointed straight ahead." if tilt >= 89.0 else
                 "Rotors not yet fully tilted — still partly lifting.")
        import os as _os
        area = math.pi / 4.0 * (cfg.lift_prop_diameter_in * 0.0254) ** 2
        t_per = cfg.weight_N * (1.0 + hover_download_fraction(cfg)) / max(cfg.num_lift_rotors, 1)
        measured = measured_lift_efficiency(cfg, t_per, area)
        if cfg.lift_prop_table is None:
            _row(rotor_status_tv, "Lift rotor efficiency",
                 f"{cfg.lift_figure_of_merit:.3f} FoM", "estimate", "na",
                 "No bench table loaded, so this is the figure of merit you "
                 "entered. Load a measured table on the Lift Rotors tab to "
                 "replace the estimate.")
        elif measured is None:
            _row(rotor_status_tv, "Lift rotor efficiency",
                 f"{cfg.lift_figure_of_merit:.3f} FoM",
                 "estimate (outside table)", "warn",
                 f"A table is loaded ({_os.path.basename(cfg.lift_prop_table_csv)}) "
                 f"but {t_per:.1f} N per rotor is outside the thrust range it "
                 f"covers, so the estimate is being used instead.")
        else:
            _row(rotor_status_tv, "Lift rotor efficiency", f"{measured:.3f} FoM",
                 "measured", "ok",
                 f"From {_os.path.basename(cfg.lift_prop_table_csv)}, against "
                 f"the {cfg.lift_figure_of_merit:.3f} you entered.")

        _row(rotor_status_tv, "Hover download fraction",
             f"{hover_download_fraction(cfg) * 100:.0f}%", "—", "na",
             "Extra thrust to lift the rotor wash striking the airframe. "
             + ("Overridden on the Lift Rotors tab." if cfg.hover_download_fraction is not None
                else f"Type default for {cfg.config_type}."))

        # Thrust each rotor must make in hover, against the propeller's own
        # rated figure. This is the check that decides whether the aircraft
        # can lift itself at all, so it belongs on Status, not only on the
        # Airframe Diagram.
        rated_N = (cfg.lift_prop_max_thrust_g / 1000.0 * 9.80665
                   if getattr(cfg, "lift_prop_max_thrust_g", 0) else None)
        if rated_N:
            _row(rotor_status_tv, "Thrust per lift rotor (hover)",
                 f"{t_per:.1f} N", f"<= {rated_N:.1f} N",
                 _classify(t_per, rated_N),
                 f"{t_per / 9.80665 * 1000:.0f} g per rotor against a rated "
                 f"{cfg.lift_prop_max_thrust_g:.0f} g.")
        else:
            _row(rotor_status_tv, "Thrust per lift rotor (hover)",
                 f"{t_per:.1f} N", "Not Specified", "na",
                 "Enter the propeller's rated thrust on the Lift Rotors tab "
                 "to check it.")

        conns = dict(getattr(cfg, "connectors", {}) or {})
        if conns:
            # Checked at hover, the largest steady current a VTOL draws.
            n_esc = max(cfg.num_lift_rotors, 1)
            seen = {"Battery": hover_I, "ESC": hover_I / n_esc,
                    "Motor": hover_I / n_esc * 1.15}
            for name, (c_cont, c_max) in conns.items():
                if name in seen:
                    # The battery connector carries the whole pack current, so
                    # it is a battery-side check; the other two sit on a motor
                    # channel.
                    _dual(batt_status_tv if name == "Battery" else motor_status_tv,
                          f"{name} connector (hover)", seen[name],
                          c_cont or None, c_max or None, "A")

        drop = float(m.get("wire_drop_V", 0.0))
        if drop > 0:
            _row(batt_status_tv, "Main wire voltage drop", f"{drop:.2f} V",
                 "—", "na",
                 f"Lost along the battery lead at cruise; "
                 f"{float(m.get('wire_loss_W', 0.0)):.1f} W as heat.")

        # Wind checks: whether the course can be held at all, and whether
        # holding station is affordable. Both only say anything once a wind
        # has been entered.
        wind = float(m.get("wind_mps", 0.0))
        if wind > 0:
            cross = abs(float(m.get("wind_cross_mps", 0.0)))
            v_air = float(m.get("airspeed_mps", 0.0))
            _row(aero_status_tv, "Crosswind vs airspeed",
                 f"{cross:.1f} m/s cross at {v_air:.1f} m/s",
                 f"< {v_air:.1f} m/s",
                 "ok" if cross < v_air * 0.9 else
                 ("warn" if cross < v_air else "bad"),
                 "The aircraft crabs into a crosswind, and that part of the "
                 "airspeed never becomes progress. At or above the airspeed "
                 "the course cannot be held at all and groundspeed along it "
                 "is zero."
                 if cross >= v_air * 0.9 else
                 f"Crabbing costs "
                 f"{v_air - float(m.get('groundspeed_mps', 0.0)) - float(m.get('wind_head_mps', 0.0)):.1f} m/s "
                 f"of along-track speed.")

            gs = float(m.get("groundspeed_mps", 0.0))
            _row(aero_status_tv, "Ground speed", f"{gs:.1f} m/s", "> 0 m/s",
                 "ok" if gs > 0.5 else "bad",
                 "Progress along the course. Zero means the wind is holding "
                 "the aircraft still or pushing it backwards — endurance is "
                 "unaffected, but the range is nil."
                 if gs <= 0.5 else
                 f"Range is set by this, not the {v_air:.1f} m/s airspeed.")

            _row(rotor_status_tv, "Station-keeping in this wind",
                 f"{float(m.get('station_power_W', 0.0)):.0f} W",
                 f"hover is {float(m.get('hover_power_W', 0.0)):.0f} W", "na",
                 f"Holding position means flying at {wind:.1f} m/s through "
                 f"the air, which the model evaluates as "
                 f"{m.get('station_regime', '')}. Translational lift makes "
                 f"that cheaper than still-air hover.")

        esc_loss = float(m.get("esc_loss_W", 0.0))
        if esc_loss > 0:
            _row(motor_status_tv, "ESC + motor loss at cruise",
                 f"{esc_loss:.1f} W", "—", "na",
                 f"{esc_loss / max(float(m.get('total_power_W', 0.0)), 1e-9) * 100:.1f}% "
                 f"of pack power. The VTOL model folds motor losses into the "
                 f"ESC efficiency, so this covers both.")

        _size_status_tables()

    # ---- Mission Plots -------------------------------------------------
    # Laid out as a labelled controls column beside the plot, matching the
    # other two simulators. The old arrangement packed an "X axis:" label
    # immediately before the VARIABLE list, so on screen that label appeared
    # to name the variable selector while the real Time/Distance choice sat
    # unlabelled to its right.
    tab_mplots = _tab("Mission Plots")
    tab_mplots.columnconfigure(0, weight=0)
    tab_mplots.columnconfigure(1, weight=1)

    _MPLOT_KEYS = [
        ("segment_code", "Segment type (—)"),
        ("airspeed_mps", "Airspeed (m/s)"),
        ("altitude_m", "Altitude (m)"),
        ("climb_rate_mps", "Climb rate (m/s)"),
        ("distance_km", "Distance travelled (km)"),
        ("total_power_W", "Total power (W)"),
        ("shaft_power_W", "Shaft power, total (W)"),
        ("rotor_shaft_W", "Lift rotor shaft power (W)"),
        ("cruise_shaft_W", "Cruise prop shaft power (W)"),
        ("specific_power_W_per_kg", "Specific power (W/kg)"),
        ("pack_current_A", "Pack current (A)"),
        ("c_rate", "C-rate (—)"),
        ("energy_remaining_Wh", "Energy remaining (Wh)"),
        ("energy_used_Wh", "Energy used (Wh)"),
        ("soc_pct", "Battery SoC (%)"),
        ("capacity_remaining_mAh", "Capacity remaining (mAh)"),
        ("reserve_margin_Wh", "Reserve margin (Wh)"),
        ("rotor_thrust_N", "Rotor thrust (N)"),
        ("wing_lift_N", "Wing lift (N)"),
        ("cruise_thrust_N", "Cruise thrust (N)"),
        ("drag_N", "Drag (N)"),
        ("lift_share_wing", "Wing lift share (—)"),
        ("tilt_deg", "Rotor tilt (deg)"),
    ]

    mplot_controls = ttk.LabelFrame(tab_mplots, text="Y-axis variables",
                                    padding=4)
    mplot_controls.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
    mplot_controls.rowconfigure(1, weight=1)
    mplot_controls.columnconfigure(0, weight=1)
    ttk.Label(mplot_controls, justify="left", wraplength=210,
              foreground="#555555",
              text="Select up to 4 variables to plot.\nTwo y-axes on the "
                   "left, two on the right.").grid(row=0, column=0,
                                                   columnspan=2, sticky="w")
    mplot_list = tk.Listbox(mplot_controls, selectmode="extended", height=16,
                            width=28, exportselection=False)
    for _k, _lbl in _MPLOT_KEYS:
        mplot_list.insert("end", _lbl)
    mplot_list.grid(row=1, column=0, sticky="nsew", pady=(4, 4))
    _mplot_sb = ttk.Scrollbar(mplot_controls, orient="vertical",
                              command=mplot_list.yview)
    _mplot_sb.grid(row=1, column=1, sticky="ns", pady=(4, 4))
    mplot_list.configure(yscrollcommand=_mplot_sb.set)
    mplot_list.bind("<MouseWheel>", lambda e: (
        mplot_list.yview_scroll(int(-1 * (e.delta / 120)), "units"), "break")[1])
    # Power against altitude is the pair that shows a VTOL's story: the
    # climb and the transition are where the energy goes.
    mplot_list.selection_set(_MPLOT_KEYS.index(("total_power_W", "Total power (W)")))
    mplot_list.selection_set(_MPLOT_KEYS.index(("altitude_m", "Altitude (m)")))

    mplot_btns = ttk.Frame(mplot_controls)
    mplot_btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 4))

    _mplot_xbar = ttk.Frame(mplot_controls)
    _mplot_xbar.grid(row=3, column=0, columnspan=2, sticky="w")
    ttk.Label(_mplot_xbar, text="X axis:").pack(side="left")
    v_mx = tk.StringVar(value="time")
    ttk.Radiobutton(_mplot_xbar, text="Time", value="time",
                    variable=v_mx).pack(side="left")
    ttk.Radiobutton(_mplot_xbar, text="Distance", value="distance",
                    variable=v_mx).pack(side="left")

    mplot_holder = ttk.Frame(tab_mplots)
    mplot_holder.grid(row=1, column=1, sticky="nsew")
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
        # A variable added to the list but never recorded would otherwise
        # raise KeyError at plot time, on the user's click rather than here.
        chosen = [c for c in chosen if c[0] in series]
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

    ttk.Button(mplot_btns, text="Plot selected",
               command=draw_mission_plots).pack(side="left")
    # Bound late: clear_mission_plots is defined below this bar, so the name
    # is resolved when the user clicks rather than when the button is built.
    ttk.Button(mplot_btns, text="Clear",
               command=lambda: clear_mission_plots()).pack(side="left", padx=6)

    def clear_mission_plots():
        _mission_state.update(series=None, phases=None, results=None)
        _destroy_canvas(_mplot_canvas)
        mplot_placeholder.configure(
            text="These plots come from a mission run.\n\nYou have just run a "
                 "fixed speed sweep, so there is no mission history to plot.")
        mplot_placeholder.grid()

    # ---- Weight Budget -------------------------------------------------
    # Table on the left, share chart on the right, as the other two
    # simulators lay it out. The table alone left most of the tab empty.
    tab_wb = _tab("Weight Budget")
    wb_note = ttk.Label(tab_wb, text="", foreground="#555555", wraplength=900,
                        justify="left")
    wb_note.grid(row=0, column=0, columnspan=2, sticky="ew")

    wb_outer = ttk.Frame(tab_wb)
    wb_outer.grid(row=1, column=0, columnspan=2, sticky="nsew")
    wb_outer.columnconfigure(0, weight=3)
    wb_outer.columnconfigure(1, weight=2)
    wb_outer.rowconfigure(0, weight=1)

    wb_left = ttk.LabelFrame(wb_outer, text="Component Weights", padding=4)
    wb_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    wb_left.columnconfigure(0, weight=1)
    wb_left.rowconfigure(0, weight=1)
    wb_tv = _tree(wb_left, [("item", "Component", 170), ("unit", "Unit (g)", 70),
                            ("qty", "Qty", 45), ("total", "Total (g)", 80),
                            ("pct", "% of Total", 75)], height=12, row=0)

    wb_right = ttk.LabelFrame(wb_outer, text="Weight Distribution", padding=4)
    wb_right.grid(row=0, column=1, sticky="nsew")
    wb_right.columnconfigure(0, weight=1)
    wb_right.rowconfigure(0, weight=1)
    _wb_canvas = {"widget": None}
    wb_chart_placeholder = ttk.Label(
        wb_right, foreground="#888888", justify="center", wraplength=260,
        text="Run anything to see where the weight goes.")
    wb_chart_placeholder.grid(row=0, column=0, padx=12, pady=30)

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
        # One ESC per driven rotor. On a vectored type the lift rotors are
        # the only rotors, so there is no separate cruise ESC to count.
        n_esc = cfg.num_lift_rotors + (0 if uses_vectored_thrust(cfg)
                                       else cfg.num_cruise_motors)
        items.append(("ESCs", cfg.esc_weight_g, n_esc))
        items.append(("Avionics", cfg.avionics_mass_g, 1))
        parts = 0.0
        chart = []
        for name, each, qty in items:
            total = float(each) * int(qty)
            if total <= 0:
                continue
            parts += total
            chart.append((name, total))
            wb_tv.insert("", "end", values=(name, f"{float(each):.0f}", qty,
                                            f"{total:.0f}", f"{total / auw * 100:.1f}%"))
        structure = cfg.aircraft_weight_g - parts
        chart.append(("Airframe / structure", structure))
        wb_tv.insert("", "end", tags=(("bad",) if structure < 0 else ()),
                     values=("Airframe / structure", "", "", f"{structure:.0f}",
                             f"{structure / auw * 100:.1f}%"))
        if cfg.payload_mass_g > 0:
            chart.append(("Payload", cfg.payload_mass_g))
            wb_tv.insert("", "end", values=("Payload", f"{cfg.payload_mass_g:.0f}", 1,
                                            f"{cfg.payload_mass_g:.0f}",
                                            f"{cfg.payload_mass_g / auw * 100:.1f}%"))
        wb_tv.insert("", "end", tags=("total",),
                     values=("ALL-UP WEIGHT", "", "", f"{auw:.0f}", "100%"))

        fig = _draw_share_figure(chart, f"Total: {auw:.0f} g")
        if fig is None:
            _destroy_canvas(_wb_canvas)
            wb_chart_placeholder.grid()
        else:
            wb_chart_placeholder.grid_remove()
            _show_figure(wb_right, _wb_canvas, fig, row=0)
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

    pb_outer = ttk.Frame(tab_pb)
    pb_outer.grid(row=1, column=0, columnspan=2, sticky="nsew")
    pb_outer.columnconfigure(0, weight=3)
    pb_outer.columnconfigure(1, weight=2)
    pb_outer.rowconfigure(0, weight=1)

    pb_left = ttk.LabelFrame(pb_outer, text="Component / Loss Breakdown", padding=4)
    pb_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    pb_left.columnconfigure(0, weight=1)
    pb_left.rowconfigure(0, weight=1)
    pb_tv = _tree(pb_left, [("item", "Component / Loss", 200), ("watts", "Power (W)", 75),
                            ("pct", "% of P_in", 70), ("voltage", "Voltage", 70),
                            ("current", "Current (A)", 75)], height=14, row=0)

    # Deliberately a stacked bar and not a pie: the pie that used to sit here
    # in the other two simulators was removed because a dozen sub-1% loss
    # slices are unreadable as wedges. The bar keeps the same information and
    # stays legible when one component is 90% of the total.
    pb_right = ttk.LabelFrame(pb_outer, text="Power Distribution", padding=4)
    pb_right.grid(row=0, column=1, sticky="nsew")
    pb_right.columnconfigure(0, weight=1)
    pb_right.rowconfigure(0, weight=1)
    _pb_canvas = {"widget": None}
    pb_chart_placeholder = ttk.Label(
        pb_right, foreground="#888888", justify="center", wraplength=260,
        text="Run a fixed speed sweep to see where the battery's power goes.")
    pb_chart_placeholder.grid(row=0, column=0, padx=12, pady=30)

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
            # Two different direct-from-pack loads share this row: the flat
            # avionics figure (used only when no rails replace it) and the
            # peripheral current, which always adds. core emits a single
            # "Peripheral devices (direct from pack)" row, so they are summed
            # here — the total is what has to balance, and it does.
            peripheral_W=((0.0 if cfg.avionics_rails else cfg.avionics_power_W)
                          + peripheral_power_W(cfg)),
            peripheral_A=(((0.0 if cfg.avionics_rails else cfg.avionics_power_W)
                           / max(cfg.battery.vnom_pack, 1e-9))
                          + cfg.periph_current_A) or None,
            rails=[{"name": f"{v:g}V", "voltage_V": v, "current_A": a,
                    "efficiency": e}
                   for v, (a, e) in sorted(cfg.avionics_rails.items())] or None)
        for r in rows:
            pb_tv.insert("", "end", tags=(r["kind"],),
                         values=(r["name"], f"{r['watts']:.1f}", f"{r['pct']:.1f}%",
                                 # core emits these as `voltage` / `current`,
                                 # not the *_V / *_A names its INPUT rails use.
                                 ("" if r.get("voltage") in (None, "")
                                  else str(r["voltage"])),
                                 ("" if r.get("current") in (None, "")
                                  else str(r["current"]))))
        pb_scope.configure(text=(
            f"At the cruise speed of {float(m.get('airspeed_mps', 0.0)):.1f} m/s "
            f"({m.get('regime', '')}). Motor losses are folded into the ESC "
            f"efficiency and rotor losses into the figure of merit, so there is "
            f"no separate motor copper-loss line."))

        # Only the leaf rows: the subtotal and total rows are sums of these,
        # so charting them too would double the whole.
        chart = [(r["name"], r["watts"]) for r in rows
                 if r["kind"] in ("delivered", "lost")]
        fig = _draw_share_figure(
            chart, f"Total from cells: {float(m.get('total_power_W', 0.0)):.0f} W",
            pie=False)
        if fig is None:
            _destroy_canvas(_pb_canvas)
            pb_chart_placeholder.grid()
        else:
            pb_chart_placeholder.grid_remove()
            _show_figure(pb_right, _pb_canvas, fig, row=0)

    def clear_power_budget():
        _clear_tree(pb_tv)
        pb_tv.insert("", "end", tags=("total",),
                     values=("Mission run — no single operating point to break down",
                             "", "", "", ""))
        pb_scope.configure(text="Press Run Fixed Speed Sweep to build a power budget.")
        _destroy_canvas(_pb_canvas)
        pb_chart_placeholder.configure(
            text="A mission has no single operating point, so there is no "
                 "power split to show.\n\nPress Run Fixed Speed Sweep to "
                 "build one.")
        pb_chart_placeholder.grid()

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
        md_placeholder.configure(text="A fixed speed sweep has no route.\n\n"
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
    # Collapsing four perturbations into Low/High hid which DIRECTION an
    # input pushes, and whether it is linear: an input whose -20% and +20%
    # both raise the output behaves very differently from one that is
    # monotonic, and Low/High cannot tell them apart.
    sens_tv = _tree(tab_sens, [("name", "Input", 190), ("m20", "-20%", 78),
                               ("m10", "-10%", 78), ("base", "baseline", 82),
                               ("p10", "+10%", 78), ("p20", "+20%", 78),
                               ("span", "swing", 96)], height=12)
    _sens_state = {"cfg": None, "mission": None, "wind": 0.0, "wind_dir": 0.0,
                   "course": 0.0}

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
        sens_tv.insert("", "end", values=(reason, "", "", "", "", "",
                                          "Press Run Sensitivity"))

    def run_sensitivity():
        base = _sens_state["cfg"]
        if base is None:
            messagebox.showinfo("Sensitivity", "Run a fixed speed sweep or a mission first.")
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
                # The same wind as the run this sensitivity is based on:
                # "Cruise range" depends on it, so a still-air sweep here
                # would disagree with the number on the Metrics tab.
                mm = compute_metrics(
                    cfg,
                    wind_mps=_sens_state.get("wind", 0.0),
                    wind_direction_deg=_sens_state.get("wind_dir", 0.0),
                    course_deg=_sens_state.get("course", 0.0))
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
                                              "", "", "", "", "", ""))
            return
        for r in rows:
            # `results` is a dict keyed by the factor, NOT a list — reading it
            # positionally would have put the numbers in whatever order the
            # dict happened to yield. Look each factor up by name.
            found = r.get("results") or {}
            cells = []
            for factor in (0.8, 0.9, 1.1, 1.2):
                value = found.get(factor)
                cells.append("—" if value is None else f"{value:.2f}")
            sens_tv.insert("", "end", values=(
                r["name"], cells[0], cells[1], f"{r['baseline']:.2f}",
                cells[2], cells[3],
                f"{r['span']:.2f}  ({r['span_pct']:.0f}%)"))

    ttk.Button(sens_bar, text="⚡  Run Sensitivity", command=run_sensitivity).pack(
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

    ttk.Button(cmp_bar, text="✖  Clear Baseline", command=clear_baseline).pack(side="right")
    ttk.Button(cmp_bar, text="📌  Pin Current as Baseline", command=pin_baseline).pack(
        side="right", padx=6)

    # ---- Airframe Diagram ---------------------------------------------
    # Drawing on the left in a titled frame, rotor loading on the right with
    # the explanation beside it — the multicopter's arrangement. The table
    # sat bare beneath the plot before, with nothing saying what it was.
    tab_ad = _tab("Airframe Diagram")
    tab_ad.columnconfigure(0, weight=3)
    tab_ad.columnconfigure(1, weight=2)

    ad_frame = ttk.LabelFrame(tab_ad, text="Plan View (to scale)", padding=4)
    ad_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
    ad_frame.columnconfigure(0, weight=1)
    ad_frame.rowconfigure(1, weight=1)
    ad_holder = ttk.Frame(ad_frame)
    ad_holder.grid(row=1, column=0, sticky="nsew")
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

    # ---- R11: per-rotor loading ----------------------------------------
    # The multicopter shows what each rotor is actually carrying. A VTOL's
    # lift rotors share the hover load evenly in still air — there is no
    # drag-induced moment to redistribute it, because the aircraft is not
    # translating — so the useful numbers here are the load per rotor and
    # how much margin is left to the propeller's rated thrust.
    rl_frame = ttk.LabelFrame(tab_ad, text="Per-Rotor Loading", padding=4)
    rl_frame.grid(row=1, column=1, sticky="nsew")
    rl_frame.columnconfigure(0, weight=1)
    rl_frame.rowconfigure(1, weight=1)
    ttk.Label(rl_frame, foreground="#555555", wraplength=380, justify="left",
              text="Rotor numbering matches the diagram.\n"
                   "In still-air hover the lift rotors share the weight "
                   "evenly: there is no translation, so no drag-induced "
                   "moment to redistribute it. Margin is measured against "
                   "the propeller's rated thrust, and the download the "
                   "airframe adds is already included in the load."
              ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
    rl_tv = _tree(rl_frame, [("rotor", "Rotor", 90), ("thrust_n", "Thrust (N)", 110),
                             ("thrust_g", "Thrust (g)", 110),
                             ("share", "Share of weight", 130),
                             ("margin", "Margin to rated", 140)], height=8, row=1)

    def update_rotor_loading(cfg, m):
        _clear_tree(rl_tv)
        n = max(int(cfg.num_lift_rotors), 1)
        total_N = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
        per_N = total_N / n
        rated_N = (cfg.lift_prop_max_thrust_g / 1000.0 * 9.80665
                   if getattr(cfg, "lift_prop_max_thrust_g", 0) else None)
        for i in range(1, n + 1):
            if rated_N:
                margin = (rated_N - per_N) / rated_N * 100.0
                tag = ("bad" if margin < 0 else "warn" if margin < 20 else "ok")
                margin_text = f"{margin:+.0f}%"
            else:
                tag, margin_text = "na", "no rated thrust entered"
            rl_tv.insert("", "end", tags=(tag,), values=(
                f"{i}", f"{per_N:.1f}", f"{per_N / 9.80665 * 1000:.0f}",
                f"{100.0 / n:.1f}%", margin_text))
        rl_tv.insert("", "end", tags=("total",), values=(
            "TOTAL", f"{total_N:.1f}", f"{total_N / 9.80665 * 1000:.0f}",
            "100%", ""))

    # ---- exports --------------------------------------------------------
    # Built by reading the tables on screen, so an export cannot disagree
    # with what the user is looking at — the two cannot drift apart because
    # there is only one source.
    _export_state = {"cfg": None, "from_mission": False, "results": None}

    def _tree_section(title, tree):
        """
        A tree's visible contents as an export section.

        Metrics nests its rows under section nodes, so a flat read of
        get_children() would export six section headings and none of the
        numbers under them. Walking the tree keeps the export and the screen
        showing the same thing, which is the only reason the export is
        trustworthy.
        """
        headers = [tree.heading(c)["text"] for c in tree.cget("columns")]
        rows = []

        def _walk(parent, depth=0):
            for iid in tree.get_children(parent):
                values = list(tree.item(iid, "values"))
                if depth and values:
                    # Indent a child row so the grouping survives into a flat
                    # CSV, where there is no other way to show nesting.
                    values[0] = f"  {values[0]}"
                rows.append(values)
                _walk(iid, depth + 1)

        _walk("")
        return (title, headers, rows)

    def _sweep_section(cfg):
        """The speed sweep behind the Fixed Speed Plots, as numbers."""
        speeds, rows = [], []
        # Plot Settings wins when a maximum was entered.
        v_max = (opt("plot_vmax")
                 or max(cfg.cruise_speed_mps * 1.6,
                        transition_speed_mps(cfg) * 1.6))
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
        sections = [_tree_section("Metrics", metrics_tv)]
        # One export section per Status sub-table, so a reader of the CSV or
        # the PDF sees the same grouping as the screen rather than one
        # undifferentiated block.
        sections += [_tree_section(_title, _tv) for _title, _tv in _STATUS_TABLES]
        sections.append(_tree_section("Weight Budget", wb_tv))
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

    # ---- View settings -------------------------------------------------
    # Every display choice lives here so a preset can set all four at once
    # and the plot builders can read the current figure size.
    _view = {
        "scale_pct":    100,
        # Sized for the 2x2 sweep. It was (11, 4.5) when the sweep was 1x2;
        # four panels in that height are unreadable.
        "plot_w":       14.0,
        "plot_h":        8.5,
        "mpl_fontsize":   9,
        "ui_fontsize":    9,
    }

    def _apply_tk_scale(pct: int) -> None:
        _view["scale_pct"] = pct
        # Tk scaling is in points per pixel; 1.3333 is the 100% baseline.
        try:
            root.tk.call("tk", "scaling", 1.3333 * pct / 100.0)
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
            "font.size":       size,
            "axes.titlesize":  size + 1,
            "axes.labelsize":  size,
            "xtick.labelsize": size - 1,
            "ytick.labelsize": size - 1,
            "legend.fontsize": size - 1,
        })
        _rerender_if_possible()

    def _apply_plot_size(w: float, h: float) -> None:
        _view["plot_w"] = w
        _view["plot_h"] = h
        _rerender_if_possible()

    def _rerender_if_possible() -> None:
        """
        Redraw the fixed-speed sweep at the new size or font.

        A mission run leaves no sweep on screen, so there is nothing to
        re-render then — clear_fixed_speed_plots has already put the
        explanatory placeholder there and redrawing would wipe it.
        """
        cfg = _export_state.get("cfg")
        if cfg is None or _export_state.get("from_mission"):
            return
        try:
            draw_plots(cfg)
        except Exception:
            pass

    # -- Window Scale --
    _scale_var = tk.IntVar(value=100)
    scale_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Window Scale", menu=scale_menu)
    for pct, label in [(75, "75 % – Compact"), (90, "90 % – Smaller"),
                       (100, "100 % – Default"), (115, "115 % – Slightly Larger"),
                       (125, "125 % – Large"), (150, "150 % – Extra Large"),
                       (175, "175 % – Very Large"), (200, "200 % – Max")]:
        scale_menu.add_radiobutton(label=label, variable=_scale_var, value=pct,
                                   command=lambda p=pct: _apply_tk_scale(p))
    view_menu.add_separator()

    # -- Plot Size --
    # Scaled around the 2x2 sweep's (14, 8.5), so "Medium" is what the tool
    # draws by default.
    _plot_size_var = tk.StringVar(value="medium")
    plot_size_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Plot Size", menu=plot_size_menu)
    for key, label, (w, h) in [
        ("small",   "Small  (11 × 6.7)",            (11.0, 6.7)),
        ("medium",  "Medium (14 × 8.5)  ← default", (14.0, 8.5)),
        ("large",   "Large  (17 × 10.3)",           (17.0, 10.3)),
        ("xlarge",  "X-Large (20 × 12.1)",          (20.0, 12.1)),
        ("xxlarge", "XX-Large (24 × 14.6)",         (24.0, 14.6)),
    ]:
        plot_size_menu.add_radiobutton(
            label=label, variable=_plot_size_var, value=key,
            command=lambda pw=w, ph=h: _apply_plot_size(pw, ph))
    view_menu.add_separator()

    # -- UI Font Size --
    _ui_font_var = tk.IntVar(value=9)
    ui_font_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="UI Font Size", menu=ui_font_menu)
    for sz, label in [(8, "8 pt  – Tiny"), (9, "9 pt  – Default"),
                      (10, "10 pt – Comfortable"), (11, "11 pt – Large"),
                      (13, "13 pt – Extra Large"), (15, "15 pt – Accessibility")]:
        ui_font_menu.add_radiobutton(label=label, variable=_ui_font_var, value=sz,
                                     command=lambda s=sz: _apply_ui_font(s))
    view_menu.add_separator()

    # -- Plot Font Size --
    _mpl_font_var = tk.IntVar(value=9)
    mpl_font_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Plot Font Size", menu=mpl_font_menu)
    for sz, label in [(7, "7 pt  – Tiny"), (8, "8 pt  – Small"),
                      (9, "9 pt  – Default"), (10, "10 pt – Medium"),
                      (12, "12 pt – Large"), (14, "14 pt – Extra Large")]:
        mpl_font_menu.add_radiobutton(label=label, variable=_mpl_font_var, value=sz,
                                      command=lambda s=sz: _apply_mpl_font(s))
    view_menu.add_separator()

    # -- Quick Presets --
    def _preset_compact():
        _scale_var.set(85);          _apply_tk_scale(85)
        _ui_font_var.set(8);         _apply_ui_font(8)
        _mpl_font_var.set(8);        _apply_mpl_font(8)
        _plot_size_var.set("small"); _apply_plot_size(9.0, 3.7)

    def _preset_default():
        _scale_var.set(100);          _apply_tk_scale(100)
        _ui_font_var.set(9);          _apply_ui_font(9)
        _mpl_font_var.set(9);         _apply_mpl_font(9)
        _plot_size_var.set("medium"); _apply_plot_size(11.0, 4.5)

    def _preset_presentation():
        _scale_var.set(140);         _apply_tk_scale(140)
        _ui_font_var.set(12);        _apply_ui_font(12)
        _mpl_font_var.set(12);       _apply_mpl_font(12)
        _plot_size_var.set("large"); _apply_plot_size(13.5, 5.5)

    def _preset_accessibility():
        _scale_var.set(160);          _apply_tk_scale(160)
        _ui_font_var.set(14);         _apply_ui_font(14)
        _mpl_font_var.set(13);        _apply_mpl_font(13)
        _plot_size_var.set("xlarge"); _apply_plot_size(16.0, 6.5)

    presets_menu = tk.Menu(view_menu, tearoff=0)
    view_menu.add_cascade(label="Quick Presets", menu=presets_menu)
    presets_menu.add_command(label="🗜  Compact",      command=_preset_compact)
    presets_menu.add_command(label="⚙  Default",       command=_preset_default)
    presets_menu.add_command(label="📊  Presentation", command=_preset_presentation)
    presets_menu.add_command(label="♿  Accessibility", command=_preset_accessibility)
    view_menu.add_separator()
    view_menu.add_command(label="Reset All to Default", command=_preset_default)

    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    def _show_about():
        """
        What is running, and — more usefully — what it does and does not
        model. The scope lines are here because every one of them has been
        asked in the form "why does my number not match X".
        """
        messagebox.showinfo(
            "About",
            f"VTOL Power Simulator\n"
            f"Version {SIM_VERSION}\n"
            f"{SIM_BUILD_NOTE}\n\n"
            "Configurations: lift+cruise, tiltrotor, tiltwing, tailsitter.\n\n"
            "Modelled: momentum-theory hover with a figure of merit, wing "
            "lift and drag from CD0 and the Oswald factor, the transition "
            "between them, vectored-thrust force balance, hover download, "
            "pack sag from a state-of-charge curve, ESC and wiring losses, "
            "and time-stepped missions.\n\n"
            "NOT modelled: rotor-to-rotor and rotor-to-wing interference, "
            "blade-element aerodynamics, control-system behaviour, "
            "structural loads, or any thermal limit. Motor losses are "
            "folded into the ESC efficiency rather than given a separate "
            "electrical model.\n\n"
            "This is a performance-level model. It is for comparing designs "
            "and sizing a pack, not for certifying one.")

    help_menu.add_command(label="About / Version", command=_show_about)
    root.config(menu=menubar)

    def _set_scope(from_mission):
        status_scope.configure(text=(
            "Mission run — every row is the WORST value reached at any point: "
            "the highest current, C-rate and power, and the LOWEST reserve margin."
            if from_mission else
            "Fixed speed run — rows are evaluated at the cruise speed, with hover "
            "checked alongside because it is the heaviest steady load a VTOL carries."))
        metrics_scope.configure(text=(
            "Mission run — values at the LAST instant flown, not an average and "
            "not the worst case. Worst-case figures are on the Status tab."
            if from_mission else
            "Fixed speed run — a steady operating point at the cruise speed."))
        _set_sens_mode(from_mission)

    # ---- R1: match the multicopter and fixed-wing tab order ----------
    # A user moving between the three should find the same tabs in the same
    # places. These are created in whatever order their code happens to sit
    # in, so put them right once everything exists.
    _TAB_ORDER = ["Fixed Speed Plots", "Status", "Metrics", "Mission Plots",
                  "Weight Budget", "Power Budget", "Airframe Diagram",
                  "Mission Diagram", "Sensitivity", "Compare"]
    _by_title = {out_nb.tab(t, "text"): t for t in out_nb.tabs()}
    for _position, _title in enumerate(_TAB_ORDER):
        if _title in _by_title:
            out_nb.insert(_position, _by_title[_title])
    out_nb.select(0)

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

    def _parse_rails(spec: str) -> dict:
        """
        "5:2:0.9, 12:1.5:0.87" into {volts: (amps, efficiency)}.

        Stored as one string so the rails save and load with every other
        field rather than needing their own serialisation path.
        """
        rails = {}
        for part in str(spec or "").split(","):
            part = part.strip()
            if not part:
                continue
            bits = part.split(":")
            if len(bits) != 3:
                continue
            try:
                volts, amps, eff = (float(b) for b in bits)
            except ValueError:
                continue
            if volts > 0 and amps >= 0 and 0 < eff <= 1:
                rails[volts] = (amps, eff)
        return rails

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

    def _airframe_weight_g() -> float:
        """
        Airframe weight, by whichever route the user chose.

        "derive structure" is the original behaviour: they give the airframe
        weight and structure falls out as the residual. "enter structure"
        runs it the other way, for when the frame is known and the components
        are still being chosen.
        """
        entered = num("weight", 6000)
        if fields["mass_mode"].get() != "enter structure":
            return entered
        structure = opt("structure_mass") or 0.0
        parts = ((opt("cell_wt") or 0.0) * int(num("series", 1)) * int(num("parallel", 1))
                 + (opt("lift_wt") or 0.0) * int(num("n_lift", 1))
                 + (opt("lift_prop_wt") or 0.0) * int(num("n_lift", 1))
                 + (opt("cruise_wt") or 0.0) * int(num("n_cruise", 0))
                 + (opt("cruise_prop_wt") or 0.0) * int(num("n_cruise", 0))
                 + (opt("avionics_mass") or 0.0))
        return structure + parts

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
            soc_model=(fields["soc_model"].get().strip() or "auto"),
            unit_mode=(fields["unit_mode"].get().strip() or "cell"),
            cells_series_per_unit=int(opt("cells_s_per_pack") or 1),
            cells_parallel_per_unit=int(opt("cells_p_per_pack") or 1),
            pack_capacity_mAh=opt("pack_cap"),
            pack_weight_g=opt("pack_wt"),
            energy_density_Wh_per_kg=opt("energy_density"),
            discharge_cont_A=opt("a_cont"),
            discharge_max_A=opt("a_max"),
            charge_current_max_A=opt("charge_a"),
            soc_bp=_parse_float_list(fields["soc_bp"].get()),
            ocv_cell_bp=_parse_float_list(fields["ocv_cell_bp"].get()),
            r_scale_bp=_parse_float_list(fields["r_scale_bp"].get()),
        )
        temp = fields["temp"].get().strip()
        return VTOLConfig(
            config_type=v_config_type.get(),
            aircraft_weight_g=_airframe_weight_g(),
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
            avionics_power_W=(opt("avionics_flat") or num("avionics", 15)),
            periph_current_A=(opt("periph_current") or 0.0),
            # The ESC tab's field wins; the old Mission/Env one is kept as a
            # fallback so configs saved before the tab existed still load.
            esc_efficiency=(opt("esc_eff_tab") or num("esc_eff", 0.96)),
            esc_resistance_ohm=opt("esc_r") or 0.0,
            esc_max_current_A=opt("esc_imax"),
            esc_weight_g=(opt("esc_wt") or 0.0),
            avionics_rails=_parse_rails(fields["avionics_rails"].get()),
            # Pressure overrides the altitude-derived value, which is what
              # a field barometer reading is for: the standard atmosphere is
              # an average, and a real day is not.
            air_density=core.air_density(num("alt", 0),
                                         float(temp) if temp else None,
                                         opt("pressure")),
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
            lift_motor_max_current_A=opt("lift_imax"),
            lift_prop_max_thrust_g=opt("lift_max_thrust") or 0.0,
            cruise_motor_max_current_A=opt("cruise_imax"),
            cruise_motor_max_power_W=opt("cruise_pmax"),
            lift_prop_table_csv=(fields["lift_table"].get().strip() or None),
            cruise_prop_table_csv=(fields["cruise_table"].get().strip() or None),
            lift_prop_weight_g=opt("lift_prop_wt") or 0.0,
            cruise_prop_weight_g=opt("cruise_prop_wt") or 0.0,
            avionics_mass_g=opt("avionics_mass") or 0.0,
        )

    # What each metric means, keyed by its label. Kept beside the table
    # rather than inside show_metrics so the row list stays readable.
    _METRIC_NOTES = {
        "Configuration": "Which VTOL layout is being flown.",
        "Regime at cruise speed": "hover, transition or cruise — whether the "
            "wing is carrying the aircraft yet.",
        "All-up weight": "Everything that leaves the ground, payload included.",
        "Wing loading": "Weight per square metre of wing. Higher means a "
            "faster stall and a rougher ride in gusts.",
        "Disc loading (hover)": "Weight per square metre of rotor disc. The "
            "single strongest driver of hover efficiency: lower is better.",
        "Aspect ratio": "Span squared over wing area. Higher gives less "
            "induced drag and a better glide.",
        "Stall speed": "Slowest the wing can fly. Transition must happen "
            "above this with margin.",
        "Transition speed": "Where the wing can carry the whole aircraft and "
            "the rotors can stop.",
        "Hover power": "Electrical power to hold station. Usually the "
            "heaviest steady load a VTOL sees.",
        "Hover endurance": "How long the usable pack lasts hovering.",
        "Cruise power": "Electrical power at the cruise speed.",
        "Cruise endurance": "How long the usable pack lasts in cruise.",
        "Cruise range": "Still-air distance at the cruise speed. Wind "
            "changes this; it does not change endurance.",
        "Hover / cruise power": "How much dearer hovering is than cruising — "
            "the number that decides whether VTOL is worth it for a mission.",
        "Pack current": "Current drawn at the cruise speed.",
        "Loaded voltage": "Pack voltage under that current, after sag.",
        "Usable energy": "Pack energy minus the reserve you chose to keep.",
        "SoC model": "Where the discharge curve came from: a chemistry "
            "preset, or a measured CSV.",
        "Rotor thrust": "Total thrust the lift rotors are producing.",
        "Wing lift": "How much of the weight the wing is carrying.",
        "Rotor tilt": "Thrust direction, 0 deg straight up to 90 deg "
            "straight ahead. Vectored types only.",
        "L/D at cruise": "Lift over drag. Higher is a more efficient wing.",
        "Wind":
            "Steady wind speed and the direction it blows FROM. It changes "
            "what the flight achieves over the ground, never what it costs: "
            "power depends on airspeed alone.",
        "Course heading":
            "The direction this fixed-speed run is flying. With the wind it "
            "sets the head and cross components below. Missions ignore it — "
            "each leg carries its own course.",
        "Head / cross wind":
            "The wind resolved along and across the course. A positive "
            "headwind opposes the aircraft. The crosswind is what forces a "
            "crab, and crabbing spends airspeed that never becomes progress.",
        "Ground speed":
            "Progress along the course: sqrt(airspeed^2 - crosswind^2) minus "
            "the headwind. This, not the airspeed, is what sets the range.",
        "Station-keeping power":
            "Power to hold POSITION in this wind. Holding station means "
            "flying at the wind speed through the air, so translational lift "
            "makes it cheaper than still-air hover — in a strong enough wind "
            "the wing takes the load and the regime stops being hover at "
            "all. At zero wind this is exactly the hover figure.",
        "Station-keeping endurance":
            "How long the usable pack holds position in this wind. Longer "
            "than hover endurance whenever there is wind to lean on.",
        "Cruise speed":
            "The airspeed every fixed-speed figure on this tab is evaluated "
            "at, in the three units the job is usually quoted in.",
        "Pack configuration":
            "Cells in series x cells in parallel, and which entry mode "
            "produced them. Series sets voltage, parallel sets capacity.",
        "Pack capacity":
            "Total pack capacity. Parallel units multiply this; series units "
            "raise voltage instead, so a 2S pack of the same cells holds the "
            "same mAh at twice the volts.",
        "Energy density":
            "Pack energy over pack mass. Derived from what you entered unless "
            "you typed a figure, so comparing the two catches a wrong cell "
            "weight. Roughly 150-200 Wh/kg for LiPo, 200-260 for Li-ion.",
        "Discharge limit":
            "What the pack is rated to deliver. An amp figure is used as "
            "given; otherwise it is the C-rate times capacity.",
        "Charge time":
            "Capacity over the rated charge current — the constant-current "
            "part only. A real charge takes longer once it tapers, and the "
            "taper depends on the charger, so it is not modelled.",
        "Avionics at the pack":
            "What the avionics cost the battery, regulator losses included. "
            "With rails entered this is the sum of each rail's delivered "
            "power divided by its converter efficiency; otherwise it is the "
            "flat figure.",
        "Peripheral current":
            "Current drawn straight from the pack, bypassing the regulated "
            "rails. Adds to the avionics load rather than replacing it.",
        "Peripheral load at the pack":
            "That current at nominal pack voltage. It is charged against "
            "every flight regime, hover included, because a load wired to "
            "the main bus does not stop when the aircraft transitions.",
    }

    # Which sections the user had folded, remembered across runs. Without
    # this, collapsing "Battery" and pressing Run would silently reopen it.
    _metrics_open = {}
    _metrics_section = {"node": ""}

    def _metrics_add_section(title: str):
        node = metrics_tv.insert("", "end", text="", tags=("section",),
                                 values=(title, "", ""),
                                 open=_metrics_open.get(title, True))
        _metrics_section["node"] = node
        return node

    def _metrics_row(label: str, value: str, note: str = ""):
        metrics_tv.insert(_metrics_section["node"], "end", text="",
                          values=(label, value,
                                  note or _METRIC_NOTES.get(str(label).strip(), "")))

    def _metrics_remember_open(_e=None):
        for node in metrics_tv.get_children(""):
            title = metrics_tv.item(node, "values")[0]
            _metrics_open[title] = bool(metrics_tv.item(node, "open"))
    metrics_tv.bind("<<TreeviewOpen>>", _metrics_remember_open, add="+")
    metrics_tv.bind("<<TreeviewClose>>", _metrics_remember_open, add="+")

    def _metrics_toggle_section(evt):
        """
        Clicking anywhere on a section heading folds it, not just the small
        disclosure triangle — which is a 10-pixel target and the only way to
        do it otherwise.
        """
        row = metrics_tv.identify_row(evt.y)
        if not row or metrics_tv.parent(row):
            return
        if metrics_tv.identify_region(evt.x, evt.y) == "tree":
            return          # the triangle already handles this one
        metrics_tv.item(row, open=not metrics_tv.item(row, "open"))
        _metrics_remember_open()
    metrics_tv.bind("<Button-1>", _metrics_toggle_section, add="+")

    def _speed(mps: float) -> str:
        """
        A speed in m/s with the units a pilot and a customer actually use.

        The other two simulators show all three. A VTOL spec sheet quotes
        km/h, an airspace filing quotes knots, and the model works in m/s —
        converting by hand between them is where errors get made.
        """
        return f"{mps:.2f} m/s  ({mps * 3.6:.1f} km/h / {mps * 1.94384:.1f} kt)"

    def _mass(grams: float) -> str:
        """Mass in grams, with kilograms once it stops being a small number."""
        if grams >= 1000.0:
            return f"{grams:.0f} g  ({grams / 1000.0:.2f} kg)"
        return f"{grams:.0f} g"

    def show_metrics(m):
        for item in metrics_tv.get_children():
            metrics_tv.delete(item)

        _metrics_add_section("Aircraft")
        _metrics_row("Configuration", str(m["config_type"]))
        _metrics_row("Regime at cruise speed", str(m["regime"]))
        _metrics_row("All-up weight",
                     f"{_mass(m['all_up_weight_g'])}  ({m['weight_N']:.1f} N)")
        _metrics_row("Wing loading", f"{m['wing_loading_N_m2']:.1f} N/m²")
        _metrics_row("Disc loading (hover)", f"{m['disc_loading_N_m2']:.1f} N/m²")
        _metrics_row("Aspect ratio", f"{m['aspect_ratio']:.2f}")

        _metrics_add_section("Speeds")
        _metrics_row("Stall speed", _speed(m["stall_speed_mps"]))
        _metrics_row("Transition speed", _speed(m["transition_speed_mps"]))
        _metrics_row("Cruise speed", _speed(m["airspeed_mps"]))

        _metrics_add_section("Hover")
        _metrics_row("Hover power", f"{m['hover_power_W']:.0f} W")
        _metrics_row("Hover endurance",
                     f"{m['hover_endurance_min']:.1f} min  "
                     f"({m['hover_endurance_min'] * 60:.0f} s)")

        _metrics_add_section("Cruise")
        _metrics_row("Cruise power", f"{m['total_power_W']:.0f} W")
        _metrics_row("  rotor share", f"{m['rotor_shaft_W']:.0f} W")
        _metrics_row("  cruise prop share", f"{m['cruise_shaft_W']:.0f} W")
        _metrics_row("Cruise endurance", f"{m['cruise_endurance_min']:.1f} min")
        _metrics_row("Cruise range",
                     f"{m['cruise_range_km']:.2f} km  "
                     f"({m['cruise_range_km'] * 0.539957:.2f} NM / "
                     f"{m['cruise_range_km'] * 0.621371:.2f} mi)"
                     + ("" if m["wind_mps"] <= 0 else
                        f"   [still air {m['cruise_range_still_air_km']:.2f} km]"))
        _metrics_row("Hover / cruise power",
                     f"{m['hover_to_cruise_power_ratio']:.2f} x")

        _metrics_add_section("Wind")
        _metrics_row("Wind", _speed(m["wind_mps"])
                     + (f"  from {m['wind_direction_deg']:.0f} deg"
                        if m["wind_mps"] > 0 else ""))
        _metrics_row("Course heading", f"{m['course_deg']:.0f} deg")
        _metrics_row("Head / cross wind",
                     f"{m['wind_head_mps']:+.2f} / {m['wind_cross_mps']:+.2f} m/s")
        _metrics_row("Ground speed", _speed(m["groundspeed_mps"]))
        _metrics_row("Station-keeping power",
                     f"{m['station_power_W']:.0f} W  ({m['station_regime']})")
        _metrics_row("Station-keeping endurance",
                     f"{m['station_endurance_min']:.1f} min")

        _metrics_add_section("Battery")
        # NOT "Configuration": _METRIC_NOTES is keyed by label, and the
        # Aircraft section already owns that one — a duplicate label silently
        # inherits the other section's explanation.
        _metrics_row("Pack configuration",
                     f"{m['battery_series_cells']}S x {m['battery_parallel_cells']}P  "
                     f"({m['battery_total_cells']} cells, entered by "
                     f"{m['battery_unit_mode']})")
        _metrics_row("Pack capacity",
                     f"{m['battery_capacity_mAh']:.0f} mAh  "
                     f"({m['battery_capacity_mAh'] / 1000.0:.2f} Ah)")
        _metrics_row("Pack current", f"{m['pack_current_A']:.2f} A")
        _metrics_row("Loaded voltage", f"{m['v_load_V']:.2f} V")
        _metrics_row("Usable energy", f"{m['usable_Wh']:.1f} Wh")
        _metrics_row("Battery weight",
                     f"{_mass(m['battery_weight_g'])}  "
                     f"({m['battery_weight_g'] / max(m['all_up_weight_g'], 1e-9) * 100:.0f}% of AUW)")
        _metrics_row("Energy density",
                     f"{m['battery_energy_density_Wh_per_kg']:.0f} Wh/kg")
        _metrics_row("Discharge limit",
                     "not specified" if m["battery_cont_A"] is None else
                     (f"{m['battery_cont_A']:.0f} A cont"
                      + ("" if m["battery_max_A"] is None
                         else f" / {m['battery_max_A']:.0f} A burst")))
        _metrics_row("Charge time",
                     "no charge current entered" if m["battery_charge_time_h"] is None
                     else f"{m['battery_charge_time_h']:.2f} h")
        _metrics_row("SoC model", str(m["soc_model"]))

        _metrics_add_section("Systems Load")
        _metrics_row("Avionics at the pack",
                     f"{m['avionics_input_power_W']:.1f} W")
        _metrics_row("Peripheral current", f"{m['periph_current_A']:.2f} A")
        _metrics_row("Peripheral load at the pack",
                     f"{m['peripheral_power_W']:.1f} W")

        _metrics_add_section("Aerodynamics")
        _metrics_row("Stopped-rotor drag area",
                     f"{m['stopped_rotor_drag_area_m2']*1e4:.0f} cm²")

    def draw_plots(cfg):
        v_stall = stall_speed_mps(cfg)
        v_trans = transition_speed_mps(cfg)
        v_cruise = float(cfg.cruise_speed_mps)
        # Plot Settings wins when a maximum was entered, so the sweep and the
        # export cover the same range.
        v_max = (opt("plot_vmax")
                 or max(v_cruise * 1.6, v_trans * 1.6))
        steps = max(int(v_max / 0.5), 8)
        speeds = [v_max * i / steps for i in range(1, steps + 1)]

        powers, rotor_W, cruise_W = [], [], []
        endurance, ranges, share, rotor_N, wing_N = [], [], [], [], []
        usable_Wh = cfg.battery.usable_Wh
        # The sweep is flown in the same wind as the run. Range is ground
        # distance, so a headwind moves the best-range speed UP — the
        # aircraft has to fly faster to spend less time being pushed back.
        # Plotting still-air range in a wind would hide exactly that.
        _sweep_head, _sweep_cross = core.wind_components_mps(
            num("wind", 0.0), num("wind_dir", 0.0), num("course_deg", 0.0))
        for v in speeds:
            p = power_at_airspeed(cfg, v)
            total = p["total_power_W"]
            powers.append(total)
            rotor_W.append(p["rotor_shaft_W"])
            cruise_W.append(p["cruise_shaft_W"])
            # Endurance and range at a STEADY speed, which is what a
            # fixed-speed sweep describes. A real flight also pays for the
            # climb and the transition; the mission run is what prices those.
            mins = usable_Wh / max(total, 1e-9) * 60.0
            endurance.append(mins)
            ranges.append(mins * 60.0 * core.groundspeed_along_track_mps(
                v, _sweep_head, _sweep_cross) / 1000.0)
            share.append(float(p.get("lift_share_wing",
                                     1.0 if p.get("regime") == "cruise" else 0.0)))
            rotor_N.append(float(p.get("rotor_thrust_N", 0.0)))
            wing_N.append(float(p.get("wing_lift_N", 0.0)))

        best_e = max(range(len(speeds)), key=lambda i: endurance[i])
        best_r = max(range(len(speeds)), key=lambda i: ranges[i])

        fig, axes = core.make_figure(
            2, 2, figsize=(_view["plot_w"], _view["plot_h"]))
        fig.suptitle(f"{cfg.config_type} Performance", fontsize=13,
                     fontweight="bold")

        def _marks(ax, legend=False):
            """The three speeds every panel is read against."""
            ax.axvline(v_trans, color="#6A1B9A", linestyle=":", linewidth=1.2,
                       label=f"transition {v_trans:.1f} m/s" if legend else None)
            ax.axvline(v_cruise, color="gray", linestyle="-.", linewidth=1.0,
                       alpha=0.8,
                       label=f"cruise {v_cruise:.1f} m/s" if legend else None)
            ax.axvline(v_stall, color="#EF6C00", linestyle=":", linewidth=1.0,
                       alpha=0.8,
                       label=f"stall {v_stall:.1f} m/s" if legend else None)

        # 1. Power required, against the hover figure it has to beat.
        ax1 = axes[0, 0]
        ax1.plot(speeds, powers, color="#C62828", label="Total")
        ax1.axhline(hover_power_W(cfg)["total_power_W"], color="#1565C0",
                    linestyle="--", label="Hover")
        _marks(ax1, legend=True)
        ax1.set_xlabel("Airspeed (m/s)"); ax1.set_ylabel("Power (W)")
        ax1.set_title("Power vs Airspeed"); ax1.grid(alpha=0.3)
        ax1.legend(fontsize=7)

        # 2. Where that power is going.
        ax2 = axes[0, 1]
        ax2.stackplot(speeds, rotor_W, cruise_W,
                      labels=["Lift rotors", "Cruise prop"],
                      colors=["#90CAF9", "#A5D6A7"])
        _marks(ax2)
        ax2.set_xlabel("Airspeed (m/s)"); ax2.set_ylabel("Shaft power (W)")
        ax2.set_title("Where the power goes"); ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7, loc="upper center")

        # 3. What that costs in time and distance — the two speeds a mission
        # is actually planned around, and they are not the same speed.
        ax3 = axes[1, 0]
        ax3b = ax3.twinx()
        l1, = ax3.plot(speeds, endurance, color="royalblue",
                       label="Endurance (min)")
        l2, = ax3b.plot(speeds, ranges, color="darkorange", linestyle="--",
                        label="Range (km)")
        ax3.axvline(speeds[best_e], color="royalblue", linestyle=":", linewidth=1.2)
        ax3b.axvline(speeds[best_r], color="darkorange", linestyle=":", linewidth=1.2)
        ax3.set_xlabel("Airspeed (m/s)"); ax3.set_ylabel("Endurance (min)")
        ax3b.set_ylabel("Range (km)")
        ax3.set_title(f"Endurance & Range  (best {speeds[best_e]:.1f} / "
                      f"{speeds[best_r]:.1f} m/s)")
        ax3.grid(alpha=0.3)
        ax3.legend(handles=[l1, l2], fontsize=7, loc="lower right")

        # 4. Who is carrying the weight. This is the panel that makes a VTOL
        # a VTOL: below the transition the rotors are still lifting, and the
        # wing takes over as the share reaches 1.
        ax4 = axes[1, 1]
        # Handles collected as they are drawn. ax.get_lines() would sweep up
        # the three axvline markers too, and they arrive in the legend as
        # "_child1", "_child2", "_child3".
        h_share, = ax4.plot(speeds, [sh * 100.0 for sh in share],
                            color="#2E7D32", label="Wing share of lift (%)")
        ax4b = ax4.twinx()
        h_rotor, = ax4b.plot(speeds, rotor_N, color="#1565C0", linestyle="--",
                             label="Rotor thrust (N)")
        h_wing, = ax4b.plot(speeds, wing_N, color="#8E24AA", linestyle=":",
                            label="Wing lift (N)")
        _marks(ax4)
        ax4.set_ylim(-2, 105)
        ax4.set_xlabel("Airspeed (m/s)"); ax4.set_ylabel("Wing share (%)")
        ax4b.set_ylabel("Force (N)")
        ax4.set_title("Lift Handover")
        ax4.grid(alpha=0.3)
        handles = [h_share, h_rotor, h_wing]
        ax4.legend(handles, [h.get_label() for h in handles], fontsize=7,
                   loc="center right")

        # w_pad keeps the right-hand y-label of one panel clear of the
        # left-hand y-label of the next: both bottom panels carry a twin
        # axis, so there are four y-labels competing for two gaps.
        # Constrained layout, not tight_layout: the plot pane squashes the
        # figure to its own width (about half the requested 14in), and
        # tight_layout's padding was tuned for the full width — at the real
        # rendered size it clipped the left-hand y-labels off the figure and
        # ran the twin-axis labels of the two bottom panels into each other.
        # Constrained layout measures the labels that are actually there,
        # including the twinx ones it is given, and adapts.
        fig.set_layout_engine("constrained", w_pad=0.06, h_pad=0.06,
                              wspace=0.08, hspace=0.08)

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
        _refresh_plot_scrollregion()

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
                  text="These plots come from a fixed speed sweep.\n\nYou have "
                       "just run a mission, so there is nothing to show here.").grid(
            row=0, column=0, padx=20, pady=40)
        _refresh_plot_scrollregion()

    def run_single_point():
        try:
            cfg = build_config()
            m = compute_metrics(cfg,
                                wind_mps=num("wind", 0.0),
                                wind_direction_deg=num("wind_dir", 0.0),
                                course_deg=num("course_deg", 0.0))
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
        update_rotor_loading(cfg, m)
        _export_state.update(cfg=cfg, from_mission=False, results=None)
        update_power_budget(cfg, m)
        clear_mission_plots()
        clear_mission_diagram()
        _set_scope(False)
        _sens_state.update(cfg=cfg, mission=None,
                           wind=num("wind", 0.0),
                           wind_dir=num("wind_dir", 0.0),
                           course=num("course_deg", 0.0))
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
        update_rotor_loading(cfg, last)
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
        v_loaded_cfg.set(os.path.basename(path))
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
        # The rails live in a hidden field, so the table has to be repainted
        # from it — otherwise a loaded config would apply rails the user
        # cannot see or edit.
        _av_refresh()
        v_loaded_cfg.set(os.path.basename(path))
        log(f"Loaded {os.path.basename(path)}\n"
            f"Press Run Fixed Speed Sweep to evaluate it.\n")

    # Laid out with grid, left to right, in the same order as the other two
    # simulators. pack(side="right") reversed the right-hand group, so the
    # bar read Export CSV ... Save Config — the exact opposite of the
    # multicopter and fixed-wing, which defeats muscle memory.
    buttons = ttk.Frame(root, padding=6)
    buttons.grid(row=2, column=0, columnspan=2, sticky="ew")
    buttons.columnconfigure(1, weight=1)
    ttk.Button(buttons, text="▶  Run Fixed Speed Sweep",
               command=run_single_point).grid(row=0, column=0, padx=(0, 6), pady=4)
    ttk.Button(buttons, text="📋  Run Mission (JSON)",
               command=run_mission).grid(row=0, column=1, padx=(0, 6), pady=4,
                                         sticky="w")
    ttk.Button(buttons, text="💾  Save Config",
               command=save_config).grid(row=0, column=2, padx=4, pady=4)
    ttk.Button(buttons, text="📂  Load Config",
               command=load_config).grid(row=0, column=3, padx=4, pady=4)
    ttk.Button(buttons, text="📊  Export CSV",
               command=lambda: export_csv()).grid(row=0, column=4, padx=4, pady=4)
    ttk.Button(buttons, text="📗  Export Excel",
               command=lambda: export_excel()).grid(row=0, column=5, padx=4, pady=4)
    ttk.Button(buttons, text="📄  Generate Report",
               command=lambda: generate_report()).grid(row=0, column=6, padx=4,
                                                       pady=4)

    # The startup banner is the only orientation a first-time user gets, so
    # it says where the controls are rather than only what the tool is.
    log("\n".join([
        f"VTOL Power Simulator  v{SIM_VERSION}",
        "=" * 52,
        "All four configurations are implemented: lift+cruise, tiltrotor,",
        "tiltwing and tailsitter.",
        "",
        "Input detail is set to Simple (selector above the input tabs).",
        "Hover the blue ? beside any field for an explanation and a",
        "typical value. Switch to Advanced to reveal every input.",
        "Metrics sections collapse - click a section heading to fold it.",
        "",
        "Load Config -> examples/configs/ for ready-made aircraft.",
        "Run Mission (JSON) -> examples/missions/ for flight profiles.",
        "",
        "Press Run Fixed Speed Sweep to begin.",
    ]) + "\n")

    root.mainloop()


if __name__ == "__main__":
    main()
