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
from matplotlib.lines import Line2D as _Line2D
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

SIM_VERSION = "1.11.0"
SIM_BUILD_NOTE = "VTOL simulator - lift+cruise, tiltrotor, tiltwing, tailsitter; motor electrical model; multicopter and fixed-wing outputs"

G0 = core.G0

CONFIG_TYPES = ["lift+cruise", "tiltrotor", "tiltwing", "tailsitter"]
IMPLEMENTED_CONFIG_TYPES = {"lift+cruise", "tiltrotor", "tiltwing", "tailsitter"}

# Rotor inflow efficiency against advance ratio, the multicopter's defaults.
# Off unless enabled: the forward-flight induced-velocity solver already
# carries translational lift, and this map is an empirical correction on top.
DEFAULT_INFLOW_MU_BP = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.50]
DEFAULT_INFLOW_EFF_BP = [1.00, 1.04, 1.08, 1.06, 1.00, 0.94, 0.88]


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
                 r_scale_bp: Optional[List[float]] = None,
                 # --- thermal limits, checked on Status --------------
                 max_time_s: Optional[float] = None,
                 temp_limit_C: float = 55.0):
        self.chemistry = chemistry
        self.max_time_s = (None if max_time_s in (None, "") else float(max_time_s))
        self.temp_limit_C = float(temp_limit_C or 55.0)

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
                 cruise_prop_table_csv: Optional[str] = None,
                 # --- motor electrical detail (both rotor groups) --------
                 lift_motor_i0_A: float = 0.5,
                 lift_motor_v0_V: Optional[float] = None,
                 cruise_motor_i0_A: float = 0.5,
                 cruise_motor_v0_V: Optional[float] = None,
                 lift_motor_max_time_s: Optional[float] = None,
                 cruise_motor_max_time_s: Optional[float] = None,
                 lift_motor_temp_limit_C: float = 100.0,
                 cruise_motor_temp_limit_C: float = 100.0,
                 lift_motor_v_unit: str = "S",
                 lift_motor_rating_min: Optional[float] = None,
                 lift_motor_rating_max: Optional[float] = None,
                 cruise_motor_v_unit: str = "S",
                 cruise_motor_rating_min: Optional[float] = None,
                 cruise_motor_rating_max: Optional[float] = None,
                 lift_motor_pole_count: int = 14,
                 cruise_motor_pole_count: int = 14,
                 lift_motor_size: str = "",
                 cruise_motor_size: str = "",
                 # --- propeller detail -----------------------------------
                 lift_prop_blades: int = 2,
                 cruise_prop_blades: int = 2,
                 lift_prop_max_rpm: Optional[float] = None,
                 cruise_prop_max_rpm: Optional[float] = None,
                 lift_prop_tconst: Optional[float] = None,
                 lift_prop_pconst: Optional[float] = None,
                 cruise_prop_tconst: Optional[float] = None,
                 cruise_prop_pconst: Optional[float] = None,
                 cruise_prop_max_thrust_g: float = 0.0,
                 cruise_prop_eff_model: str = "constant",
                 # --- ESC detail -----------------------------------------
                 esc_cont_current_A: Optional[float] = None,
                 esc_idle_current_A: float = 0.0,
                 esc_max_time_s: Optional[float] = None,
                 esc_temp_limit_C: float = 90.0,
                 esc_v_unit: str = "S",
                 esc_rating_min: Optional[float] = None,
                 esc_rating_max: Optional[float] = None,
                 # --- lift-rotor layout ----------------------------------
                 lift_rotor_layout: str = "flat",
                 coaxial_spacing_m: Optional[float] = None,
                 inflow_map_enabled: bool = False,
                 inflow_mu_bp: Optional[List[float]] = None,
                 inflow_eff_bp: Optional[List[float]] = None,
                 # --- extra airframe drag beyond the wing's CD0 ----------
                 drag_model_mode: str = "auto",
                 parasite_drag_cd: Optional[float] = None,
                 parasite_area_m2: Optional[float] = None,
                 profile_drag_cd: Optional[float] = None,
                 profile_area_m2: Optional[float] = None,
                 body_length_m: Optional[float] = None,
                 body_width_m: Optional[float] = None,
                 body_height_m: Optional[float] = None,
                 arm_length_m: Optional[float] = None,
                 arm_width_m: Optional[float] = None,
                 drag_cg_offset_m: float = 0.0,
                 # --- hover attitude limits ------------------------------
                 max_tilt_deg: float = 25.0,
                 max_pitch_deg: Optional[float] = None,
                 max_roll_deg: Optional[float] = None,
                 # --- runway, climb and turn -----------------------------
                 mu_roll: float = 0.04,
                 mu_brake: float = 0.30,
                 CL_takeoff: float = 0.80,
                 cruise_altitude_m: Optional[float] = None,
                 # --- run settings the single-point figures read ---------
                 bank_deg: float = 0.0,
                 climb_rate_mps: float = 0.0,
                 descent_rate_mps: float = 0.0,
                 reserve_percent: Optional[float] = None,
                 transient_dt_s: Optional[float] = None,
                 min_climb_mps: Optional[float] = None,
                 field_takeoff_m: Optional[float] = None,
                 field_landing_m: Optional[float] = None,
                 ambient_temp_C: Optional[float] = None,
                 pressure_Pa: Optional[float] = None):
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

        def _opt(x):
            return None if x in (None, "") else float(x)

        # Motor electrical detail. Kv, Rm and I0 drive the motor model below;
        # the ratings, limits and size are checked on Status or shown on
        # Metrics. Blank ratings mean "not specified", never an invented limit.
        self.lift_motor_i0_A = max(float(lift_motor_i0_A or 0.0), 0.0)
        self.cruise_motor_i0_A = max(float(cruise_motor_i0_A or 0.0), 0.0)
        self.lift_motor_v0_V = _opt(lift_motor_v0_V)
        self.cruise_motor_v0_V = _opt(cruise_motor_v0_V)
        self.lift_motor_max_time_s = _opt(lift_motor_max_time_s)
        self.cruise_motor_max_time_s = _opt(cruise_motor_max_time_s)
        self.lift_motor_temp_limit_C = float(lift_motor_temp_limit_C or 100.0)
        self.cruise_motor_temp_limit_C = float(cruise_motor_temp_limit_C or 100.0)
        self.lift_motor_v_unit = str(lift_motor_v_unit or "S").strip().upper()[:1] or "S"
        self.cruise_motor_v_unit = str(cruise_motor_v_unit or "S").strip().upper()[:1] or "S"
        self.lift_motor_rating_min = _opt(lift_motor_rating_min)
        self.lift_motor_rating_max = _opt(lift_motor_rating_max)
        self.cruise_motor_rating_min = _opt(cruise_motor_rating_min)
        self.cruise_motor_rating_max = _opt(cruise_motor_rating_max)
        self.lift_motor_pole_count = max(int(lift_motor_pole_count or 14), 2)
        self.cruise_motor_pole_count = max(int(cruise_motor_pole_count or 14), 2)
        self.lift_motor_size = str(lift_motor_size or "")
        self.cruise_motor_size = str(cruise_motor_size or "")

        self.lift_prop_blades = max(int(lift_prop_blades or 2), 1)
        self.cruise_prop_blades = max(int(cruise_prop_blades or 2), 1)
        self.lift_prop_max_rpm = _opt(lift_prop_max_rpm)
        self.cruise_prop_max_rpm = _opt(cruise_prop_max_rpm)
        self.lift_prop_tconst = _opt(lift_prop_tconst)
        self.lift_prop_pconst = _opt(lift_prop_pconst)
        self.cruise_prop_tconst = _opt(cruise_prop_tconst)
        self.cruise_prop_pconst = _opt(cruise_prop_pconst)
        self.cruise_prop_max_thrust_g = max(float(cruise_prop_max_thrust_g or 0.0), 0.0)
        model = str(cruise_prop_eff_model or "constant").strip().lower()
        self.cruise_prop_eff_model = "curve" if model == "curve" else "constant"

        # ESC detail. The efficiency above is the ESC's switching loss; its
        # resistance and idle current add on top when entered.
        self.esc_cont_current_A = _opt(esc_cont_current_A)
        self.esc_idle_current_A = max(float(esc_idle_current_A or 0.0), 0.0)
        self.esc_max_time_s = _opt(esc_max_time_s)
        self.esc_temp_limit_C = float(esc_temp_limit_C or 90.0)
        self.esc_v_unit = str(esc_v_unit or "S").strip().upper()[:1] or "S"
        self.esc_rating_min = _opt(esc_rating_min)
        self.esc_rating_max = _opt(esc_rating_max)

        layout = str(lift_rotor_layout or "flat").strip().lower()
        self.lift_rotor_layout = "coaxial" if layout.startswith("coax") else "flat"
        self.coaxial_spacing_m = _opt(coaxial_spacing_m)
        self.inflow_map_enabled = bool(inflow_map_enabled)
        self.inflow_mu_bp = list(inflow_mu_bp or DEFAULT_INFLOW_MU_BP)
        self.inflow_eff_bp = list(inflow_eff_bp or DEFAULT_INFLOW_EFF_BP)
        if len(self.inflow_mu_bp) != len(self.inflow_eff_bp) or len(self.inflow_mu_bp) < 2:
            self.inflow_mu_bp = list(DEFAULT_INFLOW_MU_BP)
            self.inflow_eff_bp = list(DEFAULT_INFLOW_EFF_BP)

        mode = str(drag_model_mode or "auto").strip().lower()
        self.drag_model_mode = mode if mode in ("auto", "manual", "geometry") else "auto"
        self.parasite_drag_cd = _opt(parasite_drag_cd)
        self.parasite_area_m2 = _opt(parasite_area_m2)
        self.profile_drag_cd = _opt(profile_drag_cd)
        self.profile_area_m2 = _opt(profile_area_m2)
        self.body_length_m = _opt(body_length_m)
        self.body_width_m = _opt(body_width_m)
        self.body_height_m = _opt(body_height_m)
        self.arm_length_m = _opt(arm_length_m)
        self.arm_width_m = _opt(arm_width_m)
        self.drag_cg_offset_m = float(drag_cg_offset_m or 0.0)

        self.max_tilt_deg = min(max(float(max_tilt_deg or 25.0), 1.0), 85.0)
        self.max_pitch_deg = _opt(max_pitch_deg)
        self.max_roll_deg = _opt(max_roll_deg)

        self.mu_roll = max(float(mu_roll if mu_roll is not None else 0.04), 0.0)
        self.mu_brake = max(float(mu_brake if mu_brake is not None else 0.30), 0.0)
        self.CL_takeoff = max(float(CL_takeoff or 0.8), 1e-3)
        self.cruise_altitude_m = (None if cruise_altitude_m in (None, "")
                                  else max(float(cruise_altitude_m), 0.0))

        # Carried on the aircraft rather than passed alongside it, so that
        # Sensitivity and Compare — which re-run a copy of the config — see
        # exactly the settings the original run used.
        self.bank_deg = min(max(float(bank_deg or 0.0), 0.0), 80.0)
        climb, descent = max(float(climb_rate_mps or 0.0), 0.0), max(float(descent_rate_mps or 0.0), 0.0)
        # Climbing and descending at once is contradictory; the multicopter
        # keeps the climb, and so does this.
        self.climb_rate_mps = climb
        self.descent_rate_mps = 0.0 if climb > 0 else descent
        self.reserve_percent = _opt(reserve_percent)
        self.transient_dt_s = _opt(transient_dt_s)
        self.min_climb_mps = _opt(min_climb_mps)
        self.field_takeoff_m = _opt(field_takeoff_m)
        self.field_landing_m = _opt(field_landing_m)
        # Density is what the physics uses; these are kept for display and
        # for the thermal estimates, which need an ambient to rise from.
        self._ambient_temp_C = _opt(ambient_temp_C)
        self.pressure_Pa = _opt(pressure_Pa)

    @property
    def ambient_temp_C(self) -> float:
        """Entered temperature, else the standard atmosphere's at the field."""
        if self._ambient_temp_C is not None:
            return self._ambient_temp_C
        return 15.0 - core.LAPSE_K_PER_M * self.reference_altitude_m

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


def extra_drag_areas(cfg: VTOLConfig) -> Dict[str, object]:
    """
    Drag of the airframe BEYOND the wing's CD0, as Cd x area, in m^2.

    The multicopter's inputs, carried over: a parasite term (the frontal
    silhouette, met in forward flight) and a profile term (the side
    silhouette, met when hovering level in a wind). Entered as Cd and area,
    or derived from a box body and square-tube booms.

    Everything blank gives zero, so CD0 remains the whole-aircraft drag it
    has always been. Use these when CD0 describes the wing alone and the
    fuselage, booms or payload pod are to be added separately.

    manual:   use the entered Cd and area only
    geometry: derive from the body and boom dimensions
    auto:     entered Cd and area if any, else geometry if given, else none
    """
    manual_front = ((cfg.parasite_drag_cd or 0.0) * (cfg.parasite_area_m2 or 0.0))
    manual_side = ((cfg.profile_drag_cd or 0.0) * (cfg.profile_area_m2 or 0.0))
    have_manual = any(x for x in (cfg.parasite_drag_cd, cfg.parasite_area_m2,
                                  cfg.profile_drag_cd, cfg.profile_area_m2))
    have_geometry = bool(cfg.body_width_m and cfg.body_height_m)
    mode = cfg.drag_model_mode
    if mode == "manual" or (mode == "auto" and have_manual):
        return {"frontal_CdA": manual_front, "side_CdA": manual_side,
                "source": "entered"}
    if not have_geometry or mode == "manual":
        return {"frontal_CdA": 0.0, "side_CdA": 0.0, "source": "none"}

    CD_BOX, CD_BOOM = 1.05, 1.10          # the multicopter's constants
    w, h = float(cfg.body_width_m), float(cfg.body_height_m)
    length = float(cfg.body_length_m or 0.0)
    tube = float(cfg.arm_width_m or 0.02)
    boom_len = float(cfg.arm_length_m or 0.0)
    n_booms = max(cfg.num_lift_rotors // (2 if cfg.lift_rotor_layout == "coaxial" else 1), 1)
    if uses_vectored_thrust(cfg):
        # Nacelle pylons on the wing: the multicopter's 70% projection.
        boom_front = n_booms * tube * boom_len * 0.7
    else:
        # A lift+cruise's booms run fore and aft, so they meet the airflow
        # end-on in cruise and broadside only from the side.
        boom_front = n_booms * tube * tube
    boom_side = n_booms * tube * boom_len
    return {"frontal_CdA": CD_BOX * w * h + CD_BOOM * boom_front,
            "side_CdA": CD_BOX * length * h + CD_BOOM * boom_side,
            "source": "geometry"}


def body_drag_N(cfg: VTOLConfig, airspeed_mps: float) -> float:
    """Forward-flight drag of the fuselage and booms beyond CD0. Zero by default."""
    v = max(float(airspeed_mps), 0.0)
    return 0.5 * cfg.air_density * v * v * float(extra_drag_areas(cfg)["frontal_CdA"])


def hover_wind_drag_N(cfg: VTOLConfig, wind_mps: float) -> float:
    """
    Drag on the airframe hovering level in a wind, from the side silhouette.

    Falls back to the frontal term when only that was given, since a
    hovering aircraft can meet the wind from any side.
    """
    areas = extra_drag_areas(cfg)
    cda = float(areas["side_CdA"] or areas["frontal_CdA"])
    v = max(float(wind_mps), 0.0)
    return 0.5 * cfg.air_density * v * v * cda


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


# ============================================================
# MOTOR ELECTRICAL MODEL
# ============================================================
#
# Shaft power becomes electrical power through the motor, and the motor has
# losses of its own that neither the figure of merit (a rotor number) nor
# the propeller efficiency (a propeller number) contains:
#
#     I      = Q / Kt + I0            Q = P_shaft / omega,  Kt = 60 / (2 pi Kv)
#     V_emf  = RPM / Kv
#     V_term = V_emf + I * Rm
#     P_elec = V_term * I  =  P_shaft + I0 * V_emf + I^2 * Rm
#
# so the loss is the no-load (iron and friction) term plus copper. RPM comes
# from the thrust through a thrust coefficient: TConst if entered, else one
# fitted from a bench table's RPM column, else the shared estimate from
# diameter, pitch and blade count. At an axial airspeed V the blade meets the
# air at a lower angle and thrust falls roughly linearly with advance ratio,
#
#     T = C_T * rho * n^2 * D^4 * (1 - J / J0),    J = V / (n D),  J0 ~ P / D
#
# which is solved for n in closed form.
#
# A motor whose Kv is 0 or blank has no electrical model: its loss is taken
# as zero, so the ESC efficiency is then the only conversion loss, which is
# how every version before this one behaved.


def _group(cfg: VTOLConfig, group: str) -> dict:
    """
    The hardware of one rotor group.

    A vectored type (tiltrotor, tiltwing, tailsitter) has ONE set of rotors
    that both lifts and cruises, so its "cruise" group is the lift group.
    """
    if group == "cruise" and not uses_vectored_thrust(cfg):
        return {
            "name": "cruise", "n": cfg.num_cruise_motors,
            "d_in": cfg.cruise_prop_diameter_in, "p_in": cfg.cruise_prop_pitch_in,
            "blades": cfg.cruise_prop_blades, "kv": cfg.cruise_motor_kv,
            "rm": cfg.cruise_motor_resistance, "i0": cfg.cruise_motor_i0_A,
            "v0": cfg.cruise_motor_v0_V, "tconst": cfg.cruise_prop_tconst,
            "pconst": cfg.cruise_prop_pconst, "table": cfg.cruise_prop_table,
            "imax": cfg.cruise_motor_max_current_A,
            "pmax": cfg.cruise_motor_max_power_W,
            "max_rpm": cfg.cruise_prop_max_rpm,
            "poles": cfg.cruise_motor_pole_count,
        }
    return {
        "name": "lift", "n": cfg.num_lift_rotors,
        "d_in": cfg.lift_prop_diameter_in, "p_in": cfg.lift_prop_pitch_in,
        "blades": cfg.lift_prop_blades, "kv": cfg.lift_motor_kv,
        "rm": cfg.lift_motor_resistance, "i0": cfg.lift_motor_i0_A,
        "v0": cfg.lift_motor_v0_V, "tconst": cfg.lift_prop_tconst,
        "pconst": cfg.lift_prop_pconst, "table": cfg.lift_prop_table,
        "imax": cfg.lift_motor_max_current_A,
        "pmax": cfg.lift_motor_max_power_W,
        "max_rpm": cfg.lift_prop_max_rpm,
        "poles": cfg.lift_motor_pole_count,
    }


def prop_coefficients(cfg: VTOLConfig, group: str) -> Dict[str, object]:
    """
    Thrust and power coefficients for one group, and where they came from.

    Priority: entered TConst/PConst, then a fit to a bench table with an RPM
    column, then the shared estimate. The source is returned because a
    coefficient that is a +/-30% estimate should say so on Metrics.
    """
    g = _group(cfg, group)
    fit = None
    if g["table"] is not None and "RPM" in g["table"]:
        # Keyed by the table and diameter: a sensitivity lever that scales the
        # diameter must not reuse a coefficient fitted at the old one.
        cache = cfg.__dict__.setdefault("_coeff_cache", {})
        key = (group, id(g["table"]), round(float(g["d_in"]), 6))
        if key not in cache:
            cache[key] = core.derive_prop_coefficients_from_table(g["table"], g["d_in"])
        fit = cache[key]
    if g["tconst"]:
        c_t, source = float(g["tconst"]), "entered"
    elif fit and fit.get("c_t"):
        c_t, source = float(fit["c_t"]), "bench table"
    else:
        c_t, source = core.estimate_prop_thrust_coefficient(
            g["d_in"], g["p_in"], g["blades"]), "estimate"
    if g["pconst"]:
        c_p = float(g["pconst"])
    elif fit and fit.get("c_p"):
        c_p = float(fit["c_p"])
    else:
        fom = cfg.lift_figure_of_merit if g["name"] == "lift" else 0.65
        c_p = core.estimate_prop_power_coefficient(c_t, fom)
    return {"c_t": c_t, "c_p": c_p, "source": source}


def prop_rpm(cfg: VTOLConfig, group: str, thrust_per_rotor_N: float,
             axial_mps: float = 0.0) -> float:
    """
    RPM of one rotor making `thrust_per_rotor_N` with `axial_mps` of
    freestream through its disc.

    Static thrust from the coefficient, reduced linearly with advance ratio
    to zero at J0, the advance ratio at which a fixed-pitch blade stops
    producing thrust. Published APC data puts J0 at about 1.2 x pitch /
    diameter (a 10x7 reaches zero thrust near J = 0.85). Solving
    T = C_T rho D^4 (n^2 - n V / (D J0)) for n gives the closed form below;
    at V = 0 it is the familiar static result.
    """
    t = max(float(thrust_per_rotor_N), 0.0)
    if t <= 0:
        return 0.0
    g = _group(cfg, group)
    d = max(float(g["d_in"]) * 0.0254, 1e-6)
    rho = max(float(cfg.air_density), 1e-9)
    c_t = max(float(prop_coefficients(cfg, group)["c_t"]), 1e-6)
    j0 = max(1.2 * float(g["p_in"]) / max(float(g["d_in"]), 1e-6), 0.2)
    a = max(float(axial_mps), 0.0) / (d * j0)
    b = t / (c_t * rho * d ** 4)
    n_rev_s = 0.5 * (a + math.sqrt(a * a + 4.0 * b))
    return n_rev_s * 60.0


def motor_operating_point(cfg: VTOLConfig, group: str, thrust_per_rotor_N: float,
                          shaft_per_rotor_W: float, axial_mps: float = 0.0,
                          measured: bool = False) -> Dict[str, float]:
    """
    One motor's electrical state for a given thrust and shaft power.

    `measured` is True where a bench table covers this thrust: the table's
    power was taken at the ESC input, so it already contains the motor's
    losses, and the figure of merit derived from it absorbs them. What is
    handed in as `shaft_per_rotor_W` is then the power the motor DRAWS, not
    what it delivers, and the model splits that measured input into shaft
    output and loss. Adding the model's loss on top instead counted the motor
    twice: the plots and Metrics showed 14-29% more power than the bench
    recorded. The split loss is reported but not charged.
    """
    g = _group(cfg, group)
    shaft = max(float(shaft_per_rotor_W), 0.0)
    rpm = prop_rpm(cfg, group, thrust_per_rotor_N, axial_mps) if shaft > 0 else 0.0
    kv = float(g["kv"] or 0.0)
    v_pack = max(float(cfg.battery.vnom_pack), 1e-9)
    out = {
        "rpm": rpm, "thrust_N": max(float(thrust_per_rotor_N), 0.0),
        "shaft_W": shaft, "current_A": 0.0, "v_emf_V": 0.0, "v_term_V": 0.0,
        "elec_W": shaft, "loss_W": 0.0, "copper_W": 0.0, "iron_W": 0.0,
        "efficiency": 1.0, "throttle": 0.0, "saturated": False,
        "torque_Nm": 0.0, "kt": 0.0, "i0_A": 0.0, "measured": bool(measured),
        "modelled": kv > 0,
    }
    if shaft <= 0 or rpm <= 0:
        return out
    omega = rpm * 2.0 * math.pi / 60.0
    torque = shaft / max(omega, 1e-9)
    out["torque_Nm"] = torque
    if kv <= 0:
        # No electrical model: the current is what the shaft power costs at
        # pack voltage, so Status still has a number to check.
        out["current_A"] = shaft / v_pack
        out["throttle"] = float("nan")
        return out
    kt = 60.0 / (2.0 * math.pi * kv)
    v_emf = rpm / kv
    # No-load current is measured at one voltage; core loss grows with speed,
    # and the usual approximation scales the current with sqrt(speed).
    i0 = float(g["i0"] or 0.0)
    if g["v0"]:
        i0 *= math.sqrt(max(v_emf, 0.0) / max(float(g["v0"]), 1e-9))
    rm = float(g["rm"] or 0.0)
    if measured:
        # P_in = I * V_term = I * (V_emf + I * Rm), solved for I in the form
        # that stays finite at Rm = 0. Whatever the motor does not lose is
        # the shaft output; a table that measured less than the model's own
        # no-load draw leaves none.
        elec = shaft
        current = 2.0 * elec / (v_emf + math.sqrt(v_emf * v_emf + 4.0 * rm * elec))
        torque = max(current - i0, 0.0) * kt
        shaft = torque * omega
    else:
        current = torque / kt + i0
        elec = shaft + current * current * rm + i0 * v_emf
    copper = current * current * rm
    iron = i0 * v_emf
    v_term = v_emf + current * rm
    out.update({
        "shaft_W": shaft, "torque_Nm": torque,
        "current_A": current, "v_emf_V": v_emf, "v_term_V": v_term,
        "elec_W": elec, "copper_W": copper, "iron_W": iron,
        "loss_W": 0.0 if measured else copper + iron,
        "efficiency": shaft / max(elec, 1e-9),
        "throttle": v_term / v_pack, "saturated": v_term > v_pack,
        "kt": kt, "i0_A": i0,
    })
    return out


def lift_table_covers(cfg: VTOLConfig, thrust_per_rotor_N: float) -> bool:
    """True where the lift bench table measures this thrust."""
    area = math.pi / 4.0 * (cfg.lift_prop_diameter_in * 0.0254) ** 2
    return measured_lift_efficiency(cfg, thrust_per_rotor_N, area) is not None


def cruise_table_covers(cfg: VTOLConfig, thrust_per_motor_N: float) -> bool:
    """True where the cruise bench table measures this thrust."""
    return measured_cruise_efficiency(cfg, thrust_per_motor_N,
                                      cfg.cruise_disc_area_m2) is not None


def motor_operating_curve(cfg: VTOLConfig, group: str, thrusts_per_rotor_N,
                          airspeed_mps: float = 0.0) -> Dict[str, list]:
    """
    One motor across a range of thrusts at a single airspeed, through the
    same chain drive_chain() uses, so a run's operating point lies on the
    curve drawn at that run's airspeed.

    The lift rotors meet the freestream edgewise with no axial flow; the
    cruise propeller of a lift+cruise meets it head-on, so the airspeed is
    also its axial speed and sets both its power and its RPM.
    """
    g = _group(cfg, group)
    n = max(int(g["n"]), 1)
    v = max(float(airspeed_mps), 0.0)
    out = {"thrust_N": [], "elec_W": [], "current_A": [], "g_per_W": [],
           "rpm": [], "efficiency": []}
    for t in thrusts_per_rotor_N:
        t = float(t)
        if group == "lift":
            shaft = rotor_power_W(cfg, t * n, v) / n
            op = motor_operating_point(cfg, group, t, shaft, 0.0,
                                       measured=lift_table_covers(cfg, t))
        else:
            shaft = cruise_prop_power_W(cfg, t * n, v) / n
            op = motor_operating_point(cfg, group, t, shaft, v,
                                       measured=cruise_table_covers(cfg, t))
        out["thrust_N"].append(t)
        out["elec_W"].append(op["elec_W"])
        out["current_A"].append(op["current_A"])
        out["g_per_W"].append(t / G0 * 1000.0 / max(op["elec_W"], 1e-9))
        out["rpm"].append(op["rpm"])
        out["efficiency"].append(op["efficiency"])
    return out


def coaxial_power_multiplier(cfg: VTOLConfig, thrust_per_rotor_N: float,
                             airspeed_mps: float = 0.0) -> float:
    """
    Interference penalty for stacked lift rotors, the multicopter's model.

    About 1.18 at a 0.2 D spacing in hover, falling with spacing and easing
    as the freestream sweeps the upper rotor's wake clear of the lower one.
    1.0 for a flat layout.
    """
    if cfg.lift_rotor_layout != "coaxial":
        return 1.0
    d = cfg.lift_prop_diameter_in * 0.0254
    if d <= 0:
        return 1.18
    spacing = cfg.coaxial_spacing_m if cfg.coaxial_spacing_m else 0.20 * d
    inc = 0.25 * math.exp(-3.0 * max(float(spacing) / d, 0.0)) + 0.03
    v = max(float(airspeed_mps), 0.0)
    if v > 0 and thrust_per_rotor_N > 0:
        area = math.pi / 4.0 * d * d
        v_hover = math.sqrt(thrust_per_rotor_N / max(2.0 * cfg.air_density * area, 1e-9))
        inc *= 1.0 - 0.30 * v / (v + max(v_hover, 1e-9))
    return 1.0 + inc


def inflow_multiplier(cfg: VTOLConfig, thrust_per_rotor_N: float,
                      edgewise_mps: float, axial_mps: float = 0.0) -> Tuple[float, float, float]:
    """
    (power multiplier, advance ratio mu, inflow efficiency) for a lift rotor
    meeting `edgewise_mps` of freestream across its disc.

    The multicopter's empirical map, off by default. mu = V / (Omega R).
    """
    rpm = prop_rpm(cfg, "lift", thrust_per_rotor_N, axial_mps)
    r = max(cfg.lift_prop_diameter_in * 0.0254 / 2.0, 1e-9)
    omega = rpm * 2.0 * math.pi / 60.0
    mu = max(float(edgewise_mps), 0.0) / max(omega * r, 1e-9) if omega > 0 else 0.0
    if not cfg.inflow_map_enabled or edgewise_mps <= 0:
        return 1.0, mu, 1.0
    eta = max(float(np.interp(mu, cfg.inflow_mu_bp, cfg.inflow_eff_bp)), 0.2)
    return 1.0 / eta, mu, eta


def rotor_power_W(cfg: VTOLConfig, thrust_N: float, airspeed_mps: float = 0.0,
                  climb_rate_mps: float = 0.0) -> float:
    """
    Shaft power for the lift rotors to make `thrust_N` in total.

    Momentum theory with a figure of merit for real losses, using the shared
    forward-flight inflow solver so a rotor climbing away or translating in
    the transition is not charged its hover induced power. A coaxial layout
    and the optional inflow map scale the result.
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
    mult = coaxial_power_multiplier(cfg, t_per, airspeed_mps)
    mult *= inflow_multiplier(cfg, t_per, airspeed_mps, climb_rate_mps)[0]
    return ideal_per / fom * n * mult


def cruise_prop_efficiency_at(cfg: VTOLConfig, airspeed_mps: float) -> float:
    """
    Cruise propeller efficiency at an airspeed.

    "constant" returns the entered figure everywhere. "curve" is the
    fixed-wing model: the entered figure is the PEAK, reached near 60% of the
    full-throttle pitch speed, and it falls away either side of it.
    """
    peak = max(float(cfg.cruise_prop_efficiency), 0.10)
    if cfg.cruise_prop_eff_model != "curve":
        return peak
    g = _group(cfg, "cruise")
    pitch_m = float(g["p_in"]) * 0.0254
    kv = float(g["kv"] or 0.0)
    if kv <= 0 or pitch_m <= 0:
        return peak
    v_pitch = pitch_m * kv * float(cfg.battery.vmax_pack) / 60.0
    if v_pitch <= 1e-6:
        return peak
    x = max(float(airspeed_mps), 0.0) / v_pitch
    shape = (x * (1.0 - x)) / (0.60 * 0.40)
    return min(max(peak * max(shape, 0.0), 0.25 * peak), peak)


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
    eff = (measured_cruise_efficiency(cfg, t_per, area)
           or cruise_prop_efficiency_at(cfg, v))
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


def electrical_power_W(cfg: VTOLConfig, shaft_power_W: float,
                       motor_loss_W: float = 0.0) -> float:
    """
    Shaft power to pack power: through the motors and ESC, plus the avionics
    load, plus the main wire run if one was entered.

    The simple form, for callers that already know the motor loss. The
    flight regimes use drive_chain, which works the loss out per motor.
    """
    base = ((shaft_power_W + motor_loss_W) / cfg.esc_efficiency
            + avionics_input_power_W(cfg)
            + peripheral_power_W(cfg))
    return base + wire_loss_W(cfg, base)


def n_escs(cfg: VTOLConfig) -> int:
    """One ESC per driven rotor: lift rotors, plus cruise motors on a lift+cruise."""
    return cfg.num_lift_rotors + (0 if uses_vectored_thrust(cfg) else cfg.num_cruise_motors)


def drive_chain(cfg: VTOLConfig,
                rotor_thrust_N: float, rotor_shaft_W: float,
                cruise_thrust_N: float = 0.0, cruise_shaft_W: float = 0.0,
                rotor_axial_mps: float = 0.0,
                cruise_axial_mps: float = 0.0) -> Dict[str, object]:
    """
    Everything between the shafts and the cells, for one operating point.

        motor input = shaft + motor loss                      (per motor model)
        ESC loss    = motor input x (1/eta - 1)               (under load)
                      + idle current x pack V per ESC         (if entered)
        pack power  = motor input + ESC loss + avionics + peripherals + wire

    Returns the total and every piece of it, so the Power Budget, Status and
    Metrics all read the same numbers rather than re-deriving them.
    """
    n_lift = max(cfg.num_lift_rotors, 1)
    t_lift = max(float(rotor_thrust_N), 0.0) / n_lift
    lift = motor_operating_point(
        cfg, "lift", t_lift, max(float(rotor_shaft_W), 0.0) / n_lift,
        rotor_axial_mps, measured=lift_table_covers(cfg, t_lift))

    if uses_vectored_thrust(cfg):
        cruise = lift
        n_cruise = 0
    else:
        n_cruise = max(cfg.num_cruise_motors, 1)
        t_cruise = max(float(cruise_thrust_N), 0.0) / n_cruise
        cruise = motor_operating_point(
            cfg, "cruise", t_cruise, max(float(cruise_shaft_W), 0.0) / n_cruise,
            cruise_axial_mps, measured=cruise_table_covers(cfg, t_cruise))

    shaft = max(float(rotor_shaft_W), 0.0) + max(float(cruise_shaft_W), 0.0)
    # A group a bench table covers is charged nothing here: its measured
    # power already contains the motor (see motor_operating_point).
    charged = [(n, op) for n, op in ((n_lift, lift), (n_cruise, cruise))
               if n and not op["measured"]]
    motor_loss = sum(n * op["loss_W"] for n, op in charged)
    copper = sum(n * op["copper_W"] for n, op in charged)
    iron = sum(n * op["iron_W"] for n, op in charged)
    motor_in = shaft + motor_loss

    # The efficiency already covers everything the ESC loses under load, so
    # its resistance does not ADD a loss — it says how much of that loss is
    # conduction (I^2 R) rather than switching, for the Power Budget. The
    # idle current is different: a standby draw the ESCs take whether or not
    # their motor is turning, so it adds.
    esc_load = motor_in * (1.0 / max(cfg.esc_efficiency, 1e-9) - 1.0)
    esc_cond = min(cfg.esc_resistance_ohm * (
        n_lift * lift["current_A"] ** 2 + n_cruise * cruise["current_A"] ** 2), esc_load)
    esc_idle = cfg.esc_idle_current_A * n_escs(cfg) * cfg.battery.vnom_pack
    esc_loss = esc_load + esc_idle

    base = motor_in + esc_loss + avionics_input_power_W(cfg) + peripheral_power_W(cfg)
    wire = wire_loss_W(cfg, base)
    # The ESCs sit at the far end of the main lead, so the voltage they can
    # put across a motor is the pack's less the lead's I*R drop. The power
    # is already right — the lead's I^2 R is in `wire` — but the throttle
    # each motor needs, and whether it runs out of voltage, are judged
    # against what actually arrives. No wiring entered leaves them as they were.
    r_wire = float(getattr(cfg, "wire_resistance_ohm", 0.0) or 0.0)
    if r_wire > 0:
        v_pack = max(float(cfg.battery.vnom_pack), 1e-9)
        supply = max(v_pack - (base + wire) / v_pack * r_wire, 1e-9)
        for op in {id(lift): lift, id(cruise): cruise}.values():
            if op.get("v_term_V", 0.0) > 0 and math.isfinite(op.get("throttle", float("nan"))):
                op["throttle"] = op["v_term_V"] / supply
                op["saturated"] = op["v_term_V"] > supply
    return {
        "total_power_W": base + wire,
        "motor_input_W": motor_in,
        "motor_loss_W": motor_loss,
        "motor_copper_W": copper,
        "motor_iron_W": iron,
        "esc_loss_W": esc_loss,
        "esc_conduction_W": esc_cond,
        "esc_switching_W": esc_load - esc_cond,
        "esc_idle_W": esc_idle,
        "wire_loss_W": wire,
        "drive_efficiency": shaft / max(motor_in, 1e-9) if shaft > 0 else 1.0,
        "lift_motor": lift,
        "cruise_motor": cruise,
    }


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
    climb = max(float(climb_rate_mps), 0.0)
    thrust_N = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
    shaft = rotor_power_W(cfg, thrust_N, airspeed_mps=0.0, climb_rate_mps=climb)
    # Vertical climb adds potential power directly.
    shaft += thrust_N * climb
    chain = drive_chain(cfg, thrust_N, shaft, rotor_axial_mps=climb)
    point = {
        "regime": "hover",
        "airspeed_mps": 0.0,
        "rotor_thrust_N": thrust_N,
        "wing_lift_N": 0.0,
        "cruise_thrust_N": 0.0,
        "rotor_shaft_W": shaft,
        "cruise_shaft_W": 0.0,
        "shaft_power_W": shaft,
    }
    point.update(chain)
    return point


def cruise_power_W(cfg: VTOLConfig, airspeed_mps: float) -> Dict[str, float]:
    """
    Power in wing-borne cruise, lift rotors stopped.

    Valid only above the stall speed; below it the wing cannot carry the
    aircraft and the transition model applies instead.
    """
    v = max(float(airspeed_mps), 1e-6)
    lift_N = cfg.weight_N
    drag_N = (wing_drag_N(cfg, v, lift_N) + stopped_rotor_drag_N(cfg, v)
              + body_drag_N(cfg, v))
    shaft = cruise_prop_power_W(cfg, drag_N, v)
    chain = drive_chain(cfg, 0.0, 0.0, drag_N, shaft, cruise_axial_mps=v)
    point = {
        "regime": "cruise",
        "airspeed_mps": v,
        "rotor_thrust_N": 0.0,
        "wing_lift_N": lift_N,
        "cruise_thrust_N": drag_N,
        "drag_N": drag_N,
        "rotor_shaft_W": 0.0,
        "cruise_shaft_W": shaft,
        "shaft_power_W": shaft,
    }
    point.update(chain)
    return point


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
    drag_N = wing_drag_N(cfg, v, lift_N) + rotor_drag_N + body_drag_N(cfg, v)
    cruise_thrust_N = drag_N

    rotor_shaft = rotor_power_W(cfg, rotor_thrust_N, airspeed_mps=v)
    cruise_shaft = cruise_prop_power_W(cfg, cruise_thrust_N, v)
    shaft = rotor_shaft + cruise_shaft
    chain = drive_chain(cfg, rotor_thrust_N, rotor_shaft,
                        cruise_thrust_N, cruise_shaft, cruise_axial_mps=v)

    point = {
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
    }
    point.update(chain)
    return point


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
    efficiency = (1.0 - blend) * hover_eff + blend * cruise_prop_efficiency_at(cfg, v)
    edgewise = v * math.cos(math.radians(tilt))
    mult = coaxial_power_multiplier(cfg, t_per, v)
    mult *= inflow_multiplier(cfg, t_per, edgewise, v_axial)[0]
    return t_per * (v_axial + vi) / max(efficiency, 0.05) * n * mult


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

    climb = max(float(climb_rate_mps), 0.0)
    vertical_N = max(weight * (1.0 + download) - lift_N, 0.0)
    drag_N = (wing_drag_N(cfg, v, lift_N) + body_drag_N(cfg, v)) if v > 1e-6 else 0.0

    thrust_N = math.hypot(vertical_N, drag_N)
    tilt_deg = math.degrees(math.atan2(drag_N, max(vertical_N, 1e-12)))

    shaft = vectored_rotor_power_W(cfg, thrust_N, tilt_deg, v)
    shaft += thrust_N * climb * math.cos(math.radians(tilt_deg))
    axial = v * math.sin(math.radians(tilt_deg)) + climb * math.cos(math.radians(tilt_deg))
    chain = drive_chain(cfg, thrust_N, shaft, rotor_axial_mps=axial)

    if v < 1e-6:
        regime = "hover"
    elif lift_N >= weight - 1e-9:
        regime = "cruise"
    else:
        regime = "transition"

    point = {
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
    }
    point.update(chain)
    return point


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
# PERFORMANCE — carried over from the fixed-wing and multicopter
# ============================================================
#
# A VTOL is both aircraft, so it answers both sets of questions: the
# multicopter's (RPM, tip Mach, figure of merit, thrust-to-weight, hover wind
# limit) for its rotors, and the fixed-wing's (L/D, climb, ceiling, turns,
# glide, runway) for its wing. The formulas are the ones those two tools use,
# restated here against the VTOL's own drag and propulsion.

MU_AIR = 1.81e-5            # Pa.s, dynamic viscosity of air near 15 C
ROC_CEILING_MPS = 0.508     # 100 ft/min, the service-ceiling definition

# Lumped thermal model, per unit: a steady temperature rise of R_th x loss,
# approached with time constant R_th x C_th. These are the multicopter's
# whole-aircraft constants spread over the four motors and ESCs it assumes,
# so the two tools warm a component at the same rate for the same loss.
THERMAL = {
    "motor":   {"R": 1.40, "C": 60.0},     # C/W per motor, J/C per motor
    "esc":     {"R": 3.00, "C": 45.0},     # per ESC
    "battery": {"R": 0.25, "C": 500.0},    # the pack
}


def blade_chord_m(diameter_in: float, blades: int) -> float:
    """Average blade chord, the multicopter's coarse estimate."""
    d_m = max(float(diameter_in), 0.0) * 0.0254
    b = max(int(blades), 1)
    return 0.11 * d_m / (1.0 + 0.08 * (b - 2))


def prop_solidity(diameter_in: float, blades: int) -> float:
    """sigma = N_blades * chord / (pi * R)."""
    r_m = max(float(diameter_in), 0.0) * 0.0254 / 2.0
    if r_m <= 0:
        return 0.0
    return max(int(blades), 1) * blade_chord_m(diameter_in, blades) / (math.pi * r_m)


def tip_speed_mps(diameter_in: float, rpm: float) -> float:
    return math.pi * max(float(diameter_in), 0.0) * 0.0254 * max(float(rpm), 0.0) / 60.0


def speed_of_sound_mps(temp_C: float) -> float:
    return 331.3 * math.sqrt(max(1.0 + float(temp_C) / 273.15, 1e-6))


def density_altitude_m(rho: float) -> float:
    """Standard-atmosphere altitude with this density."""
    return 44330.8 * (1.0 - (max(float(rho), 1e-9) / 1.225) ** 0.234969)


def static_thrust_available_N(cfg: VTOLConfig, group: str) -> Tuple[float, str]:
    """
    Largest static thrust the group can make, all rotors together, and where
    that figure came from — the multicopter's order of preference:

      1. the bench table's highest measured thrust;
      2. the propeller's rated thrust as entered;
      3. the motor's max power through an ideal actuator disc;
      4. the motor at full throttle: Kv x pack voltage sets the RPM, and the
         thrust coefficient turns that into thrust.
    """
    g = _group(cfg, group)
    n = max(int(g["n"]), 1)
    rho = max(float(cfg.air_density), 1e-9)
    d = float(g["d_in"]) * 0.0254
    area = math.pi / 4.0 * d * d
    table = g["table"]
    if table is not None:
        return float(table["Thrust_g"].max()) * G0 / 1000.0 * n, "bench table"
    rated_g = (cfg.lift_prop_max_thrust_g if g["name"] == "lift"
               else cfg.cruise_prop_max_thrust_g)
    if rated_g and rated_g > 0:
        return rated_g * G0 / 1000.0 * n, "rated thrust"
    if g["pmax"]:
        per = (float(g["pmax"]) * math.sqrt(2.0 * rho * area)) ** (2.0 / 3.0)
        return per * n, "motor max power"
    if g["kv"] and float(g["kv"]) > 0:
        # Loaded, a motor reaches roughly 85% of its no-load speed.
        n_rev = float(g["kv"]) * float(cfg.battery.vnom_pack) * 0.85 / 60.0
        c_t = float(prop_coefficients(cfg, group)["c_t"])
        return c_t * rho * n_rev ** 2 * d ** 4 * n, "Kv at full throttle"
    return 0.0, "unknown"


def forward_thrust_available_N(cfg: VTOLConfig, airspeed_mps: float) -> float:
    """
    Thrust available for forward flight at an airspeed, all propulsors.

    The fixed-wing's momentum bound: the ideal power the propeller absorbs
    at its static maximum is held fixed, and at speed the thrust is whatever
    that power buys, T (V + vi) = P. At zero airspeed this is the static
    figure; as the speed rises the thrust falls roughly as P / V.
    """
    static_T, _src = static_thrust_available_N(cfg, "cruise")
    v = max(float(airspeed_mps), 0.0)
    if static_T <= 0 or v < 0.1:
        return static_T
    g = _group(cfg, "cruise")
    n = max(int(g["n"]), 1)
    rho = max(float(cfg.air_density), 1e-9)
    area = math.pi / 4.0 * (float(g["d_in"]) * 0.0254) ** 2
    t_static = static_T / n
    p_ideal = t_static * math.sqrt(t_static / (2.0 * rho * area))

    def power_needed(t):
        vi = -v / 2.0 + math.sqrt((v / 2.0) ** 2 + t / (2.0 * rho * area))
        return t * (v + vi)

    lo, hi = 0.0, t_static
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if power_needed(mid) > p_ideal:
            hi = mid
        else:
            lo = mid
    return lo * n


def wingborne_drag_N(cfg: VTOLConfig, airspeed_mps: float, load_factor: float = 1.0) -> float:
    """Drag in wing-borne flight: the wing at the CL for n x W, rotors stopped."""
    v = max(float(airspeed_mps), 1e-6)
    lift = cfg.weight_N * max(float(load_factor), 1.0)
    drag = wing_drag_N(cfg, v, lift) + body_drag_N(cfg, v)
    if not uses_vectored_thrust(cfg):
        drag += stopped_rotor_drag_N(cfg, v)
    return drag


def rate_of_climb_mps(cfg: VTOLConfig, airspeed_mps: float) -> float:
    """(T_available - D) V / W, wing-borne. Zero where the thrust cannot keep up."""
    v = max(float(airspeed_mps), 0.1)
    excess = forward_thrust_available_N(cfg, v) - wingborne_drag_N(cfg, v)
    return max(excess * v / max(cfg.weight_N, 1e-9), 0.0)


def climb_angle_deg(cfg: VTOLConfig, airspeed_mps: float) -> float:
    """asin((T - D) / W), wing-borne."""
    v = max(float(airspeed_mps), 0.1)
    ratio = ((forward_thrust_available_N(cfg, v) - wingborne_drag_N(cfg, v))
             / max(cfg.weight_N, 1e-9))
    return math.degrees(math.asin(min(ratio, 1.0))) if ratio > 0 else 0.0


def _wingborne_speeds(cfg: VTOLConfig, v_max: Optional[float] = None, steps: int = 120):
    """Speeds from the transition speed up, where the wing carries the aircraft."""
    v_lo = max(transition_speed_mps(cfg), 1.0)
    v_hi = max(v_max or max(cfg.cruise_speed_mps * 1.8, v_lo * 2.5), v_lo + 1.0)
    return [v_lo + (v_hi - v_lo) * i / steps for i in range(steps + 1)]


def best_climb(cfg: VTOLConfig, need_angle: bool = True) -> Dict[str, float]:
    """Best rate of climb (Vy) and best angle (Vx), wing-borne."""
    best = {"vy_mps": 0.0, "max_roc_mps": 0.0, "vx_mps": 0.0, "max_climb_angle_deg": 0.0}
    w = max(cfg.weight_N, 1e-9)
    for v in _wingborne_speeds(cfg, steps=80):
        excess = forward_thrust_available_N(cfg, v) - wingborne_drag_N(cfg, v)
        if excess <= 0:
            continue
        roc = excess * v / w
        if roc > best["max_roc_mps"]:
            best["vy_mps"], best["max_roc_mps"] = v, roc
        if need_angle:
            angle = math.degrees(math.asin(min(excess / w, 1.0)))
            if angle > best["max_climb_angle_deg"]:
                best["vx_mps"], best["max_climb_angle_deg"] = v, angle
    return best


def service_ceiling_m(cfg: VTOLConfig, max_alt_m: float = 8000.0) -> float:
    """
    Standard-atmosphere altitude where the best wing-borne climb falls to
    100 ft/min, the fixed-wing's definition. Infinite if it is still above
    that at `max_alt_m`; the field elevation if it is already below.
    """
    import copy

    def roc_at(alt):
        trial = copy.copy(cfg)
        trial.__dict__ = dict(cfg.__dict__)
        trial.__dict__.pop("_coeff_cache", None)
        trial.air_density = core.air_density(alt)
        return best_climb(trial, need_angle=False)["max_roc_mps"]

    base = max(cfg.reference_altitude_m, 0.0)
    if roc_at(base) < ROC_CEILING_MPS:
        return base
    lo, hi, alt = base, None, base + 500.0
    while alt <= max_alt_m:
        if roc_at(alt) < ROC_CEILING_MPS:
            hi = alt
            break
        lo, alt = alt, alt + 500.0
    if hi is None:
        return float("inf")
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if roc_at(mid) >= ROC_CEILING_MPS:
            lo = mid
        else:
            hi = mid
    return lo


def best_speeds(cfg: VTOLConfig, v_max: Optional[float] = None,
                wind_head_mps: float = 0.0, wind_cross_mps: float = 0.0) -> Dict[str, float]:
    """
    Best-endurance and best-range airspeeds over the whole speed range,
    hover to fast cruise — a VTOL can loiter at any speed, so the search
    starts at zero rather than at the stall. Range is over the ground in the
    given wind, so a headwind moves the best-range speed up.
    """
    v_hi = v_max or max(cfg.cruise_speed_mps * 1.8, transition_speed_mps(cfg) * 1.8)
    speeds = [v_hi * i / 240 for i in range(241)]
    usable = cfg.battery.usable_Wh
    best_e, best_r = (0.0, -1.0), (0.0, -1.0)
    for v in speeds:
        p = float(power_at_airspeed(cfg, v)["total_power_W"])
        minutes = usable / max(p, 1e-9) * 60.0
        km = minutes * 60.0 * max(core.groundspeed_along_track_mps(
            v, wind_head_mps, wind_cross_mps), 0.0) / 1000.0
        if minutes > best_e[1]:
            best_e = (v, minutes)
        if km > best_r[1]:
            best_r = (v, km)
    return {"best_endurance_speed_mps": best_e[0], "best_endurance_min": best_e[1],
            "best_range_speed_mps": best_r[0], "best_range_km": best_r[1]}


def glide(cfg: VTOLConfig) -> Dict[str, float]:
    """
    Unpowered glide, rotors stopped: the fixed-wing's min-sink search and the
    analytic best-L/D speed, against the VTOL's full drag.
    """
    v_lo = max(stall_speed_mps(cfg) * 1.05, 1.0)
    best_v, best_sink = v_lo, float("inf")
    for i in range(301):
        v = v_lo + 40.0 * i / 300
        ld = cfg.weight_N / max(wingborne_drag_N(cfg, v), 1e-9)
        sink = v / ld
        if sink < best_sink:
            best_v, best_sink = v, sink
    q_s = 2.0 * cfg.weight_N / (cfg.air_density * cfg.wing_area_m2)
    v_md = math.sqrt(q_s) * (cfg.induced_drag_factor / max(cfg.CD0, 1e-9)) ** 0.25
    return {"min_sink_speed_mps": best_v, "min_sink_rate_mps": best_sink,
            "best_glide_speed_mps": max(v_md, v_lo),
            "ld_max_analytic": 0.5 * math.sqrt(math.pi * cfg.aspect_ratio * cfg.oswald
                                               / max(cfg.CD0, 1e-9))}


def takeoff_roll_m(cfg: VTOLConfig) -> float:
    """
    Conventional take-off roll on the forward thrust — the fixed-wing's
    Raymer estimate, s = 1.44 W^2 / (g rho S CL_to (T - mu W)), with thrust
    taken at 0.707 of the lift-off speed. Infinite if the thrust cannot
    overcome the rolling friction.
    """
    v_lof = 1.2 * stall_speed_mps(cfg)
    thrust = forward_thrust_available_N(cfg, 0.707 * v_lof)
    net = thrust - cfg.mu_roll * cfg.weight_N
    if net <= 0:
        return float("inf")
    return (1.44 * cfg.weight_N ** 2 /
            (G0 * cfg.air_density * cfg.wing_area_m2 * cfg.CL_takeoff * net))


def landing_distance_m(cfg: VTOLConfig, obstacle_m: float = 15.0) -> float:
    """Conventional landing over a 15 m obstacle, the fixed-wing's method."""
    v_s = stall_speed_mps(cfg)
    v_app, v_td = 1.30 * v_s, 1.15 * v_s
    ld_app = cfg.weight_N / max(wingborne_drag_N(cfg, v_app), 1e-9)
    q_td = 0.5 * cfg.air_density * v_td ** 2
    cl_ground = min(0.25, cfg.CL_max)
    drag = q_td * cfg.wing_area_m2 * (cfg.CD0 + cfg.induced_drag_factor * cl_ground ** 2)
    lift = q_td * cfg.wing_area_m2 * cl_ground
    decel = (max(cfg.mu_brake, 0.08) * max(cfg.weight_N - lift, 0.0) + drag) / (cfg.weight_N / G0)
    if decel <= 0:
        return float("inf")
    return obstacle_m * ld_app + 0.5 * v_td + v_td ** 2 / (2.0 * decel)


def turn(cfg: VTOLConfig, airspeed_mps: float, bank_deg: float) -> Dict[str, float]:
    """
    A coordinated, level, wing-borne turn: the fixed-wing's figures, plus
    what the turn costs in power — the wing flies at n x W, so its induced
    drag rises by n squared.
    """
    v = max(float(airspeed_mps), 0.1)
    phi = math.radians(min(max(float(bank_deg), 0.0), 80.0))
    n = 1.0 / max(math.cos(phi), 1e-6)
    tan_phi = math.tan(phi)
    radius = v * v / (G0 * tan_phi) if tan_phi > 1e-6 else float("inf")
    rate = math.degrees(G0 * tan_phi / v)
    drag = wingborne_drag_N(cfg, v, n)
    shaft = cruise_prop_power_W(cfg, drag, v)
    if uses_vectored_thrust(cfg):
        chain = drive_chain(cfg, drag, shaft, rotor_axial_mps=v)
    else:
        chain = drive_chain(cfg, 0.0, 0.0, drag, shaft, cruise_axial_mps=v)
    return {"load_factor": n, "turn_radius_m": radius, "turn_rate_deg_s": rate,
            "turn_period_s": 360.0 / rate if rate > 1e-9 else float("inf"),
            "turn_stall_speed_mps": stall_speed_mps(cfg) * math.sqrt(n),
            "turn_drag_N": drag, "turn_power_W": chain["total_power_W"]}


def hover_wind_limit_mps(cfg: VTOLConfig) -> float:
    """
    Strongest wind the aircraft can hold station against by tilting its lift
    rotors, the multicopter's estimate:

        T_h = min( sqrt(T_avail^2 - W^2),  T_avail sin(tilt limit) )
        V   = sqrt( 2 T_h / (rho CdA_side) )

    NaN when no side drag area is known — an honest "cannot say" rather than
    a divide by nearly nothing. It ignores the wing, which in a real wind
    starts lifting too; that makes it conservative.
    """
    t_avail, _src = static_thrust_available_N(cfg, "lift")
    need = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
    areas = extra_drag_areas(cfg)
    cda = float(areas["side_CdA"] or areas["frontal_CdA"])
    if cda < 1e-4:
        return float("nan")
    if t_avail <= need:
        return 0.0
    t_h = min(math.sqrt(t_avail ** 2 - need ** 2),
              t_avail * math.sin(math.radians(cfg.max_tilt_deg)))
    return math.sqrt(2.0 * t_h / (cfg.air_density * cda))


def hover_tilt_in_wind(cfg: VTOLConfig, wind_mps: float, wind_direction_deg: float,
                       course_deg: float) -> Dict[str, float]:
    """
    Tilt needed to hold station in a wind, split into pitch and roll by where
    the wind comes from relative to the heading.
    """
    drag = hover_wind_drag_N(cfg, wind_mps)
    need = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
    tilt = math.degrees(math.atan2(drag, max(need, 1e-9)))
    pitch, roll = core.pitch_roll_from_tilt(tilt, float(wind_direction_deg) - float(course_deg))
    return {"hover_tilt_deg": tilt, "hover_pitch_deg": abs(pitch),
            "hover_roll_deg": abs(roll), "hover_wind_drag_N": drag}


def thermal_steady_C(ambient_C: float, loss_W: float, part: str) -> float:
    """Steady temperature of one unit dissipating `loss_W`."""
    return float(ambient_C) + THERMAL[part]["R"] * max(float(loss_W), 0.0)


def thermal_status(*pairs) -> str:
    """
    OK / WARN / HOT from (temperature, limit) pairs: WARN within 10 C of a
    limit, HOT past one. The multicopter's absolute 95 / 115 C bands are the
    fallback where no limit applies.
    """
    worst = "OK"
    for temp, limit in pairs:
        if limit:
            if temp > limit:
                return "HOT"
            if temp > limit - 10.0:
                worst = "WARN"
        elif temp >= 115.0:
            return "HOT"
        elif temp >= 95.0:
            worst = "WARN"
    return worst


def voltage_rating_check(cfg: VTOLConfig, unit: str, lo: Optional[float],
                         hi: Optional[float]) -> Optional[Dict[str, object]]:
    """
    The pack against a component's voltage rating, in the unit it was
    entered in: series cells for "S", full-charge volts for "V". None when
    no rating was entered.
    """
    if lo is None and hi is None:
        return None
    cells = str(unit).upper().startswith("S")
    value = float(cfg.battery.series_cells) if cells else float(cfg.battery.vmax_pack)
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    fmt = (lambda x: f"{x:g}S") if cells else (lambda x: f"{x:.1f} V")
    limit = " - ".join(fmt(x) for x in (lo, hi) if x is not None)
    return {"value": fmt(value), "limit": limit, "ok": ok}


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
#
# Every field whose counterpart is in the multicopter's or the fixed-wing's
# Simple set is here too, so a user moving between the three finds the same
# inputs at the same level of detail. The Avionics tab stays fully visible,
# as it does in both of them: omitting the avionics draw is a common and
# costly beginner mistake.
VTOL_SIMPLE_FIELDS = {
    # airframe and mass
    "weight", "payload", "mass_mode", "structure_mass", "avionics_mass",
    "span", "area", "cd0", "clmax", "oswald",
    "mu_roll", "mu_brake", "cl_takeoff",
    "drag_model_mode", "parasite_drag", "parasite_area", "profile_drag",
    "profile_area", "body_length_m", "body_width_m", "body_height_m",
    "arm_length_m", "arm_width_m",
    # lift rotors
    "n_lift", "lift_layout", "coax_spacing", "lift_d", "lift_p", "lift_blades",
    "lift_kv", "lift_rm", "lift_i0", "lift_wt", "lift_imax", "lift_pmax",
    "lift_prop_wt", "lift_max_thrust", "lift_table",
    # cruise propulsion
    "n_cruise", "cruise_d", "cruise_p", "cruise_blades", "cruise_kv",
    "cruise_rm", "cruise_i0", "cruise_wt", "cruise_imax", "cruise_pmax",
    "cruise_eff", "cruise_eff_model", "cruise_prop_wt", "cruise_max_thrust",
    "cruise_table",
    # battery
    "chem", "unit_mode", "cell_cap", "series", "parallel", "cell_wt",
    "cells_s_per_pack", "cells_p_per_pack", "pack_cap", "pack_wt",
    "vmin", "vnom", "vmax", "rcell", "usable", "c_cont", "a_cont", "soc_model",
    # ESC and avionics
    "esc_cont", "esc_imax", "esc_wt",
    "avionics_flat", "periph_current", "avionics_rails",
    # mission and environment
    "cruise_v", "alt", "cruise_altitude", "temp", "wind", "wind_dir",
    "course_deg", "bank_deg", "reserve_percent", "accel", "decel", "max_tilt",
    "mission", "plot_vmax",
}


def lift_rotor_positions(cfg: VTOLConfig) -> List[Tuple[float, float]]:
    """
    Where the lift rotors sit, in metres from the CG: x to starboard, y
    toward the nose — the plan view's axes. One layout, shared by the
    Airframe Diagram and the Per-Rotor Loading table, so the numbering on
    the drawing is the numbering in the table.
    """
    span = max(float(cfg.wing_span_m), 1e-3)
    chord = max(float(cfg.wing_area_m2) / span, 1e-3)
    lift_r = float(cfg.lift_prop_diameter_in) * 0.0254 / 2.0
    n = max(int(cfg.num_lift_rotors), 1)
    per_side = max(n // 2, 1)
    positions = []
    if uses_vectored_thrust(cfg):
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
    return positions[:n]


def hover_rotor_thrusts(cfg: VTOLConfig, wind_mps: float = 0.0,
                        wind_direction_deg: float = 0.0,
                        course_deg: float = 0.0) -> List[float]:
    """
    Thrust each lift rotor makes holding station in a wind, the multicopter's
    per-rotor model: the airframe's drag acts at its height above the CG, and
    the rotors counter that moment with uneven thrust. In still air, or with
    the drag at the CG, they share the load evenly.

    Holding station in a wind is translating into it at the wind speed, so
    the travel direction relative to the nose is the wind's bearing minus
    the heading.
    """
    need = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
    drag = hover_wind_drag_N(cfg, wind_mps)
    total = math.hypot(need, drag)
    # The core wants +x forward, +y right; the plan view is x right, y forward.
    positions = [(y, x) for x, y in lift_rotor_positions(cfg)]
    return core.rotor_thrust_distribution(
        positions, total, drag_N=drag,
        translation_azimuth_deg=float(wind_direction_deg) - float(course_deg),
        drag_height_above_cg_m=cfg.drag_cg_offset_m)


def make_motor_figure(cfg: VTOLConfig, m: dict, figsize):
    """
    The multicopter's and fixed-wing's motor operating-point figure, once
    per motor type: electrical power and current, then efficiency and
    RPM, against thrust per rotor — each with this run's operating point
    marked, so it is plain how much range the motor has either side.

    The cruise motor is marked at the run's airspeed, where its propeller
    meets the freestream head-on and needs several times its static power
    for the same thrust. Drawn only at 0 m/s, its marker sat far off its
    own curve (348 W against 96 W on the default aircraft). Its curve is
    now drawn at the marker's airspeed, stopping at the thrust the
    propeller can make there, with the 0 m/s curve kept faint behind it
    for comparison with a static bench table.
    """
    groups = [("lift", "Lift motor", "hover_lift", "hover")]
    if not uses_vectored_thrust(cfg):
        groups.append(("cruise", "Cruise motor", "cruise_motor", "cruise"))
    fig, axes = core.make_figure(len(groups), 2, figsize=figsize)
    axes = np.atleast_2d(axes)
    fig.suptitle("Motor Operating Points", fontsize=12, fontweight="bold")
    faint = {"alpha": 0.35, "linewidth": 1.0}
    for row, (group, name, prefix, where) in enumerate(groups):
        g = _group(cfg, group)
        n = max(int(g["n"]), 1)
        op_N = float(m.get(f"{prefix}_thrust_N", 0.0))
        avail, _src = static_thrust_available_N(cfg, group)
        per_max = max(avail / n, op_N * 1.6, 1.0)
        static = motor_operating_curve(
            cfg, group, [per_max * i / 60 for i in range(1, 61)])
        speed = float(m.get("airspeed_mps", 0.0)) if group == "cruise" else 0.0
        if speed > 0:
            top = max(forward_thrust_available_N(cfg, speed) / n if avail > 0
                      else per_max, op_N)
            main = motor_operating_curve(
                cfg, group, [top * i / 60 for i in range(1, 61)], speed)
            where = f"{where} {speed:.1f} m/s"
        else:
            main, static = static, None
        grams = [t / G0 * 1000.0 for t in main["thrust_N"]]
        op_g = op_N / G0 * 1000.0
        ref = None
        if static is not None:
            s_grams = [t / G0 * 1000.0 for t in static["thrust_N"]]
            ref = _Line2D([], [], color="gray", label="Faint: 0 m/s (static)", **faint)

        ax = axes[row, 0]
        h1, = ax.plot(grams, main["elec_W"], color="#C62828", label="Electrical power (W)")
        axb = ax.twinx()
        h2, = axb.plot(grams, main["current_A"], color="#1565C0", linestyle="--",
                       label="Current (A)")
        if static is not None:
            ax.plot(s_grams, static["elec_W"], color="#C62828", **faint)
            axb.plot(s_grams, static["current_A"], color="#1565C0", linestyle="--", **faint)
        if op_g > 0:
            ax.axvline(op_g, color="gray", linestyle=":", linewidth=1.2)
            ax.plot([op_g], [float(m.get(f"{prefix}_elec_W", 0.0))], "o", color="#C62828")
        if g["imax"]:
            axb.axhline(float(g["imax"]), color="#1565C0", linestyle=":", linewidth=0.9)
        ax.set_xlabel(f"Thrust per rotor (g)")
        ax.set_ylabel("Power (W)")
        axb.set_ylabel("Current (A)")
        ax.set_title(f"{name}: Thrust vs Power & Current ({where} marked)")
        ax.grid(alpha=0.3)
        handles = [h for h in (h1, h2, ref) if h is not None]
        ax.legend(handles, [h.get_label() for h in handles], fontsize=7, loc="upper left")

        ax = axes[row, 1]
        h1, = ax.plot(grams, main["g_per_W"], color="#2E7D32", label="Thrust per watt (g/W)")
        h3, = ax.plot(grams, [e * 10.0 for e in main["efficiency"]], color="#6A1B9A",
                      linestyle="-.", label="Motor efficiency (/10 %)")
        axb = ax.twinx()
        h2, = axb.plot(grams, main["rpm"], color="#EF6C00", linestyle="--", label="RPM")
        if static is not None:
            ax.plot(s_grams, static["g_per_W"], color="#2E7D32", **faint)
            ax.plot(s_grams, [e * 10.0 for e in static["efficiency"]], color="#6A1B9A",
                    linestyle="-.", **faint)
            axb.plot(s_grams, static["rpm"], color="#EF6C00", linestyle="--", **faint)
        if op_g > 0:
            ax.axvline(op_g, color="gray", linestyle=":", linewidth=1.2)
        if g["max_rpm"]:
            axb.axhline(float(g["max_rpm"]), color="#EF6C00", linestyle=":", linewidth=0.9)
        ax.set_xlabel("Thrust per rotor (g)")
        ax.set_ylabel("g/W  |  efficiency / 10")
        axb.set_ylabel("RPM")
        ax.set_title(f"{name}: Thrust vs Efficiency & RPM")
        ax.grid(alpha=0.3)
        handles = [h for h in (h1, h3, h2, ref) if h is not None]
        ax.legend(handles, [h.get_label() for h in handles],
                  fontsize=7, loc="upper right")
    fig.set_layout_engine("constrained", w_pad=0.06, h_pad=0.06,
                          wspace=0.08, hspace=0.12)
    return fig


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
    positions = lift_rotor_positions(cfg)
    if not vectored:
        boom_x = span * 0.30
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
    point = dict(power_at_airspeed(cfg, v))
    usable_Wh = cfg.battery.usable_Wh

    # A commanded climb or descent at the cruise point, as the other two
    # simulators apply it: the potential power W x (climb - descent) through
    # the drive. Climbing costs it over the efficiency; descending hands it
    # back scaled by the efficiency, never below the systems' own draw.
    potential_W = cfg.weight_N * (cfg.climb_rate_mps - cfg.descent_rate_mps)
    chain_eff = max(cfg.esc_efficiency * float(point.get("drive_efficiency", 1.0)), 1e-9)
    climb_add_W = potential_W / chain_eff if potential_W >= 0 else potential_W * chain_eff
    floor_W = avionics_input_power_W(cfg) + peripheral_power_W(cfg)
    point["steady_power_W"] = float(point["total_power_W"])
    point["total_power_W"] = max(float(point["total_power_W"]) + climb_add_W, floor_W)

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
        "steady_power_W": point["steady_power_W"],
        "potential_power_W": potential_W,
        "climb_power_add_W": point["total_power_W"] - point["steady_power_W"],
        "climb_rate_cmd_mps": cfg.climb_rate_mps,
        "descent_rate_cmd_mps": cfg.descent_rate_mps,
    }
    metrics.update(_detail_metrics(cfg, point, hover, pack_I))
    metrics.update(_extended_metrics(cfg, metrics, point, hover, head, cross))
    return metrics


def _motor_metrics(cfg: VTOLConfig, prefix: str, op: dict, group: str,
                   airspeed_mps: float, axial_mps: float) -> Dict[str, object]:
    """One motor-and-propeller operating point, flattened under `prefix`."""
    g = _group(cfg, group)
    rpm = float(op.get("rpm", 0.0))
    tip = tip_speed_mps(g["d_in"], rpm)
    n_rev = rpm / 60.0
    d_m = float(g["d_in"]) * 0.0254
    omega_r = rpm * 2.0 * math.pi / 60.0 * d_m / 2.0
    return {
        f"{prefix}_rpm": rpm,
        f"{prefix}_erpm": rpm * g["poles"] / 2.0,
        f"{prefix}_thrust_N": float(op.get("thrust_N", 0.0)),
        f"{prefix}_shaft_W": float(op.get("shaft_W", 0.0)),
        f"{prefix}_elec_W": float(op.get("elec_W", 0.0)),
        f"{prefix}_current_A": float(op.get("current_A", 0.0)),
        f"{prefix}_v_emf_V": float(op.get("v_emf_V", 0.0)),
        f"{prefix}_v_term_V": float(op.get("v_term_V", 0.0)),
        f"{prefix}_copper_W": float(op.get("copper_W", 0.0)),
        f"{prefix}_iron_W": float(op.get("iron_W", 0.0)),
        f"{prefix}_motor_eff": float(op.get("efficiency", 1.0)),
        f"{prefix}_throttle": float(op.get("throttle", float("nan"))),
        f"{prefix}_saturated": bool(op.get("saturated", False)),
        f"{prefix}_torque_Nm": float(op.get("torque_Nm", 0.0)),
        f"{prefix}_measured": bool(op.get("measured", False)),
        f"{prefix}_tip_speed_mps": tip,
        f"{prefix}_tip_mach": tip / speed_of_sound_mps(cfg.ambient_temp_C),
        f"{prefix}_pitch_speed_mps": float(g["p_in"]) * 0.0254 * n_rev,
        # J for a propeller meeting the flow head-on, mu for a rotor meeting
        # it edgewise; both are reported and the reader takes the one that fits.
        f"{prefix}_advance_J": axial_mps / max(n_rev * d_m, 1e-9) if n_rev > 0 else 0.0,
        f"{prefix}_advance_mu": (airspeed_mps / max(omega_r, 1e-9)) if omega_r > 0 else 0.0,
        f"{prefix}_thrust_per_W_g": (float(op.get("thrust_N", 0.0)) / G0 * 1000.0
                                     / max(float(op.get("elec_W", 0.0)), 1e-9)
                                     if op.get("elec_W") else 0.0),
    }


def _extended_metrics(cfg: VTOLConfig, m: dict, point: dict, hover: dict,
                      head: float, cross: float) -> Dict[str, object]:
    """
    Everything the multicopter's and the fixed-wing's Metrics tabs report
    that applies to a VTOL: motors and propellers, the wing's aerodynamics,
    climb, turns, glide, runway, thermal and environment.
    """
    out: Dict[str, object] = {}
    v = float(m["airspeed_mps"])
    batt = cfg.battery
    vnom = max(batt.vnom_pack, 1e-9)

    # ---- motors and propellers ----------------------------------------
    out.update(_motor_metrics(cfg, "hover_lift", hover["lift_motor"], "lift", 0.0, 0.0))
    cruise_axial = v * (math.sin(math.radians(float(m.get("tilt_deg", 90.0))))
                        if uses_vectored_thrust(cfg) else 1.0)
    out.update(_motor_metrics(cfg, "cruise_motor", point["cruise_motor"],
                              "cruise", v, cruise_axial))
    out.update(_motor_metrics(cfg, "point_lift", point["lift_motor"], "lift", v, 0.0))
    for group in ("lift", "cruise"):
        g = _group(cfg, group)
        coeff = prop_coefficients(cfg, group)
        area = math.pi / 4.0 * (float(g["d_in"]) * 0.0254) ** 2
        out[f"{group}_c_t"] = coeff["c_t"]
        out[f"{group}_c_p"] = coeff["c_p"]
        out[f"{group}_coeff_source"] = coeff["source"]
        out[f"{group}_disc_area_m2"] = area
        out[f"{group}_solidity"] = prop_solidity(g["d_in"], g["blades"])
        out[f"{group}_chord_m"] = blade_chord_m(g["d_in"], g["blades"])
        out[f"{group}_p_over_d"] = float(g["p_in"]) / max(float(g["d_in"]), 1e-9)
        out[f"{group}_kt"] = 60.0 / (2.0 * math.pi * float(g["kv"])) if g["kv"] else float("nan")
        static_T, source = static_thrust_available_N(cfg, group)
        out[f"{group}_thrust_available_N"] = static_T
        out[f"{group}_thrust_source"] = source

    # ---- hover ---------------------------------------------------------
    need = float(hover["rotor_thrust_N"])
    hover_prop_W = max(float(hover["total_power_W"]) - avionics_input_power_W(cfg)
                       - peripheral_power_W(cfg), 1e-9)
    area_lift = cfg.lift_disc_area_m2
    ideal = need * math.sqrt(need / (2.0 * cfg.air_density * max(area_lift, 1e-9)))
    lift_avail = float(out["lift_thrust_available_N"])
    out.update({
        "hover_ideal_power_W": ideal,
        "hover_shaft_W": float(hover["shaft_power_W"]),
        "hover_propulsion_power_W": hover_prop_W,
        "hover_efficiency_gW": need / G0 * 1000.0 / hover_prop_W,
        "hover_figure_of_merit": ideal / max(float(hover["shaft_power_W"]), 1e-9),
        "hover_motor_loss_W": float(hover.get("motor_loss_W", 0.0)),
        "hover_esc_loss_W": float(hover.get("esc_loss_W", 0.0)),
        "lift_twr": lift_avail / max(cfg.weight_N, 1e-9),
        "lift_thrust_margin_pct": ((lift_avail - need) / max(lift_avail, 1e-9) * 100.0
                                   if lift_avail > 0 else float("nan")),
        # Extra mass the rotors could lift, download included, in grams.
        "max_extra_payload_g": ((lift_avail / (1.0 + hover_download_fraction(cfg))
                                 - cfg.weight_N) / G0 * 1000.0
                                if lift_avail > 0 else float("nan")),
        "payload_at_twr2_g": ((lift_avail / 2.0 - cfg.weight_N) / G0 * 1000.0
                              if lift_avail > 0 else float("nan")),
        "hover_wind_limit_mps": hover_wind_limit_mps(cfg),
    })
    out.update(hover_tilt_in_wind(cfg, m["wind_mps"], m["wind_direction_deg"],
                                  m["course_deg"]))

    # ---- the wing at the cruise point ---------------------------------
    q = 0.5 * cfg.air_density * v * v
    lift = float(point.get("wing_lift_N", 0.0))
    cl = lift / max(q * cfg.wing_area_m2, 1e-9) if v > 0.1 else 0.0
    cd_i = cfg.induced_drag_factor * cl * cl
    d_induced = q * cfg.wing_area_m2 * cd_i
    d_parasite = q * cfg.wing_area_m2 * cfg.CD0
    d_stopped = (stopped_rotor_drag_N(cfg, v) * min(max(float(m.get("lift_share_wing", 1.0)), 0.0), 1.0)
                 if not uses_vectored_thrust(cfg) else 0.0)
    d_body = body_drag_N(cfg, v)
    d_total = d_induced + d_parasite + d_stopped + d_body
    lift_slope = 2.0 * math.pi * cfg.aspect_ratio / (cfg.aspect_ratio + 2.0)
    chord = cfg.wing_area_m2 / max(cfg.wing_span_m, 1e-9)
    gl = glide(cfg)
    glide_alt = (cfg.cruise_altitude_m if cfg.cruise_altitude_m is not None
                 else cfg.reference_altitude_m)
    ld = lift / max(d_total, 1e-9) if lift > 0 else 0.0
    out.update({
        "cl_cruise": cl, "cl_margin": cfg.CL_max - cl,
        "cd_cruise": cfg.CD0 + cd_i, "cd_induced": cd_i, "cd_parasite": cfg.CD0,
        "induced_parasite_ratio": cd_i / max(cfg.CD0, 1e-9),
        "drag_induced_N": d_induced, "drag_parasite_N": d_parasite,
        "drag_stopped_rotor_N": d_stopped, "drag_body_N": d_body,
        "drag_total_N": d_total, "ld_cruise": ld,
        "ld_max": gl["ld_max_analytic"],
        "aoa_deg": math.degrees(cl / max(lift_slope, 1e-9)),
        "lift_curve_slope": lift_slope, "mean_chord_m": chord,
        "reynolds_number": cfg.air_density * v * chord / MU_AIR,
        "speed_over_stall": v / max(float(m["stall_speed_mps"]), 1e-9),
        "min_sink_speed_mps": gl["min_sink_speed_mps"],
        "min_sink_rate_mps": gl["min_sink_rate_mps"],
        "best_glide_speed_mps": gl["best_glide_speed_mps"],
        "glide_ratio": gl["ld_max_analytic"],
        "glide_reference_altitude_m": glide_alt,
        "glide_distance_km": glide_alt * gl["ld_max_analytic"] / 1000.0,
        "extra_drag_source": extra_drag_areas(cfg)["source"],
        "extra_frontal_CdA_m2": extra_drag_areas(cfg)["frontal_CdA"],
        "extra_side_CdA_m2": extra_drag_areas(cfg)["side_CdA"],
    })

    # ---- best speeds, climb, ceiling, runway, turns ----------------------
    out.update(best_speeds(cfg, wind_head_mps=head, wind_cross_mps=cross))
    out["cruise_vs_best_endurance_pct"] = (v / max(out["best_endurance_speed_mps"], 1e-9) - 1.0) * 100.0
    out["cruise_vs_best_range_pct"] = (v / max(out["best_range_speed_mps"], 1e-9) - 1.0) * 100.0
    climb = best_climb(cfg)
    out.update({
        "roc_at_cruise_mps": rate_of_climb_mps(cfg, v) if v >= float(m["transition_speed_mps"]) else 0.0,
        "max_roc_mps": climb["max_roc_mps"], "vy_mps": climb["vy_mps"],
        "max_climb_angle_deg": climb["max_climb_angle_deg"], "vx_mps": climb["vx_mps"],
        "service_ceiling_m": service_ceiling_m(cfg),
        "takeoff_roll_m": takeoff_roll_m(cfg),
        "landing_distance_m": landing_distance_m(cfg),
        "forward_thrust_available_N": forward_thrust_available_N(cfg, v),
        "forward_static_thrust_N": float(out["cruise_thrust_available_N"]),
    })
    ceiling = out["service_ceiling_m"]
    out["service_ceiling_agl_m"] = (ceiling - cfg.reference_altitude_m
                                    if math.isfinite(ceiling) else float("inf"))
    tr = turn(cfg, v, cfg.bank_deg)
    out.update({f"{k}": val for k, val in tr.items()})
    out["turn_endurance_min"] = batt.usable_Wh / max(tr["turn_power_W"], 1e-9) * 60.0
    out["loiter_circles"] = (out["turn_endurance_min"] * 60.0 / tr["turn_period_s"]
                             if math.isfinite(tr["turn_period_s"]) else 0.0)

    # ---- thrust and power at the cruise point -----------------------------
    thrust_req = float(point.get("cruise_thrust_N", 0.0))
    fwd_avail = float(out["forward_thrust_available_N"])
    total = float(m["total_power_W"])
    shaft = float(point.get("shaft_power_W", 0.0))
    out.update({
        "thrust_required_N": thrust_req,
        "thrust_margin_pct": ((fwd_avail - thrust_req) / max(fwd_avail, 1e-9) * 100.0
                              if fwd_avail > 0 else float("nan")),
        "forward_twr": float(out["forward_static_thrust_N"]) / max(cfg.weight_N, 1e-9),
        "propulsive_power_W": thrust_req * v,
        "propulsive_efficiency": thrust_req * v / max(total, 1e-9),
        "system_efficiency": shaft / max(total, 1e-9),
        "power_loading_W_per_kg": total / max(cfg.all_up_weight_g / 1000.0, 1e-9),
        "specific_range_km_per_Wh": float(m["groundspeed_mps"]) * 3.6 / max(total, 1e-9),
        "specific_endurance_min_per_Wh": 60.0 / max(total, 1e-9),
        "esc_conduction_W": float(point.get("esc_conduction_W", 0.0)),
        "esc_switching_W": float(point.get("esc_switching_W", 0.0)),
        "esc_idle_W": float(point.get("esc_idle_W", 0.0)),
    })

    # ---- battery -------------------------------------------------------
    pack_I = float(m["pack_current_A"])
    r_pack = batt.pack_resistance
    reserve_pct = cfg.reserve_percent if cfg.reserve_percent is not None else 20.0
    reserve_Wh = batt.usable_Wh * reserve_pct / 100.0
    out.update({
        "battery_chemistry": batt.chemistry,
        "pack_v_full_V": batt.vmax_pack, "pack_v_nominal_V": batt.vnom_pack,
        "pack_v_cutoff_V": batt.vmin_pack,
        "pack_sag_V": batt.vmax_pack - float(m["v_load_V"]),
        "pack_resistance_ohm": r_pack,
        "battery_loss_W": pack_I * pack_I * r_pack,
        "hover_battery_loss_W": float(m["hover_pack_current_A"]) ** 2 * r_pack,
        "capacity_Wh": batt.capacity_Wh,
        "usable_mAh": batt.capacity_mAh * batt.usable_fraction,
        "hover_c_rate": float(m["hover_pack_current_A"]) / max(batt.capacity_Ah, 1e-9),
        "reserve_percent": reserve_pct,
        "reserve_target_Wh": reserve_Wh,
        "reserve_margin_Wh": batt.usable_Wh - reserve_Wh,
        "cruise_endurance_to_reserve_min": (batt.usable_Wh - reserve_Wh) / max(total, 1e-9) * 60.0,
        "hover_endurance_to_reserve_min": ((batt.usable_Wh - reserve_Wh)
                                           / max(float(m["hover_power_W"]), 1e-9) * 60.0),
    })

    # ---- mass fractions --------------------------------------------------
    auw = max(cfg.all_up_weight_g, 1e-9)
    drive = (m["lift_motor_mass_g"] + m["cruise_motor_mass_g"] + m["lift_prop_mass_g"]
             + m["cruise_prop_mass_g"] + cfg.esc_weight_g * n_escs(cfg))
    out.update({
        "battery_mass_fraction": batt.weight_g / auw,
        "drive_mass_fraction": drive / auw,
        "payload_fraction": cfg.payload_mass_g / auw,
        "drive_mass_g": drive,
    })

    # ---- thermal: hover sizes the lift side, cruise the cruise side --------
    amb = cfg.ambient_temp_C
    lift_loss = out["hover_lift_copper_W"] + out["hover_lift_iron_W"]
    cruise_loss = out["cruise_motor_copper_W"] + out["cruise_motor_iron_W"]
    esc_hover = float(hover.get("esc_loss_W", 0.0)) / max(n_escs(cfg), 1)
    esc_cruise = float(point.get("esc_loss_W", 0.0)) / max(n_escs(cfg), 1)
    t_lift = thermal_steady_C(amb, lift_loss, "motor")
    t_cruise = thermal_steady_C(amb, cruise_loss, "motor")
    t_esc = thermal_steady_C(amb, max(esc_hover, esc_cruise), "esc")
    t_batt = thermal_steady_C(amb, out["hover_battery_loss_W"], "battery")
    motor_limit = min(cfg.lift_motor_temp_limit_C, cfg.cruise_motor_temp_limit_C)
    out.update({
        "ambient_temp_C": amb,
        "lift_motor_temp_C": t_lift, "cruise_motor_temp_C": t_cruise,
        "motor_temp_est_C": max(t_lift, t_cruise),
        "esc_temp_est_C": t_esc, "battery_temp_est_C": t_batt,
        "motor_thermal_headroom_C": motor_limit - max(t_lift, t_cruise),
        "esc_thermal_headroom_C": cfg.esc_temp_limit_C - t_esc,
        "battery_thermal_headroom_C": batt.temp_limit_C - t_batt,
        "motor_copper_loss_W_per_motor": out["hover_lift_copper_W"],
        "thermal_status": thermal_status(
            (t_lift, cfg.lift_motor_temp_limit_C), (t_cruise, cfg.cruise_motor_temp_limit_C),
            (t_esc, cfg.esc_temp_limit_C), (t_batt, batt.temp_limit_C)),
        "thermal_basis": "steady state at hover (lift side) and cruise (cruise side)",
    })

    # ---- environment -----------------------------------------------------
    out.update({
        "altitude_m": cfg.reference_altitude_m,
        "cruise_altitude_m": cfg.cruise_altitude_m,
        "pressure_Pa": cfg.pressure_Pa,
        "air_density": cfg.air_density,
        "density_ratio": cfg.air_density / 1.225,
        "density_altitude_m": density_altitude_m(cfg.air_density),
        "speed_of_sound_mps": speed_of_sound_mps(amb),
    })
    return out


def _detail_metrics(cfg: VTOLConfig, point: dict, hover: dict,
                    pack_I: float) -> Dict[str, object]:
    """
    Figures the Status, Power Budget and Weight Budget tabs need that the
    headline metrics do not carry: where the losses go, what each motor
    carries, and what each component weighs.
    """
    shaft = float(point.get("shaft_power_W", 0.0))
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
        "esc_loss_W": float(point.get("esc_loss_W", 0.0)),
        "wire_loss_W": float(point.get("wire_loss_W", 0.0)),
        "motor_loss_W": float(point.get("motor_loss_W", 0.0)),
        "motor_copper_W": float(point.get("motor_copper_W", 0.0)),
        "motor_iron_W": float(point.get("motor_iron_W", 0.0)),
        "motor_input_W": float(point.get("motor_input_W", shaft)),
        "drive_efficiency": float(point.get("drive_efficiency", 1.0)),
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
                     transient_dt_s: Optional[float] = None) -> Tuple[List[tuple], Dict[str, float]]:
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

    # The step: an explicit argument, else the aircraft's Transient step
    # field, else 0.25 s.
    dt = max(float(transient_dt_s or cfg.transient_dt_s or 0.25), 0.01)
    usable_Wh = cfg.battery.usable_Wh
    # A reserve entered on the Mission/Environment tab overrides the file's,
    # as it does on the other two simulators.
    reserve_pct = (cfg.reserve_percent if cfg.reserve_percent is not None
                   else mission.reserve_percent)
    reserve_Wh = usable_Wh * reserve_pct / 100.0
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
        "segment_code",
        # carried over from the multicopter and fixed-wing mission histories
        "commanded_airspeed_mps", "groundspeed_mps", "headwind_mps", "crosswind_mps",
        "accel_mps2", "kinetic_power_W", "potential_power_W", "battery_voltage_V",
        "battery_loss_W", "esc_loss_W", "motor_loss_W", "systems_power_W",
        "lift_motor_current_A", "lift_motor_rpm", "lift_motor_throttle",
        "lift_motor_power_W", "lift_thrust_per_rotor_N",
        "cruise_motor_current_A", "cruise_motor_rpm", "cruise_motor_throttle",
        "cruise_motor_power_W", "lift_tip_mach", "advance_ratio_mu",
        "cl_wing", "lift_drag_ratio", "motor_temp_est_C", "esc_temp_est_C",
        "battery_temp_est_C", "thermal_status", "reserve_target_Wh",
        "reserve_breach")}
    # Segment kind as a number, so it can share an axis with everything else.
    # The multicopter plots its segment type the same way.
    _KIND_CODE = {"hover": 0, "climb": 1, "descend": 2,
                  "transition": 3, "cruise": 4}
    auw_kg = max(cfg.all_up_weight_g / 1000.0, 1e-9)
    worst = {"total_power_W": 0.0, "pack_current_A": 0.0, "c_rate": 0.0,
             "remaining_Wh": usable_Wh, "peak_phase": "",
             "lift_motor_current_A": 0.0, "cruise_motor_current_A": 0.0,
             "esc_current_A": 0.0, "lift_motor_throttle": 0.0,
             "cruise_motor_throttle": 0.0, "lift_motor_power_W": 0.0,
             "cruise_motor_power_W": 0.0, "motor_temp_est_C": 0.0,
             "esc_temp_est_C": 0.0, "battery_temp_est_C": 0.0,
             "min_soc_pct": 100.0, "min_battery_voltage_V": float("inf"),
             "lift_tip_mach": 0.0, "thermal_status": "OK",
             # Seconds spent above each continuous rating, for the
             # "time at max" checks.
             "lift_over_rating_s": 0.0, "cruise_over_rating_s": 0.0,
             "esc_over_rating_s": 0.0, "battery_over_rating_s": 0.0}
    state = {"t": 0.0, "alt": 0.0, "dist": 0.0, "v": 0.0}
    # Lumped temperatures, integrated step by step from ambient, per unit.
    ambient = cfg.ambient_temp_C
    temps = {"lift": ambient, "cruise": ambient, "esc": ambient, "battery": ambient}
    n_esc = max(n_escs(cfg), 1)
    vectored = uses_vectored_thrust(cfg)

    def _record(name, kind, v, alt, power, detail=None, extra=None):
        d = detail or {}
        x = extra or {}
        current = float(x.get("pack_I", power / vnom))
        lift_op = d.get("lift_motor") or {}
        cruise_op = d.get("cruise_motor") or {}
        step = float(x.get("step_dt", 0.0))
        # Temperatures: each unit warms toward ambient + R x loss.
        lift_loss = float(lift_op.get("copper_W", 0.0)) + float(lift_op.get("iron_W", 0.0))
        cruise_loss = (0.0 if vectored else
                       float(cruise_op.get("copper_W", 0.0)) + float(cruise_op.get("iron_W", 0.0)))
        for part, key, loss in (("motor", "lift", lift_loss),
                                ("motor", "cruise", cruise_loss),
                                ("esc", "esc", float(d.get("esc_loss_W", 0.0)) / n_esc),
                                ("battery", "battery", float(x.get("battery_loss_W", 0.0)))):
            temps[key] = core.thermal_step(temps[key], ambient, loss,
                                           THERMAL[part]["R"], THERMAL[part]["C"], step)
        status = thermal_status(
            (temps["lift"], cfg.lift_motor_temp_limit_C),
            (temps["cruise"], cfg.cruise_motor_temp_limit_C),
            (temps["esc"], cfg.esc_temp_limit_C),
            (temps["battery"], cfg.battery.temp_limit_C))
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

        lift_I = float(lift_op.get("current_A", 0.0))
        cruise_I = float(cruise_op.get("current_A", 0.0))
        lift_rpm = float(lift_op.get("rpm", 0.0))
        r_lift = cfg.lift_prop_diameter_in * 0.0254 / 2.0
        omega_r = lift_rpm * 2.0 * math.pi / 60.0 * r_lift
        wing_lift = float(d.get("wing_lift_N", 0.0))
        q_s = 0.5 * cfg.air_density * v * v * cfg.wing_area_m2
        motor_temp = max(temps["lift"], temps["cruise"])
        drag = float(d.get("drag_N", 0.0) or 0.0)
        for key, value in (
                ("commanded_airspeed_mps", x.get("target_v", v)),
                ("groundspeed_mps", x.get("ground", v)),
                ("headwind_mps", x.get("head", 0.0)),
                ("crosswind_mps", x.get("cross", 0.0)),
                ("accel_mps2", x.get("accel", 0.0)),
                ("kinetic_power_W", x.get("kinetic_W", 0.0)),
                ("potential_power_W", x.get("climb_W", 0.0)),
                ("battery_voltage_V", x.get("pack_v", vnom)),
                ("battery_loss_W", x.get("battery_loss_W", 0.0)),
                ("esc_loss_W", d.get("esc_loss_W", 0.0)),
                ("motor_loss_W", d.get("motor_loss_W", 0.0)),
                ("systems_power_W", avionics_input_power_W(cfg) + peripheral_power_W(cfg)),
                ("lift_motor_current_A", lift_I),
                ("lift_motor_rpm", lift_rpm),
                ("lift_motor_throttle", lift_op.get("throttle", 0.0)),
                ("lift_motor_power_W", lift_op.get("elec_W", 0.0)),
                ("lift_thrust_per_rotor_N", lift_op.get("thrust_N", 0.0)),
                ("cruise_motor_current_A", cruise_I),
                ("cruise_motor_rpm", cruise_op.get("rpm", 0.0)),
                ("cruise_motor_throttle", cruise_op.get("throttle", 0.0)),
                ("cruise_motor_power_W", cruise_op.get("elec_W", 0.0)),
                ("lift_tip_mach", tip_speed_mps(cfg.lift_prop_diameter_in, lift_rpm)
                 / speed_of_sound_mps(ambient)),
                ("advance_ratio_mu", v / omega_r if omega_r > 1e-9 else 0.0),
                ("cl_wing", wing_lift / q_s if q_s > 1e-9 else 0.0),
                ("lift_drag_ratio", wing_lift / drag if drag > 1e-9 else 0.0),
                ("motor_temp_est_C", motor_temp),
                ("esc_temp_est_C", temps["esc"]),
                ("battery_temp_est_C", temps["battery"]),
                ("reserve_target_Wh", reserve_Wh),
                ("reserve_breach", 1.0 if remaining_Wh < reserve_Wh else 0.0)):
            value = float(value if value is not None else 0.0)
            series[key].append(value if value == value else 0.0)
        series["thermal_status"].append(status)

        # Worst values, and time spent above each continuous rating.
        esc_I = max(lift_I, cruise_I)
        for key, value in (("lift_motor_current_A", lift_I),
                           ("cruise_motor_current_A", cruise_I),
                           ("esc_current_A", esc_I),
                           ("lift_motor_power_W", float(lift_op.get("elec_W", 0.0))),
                           ("cruise_motor_power_W", float(cruise_op.get("elec_W", 0.0))),
                           ("motor_temp_est_C", motor_temp),
                           ("esc_temp_est_C", temps["esc"]),
                           ("battery_temp_est_C", temps["battery"]),
                           ("lift_tip_mach", series["lift_tip_mach"][-1])):
            worst[key] = max(worst[key], value)
        for key, op in (("lift_motor_throttle", lift_op), ("cruise_motor_throttle", cruise_op)):
            th = float(op.get("throttle", 0.0) or 0.0)
            if th == th:
                worst[key] = max(worst[key], th)
        worst["min_soc_pct"] = min(worst["min_soc_pct"], series["soc_pct"][-1])
        worst["min_battery_voltage_V"] = min(worst["min_battery_voltage_V"],
                                             series["battery_voltage_V"][-1])
        order = ("HOT", "WARN", "OK")
        if order.index(status) < order.index(worst["thermal_status"]):
            worst["thermal_status"] = status
        if cfg.lift_motor_max_current_A and lift_I > cfg.lift_motor_max_current_A:
            worst["lift_over_rating_s"] += step
        if cfg.cruise_motor_max_current_A and cruise_I > cfg.cruise_motor_max_current_A:
            worst["cruise_over_rating_s"] += step
        if cfg.esc_cont_current_A and esc_I > cfg.esc_cont_current_A:
            worst["esc_over_rating_s"] += step
        if cfg.battery.discharge_cont_A and current > cfg.battery.discharge_cont_A:
            worst["battery_over_rating_s"] += step
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
            # Both extra terms below are shaft work, so they pass through the
            # motor as well as the ESC — at the efficiency the motor is
            # running at in this step.
            lift_eff = float(point.get("lift_motor", {}).get("efficiency", 1.0) or 1.0)
            drive_eff = float(point.get("drive_efficiency", 1.0) or 1.0)
            climb_W = 0.0
            if climb_rate > 0:
                thrust = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
                climb_W = thrust * climb_rate
                # Put the climb work through the lift motors themselves, so
                # the current, RPM and loss recorded for this step are the
                # climbing ones rather than the hover ones.
                lift_op = point.get("lift_motor") or {}
                n_l = max(cfg.num_lift_rotors, 1)
                if lift_op.get("thrust_N", 0.0) > 0:
                    # A measured point was handed the motor's INPUT (the
                    # bench table already contains the motor), so the climb
                    # work adds to that, as it always has. Either way elec_W
                    # is the motor input the drive chain charges.
                    measured = lift_op.get("measured", False)
                    base_W = lift_op["elec_W"] if measured else lift_op["shaft_W"]
                    climbing = motor_operating_point(
                        cfg, "lift", lift_op["thrust_N"],
                        base_W + climb_W / n_l, climb_rate, measured=measured)
                    extra_in = n_l * (climbing["elec_W"] - lift_op["elec_W"])
                    point = dict(point)
                    point["lift_motor"] = climbing
                    if uses_vectored_thrust(cfg):
                        point["cruise_motor"] = climbing
                else:
                    extra_in = climb_W / max(lift_eff, 1e-9)
                power += extra_in / max(cfg.esc_efficiency, 1e-9)

            # The kinetic cost of the speed change, as mechanical power
            # through the drive. Decelerating releases it, scaled by regen_eff.
            kinetic = core.kinetic_power_term_W(
                cfg.all_up_weight_g, v_prev, v_next, step_dt, regen_eff=regen_eff)
            # Spending kinetic energy costs it over the efficiency; recovering
            # it (regen, negative) returns only the efficiency's share.
            kinetic_W = kinetic
            chain_eff = max(cfg.esc_efficiency * drive_eff, 1e-9)
            kinetic = kinetic / chain_eff if kinetic >= 0 else kinetic * chain_eff
            power = max(power + kinetic, 0.0)

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
            battery_loss_W = pack_I * pack_I * cfg.battery.pack_resistance * r_scale
            power += battery_loss_W

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
            _record(phase.name, kind, v_next, state["alt"], power, point, {
                "step_dt": step_dt, "target_v": target_v, "ground": ground,
                "head": head, "cross": cross,
                "accel": (v_next - v_prev) / step_dt if step_dt > 1e-12 else 0.0,
                "kinetic_W": kinetic_W, "climb_W": climb_W,
                "pack_v": pack_v - pack_I * cfg.battery.pack_resistance * r_scale,
                "pack_I": pack_I, "battery_loss_W": battery_loss_W})

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
    totals["reserve_percent"] = reserve_pct
    totals["series"] = series
    worst["reserve_margin_Wh"] = worst["remaining_Wh"] - reserve_Wh
    totals["worst"] = worst
    try:
        totals["last"] = compute_metrics(cfg, state["v"])
    except Exception:
        totals["last"] = {}
    return results, totals


# ============================================================
# ONE BUILDER FOR GUI, CLI AND BATCH
# ============================================================
#
# Every input is known by its GUI field key. The GUI hands its fields to
# config_from_fields directly; the CLI maps its flags onto the same keys
# through FIELD_TO_CLI; the batch driver translates a saved GUI config
# through the same map. One builder means the three cannot drift apart —
# they did, twice: the batch map lacked CD0 (so a batch run used the CLI's
# default drag), and the CLI had no flags for the current limits.
#
# GUI key -> CLI argument name (argparse dest). None marks a GUI-only field.
FIELD_TO_CLI: Dict[str, Optional[str]] = {
    # airframe and mass
    "weight": "weight", "payload": "payload_mass_g", "mass_mode": "mass_mode",
    "structure_mass": "structure_mass", "avionics_mass": "avionics_mass",
    "span": "wing_span", "area": "wing_area", "cd0": "CD0", "oswald": "oswald",
    "clmax": "CL_max", "clcruise": "CL_cruise_max",
    "mu_roll": "mu_roll", "mu_brake": "mu_brake", "cl_takeoff": "CL_takeoff",
    "drag_model_mode": "drag_model_mode",
    "parasite_drag": "parasite_drag", "parasite_area": "parasite_area",
    "profile_drag": "profile_drag", "profile_area": "profile_area",
    "body_length_m": "body_length_m", "body_width_m": "body_width_m",
    "body_height_m": "body_height_m", "arm_length_m": "arm_length_m",
    "arm_width_m": "arm_width_m", "drag_cg_offset_m": "drag_cg_offset_m",
    # lift rotors
    "n_lift": "num_lift_rotors", "lift_layout": "lift_rotor_layout",
    "coax_spacing": "coaxial_spacing_m",
    "lift_d": "lift_prop_diameter", "lift_p": "lift_prop_pitch",
    "lift_blades": "lift_prop_blades", "lift_kv": "lift_motor_kv",
    "lift_rm": "lift_motor_resistance", "lift_i0": "lift_motor_i0",
    "lift_v0": "lift_motor_v0", "lift_wt": "lift_motor_weight",
    "lift_imax": "lift_motor_max_current", "lift_pmax": "lift_motor_max_power",
    "lift_max_time": "lift_motor_max_time", "lift_temp_limit": "lift_motor_temp_limit",
    "lift_v_unit": "lift_motor_v_unit", "lift_s_min": "lift_motor_rating_min",
    "lift_s_max": "lift_motor_rating_max", "lift_poles": "lift_motor_pole_count",
    "lift_size": "lift_motor_size", "fom": "lift_figure_of_merit",
    "stopped_area": "stopped_rotor_drag_area", "download": "hover_download",
    "lift_prop_wt": "lift_prop_weight", "lift_max_thrust": "lift_prop_max_thrust",
    "lift_max_rpm": "lift_prop_max_rpm", "lift_tconst": "lift_prop_tconst",
    "lift_pconst": "lift_prop_pconst", "lift_table": "lift_prop_table",
    "inflow_map_enabled": "inflow_map_enabled", "inflow_mu_bp": "inflow_mu_bp",
    "inflow_eff_bp": "inflow_eff_bp",
    # cruise propulsion
    "n_cruise": "num_cruise_motors", "cruise_d": "cruise_prop_diameter",
    "cruise_p": "cruise_prop_pitch", "cruise_blades": "cruise_prop_blades",
    "cruise_kv": "cruise_motor_kv", "cruise_rm": "cruise_motor_resistance",
    "cruise_i0": "cruise_motor_i0", "cruise_v0": "cruise_motor_v0",
    "cruise_wt": "cruise_motor_weight", "cruise_imax": "cruise_motor_max_current",
    "cruise_pmax": "cruise_motor_max_power", "cruise_max_time": "cruise_motor_max_time",
    "cruise_temp_limit": "cruise_motor_temp_limit", "cruise_v_unit": "cruise_motor_v_unit",
    "cruise_s_min": "cruise_motor_rating_min", "cruise_s_max": "cruise_motor_rating_max",
    "cruise_poles": "cruise_motor_pole_count", "cruise_size": "cruise_motor_size",
    "cruise_eff": "cruise_prop_efficiency", "cruise_eff_model": "cruise_prop_eff_model",
    "cruise_prop_wt": "cruise_prop_weight", "cruise_max_thrust": "cruise_prop_max_thrust",
    "cruise_max_rpm": "cruise_prop_max_rpm", "cruise_tconst": "cruise_prop_tconst",
    "cruise_pconst": "cruise_prop_pconst", "cruise_table": "cruise_prop_table",
    # battery
    "chem": "battery_chemistry", "cell_cap": "battery_cell_capacity",
    "series": "battery_series_cells", "parallel": "battery_parallel_cells",
    "cell_wt": "battery_cell_weight_g", "vmin": "battery_voltage_min",
    "vnom": "battery_voltage_nominal", "vmax": "battery_voltage_max",
    "rcell": "battery_resistance_cell", "usable": "battery_usable_percent",
    "unit_mode": "battery_unit_mode", "cells_s_per_pack": "battery_cells_series_per_pack",
    "cells_p_per_pack": "battery_cells_parallel_per_pack",
    "pack_cap": "battery_pack_capacity", "pack_wt": "battery_pack_weight_g",
    "energy_density": "battery_energy_density", "a_cont": "battery_a_cont",
    "a_max": "battery_a_max", "charge_a": "battery_charge_current",
    "c_cont": "battery_c_cont", "c_max": "battery_c_max",
    "batt_max_time": "battery_max_time", "batt_temp_limit": "battery_temp_limit",
    "soc_model": "battery_soc_model", "soc_curve": "soc_curve",
    "soc_bp": "battery_soc_bp", "ocv_cell_bp": "battery_ocv_cell_bp",
    "r_scale_bp": "battery_r_scale_bp",
    # ESC
    "esc_eff_tab": "esc_efficiency", "esc_r": "esc_resistance",
    "esc_cont": "esc_cont_current", "esc_imax": "esc_max_current",
    "esc_idle": "esc_idle_current", "esc_wt": "esc_weight_g",
    "esc_max_time": "esc_max_time", "esc_temp_limit": "esc_temp_limit",
    "esc_v_unit": "esc_v_unit", "esc_s_min": "esc_rating_min",
    "esc_s_max": "esc_rating_max",
    # avionics and wiring
    "avionics_flat": "avionics_power", "periph_current": "peripheral_current",
    "avionics_rails": "avionics_rails",
    # wiring: the core's mapping, so all three simulators share the flags
    **{key: flag for key, flag in core.WIRING_FIELD_TO_CLI.items()},
    # mission and environment
    "cruise_v": "cruise_speed", "alt": "altitude", "cruise_altitude": "cruise_altitude",
    "temp": "temperature", "pressure": "pressure", "mission": "mission",
    "wind": "wind", "wind_dir": "wind_direction", "course_deg": "course_deg",
    "bank_deg": "bank_deg", "climb_rate": "climb_rate_mps",
    "descent_rate": "descent_rate_mps", "reserve_percent": "reserve_percent",
    "accel": "max_accel", "decel": "max_decel", "regen": "regen_eff",
    "transient_dt": "transient_dt_s", "min_climb": "min_climb_mps",
    "field_takeoff": "field_takeoff_m", "field_landing": "field_landing_m",
    "max_tilt": "max_tilt_deg", "max_pitch": "max_pitch_deg", "max_roll": "max_roll_deg",
    # GUI-only: which connector preset was picked, and the plot range
    "conn_batt": None, "conn_esc": None, "conn_motor": None, "plot_vmax": None,
}

# Fields older saved configs carry under a name that has since moved. Read
# them when the new key is absent, so a config saved before the move still
# loads to the same aircraft.
LEGACY_FIELD_ALIASES = {"avionics": "avionics_flat", "esc_eff": "esc_eff_tab"}


# Dropdown values that were renamed, to match the other two simulators.
LEGACY_FIELD_VALUES = {"mass_mode": {"derive structure": "derive airframe",
                                     "enter structure": "enter airframe"}}


def migrate_legacy_fields(values: dict) -> dict:
    """
    Copy each legacy key onto its new name when the new one is blank, and
    translate renamed dropdown values.
    """
    out = dict(values)
    for old, new in LEGACY_FIELD_ALIASES.items():
        if str(out.get(old, "")).strip() and not str(out.get(new, "")).strip():
            out[new] = out[old]
    for key, renames in LEGACY_FIELD_VALUES.items():
        value = str(out.get(key, "")).strip()
        if value in renames:
            out[key] = renames[value]
    return out


def parse_rails(spec) -> dict:
    """
    "5:2:0.9, 12:1.5:0.87" into {volts: (amps, efficiency)}.

    Stored as one string so the rails save and load with every other field
    rather than needing their own serialisation path.
    """
    if isinstance(spec, dict):
        return dict(spec)
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


def config_from_fields(values: dict, config_type: str = "lift+cruise") -> VTOLConfig:
    """
    Build the aircraft from input values keyed by GUI field name.

    Values may be strings (the GUI, a saved config) or numbers (the CLI).
    Blank means "not given": optional inputs stay None, and the rest take
    the defaults below — the same defaults the GUI fields start with.
    """
    values = migrate_legacy_fields(values or {})

    def raw(key):
        x = values.get(key)
        return "" if x is None else str(x).strip()

    def num(key, default=0.0):
        s = raw(key)
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"'{s}' is not a number (field: {key})")

    def opt(key):
        s = raw(key)
        return None if s == "" else num(key)

    def text(key, default=""):
        return raw(key) or default

    def flag(key, default=False):
        s = raw(key).lower()
        if s == "":
            return default
        return s in ("1", "true", "yes", "on")

    series, parallel = int(num("series", 6)), int(num("parallel", 2))
    battery = VTOLBattery(
        chemistry=text("chem", "LiPo"),
        cell_capacity_mAh=num("cell_cap", 5000),
        series_cells=series, parallel_cells=parallel,
        cell_weight_g=num("cell_wt", 120),
        voltage_min=num("vmin", 3.3), voltage_nominal=num("vnom", 3.7),
        voltage_max=num("vmax", 4.2),
        resistance_cell_mOhm=num("rcell", 4.0),
        usable_percent=num("usable", 80),
        discharge_c_cont=opt("c_cont"), discharge_c_max=opt("c_max"),
        soc_curve_csv=(raw("soc_curve") or None),
        soc_model=text("soc_model", "auto"),
        unit_mode=text("unit_mode", "cell"),
        cells_series_per_unit=int(opt("cells_s_per_pack") or 1),
        cells_parallel_per_unit=int(opt("cells_p_per_pack") or 1),
        pack_capacity_mAh=opt("pack_cap"), pack_weight_g=opt("pack_wt"),
        energy_density_Wh_per_kg=opt("energy_density"),
        discharge_cont_A=opt("a_cont"), discharge_max_A=opt("a_max"),
        charge_current_max_A=opt("charge_a"),
        soc_bp=_parse_float_list(raw("soc_bp")),
        ocv_cell_bp=_parse_float_list(raw("ocv_cell_bp")),
        r_scale_bp=_parse_float_list(raw("r_scale_bp")),
        max_time_s=opt("batt_max_time"),
        temp_limit_C=num("batt_temp_limit", 55.0),
    )

    n_lift = int(num("n_lift", 4))
    n_cruise = int(num("n_cruise", 1))
    # "enter airframe" runs the weight the other way: the frame is known and
    # the all-up weight is that plus the itemised parts. "enter structure" is
    # the name older configs saved it under.
    weight = num("weight", 6000)
    if text("mass_mode", "derive airframe").lower().startswith("enter"):
        vectored = str(config_type).strip().lower() in ("tiltrotor", "tiltwing", "tailsitter")
        n_esc = n_lift + (0 if vectored else n_cruise)
        weight = ((opt("structure_mass") or 0.0)
                  + battery.weight_g
                  + ((opt("lift_wt") or 0.0) + (opt("lift_prop_wt") or 0.0)) * n_lift
                  + (0.0 if vectored else
                     ((opt("cruise_wt") or 0.0) + (opt("cruise_prop_wt") or 0.0)) * n_cruise)
                  + (opt("esc_wt") or 0.0) * n_esc
                  + (opt("avionics_mass") or 0.0))

    temp = opt("temp")
    # The Wiring tab goes through the same core builder as the multicopter's
    # and fixed-wing's, so a lead costs the same in all three.
    wiring = core.wiring_from_fields(values)

    cfg = VTOLConfig(
        config_type=config_type,
        aircraft_weight_g=weight, payload_mass_g=num("payload", 0),
        wing_span_m=num("span", 2.4), wing_area_m2=num("area", 0.6),
        CD0=num("cd0", 0.035), oswald=num("oswald", 0.8),
        CL_max=num("clmax", 1.2), CL_cruise_max=num("clcruise", 0.9),
        num_lift_rotors=n_lift,
        lift_prop_diameter_in=num("lift_d", 18), lift_prop_pitch_in=num("lift_p", 6),
        lift_motor_kv=num("lift_kv", 300), lift_motor_resistance=num("lift_rm", 0.08),
        lift_motor_weight_g=num("lift_wt", 200),
        lift_figure_of_merit=num("fom", 0.65),
        num_cruise_motors=n_cruise,
        cruise_prop_diameter_in=num("cruise_d", 14),
        cruise_prop_pitch_in=num("cruise_p", 8),
        cruise_motor_kv=num("cruise_kv", 500),
        cruise_motor_resistance=num("cruise_rm", 0.06),
        cruise_motor_weight_g=num("cruise_wt", 180),
        cruise_prop_efficiency=num("cruise_eff", 0.75),
        stopped_rotor_drag_area_m2=opt("stopped_area"),
        battery=battery,
        avionics_power_W=num("avionics_flat", 15),
        avionics_rails=parse_rails(values.get("avionics_rails")),
        periph_current_A=opt("periph_current") or 0.0,
        esc_resistance_ohm=opt("esc_r") or 0.0,
        esc_max_current_A=opt("esc_imax"),
        esc_efficiency=num("esc_eff_tab", 0.96),
        esc_weight_g=opt("esc_wt") or 0.0,
        # Pressure overrides the altitude-derived value, which is what a field
        # barometer reading is for: the standard atmosphere is an average.
        air_density=core.air_density(num("alt", 0), temp, opt("pressure")),
        cruise_speed_mps=num("cruise_v", 22),
        reference_altitude_m=num("alt", 0),
        wire_resistance_ohm=wiring.resistance_ohm if wiring is not None else 0.0,
        connectors={name: (rating[0] or 0.0, rating[1] or 0.0)
                    for name, rating in (wiring.connectors if wiring else {}).items()},
        hover_download_fraction=opt("download"),
        lift_motor_max_power_W=opt("lift_pmax"),
        lift_motor_max_current_A=opt("lift_imax"),
        lift_prop_max_thrust_g=opt("lift_max_thrust") or 0.0,
        cruise_motor_max_current_A=opt("cruise_imax"),
        cruise_motor_max_power_W=opt("cruise_pmax"),
        lift_prop_weight_g=opt("lift_prop_wt") or 0.0,
        cruise_prop_weight_g=opt("cruise_prop_wt") or 0.0,
        avionics_mass_g=opt("avionics_mass") or 0.0,
        lift_prop_table_csv=(raw("lift_table") or None),
        cruise_prop_table_csv=(raw("cruise_table") or None),
        lift_motor_i0_A=num("lift_i0", 0.5), lift_motor_v0_V=opt("lift_v0"),
        cruise_motor_i0_A=num("cruise_i0", 0.5), cruise_motor_v0_V=opt("cruise_v0"),
        lift_motor_max_time_s=opt("lift_max_time"),
        cruise_motor_max_time_s=opt("cruise_max_time"),
        lift_motor_temp_limit_C=num("lift_temp_limit", 100.0),
        cruise_motor_temp_limit_C=num("cruise_temp_limit", 100.0),
        lift_motor_v_unit=text("lift_v_unit", "S"),
        lift_motor_rating_min=opt("lift_s_min"), lift_motor_rating_max=opt("lift_s_max"),
        cruise_motor_v_unit=text("cruise_v_unit", "S"),
        cruise_motor_rating_min=opt("cruise_s_min"),
        cruise_motor_rating_max=opt("cruise_s_max"),
        lift_motor_pole_count=int(num("lift_poles", 14)),
        cruise_motor_pole_count=int(num("cruise_poles", 14)),
        lift_motor_size=text("lift_size"), cruise_motor_size=text("cruise_size"),
        lift_prop_blades=int(num("lift_blades", 2)),
        cruise_prop_blades=int(num("cruise_blades", 2)),
        lift_prop_max_rpm=opt("lift_max_rpm"), cruise_prop_max_rpm=opt("cruise_max_rpm"),
        lift_prop_tconst=opt("lift_tconst"), lift_prop_pconst=opt("lift_pconst"),
        cruise_prop_tconst=opt("cruise_tconst"), cruise_prop_pconst=opt("cruise_pconst"),
        cruise_prop_max_thrust_g=opt("cruise_max_thrust") or 0.0,
        cruise_prop_eff_model=text("cruise_eff_model", "constant"),
        esc_cont_current_A=opt("esc_cont"),
        esc_idle_current_A=opt("esc_idle") or 0.0,
        esc_max_time_s=opt("esc_max_time"),
        esc_temp_limit_C=num("esc_temp_limit", 90.0),
        esc_v_unit=text("esc_v_unit", "S"),
        esc_rating_min=opt("esc_s_min"), esc_rating_max=opt("esc_s_max"),
        lift_rotor_layout=text("lift_layout", "flat"),
        coaxial_spacing_m=opt("coax_spacing"),
        inflow_map_enabled=flag("inflow_map_enabled", False),
        inflow_mu_bp=_parse_float_list(raw("inflow_mu_bp")),
        inflow_eff_bp=_parse_float_list(raw("inflow_eff_bp")),
        drag_model_mode=text("drag_model_mode", "auto"),
        parasite_drag_cd=opt("parasite_drag"), parasite_area_m2=opt("parasite_area"),
        profile_drag_cd=opt("profile_drag"), profile_area_m2=opt("profile_area"),
        body_length_m=opt("body_length_m"), body_width_m=opt("body_width_m"),
        body_height_m=opt("body_height_m"), arm_length_m=opt("arm_length_m"),
        arm_width_m=opt("arm_width_m"),
        drag_cg_offset_m=opt("drag_cg_offset_m") or 0.0,
        max_tilt_deg=num("max_tilt", 25.0),
        max_pitch_deg=opt("max_pitch"), max_roll_deg=opt("max_roll"),
        mu_roll=num("mu_roll", 0.04), mu_brake=num("mu_brake", 0.30),
        CL_takeoff=num("cl_takeoff", 0.80),
        cruise_altitude_m=opt("cruise_altitude"),
        # Run settings the single-point figures and Status read.
        bank_deg=num("bank_deg", 0.0),
        climb_rate_mps=num("climb_rate", 0.0),
        descent_rate_mps=num("descent_rate", 0.0),
        reserve_percent=opt("reserve_percent"),
        transient_dt_s=opt("transient_dt"),
        min_climb_mps=opt("min_climb"),
        field_takeoff_m=opt("field_takeoff"),
        field_landing_m=opt("field_landing"),
        ambient_temp_C=temp,
        pressure_Pa=opt("pressure"),
    )
    cfg.wiring = wiring
    return cfg


def fields_from_args(args) -> dict:
    """The CLI's flags as GUI-keyed values, for config_from_fields."""
    values = {}
    for key, dest in FIELD_TO_CLI.items():
        if dest and hasattr(args, dest):
            value = getattr(args, dest)
            if value is not None:
                values[key] = value
    return values


def load_fields_file(path: str) -> Tuple[dict, str]:
    """
    Read a GUI-saved configuration: (values keyed by field, config type).

    The rails are stored in their own field as a string, so everything the
    builder needs is in `vars`.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") not in (None, "vtol_power_sim_v1"):
        raise ValueError(
            f"{os.path.basename(path)} is a '{payload.get('schema')}' file, not "
            "a VTOL configuration.")
    return (migrate_legacy_fields(dict(payload.get("vars") or {})),
            str(payload.get("config_type") or "lift+cruise"))


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Every aircraft flag defaults to None: blank means "not given", and the
    default is applied once, in config_from_fields, exactly as for a blank
    GUI field. That is also what lets --config load a saved GUI file and the
    flags given alongside it override only what they name.
    """
    p = argparse.ArgumentParser(
        description="VTOL UAV power and endurance simulator.")
    p.add_argument("--gui", action="store_true", help="Open the graphical interface.")
    p.add_argument("--config", type=str, default=None,
                   help="A configuration saved by the GUI. Flags given as well "
                        "override the values in it.")
    p.add_argument("--config_type", type=str, default=None, choices=CONFIG_TYPES,
                   help="Airframe configuration (default lift+cruise).")

    def add(name, kind=float, help_text=None, **kw):
        p.add_argument(f"--{name}", type=kind, default=None, help=help_text, **kw)

    # ---- airframe, mass and drag -------------------------------------
    add("weight", help_text="All-up weight without payload (g).")
    add("payload_mass_g")
    add("mass_mode", str, "derive airframe | enter airframe",
        choices=["derive airframe", "enter airframe",
                 "derive structure", "enter structure"])
    add("structure_mass", help_text="Airframe mass (g), bare frame; 'enter airframe' mode only.")
    add("avionics_mass", help_text="Avionics mass (g), for the weight budget.")
    add("wing_span"); add("wing_area"); add("CD0"); add("oswald"); add("CL_max")
    add("CL_cruise_max", help_text="CL cap used during transition; below CL_max for margin.")
    add("mu_roll", help_text="Rolling friction, for a conventional take-off roll.")
    add("mu_brake", help_text="Braking friction, for a conventional landing roll.")
    add("CL_takeoff", help_text="CL at rotation, for a conventional take-off roll.")
    add("drag_model_mode", str, "Extra airframe drag beyond CD0: auto | manual | geometry",
        choices=["auto", "manual", "geometry"])
    add("parasite_drag", help_text="Frontal Cd of fuselage and booms beyond CD0.")
    add("parasite_area", help_text="Frontal area (m^2) that Cd applies to.")
    add("profile_drag", help_text="Side Cd, met hovering level in a wind.")
    add("profile_area", help_text="Side area (m^2) that Cd applies to.")
    for dim in ("body_length_m", "body_width_m", "body_height_m",
                "arm_length_m", "arm_width_m"):
        add(dim, help_text="Geometry for the derived extra drag (m).")
    add("drag_cg_offset_m", help_text="Height of the drag centre above the CG (m).")

    # ---- lift rotors ----------------------------------------------------
    add("num_lift_rotors", int)
    add("lift_rotor_layout", str, "flat | coaxial", choices=["flat", "coaxial"])
    add("coaxial_spacing_m", help_text="Vertical spacing of a coaxial pair (m).")
    add("lift_prop_diameter"); add("lift_prop_pitch"); add("lift_prop_blades", int)
    add("lift_motor_kv", help_text="0 turns the motor electrical model off.")
    add("lift_motor_resistance"); add("lift_motor_i0", help_text="No-load current (A).")
    add("lift_motor_v0", help_text="Voltage the no-load current was measured at (V).")
    add("lift_motor_weight")
    add("lift_motor_max_current", help_text="Rated current per lift motor (A).")
    add("lift_motor_max_power", help_text="Rated power per lift motor (W).")
    add("lift_motor_max_time", help_text="Seconds the lift motor may run above rating.")
    add("lift_motor_temp_limit", help_text="Lift motor temperature limit (C).")
    add("lift_motor_v_unit", str, "S (cells) or V (volts)", choices=["S", "V"])
    add("lift_motor_rating_min"); add("lift_motor_rating_max")
    add("lift_motor_pole_count", int); add("lift_motor_size", str)
    add("lift_figure_of_merit")
    add("stopped_rotor_drag_area")
    add("hover_download", help_text="Hover download as a fraction of weight.")
    add("lift_prop_weight", help_text="Mass of one lift propeller (g).")
    add("lift_prop_max_thrust", help_text="Rated static thrust of one lift propeller (g).")
    add("lift_prop_max_rpm"); add("lift_prop_tconst"); add("lift_prop_pconst")
    add("lift_prop_table", str, "Measured lift-rotor thrust/power CSV.")
    add("inflow_map_enabled", str, "1 to apply the rotor inflow map, 0 to leave it off.")
    add("inflow_mu_bp", str, "Advance-ratio breakpoints, comma separated.")
    add("inflow_eff_bp", str, "Inflow efficiency at each breakpoint.")

    # ---- cruise propulsion ---------------------------------------------
    add("num_cruise_motors", int)
    add("cruise_prop_diameter"); add("cruise_prop_pitch"); add("cruise_prop_blades", int)
    add("cruise_motor_kv", help_text="0 turns the motor electrical model off.")
    add("cruise_motor_resistance"); add("cruise_motor_i0"); add("cruise_motor_v0")
    add("cruise_motor_weight")
    add("cruise_motor_max_current", help_text="Rated current per cruise motor (A).")
    add("cruise_motor_max_power", help_text="Rated power per cruise motor (W).")
    add("cruise_motor_max_time"); add("cruise_motor_temp_limit")
    add("cruise_motor_v_unit", str, "S (cells) or V (volts)", choices=["S", "V"])
    add("cruise_motor_rating_min"); add("cruise_motor_rating_max")
    add("cruise_motor_pole_count", int); add("cruise_motor_size", str)
    add("cruise_prop_efficiency", help_text="Propeller efficiency alone; the motor "
                                            "is modelled separately.")
    add("cruise_prop_eff_model", str, "constant | curve", choices=["constant", "curve"])
    add("cruise_prop_weight"); add("cruise_prop_max_thrust"); add("cruise_prop_max_rpm")
    add("cruise_prop_tconst"); add("cruise_prop_pconst")
    add("cruise_prop_table", str, "Measured cruise-prop thrust/power CSV.")

    # ---- battery --------------------------------------------------------
    add("battery_chemistry", str)
    add("battery_cell_capacity"); add("battery_series_cells", int)
    add("battery_parallel_cells", int); add("battery_cell_weight_g")
    add("battery_voltage_min"); add("battery_voltage_nominal"); add("battery_voltage_max")
    add("battery_resistance_cell"); add("battery_usable_percent")
    add("battery_soc_model", str,
        "auto | linear | a chemistry name. 'linear' turns the discharge curve off: "
        "voltage is held at full charge all flight, which is optimistic near the "
        "end of the pack.")
    add("battery_unit_mode", str,
        "cell: series/parallel count CELLS. pack: they count finished PACKS, each "
        "of --battery_cells_series_per_pack cells in series.", choices=["cell", "pack"])
    add("battery_cells_series_per_pack", int); add("battery_cells_parallel_per_pack", int)
    add("battery_pack_capacity", help_text="Capacity of ONE pack (mAh). Pack mode only.")
    add("battery_pack_weight_g", help_text="Weight of ONE pack (g). Pack mode only.")
    add("battery_energy_density", help_text="Wh/kg. Reported only.")
    add("battery_a_cont", help_text="Continuous discharge limit (A).")
    add("battery_a_max", help_text="Burst discharge limit (A).")
    add("battery_charge_current", help_text="Maximum charge current (A).")
    add("battery_c_cont"); add("battery_c_max")
    add("battery_max_time", help_text="Seconds the pack may be held at its max rating.")
    add("battery_temp_limit", help_text="Cell temperature limit (C).")
    add("battery_soc_bp", str, "SoC breakpoints, 0..1, comma separated.")
    add("battery_ocv_cell_bp", str, "Open-circuit volts per cell at each breakpoint.")
    add("battery_r_scale_bp", str, "Resistance multiplier at each breakpoint.")
    add("soc_curve", str, "Measured pack discharge curve CSV.")

    # ---- ESC, avionics, wiring -----------------------------------------
    add("esc_efficiency", help_text="ESC efficiency under load.")
    add("esc_resistance", help_text="Splits the ESC loss into conduction and switching.")
    add("esc_cont_current", help_text="Continuous rating of one ESC (A).")
    add("esc_max_current", help_text="Burst rating of one ESC (A).")
    add("esc_idle_current", help_text="Standby draw of one ESC (A).")
    add("esc_weight_g", help_text="Mass of ONE ESC (g).")
    add("esc_max_time"); add("esc_temp_limit")
    add("esc_v_unit", str, "S (cells) or V (volts)", choices=["S", "V"])
    add("esc_rating_min"); add("esc_rating_max")
    add("avionics_power", help_text="Flat avionics draw (W), used when no rails are given.")
    add("avionics_rails", str, "Regulated rails as 'V:A:eff, V:A:eff'. Replace the flat figure.")
    add("peripheral_current", help_text="Current drawn straight from the pack (A).")
    add("wire_length", help_text="One-way battery lead length (m).")
    add("wire_awg", int, "Wire gauge (AWG).")
    add("wire_ohm_per_m", help_text="Measured wire resistance (ohm/m).")
    add("wire_temp_limit",
        help_text=f"Wire insulation temperature limit (C); default {core.WIRE_TEMP_LIMIT_C:g}.")
    for _name in ("batt", "esc", "motor"):
        add(f"connector_{_name}_cont", help_text=f"{_name} connector continuous rating (A).")
        add(f"connector_{_name}_max", help_text=f"{_name} connector burst rating (A).")
        add(f"connector_{_name}_volt", help_text=f"{_name} connector rated voltage (V).")

    # ---- flight and environment ----------------------------------------
    add("cruise_speed"); add("altitude")
    add("cruise_altitude", help_text="Height flown (m ASL), for glide distance and ceiling.")
    add("temperature"); add("pressure", help_text="Static pressure (Pa).")
    add("mission", str)
    add("wind", help_text="Steady wind speed (m/s).")
    add("wind_direction", help_text="Direction the wind comes FROM, compass degrees.")
    add("course_deg", help_text="Heading the fixed-speed run flies, compass degrees.")
    add("bank_deg", help_text="Bank angle for the turning-flight figures (deg).")
    add("climb_rate_mps", help_text="Commanded climb at the cruise point (m/s).")
    add("descent_rate_mps", help_text="Commanded descent at the cruise point (m/s).")
    add("reserve_percent", help_text="Energy reserve (%). Overrides the mission file's.")
    add("max_accel", help_text="Acceleration limit (m/s^2). 0 ignores transients.")
    add("max_decel", help_text="Deceleration limit (m/s^2). Defaults to --max_accel.")
    add("regen_eff", help_text="Fraction of braking energy recovered (0-1).")
    add("transient_dt_s", help_text="Mission time step (s).")
    add("min_climb_mps", help_text="Required climb rate, checked on Status (m/s).")
    add("field_takeoff_m", help_text="Runway available for a conventional take-off (m).")
    add("field_landing_m", help_text="Runway available for a conventional landing (m).")
    add("max_tilt_deg", help_text="Largest hover tilt the controller allows (deg).")
    add("max_pitch_deg"); add("max_roll_deg")
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
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text] or None
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


def values_from_args(args) -> Tuple[dict, str]:
    """
    Everything the CLI was told, keyed by GUI field: a --config file first,
    then any flags given on top of it.
    """
    values, config_type = {}, "lift+cruise"
    if getattr(args, "config", None):
        values, config_type = load_fields_file(args.config)
    values.update(fields_from_args(args))
    return values, (getattr(args, "config_type", None) or config_type)


def config_from_args(args) -> VTOLConfig:
    values, config_type = values_from_args(args)
    return config_from_fields(values, config_type)


def run_settings(values: dict) -> dict:
    """The per-run inputs that are not part of the aircraft."""
    def num(key, default=0.0):
        s = "" if values.get(key) is None else str(values.get(key)).strip()
        return float(s) if s else default
    return {"wind": num("wind"), "wind_dir": num("wind_dir"),
            "course": num("course_deg"), "accel": num("accel"),
            "decel": num("decel"), "regen": num("regen"),
            "mission": str(values.get("mission") or "").strip()}


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
    print_performance_summary(m, print)


def _fmt(x, fmt="{:.2f}", na="n/a") -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return na
    return fmt.format(x) if math.isfinite(x) else na


def performance_summary_lines(m: dict) -> List[str]:
    """
    The lines the multicopter and fixed-wing print after a fixed-speed run,
    for the VTOL. Shared by the CLI and the GUI's Output pane so the two say
    the same thing.
    """
    reserve_ok = float(m.get("reserve_margin_Wh", 0.0)) >= 0
    ceiling = float(m.get("service_ceiling_m", float("inf")))
    lines = [
        f"  Best endurance speed : {m['best_endurance_speed_mps']:.1f} m/s -> "
        f"{m['best_endurance_min']:.1f} min",
        f"  Best range speed     : {m['best_range_speed_mps']:.1f} m/s -> "
        f"{m['best_range_km']:.2f} km",
        # Labels as the multicopter prints them, so the batch driver's
        # parser reads the VTOL's numbers too.
        f"  Hover Efficiency     : {m['hover_efficiency_gW']:.2f} g/W",
        f"  Figure of Merit      : {m['hover_figure_of_merit']:.3f} (achieved)",
        f"  Lift thrust/weight   : {_fmt(m['lift_twr'])} ({m['lift_thrust_source']})",
        f"  Lift motor (hover)   : {m['hover_lift_rpm']:.0f} rpm, "
        f"{m['hover_lift_current_A']:.2f} A, throttle "
        f"{_fmt(m['hover_lift_throttle'] * 100, '{:.0f}')}%, "
        f"eff {m['hover_lift_motor_eff'] * 100:.1f}%",
        f"  Cruise motor         : {m['cruise_motor_rpm']:.0f} rpm, "
        f"{m['cruise_motor_current_A']:.2f} A, throttle "
        f"{_fmt(m['cruise_motor_throttle'] * 100, '{:.0f}')}%",
        f"  Lift tip Mach        : {m['hover_lift_tip_mach']:.3f}"
        + ("  (significant aeroacoustic noise likely)" if m['hover_lift_tip_mach'] > 0.6 else ""),
        f"  L/D cruise / max     : {m['ld_cruise']:.2f} / {m['ld_max']:.2f}",
        f"  Best climb rate      : {m['max_roc_mps']:.2f} m/s at {m['vy_mps']:.1f} m/s",
        f"  Service ceiling      : "
        + ("above 8000 m" if not math.isfinite(ceiling) else f"{ceiling:.0f} m"),
        f"  Motor + ESC losses   : {m['motor_loss_W']:.1f} + {m['esc_loss_W']:.1f} W",
        f"  Reserve target/margin: {m['reserve_target_Wh']:.1f} / "
        f"{m['reserve_margin_Wh']:+.1f} Wh ({m['reserve_percent']:.0f}%)",
        f"  Reserve Status       : {'OK' if reserve_ok else 'VIOLATION'}",
        f"  Motor Thermal Status : {m['thermal_status']}",
        f"  Thermal M/ESC/Batt   : {m['motor_temp_est_C']:.1f} / {m['esc_temp_est_C']:.1f} / "
        f"{m['battery_temp_est_C']:.1f} C [{m['thermal_status']}]",
        f"  Hover wind limit     : "
        + _fmt(m['hover_wind_limit_mps'], "{:.1f} m/s",
               "n/a (set a profile area or body dimensions)"),
    ]
    if m.get("hover_lift_saturated") or m.get("cruise_motor_saturated"):
        lines.append("  WARNING: a motor needs more than 100% throttle — its Kv is "
                     "too low for this pack and propeller.")
    return lines


def print_performance_summary(m: dict, out=print) -> None:
    for line in performance_summary_lines(m):
        out(line)


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
    for line in mission_summary_lines(totals):
        print(line)


def mission_summary_lines(totals: dict) -> List[str]:
    """The worst-case lines the other two simulators print after a mission."""
    w = totals.get("worst") or {}
    if "min_soc_pct" not in w:
        return []
    return [
        f"  Reserve      : {totals.get('reserve_percent', 20):.0f}% target, lowest "
        f"margin {w['reserve_margin_Wh']:+.1f} Wh "
        f"[{'OK' if w['reserve_margin_Wh'] >= 0 else 'VIOLATION'}]",
        f"  SoC (min)    : {w['min_soc_pct']:.1f}%",
        f"  Peak current : pack {w['pack_current_A']:.1f} A, lift motor "
        f"{w['lift_motor_current_A']:.1f} A, cruise motor {w['cruise_motor_current_A']:.1f} A",
        f"  Thermal peak : motor {w['motor_temp_est_C']:.1f} / ESC {w['esc_temp_est_C']:.1f} / "
        f"battery {w['battery_temp_est_C']:.1f} C [{w['thermal_status']}]",
    ]


def main() -> None:
    core.make_console_safe()
    args = build_arg_parser().parse_args()
    if args.gui:
        launch_gui(args)
        return

    values, config_type = values_from_args(args)
    try:
        cfg = config_from_fields(values, config_type)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
    run = run_settings(values)
    try:
        if run["mission"]:
            _print_mission(cfg, run["mission"], run["wind"], run["wind_dir"],
                           run["accel"], run["decel"], run["regen"])
        else:
            _print_single_point(cfg, wind_mps=run["wind"],
                                wind_direction_deg=run["wind_dir"],
                                course_deg=run["course"])
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
    r = add_row(tab_airframe, r, "All-up weight without payload (g)", "weight", 6000,
                "Everything except the payload, including battery and motors. "
                "Entered in 'derive airframe' mode; calculated, and greyed "
                "out, in 'enter airframe' mode.")
    r = add_row(tab_airframe, r, "Payload mass (g)", "payload", 0,
                "Added on top of the all-up weight without payload.")
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
                    ("lift_kv", "lift_rm", "lift_i0", "lift_wt"), row=r)
    r = add_row(tab_lift, r, "Lift motor Kv", "lift_kv", 300,
                    "RPM per volt, unloaded. With Rm and the no-load current it "
                    "drives the motor model: RPM, current, back-EMF, throttle and "
                    "the motor's own losses.\n\n"
                    "Low Kv on a big rotor is the efficient combination; high Kv "
                    "on a small one buys responsiveness at the cost of endurance. "
                    "0 turns the motor model off.")
    r = add_row(tab_lift, r, "Lift motor Rm (ohm)", "lift_rm", 0.08,
                    "Winding resistance of one lift motor. Drives the copper loss, "
                    "which grows with the SQUARE of current — so it bites hardest in "
                    "hover.")
    r = add_row(tab_lift, r, "Lift motor no-load current I0 (A)", "lift_i0", 0.5,
                    "Current the motor draws spinning with no propeller, from its "
                    "datasheet. Times back-EMF it is the iron and bearing loss, "
                    "paid however lightly the motor is loaded.")
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
                    ("cruise_kv", "cruise_rm", "cruise_i0", "cruise_wt"), row=r)
    r = add_row(tab_cruise, r, "Cruise motor Kv", "cruise_kv", 500,
                    "RPM per volt for the cruise motor. Usually higher than the lift "
                    "motors', because it turns a smaller propeller faster. 0 turns "
                    "the motor model off.")
    r = add_row(tab_cruise, r, "Cruise motor Rm (ohm)", "cruise_rm", 0.06,
                    "Winding resistance of one cruise motor, driving its copper loss.")
    r = add_row(tab_cruise, r, "Cruise motor no-load current I0 (A)", "cruise_i0", 0.5,
                    "Datasheet no-load current of the cruise motor, for its iron "
                    "and bearing loss.")
    r = add_row(tab_cruise, r, "Cruise motor weight (g)", "cruise_wt", 180,
                    "Mass of one cruise motor, for the Weight Budget.")
    r = add_section(tab_cruise, "Propeller Efficiency",
                    ("cruise_eff", "cruise_eff_model"), row=r)
    r = add_row(tab_cruise, r, "Cruise prop efficiency", "cruise_eff", 0.75,
                "The PROPELLER's efficiency in cruise — thrust power over shaft "
                "power. The motor is modelled separately from its Kv, Rm and "
                "I0, so do not fold its losses in here.\n\n"
                "Typical 0.75-0.85 for a well-matched cruise propeller. Before "
                "v1.11 this was the combined motor-and-propeller figure; divide "
                "an old value by about 0.9 to convert it.")

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
    # The avionics power and ESC efficiency that used to sit here duplicated
    # the Avionics and ESC tabs' own fields, and the two could disagree. A
    # config saved with the old keys still loads: migrate_legacy_fields maps
    # them onto the tab fields.
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

    # Mass entry mode, named and behaving as on the other two simulators.
    # The airframe is normally the residual once the itemised parts come out
    # of the all-up weight; this lets you work the other way when you know
    # the frame and want the all-up weight derived.
    _mass_row = tab_airframe.grid_size()[1]
    ttk.Label(tab_airframe, text="Mass Entry Mode").grid(
        row=_mass_row, column=0, sticky="w", pady=2)
    v_mass_mode = tk.StringVar(value="derive airframe")
    fields["mass_mode"] = v_mass_mode
    ttk.Combobox(tab_airframe, textvariable=v_mass_mode, state="readonly",
                 width=18, values=("derive airframe", "enter airframe")).grid(
        row=_mass_row, column=1, sticky="w", padx=(8, 4))
    _mm = ttk.Label(tab_airframe, text="?", foreground="#0B6BCB",
                    cursor="question_arrow")
    _mm.grid(row=_mass_row, column=2, sticky="w")
    core.Tooltip(_mm,
                 "derive airframe: you enter the all-up weight without "
                 "payload, and the Weight Budget shows the airframe as "
                 "whatever is left after the battery, motors, propellers, "
                 "ESCs and avionics.\n\n"
                 "enter airframe: you enter the airframe mass below, and the "
                 "all-up weight is the sum of that plus the itemised parts. "
                 "Useful when you know the frame but are still choosing "
                 "components.\n\n"
                 "Whichever of the two is calculated is greyed out.")
    _append_row(tab_airframe, "Airframe mass (g)", "structure_mass", "",
                "Bare frame, booms, skins and fasteners — everything that is "
                "not a component listed elsewhere. Entered in 'enter "
                "airframe' mode; greyed out in 'derive airframe' mode, where "
                "it is the Weight Budget's residual.")

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
                "Blank uses the type default: lift+cruise 0.04, tiltrotor "
                "0.10, tiltwing 0.02, tailsitter 0.02. Typical published "
                "values, not measurements — override with test data if you "
                "have it.\n\n"
                "It fades as the wing takes over, so hover and cruise meet "
                "without a step.")
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
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", pady=2)
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
        # Registered like any other row, so Simple mode and the greying-out
        # below reach the pickers too.
        _field_rows.append({"key": key, "widgets": [lbl, holder, marker]})

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
                ("accel", "decel", "regen", "transient_dt"))
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
    _append_row(tab_env, "Transient step dt (s)", "transient_dt", "",
                "Mission time step. Blank uses 0.25 s. Smaller is finer and "
                "slower; the answer should not move much below 0.5 s.")
    add_section(tab_env, "Wind", ("wind", "wind_dir", "course_deg"))
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
                "Switching and conduction efficiency of one ESC under load. "
                "The ESC alone: the motors' losses come from their own Kv, "
                "Rm and no-load current, so a datasheet figure belongs here.\n\n"
                "Typical 0.95-0.98.")
    _append_row(tab_esc, "ESC resistance (Ω)", "esc_r", "",
                "Optional. Series resistance of one ESC. It does not add a "
                "loss — the efficiency above already covers everything lost "
                "under load — but splits that loss into conduction (I²R) and "
                "switching on the Metrics tab.")
    _append_row(tab_esc, "ESC continuous current (A)", "esc_cont", "",
                "Continuous rating of one ESC. Status checks the current each "
                "ESC carries in hover and in cruise against it.")
    _append_row(tab_esc, "ESC max current (A)", "esc_imax", "",
                "Burst rating of one ESC. Above continuous but below this is "
                "amber on Status; above it is red.")
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
    _append_row(tab_wiring, "Wire temperature limit (°C)", "wire_temp_limit", "",
                f"What the insulation is rated for. Blank uses "
                f"{core.WIRE_TEMP_LIMIT_C:g} °C, typical of silicone hookup wire; "
                f"PVC is often 80-105 °C. Status estimates the lead's steady "
                f"temperature at hover current in still air — conservative, "
                f"since the lead usually sits in some airflow.")
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
        _append_row(tab_wiring, "   rated voltage (V)", f"{_prefix}_volt", "",
                    "Checked against the pack's FULL-charge voltage. Blank "
                    "skips the check; most hobby connectors publish no figure.")

        def _fill(*_a, _t=_type_var, _p=_prefix):
            # Convenience, not a lock: ratings stay editable, because burst
            # figures vary widely between manufacturers.
            defaults = core.connector_defaults(_t.get())
            if defaults:
                fields[f"{_p}_cont"].set(f"{defaults[0]:g}")
                fields[f"{_p}_max"].set(f"{defaults[1]:g}")
                volt = core.connector_voltage_default(_t.get())
                fields[f"{_p}_volt"].set(f"{volt:g}" if volt else "")
        _type_var.trace_add("write", _fill)

    # ==================================================================
    # INPUTS CARRIED OVER FROM THE MULTICOPTER AND FIXED-WING
    # ==================================================================
    # Each block below is an input one or both of the other simulators has,
    # with the same meaning. Blank or default values leave every result as it
    # was before the field existed; each tooltip says what the field feeds.

    # ---- Airframe: extra drag, conventional take-off --------------------
    add_section(tab_airframe, "Extra Airframe Drag (beyond CD0)",
                ("drag_model_mode", "parasite_drag", "parasite_area",
                 "profile_drag", "profile_area", "body_length_m", "body_width_m",
                 "body_height_m", "arm_length_m", "arm_width_m", "drag_cg_offset_m"))
    add_combo_row(tab_airframe, "Drag model mode", "drag_model_mode",
                  ("auto", "manual", "geometry"), "auto",
                  "Drag of the fuselage, booms or payload pod ON TOP of the "
                  "wing's CD0 — the multicopter's drag inputs.\n\n"
                  "auto: the Cd and areas below if entered, else the body and "
                  "boom dimensions if entered, else nothing.\n"
                  "manual: only the entered Cd and areas.\n"
                  "geometry: derive from the dimensions (box body, square-tube "
                  "booms).\n\n"
                  "Everything blank adds no drag, so CD0 stays the "
                  "whole-aircraft figure it has always been.")
    _append_row(tab_airframe, "Parasite Cd (frontal)", "parasite_drag", "",
                "Drag coefficient of the frontal silhouette, met in forward "
                "flight. Adds to the cruise and transition drag.")
    _append_row(tab_airframe, "Parasite area (m²)", "parasite_area", "",
                "Frontal area that Cd applies to.")
    _append_row(tab_airframe, "Profile Cd (side)", "profile_drag", "",
                "Drag coefficient of the side silhouette, met hovering level "
                "in a wind. Sets the hover wind limit and the hover tilt Status "
                "checks.")
    _append_row(tab_airframe, "Profile area (m²)", "profile_area", "",
                "Side area that Cd applies to.")
    _append_row(tab_airframe, "Body length (m)", "body_length_m", "",
                "Fuselage length, for the derived side area.")
    _append_row(tab_airframe, "Body width (m)", "body_width_m", "",
                "Fuselage width, for the derived frontal area.")
    _append_row(tab_airframe, "Body height (m)", "body_height_m", "",
                "Fuselage height, for both derived areas.")
    _append_row(tab_airframe, "Boom length (m)", "arm_length_m", "",
                "Length of one rotor boom or pylon. A lift+cruise's booms run "
                "fore and aft, so they present only their end in cruise.")
    _append_row(tab_airframe, "Boom width (m)", "arm_width_m", "",
                "Outer width of the boom tube. Blank uses 20 mm.")
    _append_row(tab_airframe, "Drag height above CG (m)", "drag_cg_offset_m", "",
                "Where the drag acts relative to the centre of gravity. Held in "
                "a wind, drag above the CG pitches the aircraft and the rotors "
                "must counter it with uneven thrust — shown on the Per-Rotor "
                "Loading table. Blank or 0 assumes it acts through the CG.")
    add_section(tab_airframe, "Conventional Take-off / Landing",
                ("mu_roll", "mu_brake", "cl_takeoff"))
    _append_row(tab_airframe, "Rolling friction μ", "mu_roll", "0.04",
                "Wheel friction on the runway, for the conventional take-off "
                "roll on the Metrics tab — what the aircraft would need if it "
                "took off on its cruise propeller like an aeroplane.\n\n"
                "0.02-0.05 paved, 0.08-0.12 grass.")
    _append_row(tab_airframe, "Braking friction μ", "mu_brake", "0.30",
                "Braking friction for the conventional landing roll.")
    _append_row(tab_airframe, "CL at take-off rotation", "cl_takeoff", "0.80",
                "Lift coefficient at rotation, below CL_max so the wing does "
                "not stall as it leaves the ground.")

    # ---- Lift rotors: layout, propeller, ratings --------------------------
    add_section(tab_lift, "Layout and Propeller",
                ("lift_layout", "coax_spacing", "lift_blades"))
    add_combo_row(tab_lift, "Lift rotor layout", "lift_layout", ("flat", "coaxial"),
                  "flat",
                  "coaxial: the lift rotors are stacked in pairs, one above the "
                  "other. The lower rotor works in the upper one's wake, which "
                  "costs about 18% more hover power at a 0.2 D spacing — the "
                  "multicopter's model. 'Number of lift rotors' counts every "
                  "rotor, so an X8 is 8.")
    _append_row(tab_lift, "Coaxial spacing (m)", "coax_spacing", "",
                "Vertical gap between the two rotors of a pair. Blank assumes "
                "0.2 x diameter. Wider spacing costs less.")
    _append_row(tab_lift, "Lift prop blades", "lift_blades", "2",
                "Blade count. More blades raise the thrust coefficient, so the "
                "rotor makes the same thrust at lower RPM.")
    add_section(tab_lift, "Lift Motor Ratings",
                ("lift_v0", "lift_max_time", "lift_temp_limit", "lift_v_unit",
                 "lift_s_min", "lift_s_max", "lift_poles", "lift_size"))
    _append_row(tab_lift, "Lift motor I0 test voltage (V)", "lift_v0", "",
                "The voltage the datasheet measured I0 at. Given it, the "
                "no-load current is scaled with motor speed, as iron loss is. "
                "Blank holds I0 constant.")
    _append_row(tab_lift, "Lift motor time at max (s)", "lift_max_time", "",
                "How long the motor may run above its continuous rating. A "
                "mission holding it there longer is flagged on Status.")
    _append_row(tab_lift, "Lift motor temp limit (°C)", "lift_temp_limit", "100",
                "Winding temperature the motor must stay below.")
    add_combo_row(tab_lift, "Lift motor voltage unit", "lift_v_unit", ("S", "V"), "S",
                  "Whether the rating range below is in cells (S) or volts.")
    _append_row(tab_lift, "Lift motor rating min", "lift_s_min", "",
                "Lowest pack voltage the motor is rated for.")
    _append_row(tab_lift, "Lift motor rating max", "lift_s_max", "",
                "Highest pack voltage the motor is rated for. Status checks the "
                "pack against it.")
    _append_row(tab_lift, "Lift motor pole count", "lift_poles", "14",
                "Magnet poles. Sets the electrical RPM the ESC must commutate.")
    _append_row(tab_lift, "Lift motor size", "lift_size", "",
                "Stator size, e.g. 5010. Shown on Metrics; nothing is computed "
                "from it.")
    add_section(tab_lift, "Lift Propeller Coefficients",
                ("lift_max_rpm", "lift_tconst", "lift_pconst"))
    _append_row(tab_lift, "Lift prop max RPM", "lift_max_rpm", "",
                "The propeller maker's RPM limit. Status checks hover RPM "
                "against it.")
    _append_row(tab_lift, "Lift prop TConst (C_T)", "lift_tconst", "",
                "Thrust coefficient, T = C_T rho n² D⁴. Sets RPM from thrust. "
                "Blank fits it from a bench table's RPM column, else estimates "
                "it from diameter, pitch and blades (±30%).")
    _append_row(tab_lift, "Lift prop PConst (C_P)", "lift_pconst", "",
                "Power coefficient, P = C_P rho n³ D⁵. Shown on Metrics.")
    add_section(tab_lift, "Rotor Inflow Map",
                ("inflow_map_enabled", "inflow_mu_bp", "inflow_eff_bp"))
    add_combo_row(tab_lift, "Inflow map enabled (1/0)", "inflow_map_enabled",
                  ("0", "1"), "0",
                  "The multicopter's empirical correction to lift-rotor power "
                  "against advance ratio μ = V / ΩR. Off by default: the "
                  "forward-flight inflow solver already carries translational "
                  "lift, and this is a refinement for measured data.")
    _append_row(tab_lift, "Inflow μ breakpoints", "inflow_mu_bp", "",
                "Comma separated. Blank uses 0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.50.")
    _append_row(tab_lift, "Inflow η breakpoints", "inflow_eff_bp", "",
                "Efficiency at each μ; above 1 makes the rotor cheaper. Blank "
                "uses 1.00, 1.04, 1.08, 1.06, 1.00, 0.94, 0.88.")

    # ---- Cruise: propeller, ratings --------------------------------------
    add_section(tab_cruise, "Cruise Propeller Detail",
                ("cruise_blades", "cruise_eff_model", "cruise_max_thrust"))
    _append_row(tab_cruise, "Cruise prop blades", "cruise_blades", "2",
                "Blade count of the cruise propeller.")
    add_combo_row(tab_cruise, "Prop efficiency model", "cruise_eff_model",
                  ("constant", "curve"), "constant",
                  "constant: the propeller efficiency above at every speed.\n\n"
                  "curve: the fixed-wing's model — the entered figure is the "
                  "PEAK, reached near 60% of the full-throttle pitch speed, "
                  "falling away either side. Needs the cruise motor Kv.")
    _append_row(tab_cruise, "Cruise prop rated thrust (g)", "cruise_max_thrust", "",
                "Static thrust ONE cruise propeller is rated for. Sets the "
                "thrust available for the climb figures and the conventional "
                "take-off roll; blank derives it from the motor's max power.")
    add_section(tab_cruise, "Cruise Motor Ratings",
                ("cruise_v0", "cruise_max_time", "cruise_temp_limit", "cruise_v_unit",
                 "cruise_s_min", "cruise_s_max", "cruise_poles", "cruise_size"))
    _append_row(tab_cruise, "Cruise motor I0 test voltage (V)", "cruise_v0", "",
                "The voltage the datasheet measured I0 at. Blank holds I0 constant.")
    _append_row(tab_cruise, "Cruise motor time at max (s)", "cruise_max_time", "",
                "How long the motor may run above its continuous rating.")
    _append_row(tab_cruise, "Cruise motor temp limit (°C)", "cruise_temp_limit", "100",
                "Winding temperature the motor must stay below.")
    add_combo_row(tab_cruise, "Cruise motor voltage unit", "cruise_v_unit",
                  ("S", "V"), "S", "Whether the rating range is in cells (S) or volts.")
    _append_row(tab_cruise, "Cruise motor rating min", "cruise_s_min", "",
                "Lowest pack voltage the motor is rated for.")
    _append_row(tab_cruise, "Cruise motor rating max", "cruise_s_max", "",
                "Highest pack voltage the motor is rated for.")
    _append_row(tab_cruise, "Cruise motor pole count", "cruise_poles", "14",
                "Magnet poles, for the electrical RPM.")
    _append_row(tab_cruise, "Cruise motor size", "cruise_size", "",
                "Stator size. Shown on Metrics only.")
    add_section(tab_cruise, "Cruise Propeller Coefficients",
                ("cruise_max_rpm", "cruise_tconst", "cruise_pconst"))
    _append_row(tab_cruise, "Cruise prop max RPM", "cruise_max_rpm", "",
                "The propeller maker's RPM limit.")
    _append_row(tab_cruise, "Cruise prop TConst (C_T)", "cruise_tconst", "",
                "Static thrust coefficient. Blank fits or estimates it.")
    _append_row(tab_cruise, "Cruise prop PConst (C_P)", "cruise_pconst", "",
                "Static power coefficient. Shown on Metrics.")

    # ---- Battery: thermal limits --------------------------------------------
    add_section(tab_batt, "Thermal Limits (optional)",
                ("batt_max_time", "batt_temp_limit"))
    _append_row(tab_batt, "Time at max C-rate (s)", "batt_max_time", "",
                "How long the pack may be held at its burst rating. A mission "
                "that holds it above continuous longer is flagged on Status.")
    _append_row(tab_batt, "Battery temp limit (°C)", "batt_temp_limit", "55",
                "Cell temperature the pack must stay below. Cells age fast "
                "above about 60 °C.")

    # ---- ESC: ratings -------------------------------------------------------
    add_section(tab_esc, "Ratings and Limits",
                ("esc_idle", "esc_max_time", "esc_temp_limit", "esc_v_unit",
                 "esc_s_min", "esc_s_max"))
    _append_row(tab_esc, "ESC idle current (A)", "esc_idle", "",
                "Standby draw of one ESC, paid whether or not its motor turns — "
                "so a lift+cruise pays it for the stopped lift ESCs all through "
                "cruise. Blank is none.")
    _append_row(tab_esc, "ESC time at max (s)", "esc_max_time", "",
                "How long the ESC may run above its continuous rating.")
    _append_row(tab_esc, "ESC temp limit (°C)", "esc_temp_limit", "90",
                "Temperature the ESC must stay below.")
    add_combo_row(tab_esc, "ESC voltage unit", "esc_v_unit", ("S", "V"), "S",
                  "Whether the rating range is in cells (S) or volts.")
    _append_row(tab_esc, "ESC rating min", "esc_s_min", "",
                "Lowest pack voltage the ESC is rated for.")
    _append_row(tab_esc, "ESC rating max", "esc_s_max", "",
                "Highest pack voltage the ESC is rated for. Status checks the "
                "pack against it.")

    # ---- Mission / environment: the run settings ---------------------------
    add_section(tab_env, "Flight Settings",
                ("reserve_percent", "cruise_altitude", "bank_deg", "climb_rate",
                 "descent_rate", "max_tilt", "max_pitch", "max_roll"))
    _append_row(tab_env, "Reserve percent (%)", "reserve_percent", "",
                "Energy kept back at landing, as a share of usable energy. "
                "Sets the reserve target and margin on Metrics and Status, and "
                "overrides the mission file's own reserve when entered.\n\n"
                "Blank uses the mission file's value, or 20% for a fixed-speed "
                "run.")
    _append_row(tab_env, "Cruise altitude (m)", "cruise_altitude", "",
                "Height actually flown, above sea level. Used for the glide "
                "distance and to express the ceiling above it. Blank means "
                "the same as Altitude, which is the field elevation.")
    _append_row(tab_env, "Bank angle (deg)", "bank_deg", "",
                "Bank for the Turning Flight figures on Metrics: load factor, "
                "turn radius and rate, turn stall speed and the power a "
                "sustained turn costs. Blank or 0 is straight flight.")
    _append_row(tab_env, "Climb rate cmd (m/s)", "climb_rate", "",
                "A steady climb at the cruise speed. Its potential power is "
                "added to the cruise figures, as on the other two simulators.")
    _append_row(tab_env, "Descent rate cmd (m/s)", "descent_rate", "",
                "A steady descent at the cruise speed. Its potential power is "
                "given back. Ignored when a climb is entered.")
    _append_row(tab_env, "Max hover tilt (deg)", "max_tilt", "25",
                "Largest tilt the flight controller allows in hover. Sets the "
                "hover wind limit on Metrics and the hover tilt check on "
                "Status.")
    _append_row(tab_env, "Max pitch (deg)", "max_pitch", "",
                "Pitch limit in hover, for the station-keeping check. Blank "
                "uses the tilt limit.")
    _append_row(tab_env, "Max roll (deg)", "max_roll", "",
                "Roll limit in hover. A crosswind is held with roll.")
    add_section(tab_env, "Requirements (checked on Status)",
                ("min_climb", "field_takeoff", "field_landing"))
    _append_row(tab_env, "Minimum climb rate (m/s)", "min_climb", "",
                "The wing-borne climb rate the aircraft must manage. Status "
                "checks the best climb against it.")
    _append_row(tab_env, "Take-off run available (m)", "field_takeoff", "",
                "Runway for a conventional take-off, for when the aircraft is "
                "too heavy to lift off vertically.")
    _append_row(tab_env, "Landing distance available (m)", "field_landing", "",
                "Runway for a conventional landing.")

    # ==================================================================
    # GREY OUT WHAT A DROPDOWN MAKES IRRELEVANT
    # ==================================================================
    # As on the multicopter and fixed-wing: an input the current selection
    # does not use is greyed out rather than left looking live, so it is
    # never possible to type a number that silently does nothing. Greyed
    # fields keep their values — switching back restores them — and nothing
    # here changes a result, because the model already ignores these inputs.
    _CRUISE_HARDWARE = (
        "n_cruise", "cruise_d", "cruise_p", "cruise_blades", "cruise_kv",
        "cruise_rm", "cruise_i0", "cruise_v0", "cruise_wt", "cruise_imax",
        "cruise_pmax", "cruise_prop_wt", "cruise_max_thrust", "cruise_table",
        "cruise_max_time", "cruise_temp_limit", "cruise_v_unit", "cruise_s_min",
        "cruise_s_max", "cruise_poles", "cruise_size", "cruise_max_rpm",
        "cruise_tconst", "cruise_pconst")

    def _set_enabled(widget, enabled):
        """ttk's state flag works the same for labels, entries and dropdowns."""
        try:
            widget.state(["!disabled"] if enabled else ["disabled"])
        except (AttributeError, tk.TclError):
            try:
                widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            _set_enabled(child, enabled)

    def _enable_fields(keys, enabled):
        wanted = set(keys)
        for row in _field_rows:
            if row["key"] in wanted:
                for widget in row["widgets"]:
                    _set_enabled(widget, enabled)

    def _apply_dependencies(*_a):
        pack = fields["unit_mode"].get().strip().lower() == "pack"
        _enable_fields(("cell_cap", "cell_wt"), not pack)
        _enable_fields(("cells_s_per_pack", "cells_p_per_pack", "pack_cap", "pack_wt"), pack)

        entering = fields["mass_mode"].get().strip().lower().startswith("enter")
        _enable_fields(("weight",), not entering)
        _enable_fields(("structure_mass",), entering)

        vectored = v_config_type.get() in ("tiltrotor", "tiltwing", "tailsitter")
        _enable_fields(_CRUISE_HARDWARE, not vectored)
        # Stopped-rotor drag belongs to a lift+cruise only: the vectored
        # types' rotors never stop.
        _enable_fields(("stopped_area",), not vectored)

        _enable_fields(("coax_spacing",), fields["lift_layout"].get() == "coaxial")

        mode = fields["drag_model_mode"].get().strip().lower()
        _enable_fields(("body_length_m", "body_width_m", "body_height_m",
                        "arm_length_m", "arm_width_m"), mode != "manual")
        _enable_fields(("parasite_drag", "parasite_area", "profile_drag",
                        "profile_area"), mode != "geometry")

        _enable_fields(("inflow_mu_bp", "inflow_eff_bp"),
                       fields["inflow_map_enabled"].get().strip() == "1")

        _enable_fields(("soc_curve", "soc_bp", "ocv_cell_bp", "r_scale_bp"),
                       fields["soc_model"].get().strip().lower() != "linear")

    for _var in (fields["unit_mode"], fields["mass_mode"], v_config_type,
                 fields["lift_layout"], fields["drag_model_mode"],
                 fields["inflow_map_enabled"], fields["soc_model"]):
        _var.trace_add("write", _apply_dependencies)
    _apply_dependencies()

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

    def _at_least(value, minimum, warn_frac=0.9):
        """ok at or above `minimum`, warn a little below it, bad well below."""
        try:
            v, lim = float(value), float(minimum)
        except (TypeError, ValueError):
            return "na"
        if not (math.isfinite(v) and math.isfinite(lim)):
            return "na"
        if v >= lim:
            return "ok"
        return "warn" if v >= lim * warn_frac else "bad"

    def _temp_row(tv, label, temp, limit, note=""):
        tag = "bad" if temp > limit else ("warn" if temp > limit - 10.0 else "ok")
        _row(tv, label, f"{temp:.1f} °C", f"<= {limit:.0f} °C", tag,
             note or ("Within 10 °C of the limit." if tag == "warn" else
                      "Above the limit." if tag == "bad" else "Comfortable."))

    def _rating_row(tv, label, unit, lo, hi):
        check = voltage_rating_check(_status_cfg["cfg"], unit, lo, hi)
        if check is None:
            _row(tv, label, "", "Not Specified", "na",
                 "Enter the component's voltage rating to check the pack against it.")
        else:
            _row(tv, label, check["value"], check["limit"],
                 "ok" if check["ok"] else "bad",
                 "The pack is inside the rated range." if check["ok"] else
                 "The pack is OUTSIDE the rated range — the part will be "
                 "over- or under-driven.")

    def _time_row(tv, label, seconds, limit):
        if not limit:
            if seconds > 0:
                _row(tv, label, f"{seconds:.0f} s", "Not Specified", "na",
                     "Time spent above the continuous rating. Enter a time at "
                     "max to check it.")
            return
        _row(tv, label, f"{seconds:.0f} s", f"<= {limit:.0f} s",
             _classify(seconds, limit),
             "Time spent above the continuous rating during the mission.")

    _status_cfg = {"cfg": None}

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
        _status_cfg["cfg"] = cfg
        batt = cfg.battery
        cont_A, max_A = batt.discharge_cont_A, batt.discharge_max_A
        vectored = uses_vectored_thrust(cfg)

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
            if "min_soc_pct" in worst:
                soc = float(worst["min_soc_pct"])
                _row(batt_status_tv, "Minimum state of charge", f"{soc:.1f} %", "> 0 %",
                     "ok" if soc > 0 else "bad", "Of the usable energy.")
                v_min = float(worst["min_battery_voltage_V"])
                _row(batt_status_tv, "Lowest loaded pack voltage", f"{v_min:.2f} V",
                     f">= {batt.vmin_pack:.2f} V",
                     "ok" if v_min >= batt.vmin_pack else "bad",
                     "Under load, at the emptiest point. Below the cutoff the "
                     "ESCs may cut out.")
                _temp_row(batt_status_tv, "Peak battery temperature",
                          float(worst["battery_temp_est_C"]), batt.temp_limit_C)
                _time_row(batt_status_tv, "Time above pack rating",
                          float(worst["battery_over_rating_s"]), batt.max_time_s)

                _dual(motor_status_tv, "Peak lift motor current",
                      worst["lift_motor_current_A"], cfg.lift_motor_max_current_A, None, "A")
                _time_row(motor_status_tv, "Lift motor time above rating",
                          float(worst["lift_over_rating_s"]), cfg.lift_motor_max_time_s)
                if not vectored:
                    _dual(motor_status_tv, "Peak cruise motor current",
                          worst["cruise_motor_current_A"], cfg.cruise_motor_max_current_A,
                          None, "A")
                    _time_row(motor_status_tv, "Cruise motor time above rating",
                              float(worst["cruise_over_rating_s"]), cfg.cruise_motor_max_time_s)
                _dual(motor_status_tv, "Peak ESC current", worst["esc_current_A"],
                      cfg.esc_cont_current_A, cfg.esc_max_current_A, "A")
                _time_row(motor_status_tv, "ESC time above rating",
                          float(worst["esc_over_rating_s"]), cfg.esc_max_time_s)
                for label, key in (("Peak lift motor throttle", "lift_motor_throttle"),
                                   ("Peak cruise motor throttle", "cruise_motor_throttle")):
                    if key == "cruise_motor_throttle" and vectored:
                        continue
                    th = float(worst.get(key, 0.0)) * 100.0
                    if th > 0:
                        _row(motor_status_tv, label, f"{th:.0f} %", "<= 100 %",
                             _classify(th, 100.0),
                             "Above 100% the pack cannot spin the motor fast "
                             "enough for the thrust asked of it.")
                _temp_row(motor_status_tv, "Peak motor temperature",
                          float(worst["motor_temp_est_C"]),
                          min(cfg.lift_motor_temp_limit_C, cfg.cruise_motor_temp_limit_C),
                          "Integrated through the mission from ambient.")
                _temp_row(motor_status_tv, "Peak ESC temperature",
                          float(worst["esc_temp_est_C"]), cfg.esc_temp_limit_C)
                _row(motor_status_tv, "Thermal status", str(worst["thermal_status"]),
                     "OK", {"OK": "ok", "WARN": "warn", "HOT": "bad"}.get(
                         str(worst["thermal_status"]), "na"))
                mach = float(worst["lift_tip_mach"])
                _row(rotor_status_tv, "Peak lift rotor tip Mach", f"{mach:.3f}", "<= 0.60",
                     _classify(mach, 0.60),
                     "Above about Mach 0.6 noise rises sharply and efficiency falls.")
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
        _dual(batt_status_tv, "Cruise C-rate", point_I / max(batt.capacity_Ah, 1e-9),
              batt.discharge_c_cont, batt.discharge_c_max, "C")
        v_hover = batt.voltage_under_load(hover_I)
        _row(batt_status_tv, "Pack voltage (loaded, hover)", f"{v_hover:.2f} V",
             f">= {batt.vmin_pack:.2f} V",
             "ok" if v_hover >= batt.vmin_pack * 1.05 else
             ("warn" if v_hover >= batt.vmin_pack else "bad"),
             "Full-charge pack under the hover current. Near the cutoff the "
             "ESCs may cut out before the pack is empty.")
        _row(batt_status_tv, "Total electrical power",
             f"{float(m.get('hover_power_W', 0.0)):.0f} W hover / "
             f"{float(m.get('total_power_W', 0.0)):.0f} W cruise", "—", "na")
        total = max(float(m.get("total_power_W", 0.0)), 1e-9)
        motor_in = float(m.get("motor_input_W", 0.0))
        systems = float(m.get("avionics_input_power_W", 0.0)) + float(m.get("peripheral_power_W", 0.0))
        _row(batt_status_tv, "Power split (Motor/ESC/Av)",
             f"{motor_in / total * 100:.0f} / "
             f"{float(m.get('esc_loss_W', 0.0)) / total * 100:.0f} / "
             f"{systems / total * 100:.0f} %", "—", "na",
             "At the cruise speed: motor input, ESC loss and avionics plus "
             "peripherals, as shares of pack power.")
        _row(batt_status_tv, "Usable energy", f"{float(m.get('usable_Wh', 0.0)):.1f} Wh",
             "—", "na")
        if "reserve_margin_Wh" in m:
            margin = float(m["reserve_margin_Wh"])
            _row(batt_status_tv, "Energy reserve margin", f"{margin:.1f} Wh", ">= 0 Wh",
                 "ok" if margin >= 0 else "bad",
                 f"Usable energy above the {float(m.get('reserve_percent', 20)):.0f}% "
                 f"reserve. Run a mission to see what a real flight leaves.")
        if "battery_temp_est_C" in m:
            _temp_row(batt_status_tv, "Battery temperature (est)",
                      float(m["battery_temp_est_C"]), batt.temp_limit_C,
                      "Steady state in hover, the heaviest load.")

        if cfg.lift_motor_max_power_W:
            per = float(m.get("hover_lift_elec_W", m.get("hover_power_per_lift_motor_W", 0.0)))
            _row(motor_status_tv, "Lift motor power in hover", f"{per:.0f} W",
                 f"<= {cfg.lift_motor_max_power_W:.0f} W",
                 _classify(per, cfg.lift_motor_max_power_W),
                 "Electrical input per motor. Hover is the heaviest load a lift "
                 "motor carries.")
        else:
            _row(motor_status_tv, "Lift motor power in hover",
                 f"{float(m.get('hover_lift_elec_W', m.get('hover_power_per_lift_motor_W', 0.0))):.0f} W",
                 "Not Specified", "na", "Enter a motor max power to check it.")

        # Current against the motor's own rating — the check the rated-current
        # fields always promised and never made.
        _dual(motor_status_tv, "Lift motor current in hover",
              float(m.get("hover_lift_current_A", 0.0)),
              cfg.lift_motor_max_current_A, None, "A")
        th = float(m.get("hover_lift_throttle", float("nan")))
        if math.isfinite(th):
            _row(motor_status_tv, "Lift motor throttle in hover", f"{th * 100:.0f} %",
                 "<= 100 %", "bad" if th > 1.0 else ("warn" if th > 0.85 else "ok"),
                 "Terminal voltage over pack voltage. Above about 85% there is "
                 "little left for control; above 100% the motor cannot reach "
                 "the RPM at all — a lower Kv or more cells is needed.")

        if not vectored:
            per = float(m.get("cruise_motor_elec_W", m.get("cruise_power_per_motor_W", 0.0)))
            if cfg.cruise_motor_max_power_W:
                _row(motor_status_tv, "Cruise motor power", f"{per:.0f} W",
                     f"<= {cfg.cruise_motor_max_power_W:.0f} W",
                     _classify(per, cfg.cruise_motor_max_power_W))
            else:
                _row(motor_status_tv, "Cruise motor power", f"{per:.0f} W",
                     "Not Specified", "na")
            _dual(motor_status_tv, "Cruise motor current",
                  float(m.get("cruise_motor_current_A", 0.0)),
                  cfg.cruise_motor_max_current_A, None, "A")
        th = float(m.get("cruise_motor_throttle", float("nan")))
        if math.isfinite(th) and float(m.get("cruise_motor_rpm", 0.0)) > 0:
            _row(motor_status_tv, "Cruise throttle", f"{th * 100:.0f} %", "<= 100 %",
                 "bad" if th > 1.0 else ("warn" if th > 0.9 else "ok"),
                 "At the cruise speed. Above 100% the propeller would have to "
                 "turn faster than the pack can drive it.")

        # Each ESC carries its motor's current. Hover loads the lift ESCs, and
        # cruise the cruise ESCs.
        esc_hover = float(m.get("hover_lift_current_A", 0.0))
        esc_cruise = float(m.get("cruise_motor_current_A", 0.0))
        _dual(motor_status_tv, "ESC current in hover", esc_hover,
              cfg.esc_cont_current_A, cfg.esc_max_current_A, "A")
        _dual(motor_status_tv, "ESC current in cruise", esc_cruise,
              cfg.esc_cont_current_A, cfg.esc_max_current_A, "A")
        _rating_row(motor_status_tv, "ESC voltage rating", cfg.esc_v_unit,
                    cfg.esc_rating_min, cfg.esc_rating_max)
        _rating_row(motor_status_tv, "Lift motor voltage rating", cfg.lift_motor_v_unit,
                    cfg.lift_motor_rating_min, cfg.lift_motor_rating_max)
        if not vectored:
            _rating_row(motor_status_tv, "Cruise motor voltage rating", cfg.cruise_motor_v_unit,
                        cfg.cruise_motor_rating_min, cfg.cruise_motor_rating_max)
        if "lift_motor_temp_C" in m:
            _temp_row(motor_status_tv, "Lift motor temperature (est)",
                      float(m["lift_motor_temp_C"]), cfg.lift_motor_temp_limit_C,
                      "Steady state in hover.")
            if not vectored:
                _temp_row(motor_status_tv, "Cruise motor temperature (est)",
                          float(m["cruise_motor_temp_C"]), cfg.cruise_motor_temp_limit_C,
                          "Steady state at the cruise speed.")
            _temp_row(motor_status_tv, "ESC temperature (est)",
                      float(m["esc_temp_est_C"]), cfg.esc_temp_limit_C)

        if "lift_twr" in m and float(m.get("lift_thrust_available_N", 0.0)) > 0:
            twr = float(m["lift_twr"])
            _row(motor_status_tv, "Thrust-to-weight (max available)", f"{twr:.2f}:1",
                 ">= 2.0:1", "ok" if twr >= 2.0 else ("warn" if twr >= 1.5 else "bad"),
                 f"Lift rotors at full thrust ({m['lift_thrust_source']}) over "
                 f"weight. Below 1.5 hover control is dangerously marginal.")
            pay = float(m["max_extra_payload_g"])
            _row(motor_status_tv, "Max additional payload", f"{pay:.0f} g", ">= 0 g",
                 "ok" if pay >= 0 else "bad",
                 "Extra mass liftable with no hover margin at all.")
            pay2 = float(m["payload_at_twr2_g"])
            _row(motor_status_tv, "Payload at TWR 2.0", f"{max(pay2, 0):.0f} g", ">= 0 g",
                 "ok" if pay2 > 0 else "warn",
                 "Extra mass while keeping a 2:1 margin — the usable figure.")
        hover_need = cfg.weight_N * (1.0 + hover_download_fraction(cfg))
        _row(motor_status_tv, "Thrust required / weight (hover)",
             f"{hover_need / max(cfg.weight_N, 1e-9):.2f}:1", "~1.0:1", "ok",
             "Above 1 by the hover download the rotors also have to lift.")

        esc_loss = float(m.get("esc_loss_W", 0.0)) + float(m.get("motor_loss_W", 0.0))
        if esc_loss > 0:
            _row(motor_status_tv, "ESC + motor loss at cruise",
                 f"{esc_loss:.1f} W", "—", "na",
                 f"{esc_loss / max(float(m.get('total_power_W', 0.0)), 1e-9) * 100:.1f}% "
                 f"of pack power: {float(m.get('motor_loss_W', 0.0)):.1f} W in the "
                 f"motors, {float(m.get('esc_loss_W', 0.0)):.1f} W in the ESCs.")

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

        if "cl_cruise" in m:
            cl = float(m["cl_cruise"])
            _row(aero_status_tv, "CL at cruise vs CL_max", f"{cl:.3f}",
                 f"<= {cfg.CL_max * 0.8:.2f} (80% CL_max)",
                 "ok" if cl <= cfg.CL_max * 0.8 else ("warn" if cl <= cfg.CL_max else "bad"),
                 "Flying above about 80% of CL_max leaves little margin for a "
                 "gust or a turn.")
            sos = float(m["speed_over_stall"])
            _row(aero_status_tv, "Cruise vs stall speed", f"{sos:.2f}x", ">= 1.30x",
                 "ok" if sos >= 1.3 else ("warn" if sos >= 1.1 else "bad"))
            _row(aero_status_tv, "L/D ratio (cruise)", f"{float(m['ld_cruise']):.2f}",
                 "—", "na", f"Against {float(m['ld_max']):.1f} at best.")
            _row(aero_status_tv, "Glide ratio", f"{float(m['glide_ratio']):.1f} : 1", "—", "na",
                 "Rotors stopped, motors off.")
            roc = float(m["roc_at_cruise_mps"])
            best = float(m["max_roc_mps"])
            if cfg.min_climb_mps:
                _row(aero_status_tv, "Rate of climb", f"{best:.2f} m/s best",
                     f">= {cfg.min_climb_mps:.2f} m/s", _at_least(best, cfg.min_climb_mps),
                     f"Best wing-borne climb, at {float(m['vy_mps']):.1f} m/s; "
                     f"{roc:.2f} m/s at the cruise speed.")
            else:
                _row(aero_status_tv, "Rate of climb", f"{best:.2f} m/s best",
                     "Not Specified", "na",
                     "Enter a minimum climb rate on the Mission/Environment tab to "
                     "check it.")
            _row(aero_status_tv, "Max angle of climb",
                 f"{float(m['max_climb_angle_deg']):.1f} deg", "—", "na")
            ceiling = float(m["service_ceiling_m"])
            want = cfg.cruise_altitude_m
            if want is not None:
                _row(aero_status_tv, "Service ceiling",
                     "above 8000 m" if not math.isfinite(ceiling) else f"{ceiling:.0f} m",
                     f">= {want:.0f} m", "ok" if ceiling >= want else "bad",
                     "Wing-borne ceiling against the cruise altitude.")
            else:
                _row(aero_status_tv, "Service ceiling",
                     "above 8000 m" if not math.isfinite(ceiling) else f"{ceiling:.0f} m",
                     "—", "na")
            for label, key, avail in (("Take-off ground roll", "takeoff_roll_m", cfg.field_takeoff_m),
                                      ("Landing distance (over 15 m obstacle)",
                                       "landing_distance_m", cfg.field_landing_m)):
                dist = float(m[key])
                text = "not possible" if not math.isfinite(dist) else f"{dist:.0f} m"
                if avail:
                    _row(aero_status_tv, label, text, f"<= {avail:.0f} m",
                         _classify(dist, avail) if math.isfinite(dist) else "bad",
                         "Conventional, on the runway entered on the "
                         "Mission/Environment tab.")
                else:
                    _row(aero_status_tv, label, text, "Not Specified", "na",
                         "Only matters if the aircraft is flown off a runway; "
                         "enter the runway length to check it.")
            re_n = float(m["reynolds_number"])
            _row(aero_status_tv, "Reynolds number", f"{re_n:,.0f}", ">= 100,000",
                 "ok" if re_n >= 1e5 else ("warn" if re_n >= 5e4 else "bad"),
                 "Below about 100,000 the airfoil's lift and drag degrade "
                 "noticeably from their published values.")
            _row(aero_status_tv, "Wing loading", f"{cfg.wing_loading_N_m2:.1f} N/m²",
                 "—", "na")
            _row(aero_status_tv, "Specific range",
                 f"{float(m['specific_range_km_per_Wh']):.3f} km/Wh", "—", "na")
            if cfg.bank_deg > 0:
                v_ts = float(m["turn_stall_speed_mps"])
                _row(aero_status_tv, "Turn stall speed",
                     f"{v_ts:.1f} m/s at {cfg.bank_deg:.0f} deg", f"< {v_cruise:.1f} m/s",
                     "ok" if v_ts < v_cruise * 0.9 else ("warn" if v_ts < v_cruise else "bad"),
                     "The stall speed rises with the square root of the load "
                     "factor in a turn.")

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
        if cfg.lift_prop_table is not None:
            lo = float(cfg.lift_prop_table["Thrust_g"].min())
            hi = float(cfg.lift_prop_table["Thrust_g"].max())
            grams = t_per / G0 * 1000.0
            _row(rotor_status_tv, "Table thrust range",
                 f"{grams:.0f} g in {lo:.0f}-{hi:.0f} g", "inside the table",
                 "ok" if lo <= grams <= hi else "warn",
                 "Outside the measured range the table says nothing and the "
                 "estimate is used instead.")

        if "hover_lift_rpm" in m:
            rpm = float(m["hover_lift_rpm"])
            if cfg.lift_prop_max_rpm:
                _row(rotor_status_tv, "Lift prop RPM (hover)", f"{rpm:.0f} rpm",
                     f"<= {cfg.lift_prop_max_rpm:.0f} rpm",
                     _classify(rpm, cfg.lift_prop_max_rpm),
                     "Against the propeller maker's limit.")
            else:
                _row(rotor_status_tv, "Lift prop RPM (hover)", f"{rpm:.0f} rpm",
                     "Not Specified", "na", "Enter the prop's max RPM to check it.")
            mach = float(m["hover_lift_tip_mach"])
            _row(rotor_status_tv, "Lift rotor tip Mach", f"{mach:.3f}", "<= 0.60",
                 _classify(mach, 0.60),
                 f"Hover. Above about Mach 0.6 the tip goes transonic: noise "
                 f"rises sharply and efficiency falls. Speed of sound is "
                 f"{float(m['speed_of_sound_mps']):.0f} m/s here.")
            if not vectored and float(m.get("cruise_motor_rpm", 0.0)) > 0:
                c_rpm = float(m["cruise_motor_rpm"])
                if cfg.cruise_prop_max_rpm:
                    _row(rotor_status_tv, "Cruise prop RPM", f"{c_rpm:.0f} rpm",
                         f"<= {cfg.cruise_prop_max_rpm:.0f} rpm",
                         _classify(c_rpm, cfg.cruise_prop_max_rpm))
                c_mach = float(m["cruise_motor_tip_mach"])
                _row(rotor_status_tv, "Cruise prop tip Mach", f"{c_mach:.3f}", "<= 0.60",
                     _classify(c_mach, 0.60),
                     "Helical tip speed excludes the forward speed; add it for "
                     "a fast aircraft.")
                pitch_v = float(m["cruise_motor_pitch_speed_mps"])
                _row(rotor_status_tv, "Pitch speed vs cruise",
                     f"{pitch_v:.1f} m/s vs {v_cruise:.1f} m/s", ">= 1.15x cruise",
                     "ok" if pitch_v >= v_cruise * 1.15 else
                     ("warn" if pitch_v >= v_cruise else "bad"),
                     "The propeller's pitch speed must clear the cruise speed "
                     "or it runs out of thrust. More pitch or more RPM.")
            dl = float(m["disc_loading_N_m2"])
            ideal_gW = 1000.0 / (G0 * math.sqrt(dl / (2.0 * cfg.air_density)))
            _row(rotor_status_tv, "Disk loading", f"{dl:.1f} N/m²", "Not Specified", "na",
                 f"A design choice — high disk loading buys compactness at the "
                 f"cost of hover efficiency. At this loading the ideal ceiling "
                 f"is {ideal_gW:.1f} g/W.")
            he = float(m["hover_efficiency_gW"])
            frac = he / max(ideal_gW, 1e-9)
            _row(rotor_status_tv, "Hover efficiency", f"{he:.2f} g/W",
                 f">= {0.55 * ideal_gW:.1f} g/W",
                 "ok" if frac >= 0.55 else ("warn" if frac >= 0.40 else "bad"),
                 f"{frac * 100:.0f}% of the {ideal_gW:.1f} g/W ideal for this disk "
                 f"loading — rotor and drivetrain losses together.")
            d_in = cfg.lift_prop_diameter_in
            fm_target = 0.70 if d_in >= 15 else (0.60 if d_in >= 9 else 0.45)
            fm = float(m["hover_figure_of_merit"])
            _row(rotor_status_tv, "Figure of merit", f"{fm:.3f}", f">= {fm_target:.2f}",
                 _at_least(fm, fm_target),
                 f"Achieved, penalties included. Expectation scaled for a "
                 f"{d_in:.0f} in rotor, as the multicopter scales it.")
            _row(rotor_status_tv, "Prop solidity σ", f"{float(m['lift_solidity']):.3f}",
                 "—", "na", "Blade area over disc area, estimated from diameter "
                 "and blade count.")
            limit = float(m["hover_wind_limit_mps"])
            if math.isfinite(limit):
                _row(rotor_status_tv, "Max hover wind resistance",
                     f"{limit:.1f} m/s  ({limit * 1.944:.1f} kt)", ">= 5 m/s",
                     "ok" if limit >= 8 else ("warn" if limit >= 4 else "bad"),
                     "Strongest wind the lift rotors can hold station in at the "
                     "tilt limit, wing ignored.")
            wind_now = float(m.get("wind_mps", 0.0))
            if wind_now > 0 and float(m.get("hover_wind_drag_N", 0.0)) > 0:
                tilt = float(m["hover_tilt_deg"])
                lim_pitch = cfg.max_pitch_deg or cfg.max_tilt_deg
                lim_roll = cfg.max_roll_deg or cfg.max_tilt_deg
                ok = (tilt <= cfg.max_tilt_deg and float(m["hover_pitch_deg"]) <= lim_pitch
                      and float(m["hover_roll_deg"]) <= lim_roll)
                _row(rotor_status_tv, "Hover tilt in this wind",
                     f"{tilt:.1f} deg (pitch {float(m['hover_pitch_deg']):.1f}, "
                     f"roll {float(m['hover_roll_deg']):.1f})",
                     f"<= {cfg.max_tilt_deg:.0f} deg", "ok" if ok else "bad",
                     "Tilt needed to hold station against the airframe's side "
                     "drag, against the tilt, pitch and roll limits.")
            if not vectored:
                margin = float(m["thrust_margin_pct"])
                _row(rotor_status_tv, "Cruise thrust margin", _f(margin, "{:.0f} %"),
                     ">= 30 %", _at_least(margin, 30.0, 0.5) if math.isfinite(margin) else "na",
                     "Forward thrust in hand at the cruise speed, for climbing "
                     "and accelerating.")

        # Wiring and connectors: the shared rows all three simulators show,
        # checked at hover, the largest steady current a VTOL draws. The
        # battery connector carries the whole pack current; each ESC and
        # motor connector one lift rotor's share.
        _wiring = getattr(cfg, "wiring", None)
        for _group, _name, _val, _lim, _tag, _note in core.wiring_status_rows(
                _wiring, hover_I, hover_I / max(cfg.num_lift_rotors, 1),
                float(batt.vmax_pack), float(cfg.ambient_temp_C), where="hover"):
            _row(batt_status_tv if _group == "battery" else motor_status_tv,
                 _name, _val, _lim, _tag, _note)

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
        # The multicopter's and fixed-wing's mission variables.
        ("commanded_airspeed_mps", "Commanded airspeed (m/s)"),
        ("groundspeed_mps", "Groundspeed (m/s)"),
        ("headwind_mps", "Headwind (m/s)"),
        ("crosswind_mps", "Crosswind (m/s)"),
        ("accel_mps2", "Acceleration (m/s²)"),
        ("kinetic_power_W", "Kinetic power (W)"),
        ("potential_power_W", "Climb (potential) power (W)"),
        ("battery_voltage_V", "Battery voltage, loaded (V)"),
        ("battery_loss_W", "Battery I²R loss (W)"),
        ("esc_loss_W", "ESC loss (W)"),
        ("motor_loss_W", "Motor loss (W)"),
        ("systems_power_W", "Avionics + peripherals (W)"),
        ("lift_motor_current_A", "Lift motor current (A)"),
        ("lift_motor_rpm", "Lift motor RPM"),
        ("lift_motor_throttle", "Lift motor throttle (—)"),
        ("lift_motor_power_W", "Lift motor power, each (W)"),
        ("lift_thrust_per_rotor_N", "Thrust per lift rotor (N)"),
        ("cruise_motor_current_A", "Cruise motor current (A)"),
        ("cruise_motor_rpm", "Cruise motor RPM"),
        ("cruise_motor_throttle", "Cruise motor throttle (—)"),
        ("cruise_motor_power_W", "Cruise motor power, each (W)"),
        ("lift_tip_mach", "Lift rotor tip Mach (—)"),
        ("advance_ratio_mu", "Lift rotor advance ratio μ (—)"),
        ("cl_wing", "Wing CL (—)"),
        ("lift_drag_ratio", "L/D (—)"),
        ("motor_temp_est_C", "Motor temperature (°C)"),
        ("esc_temp_est_C", "ESC temperature (°C)"),
        ("battery_temp_est_C", "Battery temperature (°C)"),
        ("reserve_target_Wh", "Reserve target (Wh)"),
        ("reserve_breach", "Reserve breached (1/0)"),
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
        chart.append(("Airframe", structure))
        wb_tv.insert("", "end", tags=(("bad",) if structure < 0 else ()),
                     values=("Airframe", "", "", f"{structure:.0f}",
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
                  f"{cfg.aircraft_weight_g:.0f} g all-up weight without payload. "
                  f"That leaves a negative airframe mass, which is impossible: "
                  f"raise the weight or correct a part."
                  if structure < 0 else
                  "The all-up weight without payload includes the battery and "
                  "motors; the airframe is what remains after the itemised "
                  "parts."))

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
            motor_copper_W=float(m.get("motor_copper_W", 0.0)),
            motor_iron_W=float(m.get("motor_iron_W", 0.0)),
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
            f"({m.get('regime', '')}). Motor losses come from the Kv / Rm / I0 "
            f"model; rotor aerodynamic losses are inside the shaft power, through "
            f"the figure of merit and propeller efficiency. A motor covered by a "
            f"bench table has its loss inside the measured power instead."))

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
    # The multicopter's and fixed-wing's outputs join the VTOL's own.
    _SENS_POINT = ["Cruise power (W)", "Cruise endurance (min)", "Cruise range (km)",
                   "Hover power (W)", "Hover endurance (min)", "Transition speed (m/s)",
                   "Stall speed (m/s)", "Pack current (A)", "Hover pack current (A)",
                   "Hover efficiency (g/W)", "Lift motor current (A)",
                   "Motor temperature (°C)", "L/D at cruise", "Best climb rate (m/s)",
                   "Take-off roll (m)"]
    _SENS_MISSION = ["Mission energy (Wh)", "Mission time (min)",
                     "Mission distance (km)", "Reserve margin (Wh)", "Peak power (W)",
                     "Minimum SoC (%)", "Peak pack current (A)", "Peak motor temp (°C)"]
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
        # The multicopter's and fixed-wing's levers.
        ("Lift motor Kv", lambda c, f: setattr(c, "lift_motor_kv", c.lift_motor_kv * f)),
        ("Lift motor resistance", lambda c, f: setattr(
            c, "lift_motor_resistance", c.lift_motor_resistance * f)),
        ("Lift prop pitch", lambda c, f: setattr(c, "lift_prop_pitch_in", c.lift_prop_pitch_in * f)),
        ("Cruise motor Kv", lambda c, f: setattr(c, "cruise_motor_kv", c.cruise_motor_kv * f)),
        ("Cruise motor resistance", lambda c, f: setattr(
            c, "cruise_motor_resistance", c.cruise_motor_resistance * f)),
        ("Cruise prop diameter", lambda c, f: setattr(
            c, "cruise_prop_diameter_in", c.cruise_prop_diameter_in * f)),
        ("Cruise prop pitch", lambda c, f: setattr(
            c, "cruise_prop_pitch_in", c.cruise_prop_pitch_in * f)),
        ("Motor no-load current", lambda c, f: (
            setattr(c, "lift_motor_i0_A", c.lift_motor_i0_A * f),
            setattr(c, "cruise_motor_i0_A", c.cruise_motor_i0_A * f))),
        ("Stopped-rotor drag area", lambda c, f: setattr(
            c, "_stopped_rotor_drag_area_m2", c.stopped_rotor_drag_area_m2 * f)),
        ("Wing span", lambda c, f: setattr(c, "wing_span_m", c.wing_span_m * f)),
        ("CL_max", lambda c, f: setattr(c, "CL_max", c.CL_max * f)),
        ("CL cap in transition", lambda c, f: setattr(
            c, "CL_cruise_max", min(c.CL_cruise_max * f, c.CL_max))),
        ("Air density", lambda c, f: setattr(c, "air_density", c.air_density * f)),
        ("Battery resistance", lambda c, f: setattr(
            c.battery, "resistance_cell", c.battery.resistance_cell * f)),
        ("ESC efficiency", lambda c, f: setattr(
            c, "esc_efficiency", min(c.esc_efficiency * f, 1.0))),
        ("Peripheral current", lambda c, f: setattr(
            c, "periph_current_A", c.periph_current_A * f)),
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
                        max_decel_mps2=_sens_state.get("decel", 0.0),
                        regen_eff=_sens_state.get("regen", 0.0))
                    worst = tot["worst"]
                    return {"Mission energy (Wh)": tot["energy_Wh"],
                            "Mission time (min)": tot["time_s"] / 60.0,
                            "Mission distance (km)": tot["distance_m"] / 1000.0,
                            "Reserve margin (Wh)": worst["reserve_margin_Wh"],
                            "Peak power (W)": worst["total_power_W"],
                            "Minimum SoC (%)": worst["min_soc_pct"],
                            "Peak pack current (A)": worst["pack_current_A"],
                            "Peak motor temp (°C)": worst["motor_temp_est_C"]}[choice]
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
                        "Transition speed (m/s)": mm["transition_speed_mps"],
                        "Stall speed (m/s)": mm["stall_speed_mps"],
                        "Pack current (A)": mm["pack_current_A"],
                        "Hover pack current (A)": mm["hover_pack_current_A"],
                        "Hover efficiency (g/W)": mm["hover_efficiency_gW"],
                        "Lift motor current (A)": mm["hover_lift_current_A"],
                        "Motor temperature (°C)": mm["motor_temp_est_C"],
                        "L/D at cruise": mm["ld_cruise"],
                        "Best climb rate (m/s)": mm["max_roc_mps"],
                        "Take-off roll (m)": mm["takeoff_roll_m"]}[choice]
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
                  ("all_up_weight_g", "All-up weight (g)", 0),
                  # The multicopter's and fixed-wing's Compare rows.
                  ("v_load_V", "Loaded voltage (V)", 2),
                  ("rotor_thrust_N", "Rotor thrust at cruise (N)", 2),
                  ("hover_efficiency_gW", "Hover efficiency (g/W)", 2),
                  ("hover_figure_of_merit", "Figure of merit (achieved)", 3),
                  ("disc_loading_N_m2", "Disc loading (N/m²)", 1),
                  ("wing_loading_N_m2", "Wing loading (N/m²)", 1),
                  ("lift_twr", "Lift thrust-to-weight", 2),
                  ("ld_cruise", "L/D at cruise", 2),
                  ("cl_cruise", "CL at cruise", 3),
                  ("roc_at_cruise_mps", "Rate of climb at cruise (m/s)", 2),
                  ("max_roc_mps", "Best climb rate (m/s)", 2),
                  ("glide_ratio", "Glide ratio", 2),
                  ("best_endurance_speed_mps", "Best endurance speed (m/s)", 2),
                  ("best_range_speed_mps", "Best range speed (m/s)", 2),
                  ("takeoff_roll_m", "Take-off roll (m)", 1),
                  ("reynolds_number", "Reynolds number", 0),
                  ("groundspeed_mps", "Groundspeed (m/s)", 2),
                  ("reserve_margin_Wh", "Reserve margin (Wh)", 2),
                  ("forward_thrust_available_N", "Forward thrust available (N)", 2),
                  ("hover_lift_current_A", "Lift motor current, hover (A)", 2),
                  ("cruise_motor_current_A", "Cruise motor current (A)", 2),
                  ("hover_lift_rpm", "Lift RPM, hover", 0),
                  ("hover_lift_tip_mach", "Lift tip Mach, hover", 3),
                  ("motor_loss_W", "Motor losses (W)", 2),
                  ("battery_loss_W", "Pack I2R loss (W)", 2),
                  ("motor_temp_est_C", "Motor temp (°C)", 1),
                  ("esc_temp_est_C", "ESC temp (°C)", 1),
                  ("battery_temp_est_C", "Battery temp (°C)", 1)]
    _CMP_MISSION = [("energy_Wh", "Mission energy (Wh)", 2),
                    ("time_min", "Mission time (min)", 2),
                    ("distance_km", "Distance (km)", 3),
                    ("hover_Wh", "Hover energy (Wh)", 2),
                    ("transition_Wh", "Transition energy (Wh)", 2),
                    ("cruise_Wh", "Cruise energy (Wh)", 2),
                    ("remaining_Wh", "Energy remaining (Wh)", 2),
                    ("peak_power_W", "Peak power (W)", 1),
                    ("peak_current_A", "Peak pack current (A)", 2),
                    ("reserve_margin_Wh", "Lowest reserve margin (Wh)", 2),
                    ("min_soc_pct", "Minimum SoC (%)", 1),
                    ("peak_lift_current_A", "Peak lift motor current (A)", 2),
                    ("peak_motor_temp_C", "Peak motor temp (°C)", 1),
                    ("peak_esc_temp_C", "Peak ESC temp (°C)", 1),
                    ("peak_battery_temp_C", "Peak battery temp (°C)", 1)]

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
        # Held in the run's wind: with a drag height above the CG entered,
        # the rotors downwind carry more than those upwind.
        thrusts = hover_rotor_thrusts(cfg, float(m.get("wind_mps", 0.0) or 0.0),
                                      float(m.get("wind_direction_deg", 0.0) or 0.0),
                                      float(m.get("course_deg", 0.0) or 0.0))
        total_N = sum(thrusts)
        rated_N = (cfg.lift_prop_max_thrust_g / 1000.0 * 9.80665
                   if getattr(cfg, "lift_prop_max_thrust_g", 0) else None)
        for i in range(1, n + 1):
            per_N = thrusts[i - 1] if i - 1 < len(thrusts) else total_N / n
            if rated_N:
                margin = (rated_N - per_N) / rated_N * 100.0
                tag = ("bad" if margin < 0 else "warn" if margin < 20 else "ok")
                margin_text = f"{margin:+.0f}%"
            else:
                tag, margin_text = "na", "no rated thrust entered"
            rl_tv.insert("", "end", tags=(tag,), values=(
                f"{i}", f"{per_N:.1f}", f"{per_N / 9.80665 * 1000:.0f}",
                f"{per_N / max(total_N, 1e-9) * 100.0:.1f}%", margin_text))
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
        """
        The speed sweep behind the Fixed Speed Plots, as numbers — built from
        the same data the panels draw, so each column is a curve on screen.
        """
        d = _sweep_data(cfg)
        cols = [("Airspeed (m/s)", "speed", 2), ("Total power (W)", "total", 1),
                ("Shaft power (W)", "shaft", 1), ("Lift rotor shaft (W)", "rotor_W", 1),
                ("Cruise prop shaft (W)", "cruise_W", 1), ("Endurance (min)", "endurance", 2),
                ("Range (km)", "range", 3), ("Regime", "regime", None),
                ("Tilt (deg)", "tilt", 1), ("Wing lift share", "share", 3),
                ("Rotor thrust (N)", "rotor_N", 2), ("Wing lift (N)", "wing_N", 2),
                ("Thrust required (N)", "thrust_req", 2),
                ("Thrust available (N)", "thrust_avail", 2),
                ("Rate of climb (m/s)", "roc", 2), ("Drag induced (N)", "drag_induced", 2),
                ("Drag CD0 (N)", "drag_cd0", 2), ("Drag stopped rotors (N)", "drag_stopped", 2),
                ("Drag fuselage/booms (N)", "drag_body", 2), ("Drag total (N)", "drag_total", 2)]
        rows = []
        for i in range(len(d["speed"])):
            row = []
            for _h, key, dec in cols:
                value = d[key][i]
                if dec is None:
                    row.append(value)
                elif value != value:
                    row.append("")
                else:
                    row.append(round(float(value), dec))
            rows.append(row)
        return ("Speed Sweep", [h for h, _k, _d in cols], rows)

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

            makers = [lambda: make_airframe_diagram_figure(cfg, figsize=(7, 5.8))]
            if not _export_state["from_mission"] and _export_state.get("metrics"):
                makers.append(lambda: make_motor_figure(
                    cfg, _export_state["metrics"], (10, 4.2 * (1 if uses_vectored_thrust(cfg) else 2))))
            for maker in makers:
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
            draw_plots(cfg, _export_state.get("metrics"))
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
            "a Kv / Rm / I0 motor model (RPM, current, throttle, losses), "
            "pack sag from a state-of-charge curve, ESC and wiring losses, "
            "climb, ceiling, turns, glide and runway figures from the "
            "fixed-wing, lumped thermal estimates, and time-stepped "
            "missions.\n\n"
            "NOT modelled: rotor-to-rotor and rotor-to-wing interference "
            "(beyond the coaxial penalty), blade-element aerodynamics, "
            "control-system behaviour or structural loads. Temperatures are "
            "lumped estimates, not a thermal model of any real part.\n\n"
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

    def field_values() -> dict:
        """Every input field's current text, keyed by field name."""
        return {key: var.get() for key, var in fields.items()}

    def build_config() -> VTOLConfig:
        """
        The aircraft on screen. Built by the same function the CLI and the
        batch driver use, so a config saved here gives the same answer there.
        """
        return config_from_fields(field_values(), v_config_type.get())

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

    def _f(x, fmt="{:.2f}", na="n/a"):
        """Format a number, or say n/a for a missing or non-finite one."""
        try:
            x = float(x)
        except (TypeError, ValueError):
            return na
        if not math.isfinite(x):
            return na
        return fmt.format(x)

    def _km(km: float) -> str:
        return f"{km:.2f} km  ({km * 0.539957:.2f} NM / {km * 0.621371:.2f} mi)"

    def _motor_section(title, prefix, m, cfg, group, label):
        """
        One motor-and-propeller operating point, the rows the multicopter's
        "Motor @ Operating Point" and the fixed-wing's motor rows carry.
        """
        g = _group(cfg, group)
        _metrics_add_section(title)
        rpm = float(m.get(f"{prefix}_rpm", 0.0))
        if rpm <= 0:
            _metrics_row(f"{label} state", "stopped",
                         "Not turning at this operating point.")
            return
        _metrics_row(f"{label} Kv", f"{_f(g['kv'], '{:.0f}')} rpm/V")
        _metrics_row(f"{label} Kt", f"{_f(m.get(f'{group}_kt'), '{:.4f}')} Nm/A",
                     "Torque per amp, 60 / (2 pi Kv).")
        _metrics_row(f"{label} no-load current", f"{_f(g['i0'])} A")
        _metrics_row(f"{label} resistance", f"{_f(float(g['rm'] or 0) * 1000, '{:.1f}')} mΩ")
        _metrics_row(f"{label} RPM", f"{rpm:.0f} rpm  ({_f(m.get(f'{prefix}_erpm'), '{:.0f}')} eRPM)",
                     "Mechanical RPM, and the electrical RPM the ESC commutates "
                     "(RPM x poles / 2).")
        _metrics_row(f"{label} current", f"{_f(m.get(f'{prefix}_current_A'))} A",
                     "Winding current: torque over Kt, plus the no-load current.")
        _metrics_row(f"{label} back-EMF", f"{_f(m.get(f'{prefix}_v_emf_V'))} V")
        _metrics_row(f"{label} terminal voltage", f"{_f(m.get(f'{prefix}_v_term_V'))} V",
                     "Back-EMF plus the I x Rm drop — what the ESC must supply.")
        throttle = float(m.get(f"{prefix}_throttle", float("nan")))
        _metrics_row(f"{label} throttle",
                     _f(throttle * 100.0, "{:.0f} %")
                     + ("  ⚠ above 100% — the pack cannot spin it this fast"
                        if m.get(f"{prefix}_saturated") else ""),
                     "Terminal voltage over pack voltage. Above 100% the motor "
                     "cannot reach the RPM this thrust needs: a lower-Kv motor, "
                     "more pitch or more cells is required.")
        _metrics_row(f"{label} electrical power", f"{_f(m.get(f'{prefix}_elec_W'), '{:.1f}')} W")
        _metrics_row(f"{label} shaft power", f"{_f(m.get(f'{prefix}_shaft_W'), '{:.1f}')} W")
        _metrics_row(f"{label} efficiency", _f(float(m.get(f"{prefix}_motor_eff", 1.0)) * 100, "{:.1f} %")
                     + ("  (inside the bench table)" if m.get(f"{prefix}_measured") else ""),
                     "Shaft power over electrical power.")
        _metrics_row(f"{label} copper loss", f"{_f(m.get(f'{prefix}_copper_W'))} W", "I² x Rm.")
        _metrics_row(f"{label} no-load loss", f"{_f(m.get(f'{prefix}_iron_W'))} W",
                     "No-load current times back-EMF: iron and bearing loss.")
        _metrics_row(f"{label} torque", f"{_f(m.get(f'{prefix}_torque_Nm'), '{:.3f}')} Nm")
        thrust = float(m.get(f"{prefix}_thrust_N", 0.0))
        _metrics_row(f"{label} thrust", f"{thrust / G0 * 1000:.0f} g  ({thrust:.2f} N)")
        _metrics_row(f"{label} thrust per watt",
                     f"{_f(m.get(f'{prefix}_thrust_per_W_g'))} g/W")
        _metrics_row(f"{label} rated current",
                     "not entered" if not g["imax"] else f"{float(g['imax']):.0f} A")
        _metrics_row(f"{label} rated power",
                     "not entered" if not g["pmax"] else f"{float(g['pmax']):.0f} W")
        size = cfg.lift_motor_size if group == "lift" else cfg.cruise_motor_size
        _metrics_row(f"{label} poles / size", f"{g['poles']} poles" + (f", {size}" if size else ""))

    def show_metrics(m, cfg=None):
        for item in metrics_tv.get_children():
            metrics_tv.delete(item)
        vectored = cfg is not None and uses_vectored_thrust(cfg)

        _metrics_add_section("Aircraft")
        _metrics_row("Configuration", str(m["config_type"]))
        _metrics_row("Regime at cruise speed", str(m["regime"]))
        _metrics_row("All-up weight",
                     f"{_mass(m['all_up_weight_g'])}  ({m['weight_N']:.1f} N)")
        _metrics_row("Payload", f"{_mass(m['payload_mass_g'])}  "
                     f"({_f(m.get('payload_fraction', 0) * 100, '{:.1f}')}% of AUW)")
        _metrics_row("Battery mass fraction", _f(m.get("battery_mass_fraction", 0) * 100, "{:.1f} %"),
                     "Pack mass over all-up weight. A VTOL carries its hover "
                     "motors all the way through cruise, so it cannot spend as "
                     "much of its weight on battery as a pure fixed-wing.")
        _metrics_row("Drive mass fraction", _f(m.get("drive_mass_fraction", 0) * 100, "{:.1f} %"),
                     "Motors, propellers and ESCs over all-up weight.")
        _metrics_row("Wing loading", f"{m['wing_loading_N_m2']:.1f} N/m²  "
                     f"({m['wing_loading_N_m2'] / G0 * 100:.1f} g/dm²)")
        _metrics_row("Disc loading (hover)", f"{m['disc_loading_N_m2']:.1f} N/m²")
        _metrics_row("Aspect ratio", f"{m['aspect_ratio']:.2f}")
        if cfg is not None:
            _metrics_row("Mean chord", f"{_f(m.get('mean_chord_m'), '{:.3f}')} m")
            _metrics_row("Induced drag factor k", f"{cfg.induced_drag_factor:.5f}",
                         "1 / (pi AR e). Multiplies CL² to give the induced drag coefficient.")
            _metrics_row("Lift rotor layout", cfg.lift_rotor_layout
                         + (f", {cfg.coaxial_spacing_m:.3f} m spacing"
                            if cfg.lift_rotor_layout == "coaxial" and cfg.coaxial_spacing_m else ""))

        _metrics_add_section("Speeds")
        _metrics_row("Stall speed", _speed(m["stall_speed_mps"]))
        _metrics_row("Transition speed", _speed(m["transition_speed_mps"]))
        _metrics_row("Cruise speed", _speed(m["airspeed_mps"]))
        if "speed_over_stall" in m:
            _metrics_row("Speed / stall margin", f"{m['speed_over_stall']:.2f} x Vs",
                         "Cruise speed over stall speed. Below about 1.3 a gust "
                         "or a turn can stall the wing.")
            _metrics_row("Best endurance speed",
                         f"{_speed(m['best_endurance_speed_mps'])}  → "
                         f"{m['best_endurance_min']:.1f} min",
                         "Where total power is lowest, searched from hover up. "
                         "It can sit in the transition, where the rotors are "
                         "still helping.")
            _metrics_row("Best range speed",
                         f"{_speed(m['best_range_speed_mps'])}  → {m['best_range_km']:.2f} km",
                         "Where distance per watt-hour is greatest, over the "
                         "ground in the entered wind.")
            _metrics_row("Min-sink glide speed",
                         f"{_speed(m['min_sink_speed_mps'])}  "
                         f"(sink {m['min_sink_rate_mps']:.2f} m/s)",
                         "Slowest descent with the motors off and the rotors stopped.")
            _metrics_row("Best glide speed (max L/D)", _speed(m["best_glide_speed_mps"]))
            _metrics_row("Cruise vs best endurance", f"{m['cruise_vs_best_endurance_pct']:+.0f} %")
            _metrics_row("Cruise vs best range", f"{m['cruise_vs_best_range_pct']:+.0f} %")

        _metrics_add_section("Hover")
        _metrics_row("Hover power", f"{m['hover_power_W']:.0f} W")
        _metrics_row("Hover endurance",
                     f"{m['hover_endurance_min']:.1f} min  "
                     f"({m['hover_endurance_min'] * 60:.0f} s)")
        if "hover_efficiency_gW" in m:
            _metrics_row("Hover endurance to reserve",
                         f"{m['hover_endurance_to_reserve_min']:.1f} min",
                         "Hover time before the reserve is reached.")
            _metrics_row("Hover efficiency", f"{m['hover_efficiency_gW']:.2f} g/W",
                         "Grams of thrust per watt of propulsion power, "
                         "avionics excluded — the multicopter's headline figure.")
            _metrics_row("Figure of merit (achieved)",
                         f"{m['hover_figure_of_merit']:.3f}",
                         "Ideal momentum power over the shaft power actually "
                         "needed, coaxial and inflow penalties included.")
            _metrics_row("Ideal hover power", f"{m['hover_ideal_power_W']:.0f} W",
                         "T x sqrt(T / 2 rho A): the least any rotor of this disc "
                         "area could need.")
            _metrics_row("Hover shaft power", f"{m['hover_shaft_W']:.0f} W")
            _metrics_row("Lift thrust available",
                         f"{m['lift_thrust_available_N'] / G0 * 1000:.0f} g  "
                         f"({m['lift_thrust_available_N']:.1f} N, {m['lift_thrust_source']})")
            _metrics_row("Lift thrust-to-weight", _f(m["lift_twr"], "{:.2f}"),
                         "Below about 1.5 there is little margin for gusts, "
                         "descent control or a motor out.")
            _metrics_row("Max extra payload", _f(m["max_extra_payload_g"], "{:.0f} g"),
                         "Mass the lift rotors could still raise at full thrust, "
                         "download included. Zero margin — not a flyable load.")
            _metrics_row("Payload at TWR 2.0", _f(m["payload_at_twr2_g"], "{:.0f} g"),
                         "Extra mass that keeps a 2:1 thrust margin, the usual "
                         "multirotor sizing rule.")
            _metrics_row("Hover wind limit",
                         _f(m["hover_wind_limit_mps"], "{:.1f} m/s",
                            "n/a — enter a profile area or body dimensions"),
                         "Strongest wind the lift rotors can hold station in at "
                         "the tilt limit. Ignores the wing's own lift, so it is "
                         "conservative.")

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
        if "specific_range_km_per_Wh" in m:
            _metrics_row("Cruise endurance to reserve",
                         f"{m['cruise_endurance_to_reserve_min']:.1f} min")
            _metrics_row("Specific range", f"{m['specific_range_km_per_Wh']:.3f} km/Wh")
            _metrics_row("Specific endurance", f"{m['specific_endurance_min_per_Wh']:.3f} min/Wh")
            _metrics_row("Commanded climb / descent",
                         f"{m['climb_rate_cmd_mps']:.2f} / {m['descent_rate_cmd_mps']:.2f} m/s  "
                         f"({m['climb_power_add_W']:+.0f} W)",
                         "A steady climb or descent at the cruise speed, entered "
                         "on the Mission/Environment tab. Its potential power "
                         "is already in the cruise power above.")

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
        if "hover_tilt_deg" in m:
            _metrics_row("Hover tilt in this wind",
                         f"{m['hover_tilt_deg']:.1f} deg  (pitch {m['hover_pitch_deg']:.1f}, "
                         f"roll {m['hover_roll_deg']:.1f})",
                         "Tilt the lift rotors need to hold station against the "
                         "airframe's side drag, split by where the wind comes "
                         "from. Zero until a profile area or body dimensions "
                         "are entered.")

        _metrics_add_section("Battery")
        # NOT "Configuration": _METRIC_NOTES is keyed by label, and the
        # Aircraft section already owns that one — a duplicate label silently
        # inherits the other section's explanation.
        _metrics_row("Pack configuration",
                     f"{m['battery_series_cells']}S x {m['battery_parallel_cells']}P  "
                     f"({m['battery_total_cells']} cells, entered by "
                     f"{m['battery_unit_mode']})")
        if "battery_chemistry" in m:
            _metrics_row("Chemistry", str(m["battery_chemistry"]))
        _metrics_row("Pack capacity",
                     f"{m['battery_capacity_mAh']:.0f} mAh  "
                     f"({m['battery_capacity_mAh'] / 1000.0:.2f} Ah)")
        _metrics_row("Pack current", f"{m['pack_current_A']:.2f} A")
        _metrics_row("Loaded voltage", f"{m['v_load_V']:.2f} V")
        if "pack_v_full_V" in m:
            _metrics_row("Pack voltage (full / nominal / cutoff)",
                         f"{m['pack_v_full_V']:.2f} / {m['pack_v_nominal_V']:.2f} / "
                         f"{m['pack_v_cutoff_V']:.2f} V")
            _metrics_row("Voltage sag at cruise",
                         f"{m['pack_sag_V']:.2f} V  "
                         f"({m['pack_sag_V'] / max(m['pack_v_full_V'], 1e-9) * 100:.1f}% of full)")
            _metrics_row("Pack resistance", f"{m['pack_resistance_ohm'] * 1000:.1f} mΩ")
            _metrics_row("Pack I²R loss", f"{m['battery_loss_W']:.2f} W at cruise, "
                         f"{m['hover_battery_loss_W']:.2f} W in hover")
            _metrics_row("Cruise C-rate", f"{m['c_rate']:.2f} C")
            _metrics_row("Hover C-rate (pack)", f"{m['hover_c_rate']:.2f} C")
            _metrics_row("Energy (total)", f"{m['capacity_Wh']:.1f} Wh")
            _metrics_row("Usable capacity", f"{m['usable_mAh']:.0f} mAh")
        _metrics_row("Usable energy", f"{m['usable_Wh']:.1f} Wh")
        if "reserve_target_Wh" in m:
            _metrics_row("Reserve target", f"{m['reserve_target_Wh']:.1f} Wh  "
                         f"({m['reserve_percent']:.0f}% of usable)")
            _metrics_row("Energy above reserve", f"{m['reserve_margin_Wh']:.1f} Wh")
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

        if "propulsive_efficiency" in m:
            _metrics_add_section("Thrust & Power")
            _metrics_row("Thrust required at cruise", f"{m['thrust_required_N']:.2f} N",
                         "Forward thrust the cruise propulsion must make — the drag.")
            _metrics_row("Forward thrust available",
                         f"{m['forward_thrust_available_N']:.2f} N at cruise, "
                         f"{m['forward_static_thrust_N']:.1f} N static  ({m['cruise_thrust_source']})")
            _metrics_row("Thrust margin at cruise", _f(m["thrust_margin_pct"], "{:.0f} %"))
            _metrics_row("Forward thrust-to-weight", _f(m["forward_twr"], "{:.2f}"),
                         "Static forward thrust over weight — what a conventional "
                         "take-off and a wing-borne climb have to work with.")
            _metrics_row("Propulsive power (T·V)", f"{m['propulsive_power_W']:.0f} W")
            _metrics_row("Shaft power at cruise", f"{m['shaft_power_W']:.0f} W")
            _metrics_row("Motor losses", f"{m['motor_loss_W']:.1f} W  "
                         f"(copper {m['motor_copper_W']:.1f}, no-load {m['motor_iron_W']:.1f})")
            _metrics_row("ESC losses", f"{m['esc_loss_W']:.1f} W",
                         "Under load, from the ESC efficiency, plus any idle draw.")
            _metrics_row("  ESC conduction / switching / idle",
                         f"{m['esc_conduction_W']:.1f} / {m['esc_switching_W']:.1f} / "
                         f"{m['esc_idle_W']:.1f} W",
                         "Conduction is I²R from the ESC resistance, if entered; "
                         "the rest of the under-load loss is switching.")
            _metrics_row("Wire loss", f"{m['wire_loss_W']:.1f} W")
            _wiring = getattr(cfg, "wiring", None)
            if _wiring is not None and _wiring.resistance_ohm > 0:
                _metrics_row("  Main lead resistance",
                             f"{_wiring.resistance_ohm * 1000:.1f} mΩ  (round trip, "
                             f"{_wiring.length_m:.2f} m one way)")
                _metrics_row("  Main lead voltage drop", f"{m.get('wire_drop_V', 0.0):.3f} V",
                             "I x R along the lead; the ESCs see the pack minus this.")
                _hover = _wiring.summary(float(m.get("hover_pack_current_A", 0.0)),
                                         cfg.ambient_temp_C)
                _t = float(_hover["temp_C"])
                _metrics_row("  Main lead temperature at hover (est)",
                             ("above 400" if not math.isfinite(_t) or _t > 400 else f"{_t:.0f}")
                             + f" °C  (limit {_wiring.temp_limit_C:.0f} °C)",
                             "Steady, still air — conservative.")
            _metrics_row("Drive efficiency", f"{m['drive_efficiency'] * 100:.1f} %",
                         "Shaft power over motor input — the motors' share of the losses.")
            _metrics_row("System efficiency", f"{m['system_efficiency'] * 100:.1f} %",
                         "Shaft power over pack power.")
            _metrics_row("Propulsive efficiency", f"{m['propulsive_efficiency'] * 100:.1f} %",
                         "Thrust power T·V over pack power: how much of the "
                         "battery's output becomes useful work against drag.")
            _metrics_row("Power loading", f"{m['power_loading_W_per_kg']:.0f} W/kg")

        _metrics_add_section("Aerodynamics")
        _metrics_row("Stopped-rotor drag area",
                     f"{m['stopped_rotor_drag_area_m2']*1e4:.0f} cm²")
        if "cl_cruise" in m:
            _metrics_row("Cruise CL", f"{m['cl_cruise']:.3f}")
            _metrics_row("CL margin to stall", f"{m['cl_margin']:.3f}")
            _metrics_row("Cruise CD", f"{m['cd_cruise']:.4f}  "
                         f"(CD0 {m['cd_parasite']:.4f} + induced {m['cd_induced']:.4f})")
            _metrics_row("Induced / parasitic ratio", f"{m['induced_parasite_ratio']:.2f}",
                         "1.0 at the best-L/D speed; above 1 you are flying slow "
                         "for your weight, below it fast.")
            _metrics_row("Induced drag", f"{m['drag_induced_N']:.2f} N")
            _metrics_row("Parasitic drag (CD0)", f"{m['drag_parasite_N']:.2f} N")
            _metrics_row("Stopped-rotor drag", f"{m['drag_stopped_rotor_N']:.2f} N")
            _metrics_row("Fuselage / boom drag", f"{m['drag_body_N']:.2f} N  "
                         f"({m['extra_drag_source']})",
                         "Drag beyond CD0 from the Extra Airframe Drag inputs.")
            _metrics_row("Total drag", f"{m['drag_total_N']:.2f} N")
            _metrics_row("L/D at cruise", f"{m['ld_cruise']:.2f}")
            _metrics_row("Max L/D (analytic)", f"{m['ld_max']:.2f}",
                         "0.5 sqrt(pi AR e / CD0): the wing and CD0 alone.")
            _metrics_row("Angle of attack", f"{m['aoa_deg']:.1f} deg",
                         "CL over the finite-wing lift slope 2 pi AR / (AR + 2), "
                         "measured from zero lift.")
            _metrics_row("Reynolds number", f"{m['reynolds_number']:,.0f}")

            _metrics_add_section("Climb & Glide")
            _metrics_row("Rate of climb at cruise", f"{m['roc_at_cruise_mps']:.2f} m/s  "
                         f"({m['roc_at_cruise_mps'] * 60:.0f} m/min)",
                         "Wing-borne, with all the forward thrust available.")
            _metrics_row("Best climb rate (Vy)",
                         f"{m['max_roc_mps']:.2f} m/s at {m['vy_mps']:.1f} m/s")
            _metrics_row("Best climb angle (Vx)",
                         f"{m['max_climb_angle_deg']:.1f} deg at {m['vx_mps']:.1f} m/s")
            ceiling = m["service_ceiling_m"]
            _metrics_row("Service ceiling (ASL)",
                         "above 8000 m" if not math.isfinite(ceiling) else f"{ceiling:.0f} m",
                         "Where the best wing-borne climb falls to 100 ft/min.")
            _metrics_row("Service ceiling (AGL)",
                         "above 8000 m" if not math.isfinite(ceiling)
                         else f"{m['service_ceiling_agl_m']:.0f} m")
            _metrics_row("Glide ratio", f"{m['glide_ratio']:.1f} : 1")
            _metrics_row("Glide distance",
                         f"{m['glide_distance_km']:.2f} km from "
                         f"{m['glide_reference_altitude_m']:.0f} m",
                         "Unpowered, rotors stopped, from the cruise altitude "
                         "(or the field elevation when none is entered).")

            _metrics_add_section("Turning Flight")
            bank = cfg.bank_deg if cfg is not None else 0.0
            if bank <= 0:
                _metrics_row("Bank angle", "0 deg (straight flight)",
                             "Enter a bank angle on the Mission/Environment tab "
                             "for the turn figures.")
            else:
                _metrics_row("Bank angle", f"{bank:.0f} deg")
                _metrics_row("Load factor", f"{m['load_factor']:.2f} g")
                _metrics_row("Turn stall speed", _speed(m["turn_stall_speed_mps"]))
                _metrics_row("Turn radius", f"{m['turn_radius_m']:.0f} m")
                _metrics_row("Turn rate", f"{m['turn_rate_deg_s']:.1f} deg/s")
                _metrics_row("Turn period", f"{m['turn_period_s']:.1f} s")
                _metrics_row("Turn power", f"{m['turn_power_W']:.0f} W  "
                             f"({m['turn_endurance_min']:.1f} min)",
                             "Sustained level turn: the wing flies at n x W, so "
                             "induced drag rises by n².")
                _metrics_row("Loiter circles", f"{m['loiter_circles']:.0f}")

            _metrics_add_section("Conventional Take-off / Landing")
            _metrics_row("Take-off ground roll",
                         _f(m["takeoff_roll_m"], "{:.0f} m", "cannot take off"),
                         "Rolling take-off on the forward thrust, as an aeroplane "
                         "— useful when the aircraft is too heavy to lift off "
                         "vertically. Raymer's estimate, the fixed-wing's method.")
            _metrics_row("Landing distance (over 15 m obstacle)",
                         _f(m["landing_distance_m"], "{:.0f} m"))

        if cfg is not None and "hover_lift_rpm" in m:
            _motor_section("Lift Motor (hover)", "hover_lift", m, cfg, "lift", "Lift motor")
            if vectored:
                _motor_section("Rotor Motors in Cruise", "cruise_motor", m, cfg,
                               "cruise", "Cruising rotor")
            else:
                _motor_section("Cruise Motor (at cruise)", "cruise_motor", m, cfg,
                               "cruise", "Cruise motor")

            _metrics_add_section("Propellers & Rotors")
            for group, name, prefix in (("lift", "Lift prop", "hover_lift"),
                                        ("cruise", "Cruise prop", "cruise_motor")):
                if group == "cruise" and vectored:
                    continue
                g = _group(cfg, group)
                _metrics_row(f"{name} size",
                             f"{g['d_in']:.1f} x {g['p_in']:.1f} in, {g['blades']} blades  "
                             f"(P/D {m[f'{group}_p_over_d']:.2f})")
                _metrics_row(f"{name} disc area",
                             f"{m[f'{group}_disc_area_m2'] * 1e4:.0f} cm² each, "
                             f"{m[f'{group}_disc_area_m2'] * g['n'] * 1e4:.0f} cm² total")
                _metrics_row(f"{name} C_T / C_P",
                             f"{m[f'{group}_c_t']:.4f} / {m[f'{group}_c_p']:.4f}  "
                             f"({m[f'{group}_coeff_source']})",
                             "Thrust and power coefficients. An estimate is good to "
                             "about ±30%; enter TConst or load a table with RPM for "
                             "better.")
                _metrics_row(f"{name} solidity σ",
                             f"{m[f'{group}_solidity']:.3f}  "
                             f"(chord {m[f'{group}_chord_m'] * 1000:.0f} mm est.)")
                where = "hover" if group == "lift" else "cruise"
                mach = float(m.get(f"{prefix}_tip_mach", 0.0))
                _metrics_row(f"{name} tip speed ({where})",
                             f"{_f(m.get(f'{prefix}_tip_speed_mps'), '{:.1f}')} m/s  "
                             f"(Mach {mach:.3f})"
                             + ("  ⚠ significant noise" if mach > 0.6 else ""))
            if not vectored:
                _metrics_row("Cruise prop pitch speed",
                             f"{_f(m.get('cruise_motor_pitch_speed_mps'), '{:.1f}')} m/s",
                             "Pitch x RPM: the speed the blade would advance per "
                             "turn in a solid. Cruise should sit well below it.")
            _metrics_row("Cruise prop advance ratio J",
                         _f(m.get("cruise_motor_advance_J"), "{:.3f}"),
                         "V / (n D). Thrust falls to zero near 1.2 x P/D.")
            _metrics_row("Lift rotor advance ratio μ",
                         _f(m.get("point_lift_advance_mu"), "{:.3f}"),
                         "V / ΩR for the lift rotors at the cruise point; zero "
                         "once they have stopped.")

            _metrics_add_section("Thermal Estimates")
            _metrics_row("Lift motor temperature",
                         f"{m['lift_motor_temp_C']:.1f} °C  (limit {cfg.lift_motor_temp_limit_C:.0f})",
                         "Steady state in hover, from ambient plus the motor's "
                         "loss times a lumped thermal resistance.")
            if not vectored:
                _metrics_row("Cruise motor temperature",
                             f"{m['cruise_motor_temp_C']:.1f} °C  (limit {cfg.cruise_motor_temp_limit_C:.0f})")
            _metrics_row("ESC temperature",
                         f"{m['esc_temp_est_C']:.1f} °C  (limit {cfg.esc_temp_limit_C:.0f})")
            _metrics_row("Battery temperature",
                         f"{m['battery_temp_est_C']:.1f} °C  (limit {cfg.battery.temp_limit_C:.0f})")
            _metrics_row("Motor thermal headroom", f"{m['motor_thermal_headroom_C']:.1f} °C")
            _metrics_row("ESC thermal headroom", f"{m['esc_thermal_headroom_C']:.1f} °C")
            _metrics_row("Battery thermal headroom", f"{m['battery_thermal_headroom_C']:.1f} °C")
            _metrics_row("Thermal status", str(m["thermal_status"]),
                         "OK, WARN within 10 °C of a limit, HOT past one. These "
                         "are steady-state estimates; a mission run integrates "
                         "them in time instead.")

            _metrics_add_section("Environment")
            _metrics_row("Field altitude", f"{m['altitude_m']:.0f} m")
            _metrics_row("Cruise altitude",
                         "same as field" if m["cruise_altitude_m"] is None
                         else f"{m['cruise_altitude_m']:.0f} m")
            _metrics_row("Temperature", f"{m['ambient_temp_C']:.1f} °C"
                         + ("" if cfg._ambient_temp_C is not None else "  (ISA)"))
            _metrics_row("Pressure", "from altitude (ISA)" if m["pressure_Pa"] is None
                         else f"{m['pressure_Pa']:.0f} Pa")
            _metrics_row("Air density", f"{m['air_density']:.4f} kg/m³  "
                         f"({(m['density_ratio'] - 1) * 100:+.1f}% vs ISA sea level)")
            _metrics_row("Density altitude", f"{m['density_altitude_m']:.0f} m  "
                         f"({m['density_altitude_m'] * 3.281:.0f} ft)")
            _metrics_row("Speed of sound", f"{m['speed_of_sound_mps']:.1f} m/s")

    def _sweep_speeds(cfg):
        """The airspeeds the Fixed Speed Plots and their export cover."""
        v_trans = transition_speed_mps(cfg)
        v_max = (opt("plot_vmax")
                 or max(float(cfg.cruise_speed_mps) * 1.6, v_trans * 1.6))
        steps = max(int(v_max / 0.5), 8)
        return [v_max * i / steps for i in range(1, steps + 1)]

    def _sweep_data(cfg):
        """
        Everything the sweep panels draw, computed once. The export reads the
        same dictionary, so a column in the file is a curve on the screen.
        """
        speeds = _sweep_speeds(cfg)
        usable_Wh = cfg.battery.usable_Wh
        v_trans = transition_speed_mps(cfg)
        # The sweep is flown in the same wind as the run. Range is ground
        # distance, so a headwind moves the best-range speed UP — the
        # aircraft has to fly faster to spend less time being pushed back.
        # Plotting still-air range in a wind would hide exactly that.
        head, cross = core.wind_components_mps(
            num("wind", 0.0), num("wind_dir", 0.0), num("course_deg", 0.0))
        data = {k: [] for k in (
            "speed", "total", "shaft", "rotor_W", "cruise_W", "endurance", "range",
            "share", "rotor_N", "wing_N", "thrust_req", "thrust_avail", "roc",
            "drag_induced", "drag_cd0", "drag_stopped", "drag_body", "drag_total",
            "regime", "tilt")}
        for v in speeds:
            p = power_at_airspeed(cfg, v)
            total = float(p["total_power_W"])
            data["speed"].append(v)
            data["total"].append(total)
            data["shaft"].append(float(p["shaft_power_W"]))
            data["rotor_W"].append(float(p["rotor_shaft_W"]))
            data["cruise_W"].append(float(p["cruise_shaft_W"]))
            # Endurance and range at a STEADY speed, which is what a
            # fixed-speed sweep describes. A real flight also pays for the
            # climb and the transition; the mission run is what prices those.
            mins = usable_Wh / max(total, 1e-9) * 60.0
            data["endurance"].append(mins)
            data["range"].append(mins * 60.0 * core.groundspeed_along_track_mps(
                v, head, cross) / 1000.0)
            data["share"].append(float(p.get("lift_share_wing",
                                             1.0 if p.get("regime") == "cruise" else 0.0)))
            data["rotor_N"].append(float(p.get("rotor_thrust_N", 0.0)))
            data["wing_N"].append(float(p.get("wing_lift_N", 0.0)))
            data["thrust_req"].append(float(p.get("cruise_thrust_N", 0.0)))
            data["thrust_avail"].append(forward_thrust_available_N(cfg, v))
            data["roc"].append(rate_of_climb_mps(cfg, v) if v >= v_trans else float("nan"))
            q = 0.5 * cfg.air_density * v * v
            lift = float(p.get("wing_lift_N", 0.0))
            cl = lift / max(q * cfg.wing_area_m2, 1e-9)
            d_i = q * cfg.wing_area_m2 * cfg.induced_drag_factor * cl * cl
            d_0 = q * cfg.wing_area_m2 * cfg.CD0
            d_s = (0.0 if uses_vectored_thrust(cfg) else
                   stopped_rotor_drag_N(cfg, v) * min(max(lift / max(cfg.weight_N, 1e-9), 0.0), 1.0))
            d_b = body_drag_N(cfg, v)
            data["drag_induced"].append(d_i)
            data["drag_cd0"].append(d_0)
            data["drag_stopped"].append(d_s)
            data["drag_body"].append(d_b)
            data["drag_total"].append(d_i + d_0 + d_s + d_b)
            data["regime"].append(str(p.get("regime", "")))
            data["tilt"].append(float(p.get("tilt_deg", 0.0)))
        return data

    _canvas_motor = {"widget": None}

    def draw_plots(cfg, m=None):
        v_stall = stall_speed_mps(cfg)
        v_trans = transition_speed_mps(cfg)
        v_cruise = float(cfg.cruise_speed_mps)
        d = _sweep_data(cfg)
        speeds = d["speed"]
        best_e = max(range(len(speeds)), key=lambda i: d["endurance"][i])
        best_r = max(range(len(speeds)), key=lambda i: d["range"][i])

        # Four rows of two: the multicopter's and fixed-wing's panels below
        # the VTOL's own four. Two columns keeps the label layout that was
        # tuned for the rendered width; the pane scrolls for the extra rows.
        fig, axes = core.make_figure(
            4, 2, figsize=(_view["plot_w"], _view["plot_h"] * 2.0))
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

        # 1. Power required, electrical and mechanical, against hover.
        ax1 = axes[0, 0]
        ax1.plot(speeds, d["total"], color="#C62828", label="Total (electrical)")
        ax1.plot(speeds, d["shaft"], color="#2E7D32", linestyle="--",
                 label="Shaft (mechanical)")
        ax1.axhline(hover_power_W(cfg)["total_power_W"], color="#1565C0",
                    linestyle="--", label="Hover")
        _marks(ax1, legend=True)
        ax1.set_xlabel("Airspeed (m/s)"); ax1.set_ylabel("Power (W)")
        ax1.set_title("Power vs Airspeed — electrical and mechanical"); ax1.grid(alpha=0.3)
        ax1.legend(fontsize=7)

        # 2. Where that power is going.
        ax2 = axes[0, 1]
        ax2.stackplot(speeds, d["rotor_W"], d["cruise_W"],
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
        l1, = ax3.plot(speeds, d["endurance"], color="royalblue",
                       label="Endurance (min)")
        l2, = ax3b.plot(speeds, d["range"], color="darkorange", linestyle="--",
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
        h_share, = ax4.plot(speeds, [sh * 100.0 for sh in d["share"]],
                            color="#2E7D32", label="Wing share of lift (%)")
        ax4b = ax4.twinx()
        h_rotor, = ax4b.plot(speeds, d["rotor_N"], color="#1565C0", linestyle="--",
                             label="Rotor thrust (N)")
        h_wing, = ax4b.plot(speeds, d["wing_N"], color="#8E24AA", linestyle=":",
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

        # 5. Forward thrust needed against what the propulsion can give —
        # the fixed-wing's panel. The gap is what climbing and accelerating
        # have to work with.
        ax5 = axes[2, 0]
        ax5.plot(speeds, d["thrust_req"], color="#C62828", label="Required (drag)")
        ax5.plot(speeds, d["thrust_avail"], color="#2E7D32", linestyle="--",
                 label="Available")
        _marks(ax5)
        ax5.set_xlabel("Airspeed (m/s)"); ax5.set_ylabel("Forward thrust (N)")
        ax5.set_title("Thrust Required vs Available"); ax5.grid(alpha=0.3)
        ax5.legend(fontsize=7)

        # 6. Climb rate, wing-borne, from the thrust in hand.
        ax6 = axes[2, 1]
        ax6.plot(speeds, d["roc"], color="#1565C0", label="Rate of climb")
        finite = [(v, r) for v, r in zip(speeds, d["roc"]) if r == r]
        if finite:
            vy, best = max(finite, key=lambda x: x[1])
            ax6.plot([vy], [best], "o", color="#1565C0")
            ax6.annotate(f"Vy {vy:.1f} m/s, {best:.1f} m/s", (vy, best),
                         textcoords="offset points", xytext=(6, -12), fontsize=7)
        if cfg.min_climb_mps:
            ax6.axhline(cfg.min_climb_mps, color="#C62828", linestyle=":",
                        label=f"required {cfg.min_climb_mps:.1f} m/s")
        ax6.axhline(0, color="gray", linewidth=0.8)
        _marks(ax6)
        ax6.set_xlabel("Airspeed (m/s)"); ax6.set_ylabel("Climb rate (m/s)")
        ax6.set_title("Rate of Climb vs Airspeed (wing-borne)"); ax6.grid(alpha=0.3)
        ax6.legend(fontsize=7)

        # 7. Drag, split into its parts — the multicopter's and fixed-wing's
        # drag panels, plus the VTOL's own stopped rotors.
        ax7 = axes[3, 0]
        ax7.plot(speeds, d["drag_total"], color="black", label="Total")
        ax7.plot(speeds, d["drag_induced"], color="#1565C0", linestyle="--", label="Induced")
        ax7.plot(speeds, d["drag_cd0"], color="#C62828", linestyle="--", label="Parasitic (CD0)")
        if any(x > 0 for x in d["drag_stopped"]):
            ax7.plot(speeds, d["drag_stopped"], color="#EF6C00", linestyle=":",
                     label="Stopped rotors")
        if any(x > 0 for x in d["drag_body"]):
            ax7.plot(speeds, d["drag_body"], color="#6A1B9A", linestyle=":",
                     label="Fuselage / booms")
        _marks(ax7)
        ax7.set_xlabel("Airspeed (m/s)"); ax7.set_ylabel("Drag (N)")
        ax7.set_title("Drag vs Airspeed"); ax7.grid(alpha=0.3)
        ax7.legend(fontsize=7)

        # 8. The drag polar, with this cruise point on it.
        ax8 = axes[3, 1]
        cls = [cfg.CL_max * i / 60 for i in range(61)]
        ax8.plot([cfg.CD0 + cfg.induced_drag_factor * c * c for c in cls], cls,
                 color="#1565C0", label="CD = CD0 + k CL²")
        if m is not None and "cl_cruise" in m:
            ax8.plot([m["cd_cruise"]], [m["cl_cruise"]], "o", color="#C62828",
                     label=f"cruise CL {m['cl_cruise']:.2f}")
        cl_md = math.sqrt(cfg.CD0 / max(cfg.induced_drag_factor, 1e-9))
        ax8.plot([0, 2 * cfg.CD0 * 1.6], [0, cl_md * 1.6], color="gray",
                 linestyle=":", label=f"best L/D, CL {cl_md:.2f}")
        ax8.axhline(cfg.CL_max, color="#C62828", linestyle=":", linewidth=1, label="CL_max")
        ax8.set_xlabel("CD"); ax8.set_ylabel("CL")
        ax8.set_title("Drag Polar (CD vs CL)"); ax8.grid(alpha=0.3)
        ax8.legend(fontsize=7, loc="lower right")

        # w_pad keeps the right-hand y-label of one panel clear of the
        # left-hand y-label of the next: two panels carry a twin axis, so
        # there are y-labels competing for the gap between columns.
        # Constrained layout, not tight_layout: the plot pane squashes the
        # figure to its own width (about half the requested 14in), and
        # tight_layout's padding was tuned for the full width — at the real
        # rendered size it clipped the left-hand y-labels off the figure and
        # ran the twin-axis labels of the two panels into each other.
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
        _destroy_canvas(_canvas_motor)
        # Remove the mission placeholder if one is showing.
        for child in plot_holder.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_holder)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        _canvas["widget"] = canvas
        if m is not None:
            try:
                rows = 1 if uses_vectored_thrust(cfg) else 2
                mfig = make_motor_figure(cfg, m, (_view["plot_w"], _view["plot_h"] * 0.55 * rows))
                mcanvas = FigureCanvasTkAgg(mfig, master=plot_holder)
                mcanvas.draw()
                mcanvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(8, 0))
                _canvas_motor["widget"] = mcanvas
            except Exception:
                pass
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
        _canvas_motor["widget"] = None
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
        show_metrics(m, cfg)
        draw_plots(cfg, m)
        update_status(cfg, m)
        update_weight_budget(cfg, m)
        draw_airframe_diagram(cfg)
        update_rotor_loading(cfg, m)
        _export_state.update(cfg=cfg, from_mission=False, results=None, metrics=m)
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
            f"minimise time in hover.\n"
            + "\n".join(line.strip() for line in performance_summary_lines(m)) + "\n")

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
            show_metrics(last, cfg)
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
                           accel=opt("accel") or 0.0, decel=opt("decel") or 0.0,
                           regen=opt("regen") or 0.0)
        clear_sensitivity("Mission re-run — sensitivity is out of date")
        _cmp.update(cur_mission=True, current={
            "energy_Wh": totals["energy_Wh"], "time_min": totals["time_s"] / 60.0,
            "distance_km": totals["distance_m"] / 1000.0,
            "hover_Wh": totals["hover_Wh"], "transition_Wh": totals["transition_Wh"],
            "cruise_Wh": totals["cruise_Wh"], "remaining_Wh": totals["remaining_Wh"],
            "peak_power_W": worst.get("total_power_W"),
            "peak_current_A": worst.get("pack_current_A"),
            "reserve_margin_Wh": worst.get("reserve_margin_Wh"),
            "min_soc_pct": worst.get("min_soc_pct"),
            "peak_lift_current_A": worst.get("lift_motor_current_A"),
            "peak_motor_temp_C": worst.get("motor_temp_est_C"),
            "peak_esc_temp_C": worst.get("esc_temp_est_C"),
            "peak_battery_temp_C": worst.get("battery_temp_est_C")})
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
        lines += [line.strip() for line in mission_summary_lines(totals)]
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
        # Older configs carry the avionics power and ESC efficiency under the
        # Mission/Env keys those fields used to have; map them onto the tab
        # fields that replaced them, or the loaded aircraft would silently
        # take the tab defaults instead.
        for key, value in migrate_legacy_fields(payload.get("vars") or {}).items():
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
