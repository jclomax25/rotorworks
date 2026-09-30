# Example configurations and missions

Everything here is a starting point, not a recommendation. Load one, change
one number, and watch what moves — that is a faster way to learn the tools
than filling sixty empty fields.

---

## Configs (`examples/configs/`)

Load with **Load Config** in any of the three GUIs, or pass to the batch
driver with `--gui-config`. The VTOL CLI also reads them directly:
`python vtol-power-sim-gui.py --config examples/configs/<file>.json`, with any
other flag overriding the value in the file.

### Multicopter

| File | What it is | Why it is interesting |
|---|---|---|
| `multicopter_3in_cinewhoop_4S.json` | 3-inch ducted cinewhoop, 320 g, 4S 850 mAh | Small and heavy for its size. Shows what high disk loading costs you. |
| `multicopter_5in_freestyle_6S.json` | 5-inch freestyle FPV, 700 g, 6S 1300 mAh | Aggressive, short flights, 45° tilt limit. |
| `multicopter_7in_longrange_6S.json` | 7-inch long range, 1250 g, 6S2P Li-ion | Efficiency build. Compare its specific range against the 5-inch. |
| `multicopter_450_survey_4S.json` | 450-class survey quad, 1450 g + 350 g payload | Pack-mode battery, gimbal on a 12 V rail. |
| `multicopter_heavylift_X8_12S.json` | Coaxial X8, 6.5 kg + 5 kg payload, 12S | `motor_configuration=coaxial` — the lower prop works in the upper prop's wash. |

### Fixed-wing

| File | What it is | Why it is interesting |
|---|---|---|
| `fixedwing_1m5_foam_trainer_3S.json` | 1.5 m foam trainer, 1050 g, 3S | High CD0 (0.042), modest CL_max — typical flat-plate foam wing. |
| `fixedwing_900mm_fpv_wing_4S.json` | 900 mm flying wing, 980 g, 4S | Low aspect ratio, reflexed airfoil, fast cruise. |
| `fixedwing_2m_survey_4S.json` | 2 m surveyor, 2600 g + 400 g payload, 4S | Cleaner airframe, cambered airfoil. |
| `fixedwing_3m_endurance_6S_liion.json` | 3 m endurance, 4200 g, 6S4P Li-ion | CD0 0.019, Oswald 0.92. Compare specific range against the trainer — that gap *is* the value of a clean airframe. |

### VTOL

| File | What it is | Why it is interesting |
|---|---|---|
| `vtol_2m4_lift_cruise_survey.json` | Generic 2.4 m lift+cruise, 6 kg, four lift rotors and a pusher | A clean starting point to edit. Change **Configuration** to `tiltrotor` and run again: hover gets dearer, cruise cheaper. |
| `vtol_trinity_f90_lift_cruise.json` | **Quantum-Systems Trinity F90+**, 5 kg | A real aircraft. Flies 90.0 min on a climb/transition/cruise/land profile to a 15% reserve against the published 90, and can still hover at the published 4500 m ceiling. |
| `vtol_wingtraone_gen2_tailsitter.json` | **Wingtra WingtraOne GEN II**, tailsitter | A real tailsitter: 58.4 min against the published 59. |

The two real aircraft separate what the manufacturer **publishes** from what
is **inferred** (wing area, drag, motor electrical parameters), and say in
`_calibration`, `_validation` and `_motor_model_migration` exactly what was
fitted, what was checked, and what changed in v1.11:

- **The Trinity hovers on three rotors.** Its flight drive is two front motors
  that fold and stop in cruise and one rear motor that tilts forward to do
  all the cruising — all three lift in hover. An earlier version of the file
  hovered on two, which the motor model showed up as 90% hover throttle and
  a hover that could not reach the published 4500 m ceiling.
- **`cruise_eff` is the propeller alone.** Before v1.11 the VTOL had no motor
  model and this field was the combined motor-and-propeller figure. Each
  file's value was converted by dividing out the motor efficiency the model
  now computes at its cruise point, so the cruise leg costs what it did and
  the motor's loss is itemised. CD0 was not re-tuned.

The VTOL files set every input the model had before v1.11. The ratings and
limits added in v1.11 — motor and ESC current and voltage ratings, time at
maximum, temperature limits, reserve percent — are left blank, so Status
reports those checks as **Not Specified** until you enter your own parts.

---

## Missions (`examples/missions/`)

Set the path in the **Mission JSON** field on the Mission/Environment tab,
then press **Run Mission**. `mc_*` files are for the multicopter, `fw_*` for
the fixed-wing and `vtol_*` for the VTOL.

### Multicopter

| File | Profile |
|---|---|
| `mc_01_takeoff_hover_land.json` | Climb to 30 m, hover 8 min, land. The simplest check of hover endurance. |
| `mc_02_takeoff_square_land.json` | 400 m square, one leg per compass heading. With wind set, each leg sees a different head/cross component. |
| `mc_03_survey_lawnmower.json` | Transit out, four survey lines with turns, transit home. 25% reserve + 8 Wh RTH. |
| `mc_04_delivery_out_and_back.json` | 2.5 km out, hover drop, 2.5 km back. Headwind out, tailwind home. |
| `mc_05_endurance_speed_sweep.json` | Two minutes each at 0/5/10/15/20 m/s. Read the power curve off Mission Plots. |

### Fixed-wing

| File | Profile |
|---|---|
| `fw_01_takeoff_cruise_land.json` | Climb, cruise 8 km, descend, land. |
| `fw_02_takeoff_square_land.json` | 1.5 km square with 30° banked turns. Watch load factor and turn stall speed on the Status tab. |
| `fw_03_survey_lawnmower.json` | 3 km transit, four 2 km lines, 3 km home. Carries RTH *and* diversion reserves. |
| `fw_04_loiter_on_station.json` | Transit 10 km, loiter 30 min in a 20° orbit, return. |
| `fw_05_speed_sweep.json` | Three minutes each at 14/18/22/26 m/s. Compare against the best-endurance and best-range speeds on the Metrics tab. |

### VTOL

Each phase carries a `kind` — `climb`, `hover`, `descend`, `transition` or
`cruise`. A **Reserve percent** entered on the Mission/Environment tab
overrides the file's `reserve_percent`.

| File | Sized for | Profile |
|---|---|---|
| `vtol_01_lift_cruise_survey.json` | `vtol_2m4_lift_cruise_survey` | Climb, transition, 24 km of transit and survey, land. Read the hover / transition / cruise energy split. |
| `vtol_02_corridor_powerline.json` | Trinity F90+ | Power-line corridor, out and back. With an 8 m/s wind from 090 the slow 14 m/s imaging leg goes from 31 to 72 min while the faster return saves only 8, and the flight from 62 to 97 min: a headwind hurts most on the leg you are already flying slowly. |
| `vtol_03_block_survey_with_hold.json` | WingtraOne GEN II | 400 ha block with a 90 s hold. The hold costs about 1080 W in still air against about 145 W in cruise, and falls to about 210 W in a 12 m/s wind — translational lift. |
| `vtol_04_delivery_hover_drop.json` | Trinity F90+ | Out, winch drop at 25 m, home. The drop hover costs about 970 W against 116 W in cruise, more than eight times. |

One caveat worth knowing: **payload mass does not change mid-mission.** The
delivery mission still carries its parcel on the way home. For a real
loaded-vs-empty comparison, run it twice with different Payload Mass values.

---

## Using these with the batch driver

`--gui-config` accepts these files directly:

```bash
# How does payload eat into endurance?
python rotorworks-batch.py sweep \
    --sim multicopter --mode simple \
    --gui-config examples/configs/multicopter_450_survey_4S.json \
    --sweep-var payload_mass_g --values 0,250,500

# Where is the best cruise speed?
python rotorworks-batch.py sweep \
    --sim fixedwing --mode simple \
    --gui-config examples/configs/fixedwing_2m_survey_4S.json \
    --sweep-var cruise_speed --values 14,18,22
```

### `--mode simple` vs `--mode advanced`

`--mode` mirrors the Simple/Advanced toggle in the GUIs, and it is a **guard
rail, not a physics switch**. The simulators compute identical numbers either
way. What the mode controls is which parameters the batch driver will let you
set or sweep:

- `simple` — only the inputs the GUI shows in Simple view. Sweeping or
  overriding anything else is a hard error naming the offending parameter.
  Keeps a sizing study from silently perturbing an inflow-map breakpoint.
- `advanced` (default) — everything is available.

The check covers `--set` overrides, `--sweep-var`, `--design-var`, and every
override in a `--runs-file`. A `--gui-config` is exempt, since a saved
aircraft legitimately contains advanced fields.

## What each config is there to show

Every example now carries the full input set, so loading one is a tour of the
software rather than a minimal case.

| Config | What it demonstrates |
|---|---|
| `multicopter_3in_cinewhoop_4S` | High disk loading and its efficiency ceiling; small-rotor figure of merit |
| `multicopter_5in_freestyle_6S` | **Translation direction set to 45°** — see drag and the pitch/roll split change |
| `multicopter_7in_longrange_6S` | Efficiency-focused cruise; Li-ion SoC curve |
| `multicopter_450_survey_4S` | Payload, avionics rails, tight tilt limits |
| `multicopter_heavylift_X8_12S` | Coaxial X8, three avionics rails, 12S ratings |
| `Jaguar-quad` | A real user design. Load your own prop table to see measured TConst/PConst |
| `fixedwing_1m5_foam_trainer_3S` | High CD0, low Reynolds number, short field |
| `fixedwing_2m_survey_4S` | Clean airframe, camera payload, 150 m strip |
| `fixedwing_3m_endurance_6S_liion` | High aspect ratio; best specific range of the set |
| `fixedwing_900mm_fpv_wing_4S` | Hand-launched (no take-off run entered), fast cruise |
| `vtol_2m4_lift_cruise_survey` | Lift+cruise VTOL: transition power peak and hover/cruise energy split |
| `vtol_trinity_f90_lift_cruise` | A real lift+cruise, validated against three published figures; the motor model at work on a three-rotor hover |
| `vtol_wingtraone_gen2_tailsitter` | A real tailsitter: one set of rotors that both lifts and cruises |

The multicopter and fixed-wing configs exercise: avionics mass, mass entry
mode, per-axis tilt limits, motor/ESC/battery temperature limits,
time-at-maximum allowances, and voltage rating ranges. (The VTOL configs leave
the v1.11 ratings blank; see above.) The fixed-wing configs add field lengths and a minimum climb
rate, which is what their take-off, landing and thrust-margin checks are judged
against.

**Masses are self-consistent.** Every config leaves a positive airframe mass
once battery, motors, ESCs, propellers and avionics are subtracted from the
all-up weight — the simulators now refuse anything else.
