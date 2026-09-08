# Example configurations and missions

Everything here is a starting point, not a recommendation. Load one, change
one number, and watch what moves — that is a faster way to learn the tools
than filling sixty empty fields.

---

## Configs (`examples/configs/`)

Load with **Load Config** in either GUI, or pass to the batch driver with
`--gui-config`.

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

---

## Missions (`examples/missions/`)

Set the path in the **Mission JSON** field on the Mission/Environment tab,
then press **Run Mission**. `mc_*` files are for the multicopter, `fw_*` for
the fixed-wing.

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

All configs now exercise: avionics mass, mass entry mode, per-axis tilt limits,
motor/ESC/battery temperature limits, time-at-maximum allowances, and voltage
rating ranges. The fixed-wing configs add field lengths and a minimum climb
rate, which is what their take-off, landing and thrust-margin checks are judged
against.

**Masses are self-consistent.** Every config leaves a positive airframe mass
once battery, motors, ESCs, propellers and avionics are subtracted from the
all-up weight — the simulators now refuse anything else.
