# RotorWorks UAV Power Simulators

**UASforge / dronefoundry**  
*Simulators v2.12.2*

A suite of cross-platform UAV powertrain performance tools:

| Tool | What it does |
|---|---|
| `rotorworks_core.py` | Shared code both simulators import — **must sit beside them** |
| `multicopter-power-sim-gui.py` | Multicopter performance simulator (GUI + CLI) |
| `fixedwing-power-sim-gui.py` | Fixed-wing performance simulator (GUI + CLI) |
| `rotorworks-batch.py` | Batch driver: parameter sweeps, sizing studies, scripted runs |
| `drag_coefficient_calculator.py` | Measures drag coefficients from photographs |

All four support **single-point analysis**; the two simulators also support
**mission simulation**, configurable **battery / motor / ESC / prop / avionics
rails**, **status limit checks**, and **plots** including mission time-series.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Simple vs Advanced mode](#simple-vs-advanced-mode)
- [Example configs and missions](#example-configs-and-missions)
- [GUI reference](#gui-reference)
- [CLI reference](#cli-reference)
- [Batch driver](#batch-driver-rotorworks-batchpy)
- [Drag coefficient calculator](#drag-coefficient-calculator)
- [Mission JSON format](#mission-json-format)
- [Motor/prop CSV table format](#motorprop-csv-table-format)
- [Modeling notes and assumptions](#modeling-notes-and-assumptions)
- [Troubleshooting](#troubleshooting)

---

## Project layout

```
rotorworks_core.py              shared: SoC model, atmosphere, wind, tooltips
multicopter-power-sim-gui.py    multicopter simulator (GUI + CLI)
fixedwing-power-sim-gui.py      fixed-wing simulator (GUI + CLI)
rotorworks-batch.py             sweeps, sizing studies, scripted runs
drag_coefficient_calculator.py  drag coefficients from photographs
examples/                       9 aircraft configs, 10 missions
tests/                          219 pytest tests
```

`rotorworks_core.py` holds the aircraft-agnostic code both simulators depend
on: the battery state-of-charge model, the standard atmosphere, wind
resolution, the thermal step, GUI tooltips, and assorted parsing helpers. Two
copies of that code previously drifted apart and caused real bugs — a tooltip
fix that had to be applied twice, and two atmosphere implementations that
disagreed about pressure overrides. **Both simulators import it, so it must
sit in the same folder.** They exit with a clear message if it is missing.

What deliberately stays per-simulator: `BatteryConfig` (the two constructors
take genuinely different parameters), `ESCConfig` / `AvionicsConfig`, and all
export, plotting and GUI-construction code, which reaches into
simulator-specific attributes.

## Requirements

Python **3.9+**.

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```txt
numpy>=1.21
scipy>=1.8
pandas>=1.4
matplotlib>=3.6
Pillow>=9.0
openpyxl>=3.0
reportlab>=3.6
```

What each is needed for:

| Package | Needed for |
|---|---|
| numpy, scipy, pandas, matplotlib | Core simulation and plotting (required) |
| `Pillow` | Photo loading in the drag calculator only |
| `openpyxl` | **Export Excel** button only |
| `reportlab` | **Generate Report** (PDF) button only |

`tkinter` ships with the Python standard library and is **not** a pip install.
On Debian/Ubuntu it is packaged separately — see
[Troubleshooting](#troubleshooting).

---

## Quick start

### GUI

```bash
python multicopter-power-sim-gui.py --gui
python fixedwing-power-sim-gui.py  --gui
python drag_coefficient_calculator.py
```

Running either simulator with **no arguments at all** also opens the GUI-less
CLI path and will report which required arguments are missing. Pass `--gui`
to go straight to the graphical interface.

Fastest way to get a feel for it:

1. Launch the GUI.
2. **Load Config** → `examples/configs/multicopter_450_survey_4S.json`
3. Press **▶ Run Single-Point**.
4. Read the **Metrics** and **Status** tabs.
5. Change one number and run again.

### CLI

Weights are in **grams**, speeds in **m/s**, resistances in **ohms**.

```bash
python multicopter-power-sim-gui.py \
  --num_motors 4 --weight 1450 --speed 10 \
  --battery_unit_mode pack --battery_pack_capacity 5200 \
  --battery_pack_weight_g 520 \
  --battery_series_units 1 --battery_parallel_units 1 \
  --battery_cells_series_per_unit 4 \
  --battery_operating_voltage_min 3.3 \
  --battery_operating_voltage_nominal 3.7 \
  --battery_operating_voltage_max 4.2 \
  --battery_resistance_cell 4.0 --battery_discharge_percent 80 \
  --battery_discharge_c_cont 25 \
  --motor_kv 920 --motor_resistance 0.115 \
  --prop_diameter 10 --prop_pitch 4.5 \
  --avionics_voltage_tree "5.0:(1.2,0.9), 12.0:(1.5,0.87)"
```

Mission run (either simulator):

```bash
python multicopter-power-sim-gui.py <config args...> \
  --mission examples/missions/mc_01_takeoff_hover_land.json
```

Full option list:

```bash
python multicopter-power-sim-gui.py --help
python fixedwing-power-sim-gui.py --help
```

---

## Simple vs Advanced mode

Both GUIs open in **Simple** mode. A selector sits directly above the input
tabs.

- **Simple** shows only the inputs needed to size a powertrain
  (multicopter 62 of 94 fields; fixed-wing 58 of 77).
- **Advanced** shows everything: SoC breakpoint curves, inflow maps,
  transient accel/decel tuning, `TConst`/`PConst`, regen efficiency, and
  reference-only fields such as pole count.

This is a **view setting only**. Hidden fields keep their values, and
switching modes never changes a computed result.

Every input has a blue **`?`** beside it. Hover it (or the field label) for a
plain-language description plus a typical range or where to find the number —
89 tooltips on the multicopter, 72 on the fixed-wing.

The **Avionics** tab stays visible in Simple mode by design. Omitting rail
loads is one of the most common reasons a beginner's endurance estimate comes
out optimistic.

The batch driver has a matching `--mode simple|advanced` flag; see
[Batch driver](#batch-driver-rotorworks-batchpy).

---

## Example configs and missions

`examples/` ships with 9 aircraft configs and 10 missions. Load a config with
**Load Config**, or feed it to the batch driver with `--gui-config`.

### Configs — multicopter

| File | Aircraft | Why it is interesting |
|---|---|---|
| `multicopter_3in_cinewhoop_4S.json` | 3in ducted, 320 g, 4S 850 mAh | High disk loading — see what it costs |
| `multicopter_5in_freestyle_6S.json` | 5in freestyle, 700 g, 6S 1300 mAh | Aggressive, 45° tilt limit |
| `multicopter_7in_longrange_6S.json` | 7in, 1250 g, 6S2P Li-ion | Compare specific range to the 5in |
| `multicopter_450_survey_4S.json` | 450-class, 1450 g + 350 g payload | Pack-mode battery, 12 V gimbal rail |
| `multicopter_heavylift_X8_12S.json` | Coaxial X8, 6.5 kg + 5 kg payload | Coaxial interference penalty |

### Configs — fixed-wing

| File | Aircraft | Why it is interesting |
|---|---|---|
| `fixedwing_1m5_foam_trainer_3S.json` | 1.5 m foam trainer, 1050 g | High CD0 (0.042), flat-plate wing |
| `fixedwing_900mm_fpv_wing_4S.json` | 900 mm flying wing, 980 g | Low aspect ratio, fast cruise |
| `fixedwing_2m_survey_4S.json` | 2 m surveyor, 2600 g + 400 g | Cambered airfoil, cleaner airframe |
| `fixedwing_3m_endurance_6S_liion.json` | 3 m endurance, 4200 g, 6S4P | CD0 0.019, Oswald 0.92 — the gap vs the trainer *is* the value of a clean airframe |

### Missions

`mc_*` are for the multicopter, `fw_*` for the fixed-wing.

| Multicopter | Fixed-wing | Profile |
|---|---|---|
| `mc_01_takeoff_hover_land` | `fw_01_takeoff_cruise_land` | Simplest possible flight |
| `mc_02_takeoff_square_land` | `fw_02_takeoff_square_land` | Square circuit, one leg per heading |
| `mc_03_survey_lawnmower` | `fw_03_survey_lawnmower` | Mapping pattern with reserves |
| `mc_04_delivery_out_and_back` | `fw_04_loiter_on_station` | Out-and-back drop / ISR loiter |
| `mc_05_endurance_speed_sweep` | `fw_05_speed_sweep` | Diagnostic: parks at several speeds |

> **Payload mass does not change mid-mission.** The delivery mission still
> carries its parcel home. For a true loaded-vs-empty comparison, run it twice
> with different payload values.

---

## GUI reference

### Input tabs

| Tab | Contents |
|---|---|
| **Airframe** | Weight, payload, motor count and layout, body geometry, tilt limit, drag model, peripheral current |
| **Battery** | Cell or pack mode, capacity, cell counts, resistance, discharge limits, SoC model |
| **Motor** | Kv, resistance, idle current, ratings, weight |
| **ESC** | Voltage rating (**in S cells, not volts**), continuous/max current, resistance |
| **Avionics** | Voltage rails: V, A, converter efficiency |
| **Prop** | Diameter, pitch, blades, limits, optional CSV test table |
| **Mission / Env** | Speed, wind, altitude, temperature, reserves, or a mission JSON |

### Output tabs

| Tab | Contents |
|---|---|
| **Metrics** | Full single-point summary, grouped into sections (see below) |
| **Status** | Colour-coded limit checks: green OK, yellow warn, red violation |
| **Weight Budget** | Component mass table plus stacked-bar and pie charts |
| **Airframe Diagram** | To-scale plan view; propeller overlap and tip clearance |
| **Sensitivity** | Ranks inputs by influence on flight time, range or power |
| **Compare** | Deltas against a pinned baseline configuration |
| **Plots** | Performance curves vs speed |
| **Mission Plots** | Time-series for a mission run, multi-unit y-axes |

**Metrics** sections: Battery, Motor @ Operating Point, Total Drive & Power,
Propeller & Rotor, Flight Performance, Propulsion Efficiency, Thermal
Estimates, Environment & Design. The fixed-wing adds Aerodynamics at Cruise,
Turning Flight, Climb Performance, Optimal Speeds, and Endurance & Range.

**Status** groups: Battery, Motor/ESC, and Propeller (multicopter) or
Aerodynamic (fixed-wing). Checks include voltage sag, C-rate vs rating,
hover/forward thrust-to-weight, disk loading, hover efficiency, figure of
merit, wind resistance, ESC S-rating vs pack, stall margin, CL margin,
Reynolds number, and service ceiling.

### Buttons

| Button | Action |
|---|---|
| **▶ Run Single-Point** | Evaluate one operating point |
| **📋 Run Mission (JSON)** | Run the loaded mission phase-by-phase |
| **💾 Save Config** / **📂 Load Config** | JSON round-trip of every field |
| **📊 Export CSV** | Performance sweep + metrics |
| **📗 Export Excel** | Three sheets: sweep, metrics, weight budget |
| **📄 Generate Report** | PDF: inputs, metrics, colour-coded checks, all plots |

UI scale is under **View → UI Scale** (150–200% helps on high-DPI displays).

---

## CLI reference

Both simulators run headless with no `--gui`. Key arguments:

### Shared

| Argument | Units / values | Notes |
|---|---|---|
| `--weight` | grams | Base weight **excluding** payload |
| `--payload_mass_g` | grams | Added on top of base weight |
| `--altitude` | metres ASL | Sets air density |
| `--temperature` | °C | Optional; blank uses ISA |
| `--wind` | m/s | Wind speed |
| `--wind_direction_deg` | degrees | Direction wind comes **from** |
| `--course_deg` | degrees | Direction of travel |
| `--mission` | path | Mission JSON; omit for single-point |
| `--avionics_voltage_tree` | string | `"5.0:(2,0.9), 12.0:(1.5,0.85)"` = 2 A at 5 V (90% eff), 1.5 A at 12 V (85%) |
| `--reserve_percent` | percent | Landing reserve |
| `--gui` | flag | Open the GUI instead |
| `--plot` | flag | Show plot window |

### Battery (both)

`--battery_unit_mode cell|pack` selects which set applies.

| Argument | Notes |
|---|---|
| `--battery_cell_capacity` / `--battery_pack_capacity` | mAh, per **cell** or per **pack** |
| `--battery_cell_weight_g` / `--battery_pack_weight_g` | grams per unit |
| `--battery_series_units` | Units in series — sets **voltage** |
| `--battery_parallel_units` | Units in parallel — sets **capacity** |
| `--battery_cells_series_per_unit` | Cells in series inside one pack |
| `--battery_operating_voltage_min/nominal/max` | Volts **per cell** |
| `--battery_resistance_cell` | milliohms per cell |
| `--battery_discharge_percent` | Usable fraction, e.g. 80 |
| `--battery_discharge_cont_A` *or* `--battery_discharge_c_cont` | Either an amp figure or a C-rate |
| `--battery_soc_model` | `auto` (preset from chemistry), `linear`, or `lipo`/`liion`/`lifepo4` |
| `--battery_soc_curve_csv` | Measured discharge curve. Columns: `soc, ocv_cell, r_scale` |
| `--battery_soc_bp`, `--battery_ocv_cell_bp`, `--battery_r_scale_bp` | Custom curve as comma-separated lists |

> Series raises voltage, parallel raises capacity. Two 6S 5000 mAh packs in
> series is 12S **5000 mAh** (222 Wh), not 10000 mAh.

### Multicopter-specific

`--num_motors`, `--speed`, `--orientation hover|forward`, `--max_tilt_deg`,
`--motor_configuration flat|coaxial`, `--coaxial_spacing_m`,
`--drag_model_mode auto|manual`, `--parasite_area`, `--parasite_drag`,
`--profile_area`, `--profile_drag`, `--body_length_m`, `--body_width_m`,
`--body_height_m`, `--arm_length_m`, `--arm_width_m`

### ESC (both)

`--esc_voltage_rating` (in **S cells**), `--esc_cont_current`,
`--esc_max_current`, `--esc_idle_current`, `--esc_resistance`, `--esc_weight`.

Supplying any one of these builds an ESC and includes its losses. Omit them
all and ESC losses are simply not modelled.

### Fixed-wing-specific

`--cruise_speed`, `--wing_span`, `--wing_area`, `--CD0`, `--CL_max`,
`--oswald`, `--CL_takeoff`, `--mu_roll`, `--mu_brake`, `--prop_efficiency`,
`--bank_deg`, `--cruise_altitude`, `--prop_eff_model`

`--prop_efficiency` is the **peak** efficiency. `--prop_eff_model` is `curve`
(default, varies with advance ratio) or `constant` (flat, pre-2.4.0).
`--cruise_altitude` sets the height used for the glide-distance estimate;
`--altitude` remains the field elevation.

Only genuinely load-bearing arguments are required. Ratings used purely for
status checks (motor max current/power, charge current, energy density) can
be omitted — the corresponding check is skipped rather than the run failing.

---

## Batch driver (`rotorworks-batch.py`)

Three subcommands, all of which accept `--gui-config` and `--mode`.

### Sweep — one-variable sensitivity

```bash
python rotorworks-batch.py sweep \
  --sim multicopter --mode simple \
  --gui-config examples/configs/multicopter_450_survey_4S.json \
  --sweep-var payload_mass_g --values 0,300,600
```

`--start/--stop/--step` works instead of `--values`. Add `--plot-metric` to
choose which metrics get charted.

### Size — constraint-driven grid search

```bash
python rotorworks-batch.py size \
  --sim multicopter --mode simple \
  --gui-config examples/configs/multicopter_450_survey_4S.json \
  --design-var "prop_diameter:9:11:1" \
  --design-var "battery_pack_capacity=4000,5200" \
  --target-min flight_time_min=15 \
  --objective maximize:flight_time_min
```

### Batch — explicit scripted runs

```bash
python rotorworks-batch.py batch \
  --sim fixedwing --mode simple \
  --gui-config examples/configs/fixedwing_2m_survey_4S.json \
  --runs-file runs.json
```

```json
{"runs": [
  {"name": "slow",   "overrides": {"cruise_speed": 15}},
  {"name": "cruise", "overrides": {"cruise_speed": 19}},
  {"name": "fast",   "overrides": {"cruise_speed": 24}}
]}
```

### Argument precedence

Lowest to highest: `--gui-config` → `--base-args-file` → `--set key=value`.

`--gui-config` reads the same JSON the GUI writes, translating GUI field
names into simulator CLI arguments (including folding the avionics rail table
into `--avionics_voltage_tree`).

### `--mode simple` vs `--mode advanced`

Mirrors the GUI toggle and is a **guard rail, not a physics switch** — the
simulators compute identical numbers either way. What it controls is which
parameters the batch driver will let you set or sweep:

- `simple` — only the Simple-view inputs. Anything else is a hard error
  naming the offending parameter, so a sizing study cannot silently perturb
  an inflow-map breakpoint.
- `advanced` (default) — everything available.

Enforced against `--set`, `--sweep-var`, `--design-var`, and every override
in a `--runs-file`. A `--gui-config` is exempt, since a saved aircraft
legitimately contains advanced fields.

Other flags: `--sim-script` (explicit simulator path), `--timeout`,
`--output-dir`, `--print-commands`.

Outputs: CSV and JSON summaries plus PNG plots in a timestamped directory.

---

## Drag coefficient calculator

Measures real drag numbers from photographs, following the ArduPilot
[airspeed estimation](https://ardupilot.org/copter/docs/airspeed-estimation.html)
method.

### Tab 1 — Body drag (BCOEF)

Per view (front, side, top):

1. **Load Image**
2. **Set Scale** — click two points a known distance apart (motor-to-motor
   wheelbase works well), enter the distance in cm
3. **Draw Outline** — click around the silhouette, excluding propeller
   blades. Right-click undoes; double-click or clicking near vertex 1 closes.

Enter mass and body Cd, then **Calculate**. Outputs:

| Output | Goes where |
|---|---|
| `EK3_DRAG_BCOEF_X` / `_Y` | ArduPilot parameters |
| `parasite_area` + `parasite_drag_coefficient` | Simulator, from the **front** view |
| `profile_area` + `profile_drag_coefficient` | Simulator, from the **side** view |

The top view is reference only (vertical/descent drag).

### Tab 2 — Propeller drag (MCOEF)

Actuator-disk estimate from mass, motor count, prop diameter, and air
density:

```
v_h   = sqrt( (W/N) / (2·ρ·A_disk) )
MCOEF = g / (2·v_h)
```

Typical range 0.1–1.0. Treat a flight-test-derived MCOEF as ground truth;
this gives you a starting value before flying.

### BCOEF vs the simulator's Cd

ArduPilot's `EK3_DRAG_BCOEF` is a **ballistic coefficient** in kg/m², not a
dimensionless drag coefficient:

```
BCOEF = mass / (Cd × projected_area)
```

ArduPilot's guide uses `BCOEF = mass / area`, implicitly assuming Cd = 1.0.
The simulator takes Cd and area separately. With Cd = 1.0 the two are
numerically identical. Verified against ArduPilot's own IRIS example
(BCOEF_X 71.4, BCOEF_Y 66.8).

---

## Mission JSON format

Ordered phases, each with a name, a speed, and **either** a duration (seconds)
**or** a distance (metres).

```json
{
  "reserve_percent": 20,
  "rth_reserve_Wh": 0,
  "diversion_reserve_Wh": 0,
  "wind_direction_deg": 0,
  "phases": [
    {"name": "Takeoff climb", "speed": 0.0,  "duration": 20,   "altitude": 30, "climb_rate_mps": 1.5},
    {"name": "Cruise north",  "speed": 10.0, "distance": 400,  "altitude": 30, "course_deg": 0},
    {"name": "Hover",         "speed": 0.0,  "duration": 120,  "altitude": 30},
    {"name": "Land",          "speed": 0.0,  "duration": 30,   "altitude": 0,  "descent_rate_mps": 1.0}
  ]
}
```

### Peripheral current vs the Avionics tab

Two separate ways to account for non-motor draw, and they must not overlap:

| Input | Use it for |
|---|---|
| **Peripheral Current** (Airframe tab) | Devices wired **directly to pack voltage** with no regulator — a heater, a pump, a payload on raw battery. Drawn at pack voltage, no conversion loss. |
| **Avionics tab** | Anything on a **regulated rail** — 5 V flight controller, 12 V VTX. Converter efficiency is applied, so the pack sees more current than the rail draws. |

Enter each device in exactly one of them.

### Transient (acceleration) settings

Settable per mission (in the JSON) or from the Mission/Environment tab in
either GUI. A value in the mission file wins; the GUI fields fill in the rest.

| Field | Default | Meaning |
|---|---|---|
| `transient_dt_s` | 0.5 | Integration step for the speed ramp |
| `max_accel_mps2` | 1.5 (FW) / 2.0 (MC) | Acceleration limit |
| `max_decel_mps2` | 2.0 (FW) / 2.5 (MC) | Deceleration limit |
| `decel_regen_eff` | 0.0 | Fraction of braking energy recovered — 0 is honest for a fixed-pitch prop |

A phase that commands a different speed from the previous one spends a ramp
segment reaching it. The ramp costs time, distance and energy, all taken out
of that phase's budget, so a mission of constant-speed phases is unaffected.

### Phase fields

| Field | Type | Notes |
|---|---|---|
| `name` | string | Shown in results and plots |
| `speed` | m/s | Airspeed for this leg |
| `duration` | seconds | Mutually exclusive with `distance` |
| `distance` | metres | Mutually exclusive with `duration` |
| `altitude` | metres ASL | Air density recomputed per phase |
| `course_deg` | degrees | Direction of travel, for wind resolution |
| `climb_rate_mps` / `descent_rate_mps` | m/s | Optional |
| `bank_deg` | degrees | **Fixed-wing only** — banked turns and loiter |

### Profile-level fields

`reserve_percent`, `rth_reserve_Wh`, `diversion_reserve_Wh`,
`wind_direction_deg`, and (multicopter only) `transient_dt_s`,
`max_accel_mps2`, `max_decel_mps2`, `decel_regen_eff`.

> **Field names are `duration`, `speed`, and `altitude`** — not `duration_s`,
> `airspeed_mps`, or `altitude_m`.

### Outputs

Per-phase time and distance with a status flag, a total row, worst-case
metrics driving the **Status** checks, and time-series (speed, altitude,
tilt, voltage, current, power, RPM, thrust) on the **Mission Plots** tab.

---

## Motor/prop CSV table format

A measured thrust table beats the analytic model and is worth supplying if
you have one.

**Required columns** (naming variants accepted): **Thrust (g)**, **Power (W)**

**Optional**: `RPM`, `Current (A)`, `Voltage (V)`, `Throttle` (e.g. `40%`),
`Efficiency (g/W)`, `Operating Temperature (℃)`, `Torque (N*m)`

Notes:

- The header row does **not** need to be the first line — extra preamble rows
  are skipped.
- Column names are normalised automatically.
- Power interpolation is **thrust-based**.
- A `Throttle` column overrides the analytic throttle estimate in
  single-point mode.

---

## Modeling notes and assumptions

### Multicopter

- Forward-flight tilt: `tilt = atan(D / W)`, with `max_tilt_deg` enforced.
- Rotor inflow uses Glauert's forward-flight momentum theory:
  `vi = v_h² / sqrt((V·cos a)² + (V·sin a + vi)²)`, solved iteratively, where
  `v_h = sqrt(T / (2·ρ·A))` and `a` is the disk incidence (the tilt angle).
  At `V = 0` this reduces to the hover value; at speed the rotor meets air
  that is already moving, so induced power falls sharply.
- Shaft power is `P = T·(V·sin a + vi)`. The first term is the propulsive
  power overcoming airframe drag — by the tilt balance `T·sin a = D`, so it
  equals `D·V` exactly. Together these produce the **power bucket**: a
  minimum roughly 10–25% below hover power somewhere around 8–14 m/s, which
  is what sets the real best-endurance and best-range speeds.
- Coaxial interference scales with spacing ratio `s / D`.
- Forward-flight drag uses the **frontal** silhouette; hover/lateral drag
  uses the **side** silhouette. These are separate terms and are not summed.
- Geometry drag fallback: box body plus square-tube arms. Rotor disk drag is
  not assumed unless included via CdA.

### Fixed-wing

- **Multiple motors** are supported. Total thrust is divided across them, so
  a twin or triple tractor spreads the same thrust over more disc area and
  needs slightly less induced power than a single. Thrust available and
  motor/ESC losses scale with motor count too.
- **Tractor vs pusher is not modelled.** Momentum theory does not distinguish
  them, and the simulator has no layout flag. The real differences — a pusher
  running in the wing and fuselage wake, a tractor blowing accelerated air
  over the wing — are worth a few percent of propulsive efficiency. Model a
  pusher by entering a slightly lower **Prop Efficiency η** (typically 2-5%
  below the equivalent tractor). The Airframe Diagram always draws props at
  the leading edge regardless.
- Drag polar: `CD = CD0 + k·CL²`, `k = 1/(π·AR·e)`.
- Propeller power uses **forward-flight** momentum theory:
  `vi = −V/2 + sqrt((V/2)² + T/(2ρA))`, `P_shaft = T·(V + vi)`.
  At V = 0 this reduces to the static hover form.
- Propeller efficiency **varies with advance ratio**. `Prop Efficiency η` is
  the PEAK value, reached near 60% of pitch speed; efficiency falls off toward
  static (blade stalled) and toward pitch speed (blade at zero incidence).
  Set `Prop Eff Model` to `constant` — or pass
  `--prop_eff_model constant` — for the older flat behaviour.
- **Glide distance** is measured from `Cruise Altitude`, not from the field
  elevation in `Altitude`. Leave Cruise Altitude blank and it falls back to
  the field elevation, which is why the figure reads 0 m at a sea-level field.
- Landing distance is the FAA-style figure **over a 15 m (50 ft) obstacle**,
  so it is dominated by the `15 m × L/D` approach segment. Lift is dumped for
  the ground roll (`CL_ground = 0.25`) so the brakes see the aircraft weight.
- Glide distance uses the altitude on the Mission/Env tab. At altitude 0 it
  is legitimately 0 — enter your cruise altitude for a meaningful number.

### Shared

- ESC losses: `P_loss = I²R + I_idle·V`
- Avionics: `I_pack = Σ(V·I / eff) / V_pack`
- Battery: capacity scales with **parallel** count, voltage with **series**
  count; total energy `E = C_Ah × V_pack`.
- **State of charge**: both simulators model pack open-circuit voltage and
  internal resistance as functions of SoC, using a chemistry preset
  (LiPo / Li-ion / LiFePO4), a CSV you measured, or breakpoints you supply.
  Resistance rises steeply below about 20% SoC, which is what makes voltage
  sag worse late in a flight. Set `--battery_soc_model linear` to disable the
  curve and anchor voltage at full charge (the older behaviour). Mission runs
  track SoC phase-by-phase; single-point runs evaluate at full charge.
- ISA atmosphere with optional temperature override.
- Thermal figures are first-order estimates anchored to component ratings,
  not a transient thermal model.

This is a **performance-level model**, not CFD or transient motor dynamics.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tkinter'` (Linux)

```bash
sudo apt install python3-tk
```

### I do not see the Simple/Advanced selector or the `?` help markers

You are running an older copy of the script. Check with **Help → About /
Version** — this release is **v2.12.2**, and the version also appears in the
window title bar and at the top of the Output pane on startup.

From a terminal:

```bash
grep -c "Input detail" multicopter-power-sim-gui.py   # 1 = current, 0 = old
```

The selector sits directly above the Drone/Battery/Motor tabs, and every
input row has a blue `?` to its right.

### Hovering a `?` shows nothing and the terminal prints `NameError: name 'tk' is not defined`

Fixed in **v2.1.1**. Earlier builds defined the tooltip helper at module level
while `tkinter` is imported lazily inside `launch_gui()`, so the name was out
of scope at hover time. Update to v2.1.1 (check **Help → About / Version**).

### UI text is too small

**View → UI Scale**, 150–200%.

### Export Excel or Generate Report does nothing

Install the optional dependency:

```bash
pip install openpyxl reportlab
```

### Drag calculator will not load images

```bash
pip install Pillow
```

The Propeller Drag (MCOEF) tab works without Pillow.

### "Missing required CLI args"

The message lists exactly what is missing. Note that cell-mode and pack-mode
capacity are alternatives — supply whichever matches
`--battery_unit_mode` — and that a current limit can be given as either
`--battery_discharge_cont_A` or `--battery_discharge_c_cont`.

### "Cannot maintain speed"

Required tilt exceeds `max_tilt_deg`. Reduce speed, drag, or weight, or raise
the tilt limit.

### Flight time looks impossibly long

Check the battery configuration first. Series count sets voltage, parallel
count sets capacity — if you entered a series stack expecting more mAh, the
energy figure will be wrong. The **Metrics → Battery** section reports
Wh/kg; anything above ~300 Wh/kg means an input is wrong, since no current
cell chemistry reaches that.

### The GUI and the CLI give different answers for the same config

They should not, as of **v2.3.0**. Before that, the fixed-wing CLI had no ESC
arguments, so a batch run silently dropped ESC losses that the GUI included
(about 1% on a typical config). If you still see a mismatch, check that the
`--gui-config` path is being used rather than hand-written `--set` overrides.

### Efficiency or endurance changed after updating

Several physics corrections have landed. Results from older versions are not
comparable — this is expected, not a regression.

**v2.4.0**
- Multicopter forward-flight power drops roughly 20% around 8–14 m/s, because
  rotor inflow is now speed-dependent (Glauert) instead of frozen at the hover
  value. Hover power is unchanged.
- Multicopter best-endurance and best-range speeds now report real values.
  Before, the power curve had no minimum, so both pinned to the ends of the
  search range (typically 0.5 m/s and the plot maximum).
- Fixed-wing power changes wherever cruise sits away from ~60% of pitch speed,
  because propeller efficiency now varies with advance ratio.
- Fixed-wing glide distance now uses Cruise Altitude rather than the field
  elevation.

**v2.12.2**
- **Fixed a crash on every multicopter run with a propeller table loaded.**
  The Status tab's new table-range check referenced a metrics variable that
  does not exist in the multicopter, raising `name 'm' is not defined`. It
  only fired when a table was present, and no GUI test loaded one — so the
  whole suite passed while the feature was broken for exactly the users who
  had test data. There is now a GUI test that loads a table via the Browse
  button and runs.

**v2.12.1**
- **Multicopter table runs double-counted the forward-flight inflow.** The
  legacy empirical inflow map was still being applied on top of the Glauert
  correction added in 2.6.0. It only triggered when an RPM was available —
  that is, only with a measured table — so `compute_operating_metrics` and
  `estimate_flight_time_minutes` disagreed by up to 10% for exactly those
  runs, and reported endurance did not match reported power. The map is no
  longer applied; advance ratio and inflow efficiency are still reported as
  diagnostics.
- **Multicopter table runs lost the power bucket.** Reading a static hover
  table directly made power depend on thrust alone, so the curve rose
  monotonically and best-endurance pinned to the search minimum. The table now
  supplies the measured efficiency, which is applied to forward-flight
  momentum theory — the same treatment the fixed-wing got in 2.11.0. Hover is
  unchanged, since the two forms coincide there.
- **New Status check: "Table thrust range".** Warns when the operating point
  sits outside the thrust the table actually measured, and flags it red below
  50% of the table minimum. Pairing a table with a different propeller is easy
  to do by accident and silently turns every table figure into a deep
  extrapolation.

**v2.12.0**
- **Fixed-wing thrust available now falls with airspeed.** It previously
  returned the STATIC bench figure at every speed. On the 3 m reference
  glider that produced a best climb of **3597 m/min at 56 m/s**, using 66 N
  of thrust at a speed where the propeller — pitch speed 25 m/s at its highest
  tested RPM — would be windmilling. The implied climb power was 2469 W
  against a measured maximum of 1680 W, so it broke energy conservation.
  Thrust is now bounded by momentum theory at the available shaft power:
  `P = T*(V + vi)`, solved for T. Best climb becomes **1526 m/min at
  25.8 m/s**, needing 1047 W against 1344 W available.
- **Take-off roll** now evaluates thrust at 0.707 x lift-off speed, the
  standard representative point, instead of assuming a flat 75% of static.
- Climb rate, best-climb speed and take-off distance all change as a result.
  Endurance and range at a fixed cruise speed are essentially unaffected.

**v2.11.2**
- **Propeller-table extrapolation below the measured range is now physical.**
  A polynomial fitted to the measured band and run downward crossed zero: on
  the sample table (1426-6733 g) the operating point read **-12.7 W** with an
  implied 53 g/W.
  Below-range power now uses `P = a*T^1.5 + b`, fitted to the table itself.
  The first term is momentum theory (ideal static power goes as T^1.5); `b`
  is the loss that does not vanish with thrust — motor no-load current, iron
  and ESC quiescent draw. On the sample table this fits to 2.7% with
  b = 20.5 W.
  A pure power law was tried first and rejected: it assumes efficiency is
  constant, but the measured efficiency is already falling (0.508 mid-range
  to 0.441 at the lowest point), so it predicted efficiency rising without
  bound — 40 g/W at 50 g of thrust against a best measured 7.6 g/W. With the
  fixed-loss term, efficiency peaks near 9.3 g/W around 580 g and then
  collapses toward zero, which is what a real motor does.
- The operating-curve chart now says **EXTRAPOLATED** in red, with the
  measured range, whenever the operating point falls outside the table.
- The chart's operating markers use the same lookup as the model, instead of
  a second hand-rolled interpolation that could disagree with it.

**v2.11.1**
- **Fixed the GUI freezing when a propeller table is loaded.** `max_thrust_N`
  called a pandas reduction on the table for every thrust evaluation, and the
  climb-rate and best-speed searches evaluate thrust around a thousand times
  per run. A single point took 49 ms and a 201-point plot sweep took roughly
  10 seconds — long enough for the window manager to report "not responding".
  Table bounds and columns are now cached when the table loads: 4.5 ms per
  point, and the same sweep takes 0.8 s. Results are unchanged, with a test
  asserting the cached and uncached paths agree to 1 part in 10^12.

**v2.11.0**
- **Fixed-wing bench tables now load.** The fixed-wing had its own table
  loader carrying the same `Series.astype(str)` fault fixed in the multicopter
  in 2.6.2, so a vendor export with a sparse title row raised
  `argument of type 'float' is not iterable`.
- **A static bench table is no longer read as cruise power.** Bench data is
  measured at V = 0. At 22 m/s an 18 in propeller needs about 5.7x the static
  ideal power for the same thrust, so reading the table directly overstated
  endurance by roughly 3.8x (190 min against a realistic 33 min).
  The table is now used to derive the *measured* combined motor+propeller
  efficiency, which is then applied to forward-flight momentum theory. That
  keeps the value of real test data — a measured efficiency rather than a
  guessed one — without pretending a static test describes cruise.
  Static and take-off cases still read the table directly, which is correct.
  The multicopter is unchanged: it hovers (genuinely static) and in forward
  flight its discs are nearly edgewise to the airflow.

**v2.10.0**
- **Removed the metric/imperial unit toggle** added in 2.9.0. All output is
  metric again. The converted rows were restored to inline formatting and
  regained the secondary units the toggle had collapsed (knots on speeds,
  miles and nautical miles on distances).

**v2.9.1**
- **Fixed-wing multi-motor power was wrong.** `motor_shaft_power_from_thrust`
  fed the aircraft's TOTAL thrust through a SINGLE propeller disc and never
  consulted the motor count, so a twin, a triple and a single tractor all
  reported identical power. Thrust is now divided per motor before any
  single-propeller calculation, and the result scaled back up. The same fix
  applies to measured prop tables, which describe one propeller.
  Spreading thrust over more discs now correctly lowers induced power: about
  2% for a twin and 3% for a quad on the 2 m reference airframe.
  The multicopter was already correct and is unchanged.

**v2.8.0**
- New **Sensitivity** tab. Scales each design input by ±10% and ±20%, re-runs
  the model, and ranks the inputs by how much they move the chosen output
  (flight time, range or total power). Results appear as a table and a tornado
  chart, widest swing first. An input with zero effect is still listed — that
  is a finding, not an omission.
- New **Compare** tab. Pin the current result as a baseline, then every
  subsequent run shows a signed change and a percentage against it. Rows are
  coloured by whether the change is an improvement, which depends on the
  metric: more range is better, more current draw is not.
- Both tabs read the configuration from the last run rather than re-reading
  the input fields, so they can never disagree with the numbers on screen.

**v2.7.0**
- New **Airframe Diagram** tab in both simulators, after Weight Budget. It
  draws a to-scale plan view from the entered dimensions so propeller overlap
  and tip clearance are visible rather than inferred.
  - *Multicopter*: an equilateral body polygon with one vertex per motor, an
    arm from each vertex, and a propeller disc at every rotor. A coaxial X8 is
    drawn on N/2 arms with a second dashed disc per arm, because that is what
    the aircraft physically is. Overlapping discs are drawn red and the gap is
    reported as a negative number.
  - *Fixed-wing*: wing planform (span x mean chord) with propellers on the
    leading edge — one on the centreline for a single tractor, spread
    symmetrically for multiples. Flags both disc-to-disc overlap and a disc
    reaching past the wing tip.
  - Body and arm dimensions are optional; without them the sketch uses
    proportionate assumptions and says so.

**v2.6.2**
- **Measured propeller/motor CSV tables load again.** A vendor export whose
  first line is a sparse title row (`Test Data,,,,,`) raised
  `'float' object has no attribute 'startswith'`: the header scan relied on
  `Series.astype(str)`, and pandas' newer `str` dtype leaves NaN as a real
  float instead of the string `'nan'`.
- **Fixed `'float' object is not iterable` on any prop-table lookup.**
  Extracting `_fit_propeller_curve` into the shared core dropped its
  `(coeffs, x_min, x_max)` return down to a bare list, so the caller's
  three-way unpack bound `coeffs` to a single float.
- **Motor operating-point plots are back** (thrust vs power/current, thrust
  vs efficiency/RPM). They only render when a measured table is loaded, and
  both bugs above prevented any table from loading.
- **Fixed the fixed-wing startup hang.** The reusable config loader ended
  with a `messagebox.showinfo`; a modal blocks until clicked, so calling the
  loader outside an interactive click froze the app. The confirmation now
  lives in the button handler, where a user is present to dismiss it.

**v2.6.1**
- The fixed-wing Mission/Environment tab now exposes the transient settings
  (`Transient step dt`, `Max accel`, `Max decel`, `Decel regen efficiency`).
  The physics shipped in 2.6.0; only the inputs were missing, so they could
  previously be set from a mission JSON but not from the GUI.
- A `Config:` label on the mode bar names the configuration currently loaded
  and updates on every load. The Output pane is overwritten by the first run,
  so the loaded config needed somewhere permanent to live.
- The first-launch example autoload introduced in 2.6.0 has been **removed**:
  it never worked on the fixed-wing (a definition-order bug hidden by a bare
  `except`), and fixing the ordering exposed a hang during GUI construction.

**v2.6.0**
- Fixed-wing missions now model acceleration and deceleration. A phase that
  commands a different speed spends a ramp segment reaching it, which costs
  time, distance and energy out of that phase's budget. Missions that hold a
  constant speed are unchanged.
- The Metrics tab groups rows into collapsible sections. Which sections you
  leave open is remembered across runs.
- Both simulators auto-load an example configuration on first launch.

**v2.5.0** — no behavioural change. The shared-core extraction is verified
against a 348-value golden snapshot; every number is bit-identical to v2.4.0.

**Earlier**
- Pack capacity no longer scales with the series count (it never should have).
- Fixed-wing cruise power uses forward-flight rather than static momentum
  theory — this alone changed endurance by about 5x.
- Multicopter arm drag is no longer counted twice in forward flight.

### Best endurance or best range reports a speed at the edge of the range

Fixed in v2.4.0. Before that, the multicopter power curve rose monotonically
with speed, so there was no minimum to find and the optimiser returned
whichever bound it started from.

### The window says "not responding" while running

Fixed in v2.11.1 for the propeller-table case. If you still see it, the likely
cause is a very high **Max speed for plot**: the performance charts evaluate
the model at 201 points across that range, so a large value multiplies the
work. Reduce it, or run without a prop table to confirm.

### CSV table not parsing

Confirm it contains **Thrust** and **Power** columns. Extra header rows are
fine. Open the file and check the values are numeric.

---

## Testing

A pytest suite lives in `tests/` — 300 tests covering physics, the shared
core, the CLI, the GUI, and the batch driver. See `tests/README.md`.

```bash
pip install pytest
pytest                        # everything, ~7 minutes
pytest -m "not slow"          # physics only, ~3 seconds
xvfb-run -a pytest            # headless machines (GUI tests need a display)
```

| Suite | Tests | Covers |
|---|---|---|
| `test_golden.py` | 1 | 348 stored numeric outputs; fails if any value drifts |
| `test_core.py` | 76 | The shared core, plus checks that both simulators really delegate to it |
| `test_physics.py` | 97 | Battery topology, atmosphere, rotor inflow, prop efficiency, SoC, drag, landing, drag calculator |
| `test_cli.py` | 64 | Subprocess runs of every argument path and example mission, plus edge cases and malformed input |
| `test_gui.py` | 44 | Real Tk window: hover events, mode toggle, config load, missions, exports |
| `test_batch.py` | 20 | Sweeps, sizing, mode enforcement, and GUI↔CLI consistency |

Almost every test corresponds to a bug that was shipped at some point; the
docstrings name the symptom. Two are worth knowing about: the tooltip test
fires real hover events (an earlier version counted widgets and passed while
every tooltip crashed), and the consistency test asserts that the same config
file produces identical numbers through the GUI and through the batch driver.
