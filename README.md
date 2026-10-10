# RotorWorks UAV Power Simulators

**UASforge / dronefoundry**  
*Multicopter and fixed-wing v2.42.0 · VTOL v1.11.0*

A suite of cross-platform UAV powertrain performance tools:

| Tool | What it does |
|---|---|
| `rotorworks_core.py` | Shared code all simulators import — **must sit beside them** |
| `multicopter-power-sim-gui.py` | Multicopter performance simulator (GUI + CLI) |
| `fixedwing-power-sim-gui.py` | Fixed-wing performance simulator (GUI + CLI) |
| `vtol-power-sim-gui.py` | VTOL simulator — lift+cruise, tiltrotor, tiltwing, tailsitter (GUI + CLI) |
| `rotorworks-batch.py` | Batch driver: parameter sweeps, sizing studies, scripted runs |
| `drag_coefficient_calculator.py` | Measures drag coefficients from photographs |

All three simulators support **single-point analysis** and **time-stepped
mission simulation**, configurable **battery / motor / ESC / prop / avionics
rails / wiring and connectors**, **status limit checks**, **measured
propeller tables**, **sensitivity sweeps**, **design comparison**, and
**plots** including mission time-series. They share the same tab layout, so
moving between them means the same tabs answering the same questions. The
VTOL is both aircraft at once, and reports what the multicopter reports about
its rotors and what the fixed-wing reports about its wing.

---

## Setup

### 1. Clone

```bash
git clone <your-repo-url> rotorworks
cd rotorworks
```

### 2. Create a virtual environment

A venv keeps these dependencies out of your system Python, which matters here
because matplotlib and numpy are easy to break globally.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activate script, either use `cmd` with
`.venv\Scripts\activate.bat`, or allow it for this session:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

Your prompt should now start with `(.venv)`. Everything below assumes it is
active; re-run the activate line in each new terminal.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. tkinter — the one thing pip cannot install

The GUIs use tkinter, which ships **with Python itself** rather than from PyPI.
`pip install tkinter` does not work and is not a typo you have made.

| Platform | What to do |
|---|---|
| **Windows** | Included in the python.org installer. If missing, re-run it and tick *tcl/tk and IDLE*. |
| **macOS** | Included in the python.org build. Homebrew Python needs `brew install python-tk`. |
| **Debian / Ubuntu** | `sudo apt install python3-tk` |
| **Fedora / RHEL** | `sudo dnf install python3-tkinter` |
| **Arch** | `sudo pacman -S tk` |

Check it:

```bash
python -c "import tkinter; print('tkinter OK')"
```

The CLI and the batch driver work without tkinter; only the GUIs need it.

### 5. Verify the install

```bash
python multicopter-power-sim-gui.py --gui
```

Or without a display, straight from an example config:

```bash
python rotorworks-batch.py sweep --sim multicopter \
    --gui-config examples/configs/multicopter_450_survey_4S.json \
    --sweep-var speed --values 10
```

A line ending `rc=0` with a flight time means everything is wired up.

### 6. Run the tests (optional)

```bash
pip install pytest
pytest -m "not slow and not gui"   # no subprocesses or windows, ~2 minutes
pytest                             # everything, ~30 minutes
```

Run from the repository root, where `pytest.ini` registers the test marks.
On a headless machine the GUI tests need a virtual display — `xvfb-run -a
pytest` on Linux. Without one they skip themselves rather than fail. See
`tests/README.md`.

### Keeping files together

`rotorworks_core.py` must sit **in the same directory** as the simulators;
they import it by path, not from the installed packages. Moving a simulator
somewhere else without it will fail at startup.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Simple vs Advanced mode](#simple-vs-advanced-mode)
- [Example configs and missions](#example-configs-and-missions)
- [VTOL simulator](#vtol-simulator)
- [Accuracy: what the validated aircraft show](#accuracy-what-the-validated-aircraft-show)
- [GUI reference](#gui-reference)
- [CLI reference](#cli-reference)
- [Batch driver](#batch-driver-rotorworks-batchpy)
- [Drag coefficient calculator](#drag-coefficient-calculator)
- [Mission JSON format](#mission-json-format)
- [Motor/prop CSV table format](#motorprop-csv-table-format)
- [Modeling notes and assumptions](#modeling-notes-and-assumptions)
- [Troubleshooting](#troubleshooting)
- [Release notes](#release-notes)
- [Testing](#testing)

---

## Project layout

```
rotorworks_core.py              shared: SoC model, atmosphere, wind, wiring,
                                connectors, prop tables, budgets, exports
multicopter-power-sim-gui.py    multicopter simulator (GUI + CLI)
fixedwing-power-sim-gui.py      fixed-wing simulator (GUI + CLI)
vtol-power-sim-gui.py           VTOL simulator, four configurations (GUI + CLI)
rotorworks-batch.py             sweeps, sizing studies, scripted runs
drag_coefficient_calculator.py  drag coefficients from photographs
examples/                       17 aircraft configs, 18 missions
tests/                          752 pytest tests, plus the measured data they read
pytest.ini                      registers the test marks
```

`rotorworks_core.py` holds the aircraft-agnostic code both simulators depend
on: the battery state-of-charge model, the standard atmosphere, wind
resolution, the thermal step, wiring resistance and connector ratings,
propeller bench tables, the weight and power budgets, the mission ground
track, CSV and Excel export, GUI tooltips, and assorted parsing helpers. Two
copies of that code previously drifted apart and caused real bugs — a tooltip
fix that had to be applied twice, and two atmosphere implementations that
disagreed about pressure overrides. **All three simulators import it, so it must
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
pandas>=1.4
matplotlib>=3.6
Pillow>=9.0
openpyxl>=3.0
reportlab>=3.6
```

What each is needed for:

| Package | Needed for |
|---|---|
| numpy, pandas, matplotlib | Core simulation and plotting (required) |
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
python vtol-power-sim-gui.py       --gui
python drag_coefficient_calculator.py
```

Running any simulator with **no arguments at all** also opens the GUI-less
CLI path and will report which required arguments are missing. Pass `--gui`
to go straight to the graphical interface.

Fastest way to get a feel for it:

1. Launch the GUI.
2. **Load Config** → `examples/configs/multicopter_450_survey_4S.json`
3. Press **▶ Run Fixed Speed Sweep**.
4. Read the **Metrics** and **Status** tabs.
5. Change one number and run again.

For the VTOL, load `examples/configs/vtol_2m4_lift_cruise_survey.json`, then
change **Configuration** to `tiltrotor` and run again — same airframe, and
the hover and cruise numbers move in opposite directions.

The VTOL CLI reads a GUI-saved config directly, and any flag given alongside
overrides the value in the file:

```bash
python vtol-power-sim-gui.py --config examples/configs/vtol_trinity_f90_lift_cruise.json
python vtol-power-sim-gui.py --config examples/configs/vtol_trinity_f90_lift_cruise.json \
    --cruise_speed 20 --mission examples/missions/vtol_02_corridor_powerline.json
```

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

All three GUIs open in **Simple** mode. A selector sits directly above the
input tabs.

- **Simple** shows only the inputs needed to size a powertrain
  (multicopter 62 of 94 fields; fixed-wing 58 of 77; VTOL 91 of 161). The
  VTOL's Simple set is every field whose multicopter or fixed-wing
  counterpart is in Simple view, so the three show the same inputs at the
  same level of detail.
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

`examples/` ships with 17 aircraft configs and 18 missions. Load a config with
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

### Real aircraft, validated against published data

Five configs describe aircraft that exist and whose manufacturers publish
enough to check the model against something it did not choose.

| File | Aircraft | Published check | Model | Published |
|---|---|---|---|---|
| `multicopter_dji_m300_rtk.json` | DJI Matrice 300 RTK | Hover, 6.3 kg no payload | **51.0 min** | 55 min |
| `multicopter_dji_m350_rtk.json` | DJI Matrice 350 RTK | ~8 m/s, 6.47 kg | **55.5 min** | 55 min |
| `multicopter_dji_m30.json` | DJI Matrice 30 | Hover, 3.77 kg | **35.7 min** | 36 min |
| `fixedwing_ebee_x_mapping.json` | senseFly eBee X | Endurance / range | **89.8 min / 94.9 km** | 90 min / 95 km |
| `vtol_trinity_f90_lift_cruise.json` | Quantum-Systems Trinity F90+ | Total flight time | **90.0 min** | 90 min |
| `vtol_wingtraone_gen2_tailsitter.json` | Wingtra WingtraOne GEN II | Total flight time | **58.4 min** | 59 min |

**The three DJI aircraft have nothing tuned.** DJI publishes the battery
energy, so each pack comes out at its published Wh by arithmetic, and the
power is momentum theory over the rotor's figure of merit, through the motor
model. The judgement calls are usable capacity, left at the conventional
80%, and the rotor figure of merit, left at the default for the diameter
(0.78 at 21 in, 0.71 at 16 in). That default was not fitted to these
aircraft: its 3, 9 and 15 in points are the size classes the Status check
already used, and 0.78 is a typical large-rotor value. Moving the 21 in point
between 0.75 and 0.80 moves the M300 between 49.3 and 52.1 min.

These are **propulsion-only** figures, as the tests compute them: the files
also carry about 72 W of avionics rails and an ESC, which the GUI counts.
Loaded in the GUI, the M300 hovers 42.7 min, the M350 38.7 and the M30 27.2.
DJI does not say what its avionics draw, so the rails are an assumption to
edit, not data to validate against, and the gap is documented rather than
fixed (audit M14). Usable capacity is the other unknown: DJI does not say
what state of charge its times run to, and 80% is conservative if it flies
closer to empty.

The other three fit exactly one number — `CD0` for the two VTOLs, battery
capacity for the eBee X — and are then checked against a **second** published
figure that was not fitted:

- the **eBee X** pack is sized to the 90 min endurance, and its range comes
  out at 94.9 km against a published 95;
- the **Trinity** drag is fitted to the 90 min endurance, and the resulting
  L/D_max of 16.7 sits near its separately published 14:1 glide ratio. Its
  three-rotor hover also clears the published 4500 m maximum altitude, at
  93% throttle — a third published figure the model was not fitted to.

**The M30 is the scale check.** It is half the M300's weight on 16 in rotors
instead of 21. Overall hover efficiency, ideal momentum power over electrical
power:

| | Model | Real |
|---|---|---|
| M300, 21 in rotors | 64.1% | **69.2%** |
| M30, 16 in rotors | 56.8% | **57.3%** |

Until audit M13 the multicopter had no rotor figure of merit at all: with no
bench table or TConst/PConst it priced each rotor at its ideal momentum
power, and its motor model took the copper loss off the input power instead
of adding it while charging the idle loss at full pack voltage. On the M300
those errors happened to cancel, almost exactly on 55 min; carried down to
the M30 they came out 31% optimistic (47.0 min), with the model's
small-rotor efficiency *above* the large one. With the figure of merit falling with size, the small rotor is
the less efficient one, as real hardware is, and the M30 lands.

**Below about 9 in, trust the default least.** Small-propeller figures of
merit vary widely with Reynolds number and blade design, and none of the
validated aircraft is that small. Load a measured propeller table, or enter
a measured figure of merit on the Propeller tab, and the default is bypassed.

The M300 and M350 together are a **sensitivity** check, which is worth more
than another absolute one: the M350 is heavier (6.47 vs 6.3 kg) on less energy
(526 vs 548 Wh), and the model puts it 7.4% shorter-legged in hover, which is
what those two numbers imply. A model can be right about one aircraft by
luck, but not about the difference between two documented separately. DJI
measures the M350's 55 min at about 8 m/s, so that is where its absolute
check is made; in hover the model gives 47.2 min.

Each file separates `_published` from `_inferred`, and `_calibration` says
exactly what was fitted and to what. Read those before trusting a number.

### Configs — VTOL

| File | Aircraft | Why it is interesting |
|---|---|---|
| `vtol_2m4_lift_cruise_survey.json` | 2.4 m lift+cruise, 6 kg | Generic 4+1; a clean starting point to edit |
| `vtol_trinity_f90_lift_cruise.json` | **Quantum-Systems Trinity F90+** | A real aircraft, validated against its published 90 min endurance |
| `vtol_wingtraone_gen2_tailsitter.json` | **Wingtra WingtraOne GEN II** | A real tailsitter, validated against its published 59 min endurance |

The last two describe aircraft that exist and whose manufacturers publish
endurance figures, so the model can be checked against something it did not
choose. Each file separates what is **published** (span, MTOW, payload,
battery chemistry and cell configuration, cruise speed, endurance) from what
is **inferred** (wing area, drag, motor electrical parameters), because
manufacturers do not publish the latter.

`CD0` is back-solved from the published endurance over a realistic
climb/transition/cruise/land profile flown to a 15% reserve, and both land
within about 1%:

| Aircraft | Model | Published | CD0 |
|---|---|---|---|
| Trinity F90+ | 90.0 min | 90 min | 0.0235 |
| WingtraOne GEN II | 58.4 min | 59 min | 0.033 |

Two things make this more than curve-fitting. The Trinity needed almost no
adjustment — a first-principles guess of 0.024 already gave 88.7 min — and
its calibrated drag implies an L/D_max of 16.7 against a **separately
published 14:1 glide ratio**, so two independent manufacturer numbers agree
on the same drag. And the tailsitter comes out 40% draggier than the clean
pusher layout, which is the right direction and roughly the right size for a
bluff body, exposed motor pods and landing feet carried into cruise.

**Two corrections in v1.11, neither a re-tune.** The v1.11 motor model made
`cruise_eff` mean the propeller alone, so each file's old combined
motor-and-propeller figure was divided by the motor efficiency the model now
computes at that cruise point (`_motor_model_migration` in each file). And
the Trinity now hovers on **three** rotors: its two front motors fold and stop
in cruise, but all three lift in hover. The earlier two-rotor file put hover
at 90% throttle and made hovering at the published 4500 m ceiling impossible;
with three it hovers there at 93% (`_validation`). CD0 was not changed.

### VTOL missions

Four, each a real operational profile rather than a shape chosen to exercise
the code. All were checked to complete on the aircraft they are sized for,
with reserve to spare.

| File | Profile | What it shows |
|---|---|---|
| `vtol_01_lift_cruise_survey.json` | Generic area survey | The energy split across hover, transition and cruise |
| `vtol_02_corridor_powerline.json` | Transmission-line corridor, out and back | Why reciprocal headings do **not** cancel in wind |
| `vtol_03_block_survey_with_hold.json` | 400 ha photogrammetric block with a hold | Translational lift: the hold gets cheaper in wind |
| `vtol_04_delivery_hover_drop.json` | Delivery with a low winch drop | What hovering mid-mission actually costs |

The corridor one is worth running with wind. Outbound and return are
reciprocal, so intuition says the wind should roughly cancel — it does not.
An 8 m/s headwind takes the slow 14 m/s imaging leg from 31 to 72 minutes
while the faster return saves only 8, and the flight goes from 62 to 97
minutes. **A headwind hurts most on the leg you are already flying slowly**,
which is the leg you cannot speed up without ruining the imagery.

The delivery one puts numbers on the VTOL's central trade: on the Trinity the
winch hover costs about 970 W against 116 W in cruise — **more than eight
times** — and the vertical climbs are dearer still at about 1150-1180 W,
because climbing adds the work of raising the aircraft on top of holding it
up.

Change **Configuration** in the GUI (or `--config_type`) to fly any of these
airframes as a different type and compare — but note that doing so is a
thought experiment, not a claim about a real aircraft.

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

## VTOL simulator

`vtol-power-sim-gui.py` models an aircraft that takes off vertically and then
flies on a wing. It shares `rotorworks_core.py` and the tab layout with the
other two, so everything above applies; this section covers what is specific
to it.

### The four configurations

Pick one with the **Configuration** dropdown or `--config_type`.

| Type | How it works | What it costs |
|---|---|---|
| **lift+cruise** | Separate systems: rotors that only lift, a propeller that only pushes | Carries stopped rotors through the whole cruise, dragging and doing nothing |
| **tiltrotor** | The same rotors tilt from vertical to horizontal | Rotors blow onto a flat wing in hover — about a tenth of the weight in download |
| **tiltwing** | The wing tilts with the rotors | Wing turns edge-on to the wash, so the download nearly vanishes |
| **tailsitter** | The whole aircraft pitches from vertical to horizontal | Same, but weathervanes broadside in a crosswind |

The three vectored types carry ONE set of rotors that both lifts and pushes.
Two force balances fix the thrust and where it points at every speed:

```
vertical:    T·cos(tilt) + L_wing = W·(1 + download)
horizontal:  T·sin(tilt)          = D
```

Tilt sweeps 0° in hover to 90° in wing-borne cruise. On the reference
aircraft the trade comes out as you would expect: the tiltrotor pays most to
hover (750 W against 672 for the others) and every vectored type cruises
cheaper than lift+cruise (195 W against 235 at 18 m/s), because lift+cruise
is dragging four stopped rotors the whole way.

**Tiltwing and tailsitter come out identical in still air, deliberately.** In
steady flight they genuinely are close. What separates them is dynamics —
weathervaning, control authority, slipstream lift over a tilting wing — none
of which a steady power model sees. Quantifying slipstream lift needs a
coverage factor there is no data to set, so rather than invent one to force
them apart, both default to the same download and either can be overridden
with measured data.

### Transition

The transition is where a VTOL's energy budget is decided, so it is modelled
rather than charged at a single point. As speed builds, the rotors give up
the lifting and the wing takes it:

| Airspeed | Rotor thrust | Wing lift | Cruise thrust | Power |
|---|---|---|---|---|
| 0.0 m/s | 61.2 N | 0.0 N | 0.0 N | 691 W |
| 6.7 m/s | 45.8 N | 14.8 N | 1.2 N | 339 W |
| 13.3 m/s | 0.3 N | 58.5 N | 5.5 N | 144 W |

The aircraft is also **accelerating**, and that kinetic energy comes from the
pack — reaching 22 m/s is about a third of the transition phase. The landing
transition integrates from the speed the aircraft is actually at, so it is
not charged for accelerating a second time, and regen (if any) applies there.

### Motor electrical model

Every rotor group — the lift rotors, and the cruise motor on a lift+cruise —
runs through a motor model built from **Kv, Rm and the no-load current I0**:

```
RPM     from thrust:  T = C_T·ρ·n²·D⁴·(1 − J/J0),  J0 ≈ 1.2·pitch/diameter
I       = Q/Kt + I0             Q = P_shaft/ω,  Kt = 60/(2π·Kv)
V_emf   = RPM/Kv,   V_term = V_emf + I·Rm,   throttle = V_term/V_pack
P_elec  = P_shaft + I0·V_emf + I²·Rm
```

The thrust coefficient is **TConst** if entered, else fitted from a bench
table's RPM column, else the shared estimate from diameter, pitch and blade
count (±30%). Giving I0's test voltage scales the no-load current with motor
speed, as iron loss does. **Kv = 0 turns the model off**, which reproduces
versions before v1.11 exactly.

What this changes, and why v1.11 results differ:

- **ESC efficiency now means the ESC alone**, and **Cruise prop efficiency the
  propeller alone**. Before v1.11 both had the motor folded in. Divide an old
  cruise-prop figure by about 0.9 to convert it; the shipped examples were
  converted this way and say so.
- **Hover and cruise cost about 10% more** on the reference aircraft, because
  the motor's loss is now charged rather than assumed away.
- **Where a bench table covers the thrust, no motor loss is added.** The
  table was measured at the ESC input, so its power already contains the
  motor; charging it again would count the motor twice.
- Status now checks **motor and ESC current against their ratings** and
  **throttle against 100%** — a motor whose Kv is too low for the pack and
  propeller shows up as more than 100% throttle.

### Outputs carried over from the multicopter and fixed-wing

The VTOL reports both aircraft's figures, computed against its own drag and
propulsion:

| From | Metrics | Status checks |
|---|---|---|
| **Multicopter** | Motor at operating point (RPM, eRPM, current, back-EMF, throttle, efficiency, copper and no-load loss, torque), hover efficiency, achieved figure of merit, thrust-to-weight, max extra payload, payload at TWR 2, hover wind limit, hover tilt split into pitch and roll, tip Mach, solidity, advance ratio | Motor/ESC current and voltage ratings, throttle, temperatures, thrust-to-weight, payload, disk loading, hover efficiency against its disk-loading ceiling, figure of merit, tip Mach, prop RPM, hover wind limit and tilt |
| **Fixed-wing** | CL and CD breakdown, L/D and max L/D, angle of attack, Reynolds number, best endurance / best range / min-sink / best-glide speeds, rate and angle of climb, service ceiling, turning flight, glide distance, conventional take-off roll and landing distance | CL vs CL_max, speed vs stall, L/D, glide ratio, climb against a required rate, ceiling against the cruise altitude, runway lengths, Reynolds number, pitch speed vs cruise, turn stall speed |

The Fixed Speed Plots gain the fixed-wing's **Thrust Required vs Available**,
**Rate of Climb**, **Drag vs Airspeed** and **Drag Polar** panels (eight in
all) and, below them, the **motor operating-point** figure for the lift and
cruise motors. Mission Plots offer about 30 more variables — groundspeed and
wind, kinetic and climb power, loaded voltage, losses, per-motor current, RPM
and throttle, wing CL and L/D, and temperatures integrated through the flight.
A mission now fills the Motor/ESC and Rotor groups on Status with their
worst values, as the multicopter does.

A conventional take-off roll is reported because a lift+cruise can take off
on its pusher like an aeroplane when it is too heavy to lift off vertically.
The Per-Rotor Loading table uses the **drag height above CG**: held in a wind,
drag above the centre of gravity pitches the aircraft and the downwind rotors
carry more.

### Missions

The mission model is **fully time-stepped**: every phase runs through one
loop that steps 0.25 s at a time, ramps speed within the acceleration limit,
moves altitude at the commanded rate, and charges power for what is actually
flown. Pack I²R is billed against the cells per step and scaled by state of
charge, so a long cruise ends drawing more than it began.

Phase kinds: `climb`, `hover`, `descend`, `transition`, `cruise`. A cruise
leg ends exactly when its distance is covered — if it could not reach its
commanded speed in that distance, the status says what speed it ended at
rather than pretending the pattern fits.

### VTOL-specific inputs

Besides these, the VTOL has every input the multicopter and fixed-wing have
that applies to it: motor I0 and ratings, propeller blades and coefficients,
ESC ratings, battery thermal limits, the extra-drag geometry, the coaxial
layout and inflow map, reserve percent, bank angle, climb and descent
commands, runway lengths, minimum climb rate, and hover tilt limits.

| Input | Why |
|---|---|
| **Hover download fraction** | Blank uses the type default. Typical published values, not measurements — override with test data |
| **Lift rotor table (CSV)** | Measured thrust/power replaces the figure-of-merit guess where the table covers the thrust |
| **Cruise prop table (CSV)** | Same for the pusher; ignored for the vectored types, whose rotors do the cruising |
| **SoC curve (CSV)** | A measured discharge curve outranks the chemistry preset |
| **Max acceleration / deceleration** | Blank uses 1.5 / 2.0 m/s², the fixed-wing's defaults. An unlimited acceleration put an unphysical power spike at every transition |
| **Lift rotor positions** | Hub positions from the CG, `x,y; x,y; ...` in metres, one pair per rotor. Blank generates a layout, and the Airframe Diagram says the layout is assumed |
| **Regen efficiency** | Fraction of braking energy recovered. 0 is the honest default for a fixed-pitch propeller |
| **Motor Kv / Rm / I0** | Drive the motor model; Kv = 0 switches it off |
| **Extra airframe drag** | Fuselage, booms or payload pod on top of CD0, as Cd·area or from box-body geometry. Blank adds nothing |
| **Reserve percent** | Sets the reserve on a fixed-speed run and overrides the mission file's |

Missing tables and curves **raise** rather than falling back silently: a
table you think is loaded but is not gives estimates wearing measured
clothes.

### Status checks

Status checks **hover as well as cruise**, because hover is the heaviest
steady load a VTOL carries and the case that sizes the battery, the motors
and the connectors — connector, motor-current, ESC-current and C-rate checks
therefore run at hover current as well as at cruise. It also reports the
transition/stall margin, the wing's lift share at the cruise speed (which
catches a cruise speed set *below* the transition speed, where the rotors are
still lifting), rotor tilt for the vectored types, whether the efficiency in
use is measured or estimated, and the multicopter's and fixed-wing's checks
listed above. Four groups: Battery, Motor / ESC, Rotor / Propeller and
Aerodynamic.

### What is not modelled

- **Tiltwing slipstream lift** and **tailsitter weathervaning**, as above.
- **Control authority and attitude dynamics** through the transition. The
  model says what a transition costs, not whether it is flyable.
- **Rotor spin-down time** — stopped rotors are assumed stopped.
- **Real thermal behaviour.** Temperatures are a lumped estimate — each unit
  warms toward ambient plus a fixed thermal resistance times its loss — good
  for spotting a part heading for trouble, not for predicting its
  temperature.
- Boom positions in the Airframe Diagram are a plausible arrangement, not a
  claim about a specific airframe. Only span, wing area and propeller
  diameters are real inputs, so it is for checking proportions and
  clearances, not a layout to build from.
- **Wing–nacelle interference**, and there is a measured number for how much
  that costs. NASA flight-tested the GL-10 "Greased Lightning" tiltwing — a
  3.17 m, 28 kg distributed-propulsion aircraft with eight motors on the
  wing — and published the result in NASA/TM-2017-219794:

  | | L/D max |
  |---|---|
  | NASA conceptual design tools predicted | 16 |
  | **Measured in flight test** | **7.2** |

  This model is in the same class of tool and predicts 16–17 for that
  geometry, so it would repeat the same 2.2× error. The TM attributes the gap
  to "the complexity of wing and nacelle interactions" — exactly what a
  lifting-line model with a flat-plate CD0 cannot see.

  The practical consequence: on an aircraft with **many nacelles spread
  across the wing**, expect this model to be optimistic about cruise, and
  calibrate `CD0` against measured endurance rather than trusting a
  first-principles estimate. The two shipped real-aircraft examples both have
  two or three clean nacelles, which is why their first-principles drag was
  close to right. Reaching the GL-10's measured 7.2 would need a `CD0` of
  0.165 — 5.5× a clean-airframe value, and not a number that means anything
  physically.

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
| **Wiring** | Main lead length, gauge (or measured Ω/m) and insulation temperature limit; connector type, current and voltage ratings for battery / ESC / motor |
| **Mission / Env** | Speed, wind, altitude, temperature, reserves, or a mission JSON |

The VTOL splits the Motor and Prop tabs into **Lift Rotors** and **Cruise**,
since it carries two propulsion systems, and adds a **Plot Settings** tab for
the sweep range.

### Output tabs

| Tab | Contents |
|---|---|
| **Metrics** | Full single-point summary, grouped into sections (see below) |
| **Status** | Colour-coded limit checks: green OK, yellow warn, red violation |
| **Weight Budget** | Component mass table plus a stacked-bar chart |
| **Power Budget** | Where every watt goes: delivered, losses, and the total from the cells |
| **Airframe Diagram** | To-scale plan view; propeller overlap and tip clearance |
| **Mission Diagram** | Ground track with numbered waypoints and yaw arrows, plus the altitude profile |
| **Sensitivity** | Ranks inputs by influence on flight time, range or power |
| **Compare** | Deltas against a pinned baseline configuration |
| **Fixed Speed Plots** | Performance curves vs speed |
| **Mission Plots** | Time-series for a mission run, multi-unit y-axes |

After a **mission** run, **Status** shows the WORST value each check reached
anywhere in the flight and **Metrics** shows the LAST instant flown; both say
which. The Power Budget and Fixed Speed Plots clear with a note, because a
mission has no single operating point. **Sensitivity** clears whenever a new
run makes it stale.

**Metrics** sections: Battery, Motor @ Operating Point, Total Drive & Power,
Propeller & Rotor, Flight Performance, Propulsion Efficiency, Thermal
Estimates, Environment & Design. The fixed-wing adds Aerodynamics at Cruise,
Turning Flight, Climb Performance, Optimal Speeds, and Endurance & Range. The
VTOL carries both sets — Aircraft, Speeds, Hover, Cruise, Wind, Battery,
Systems Load, Thrust & Power, Aerodynamics, Climb & Glide, Turning Flight,
Conventional Take-off / Landing, Lift Motor, Cruise Motor, Propellers &
Rotors, Thermal Estimates and Environment.

**Status** groups: Battery, Motor/ESC, and Propeller (multicopter) or
Aerodynamic (fixed-wing); the VTOL has both, as Rotor / Propeller and
Aerodynamic. Checks include voltage sag, C-rate vs rating,
hover/forward thrust-to-weight, disk loading, hover efficiency, figure of
merit, wind resistance, ESC S-rating vs pack, stall margin, CL margin,
Reynolds number, and service ceiling.

### Buttons

| Button | Action |
|---|---|
| **▶ Run Fixed Speed Sweep** | Evaluate one operating point and sweep the speed range |
| **📋 Run Mission (JSON)** | Run the loaded mission phase-by-phase |
| **💾 Save Config** / **📂 Load Config** | JSON round-trip of every field |
| **📊 Export CSV** | Performance sweep + metrics |
| **📗 Export Excel** | Three sheets: sweep, metrics, weight budget |
| **📄 Generate Report** | PDF: inputs, metrics, colour-coded checks, all plots |

UI scale is under **View → UI Scale** (150–200% helps on high-DPI displays).

---

## CLI reference

All three simulators run headless with no `--gui`. Key arguments:

### Shared

| Argument | Units / values | Notes |
|---|---|---|
| `--weight` | grams | Base weight **excluding** payload |
| `--payload_mass_g` | grams | Added on top of base weight |
| `--mass_mode` | `derive airframe` \| `enter airframe` | Multicopter and fixed-wing. `enter airframe` builds the weight from `--airframe_mass`, `--avionics_mass` and the component weights, as the GUI does |
| `--airframe_mass`, `--avionics_mass` | grams | Bare structure; avionics |
| `--translation_direction_deg` | degrees | Multicopter: direction of travel from the nose |
| `--metrics_json` | path | Also write the run's metrics, at full precision, as JSON. The batch driver reads this file rather than scraping the console |
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
`--body_height_m`, `--arm_length_m`, `--arm_width_m`,
`--prop_figure_of_merit` (rotor hover figure of merit, 0.2–0.9; blank uses
the typical value for the diameter; not used with `--prop_table` or
TConst/PConst)

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

### VTOL-specific

Every VTOL flag defaults to blank, and the default is applied once, exactly
as for a blank GUI field — the GUI, the CLI and the batch driver all build
the aircraft through one function (`config_from_fields`), so a config gives
the same answer through each. `--help` lists every flag.

| Argument | Units / values | Notes |
|---|---|---|
| `--config` | path | A GUI-saved config. Flags given as well override its values |
| `--config_type` | lift+cruise, tiltrotor, tiltwing, tailsitter | Which VTOL layout to fly |
| `--num_lift_rotors`, `--lift_prop_diameter` | count, inches | The lifting rotors |
| `--lift_figure_of_merit` | 0.2–0.9 | Ignored where a bench table covers the thrust |
| `--num_cruise_motors`, `--cruise_prop_diameter` | count, inches | Ignored for the vectored types |
| `--cruise_prop_efficiency` | 0–0.95 | Also the cruise end of the vectored efficiency blend |
| `--stopped_rotor_drag_area` | m² | Lift+cruise only: what the stopped rotors drag |
| `--hover_download` | fraction of weight | Overrides the per-type default |
| `--lift_prop_table`, `--cruise_prop_table` | CSV path | Measured thrust/power |
| `--soc_curve` | CSV path | Measured pack discharge curve |
| `--wind`, `--wind_direction` | m/s, degrees FROM | Wind for a mission |
| `--max_accel`, `--max_decel` | m/s² | 0 or omitted uses 1.5 / 2.0 |
| `--lift_rotor_positions` | `x,y; ...` m | Lift rotor hubs from the CG; omitted generates a layout |
| `--regen_eff` | 0–1 | Braking energy recovered; 0 is the honest default |
| `--wire_length`, `--wire_awg`, `--wire_ohm_per_m` | m, AWG, Ω/m | Main battery lead |
| `--wire_temp_limit` | °C | Lead insulation limit for the temperature check; default 150 |
| `--connector_batt_cont`, `_max`, `_volt` (and `esc_`, `motor_`) | A, A, V | Connector ratings, checked at hover |
| `--battery_c_cont`, `--battery_c_max` | C | Pack discharge ratings |
| `--lift_motor_kv`, `--lift_motor_resistance`, `--lift_motor_i0` (and `cruise_…`) | rpm/V, Ω, A | The motor model; Kv 0 turns it off |
| `--lift_motor_max_current`, `--esc_cont_current`, `--esc_max_current` | A | Checked on Status |
| `--lift_prop_blades`, `--lift_prop_tconst` (and `cruise_…`) | count, C_T | Set RPM from thrust |
| `--reserve_percent` | % | Overrides the mission file's reserve |
| `--bank_deg`, `--climb_rate_mps`, `--descent_rate_mps` | deg, m/s | Turn figures and a commanded climb at the cruise point |
| `--parasite_drag`, `--parasite_area`, `--body_width_m` and friends | Cd, m², m | Extra airframe drag beyond CD0 |
| `--field_takeoff_m`, `--field_landing_m`, `--min_climb_mps` | m, m/s | Requirements checked on Status |

```bash
# Same airframe, four ways
for t in lift+cruise tiltrotor tiltwing tailsitter; do
  python vtol-power-sim-gui.py --config_type "$t"
done

# A survey mission into a 6 m/s headwind, with measured rotors
python vtol-power-sim-gui.py \
  --mission examples/missions/vtol_01_lift_cruise_survey.json \
  --battery_parallel_cells 3 --wind 6 \
  --lift_prop_table tests/data/motor_prop_table.csv
```

---

## Batch driver (`rotorworks-batch.py`)

Three subcommands, all of which accept `--gui-config` and `--mode`.
`--sim` takes `multicopter`, `fixedwing` or `vtol`.

Each run's metrics come from the simulator's own `--metrics_json` file, at
full precision and under the simulator's own names. The console text is
still captured as `raw_` columns. For the VTOL, `flight_time_min` and
`flight_range_km` are the cruise endurance and range; the hover endurance
is `hover_endurance_min`. A `--gui-config` keeps the mass mode, the
multicopter's translation direction and the fixed-wing's propeller
efficiency model, so batch flies the configuration the GUI saved.

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
into `--avionics_voltage_tree`). For the VTOL the translation table is read
from the simulator itself, so it cannot drift from what the GUI builds —
before v1.11 the batch driver's own copy lacked CD0, CL_max and the
transition CL cap, and silently flew the CLI's default drag.

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
| `top_area` | Simulator, from the **top** view |

A translating multicopter is tilted, so forward drag uses
`parasite_area·cos(tilt) + top_area·sin(tilt)`, with the tilt solved together
with the drag. Without a top area the simulator estimates one from the body
and arm dimensions.

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

The VTOL uses the same file with one addition: each phase carries a `kind` —
`climb`, `hover`, `descend`, `transition` or `cruise` — because a VTOL's
phases are not all the same sort of flying. A `transition` phase is
integrated across its speed range and charged for the acceleration; a
`cruise` phase ends exactly when its distance is covered. `course_deg` is
used for wind resolution and to draw the ground track. A **Reserve percent**
entered on the VTOL's Mission/Environment tab (or `--reserve_percent`)
overrides the file's `reserve_percent`.

```json
{
  "reserve_percent": 20,
  "transition_time_s": 12,
  "phases": [
    {"name": "Climb",      "kind": "climb",      "duration": 30,
     "climb_rate_mps": 2.0, "altitude": 80},
    {"name": "Transition", "kind": "transition", "duration": 12,
     "speed": 22.0, "altitude": 80},
    {"name": "Survey leg", "kind": "cruise",     "distance": 2000,
     "speed": 22.0, "altitude": 80, "course_deg": 90},
    {"name": "Land",       "kind": "descend",    "duration": 30, "altitude": 0}
  ]
}
```

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

## Accuracy: what the validated aircraft show

Four aircraft have been checked against published manufacturer figures using
**first-principles inputs only** — no drag or efficiency fitted to them.
Ordered by how optimistic the untuned model turned out to be:

| Aircraft | Simulator | Size | Model / real |
|---|---|---|---|
| DJI Matrice 300 RTK | multicopter | 21 in rotors | **0.93** |
| Trinity F90+ | VTOL | 2.39 m span | **0.99** |
| DJI Matrice 30 | multicopter | 16 in rotors | **0.99** |
| Wingtra WingtraOne GEN II | VTOL | 1.25 m span | **1.10** |

The multicopter figures are propulsion-only (see the validation section
above for what the files' avionics rails do to them).

**Efficiency falls with size, and only the multicopter models that.** A
propeller's figure of merit falls with Reynolds number, and small motors and
ESCs are less efficient than large ones. The multicopter's rotor figure of
merit now falls with diameter (audit M13), and its two aircraft land within
7%. Before that it priced rotors as ideal: the M300 came out at 1.00 by a
cancellation of errors and the M30 at 1.31. The VTOL's lift figure of merit
is a flat 0.65 unless you enter one, and the smaller of its two aircraft is
10% optimistic.

**How to use this.** For aircraft in the 5 kg and up, 20 in rotor or 2 m span
class, expect the untuned model to be close. Below that, and below about 9 in
rotors on a multicopter, expect more scatter, and either:

- load a **measured propeller table** (Propeller tab, or `--lift_prop_table`
  / `--prop_table`), which bypasses the efficiency estimate entirely; or
- **enter or calibrate the figure of merit** (`--prop_figure_of_merit`,
  `--lift_figure_of_merit`) or `CD0` against a measured endurance, as the
  WingtraOne and eBee X configs do — each says in its `_calibration` block
  exactly what was fitted.

The multicopter's default is a curve through typical values for each size
class, not a fit to these aircraft. Moving it a few hundredths moves the
answers a few percent; a measured figure for your own propeller is always
better.

---

## Modeling notes and assumptions

### VTOL

- **Vectored types** (tiltrotor, tiltwing, tailsitter) resolve one rotor
  thrust into a lifting and a pushing component:
  `T = sqrt(D² + (W·(1+download) − L_wing)²)`, pointed at
  `tilt = atan2(D, W·(1+download) − L_wing)`. Rotor efficiency blends from
  the hover figure of merit to the propeller efficiency with the tilt, so a
  rotor is judged as a rotor when lifting and as a propeller when pushing.
- **Download** — the rotor wash striking structure below — scales with the
  rotors' share of the weight, so it is full in hover and gone once the wing
  carries the aircraft. That keeps hover and cruise the two ends of one
  continuous curve rather than two branches that disagree where they meet.
  Per-type figures are typical published values, not measurements.
- **Lift+cruise** carries its stopped rotors as drag through the cruise,
  phased in over the transition rather than switched on at a threshold.
- **Transitions** are integrated across their speed range and charged for the
  kinetic energy of accelerating; the landing transition integrates from the
  speed actually being flown, so it is not billed for accelerating twice.
- **Missions are time-stepped** at 0.25 s. Pack I²R is charged against the
  cells per step and scaled by state of charge.
- **Motors** have their own electrical model (see
  [Motor electrical model](#motor-electrical-model)); rotor aerodynamic losses
  are in the figure of merit and propeller efficiency. The Power Budget shows
  motor copper and no-load losses as their own lines.
- **ESC**: the efficiency covers everything lost under load. Its resistance
  does not add a loss; it splits that loss into conduction and switching. Its
  idle current is a standby draw on every ESC, so a lift+cruise pays it for
  the stopped lift ESCs through the whole cruise.
- **Extra drag** beyond CD0 uses the multicopter's convention: the frontal
  (parasite) term in forward flight, the side (profile) term hovering in a
  wind. A lift+cruise's booms meet the airflow end-on in cruise.
- **Coaxial lift rotors** take the multicopter's spacing-dependent penalty,
  eased by forward speed; the rotor **inflow map** is the multicopter's, off by
  default because the forward-flight inflow solver already carries
  translational lift.
- **Climb, ceiling, turns, glide and runway** figures are the fixed-wing's
  formulas against the VTOL's own drag and forward thrust. Thrust available is
  momentum-bounded: the ideal power at the static maximum is held fixed, so
  thrust falls with speed.
- **Temperatures** are a lumped model per unit — the multicopter's constants
  spread over its four motors and ESCs. Fixed-speed runs report steady state
  (hover for the lift side, cruise for the cruise side); missions integrate
  in time.

### Multicopter

- Forward-flight tilt: `tilt = atan(D / W)`, with `max_tilt_deg` enforced.
- Rotor inflow uses Glauert's forward-flight momentum theory:
  `vi = v_h² / sqrt((V·cos a)² + (V·sin a + vi)²)`, solved iteratively, where
  `v_h = sqrt(T / (2·ρ·A))` and `a` is the disk incidence (the tilt angle).
  At `V = 0` this reduces to the hover value; at speed the rotor meets air
  that is already moving, so induced power falls sharply.
- Ideal power is `P = T·(V·sin a + vi)`. The first term is the propulsive
  power overcoming airframe drag — by the tilt balance `T·sin a = D`, so it
  equals `D·V` exactly. Together these produce the **power bucket**: a
  minimum roughly 10–25% below hover power somewhere around 8–14 m/s, which
  is what sets the real best-endurance and best-range speeds.
- **Rotor figure of merit.** Without a bench table or TConst/PConst, shaft
  power is the ideal over the rotor's hover figure of merit: the value on the
  Propeller tab (`--prop_figure_of_merit`, 0.2–0.9), else a default that
  falls with diameter — 0.45 at 3 in, 0.60 at 9 in, 0.70 at 15 in, 0.78 at
  21 in and up, linear in between. In forward flight the hover loss
  `T·v_h·(1/FoM − 1)` is split into blade profile power, which does not fall
  with speed, and an induced-loss factor that scales with `vi`; the
  propulsive term is not charged it. Until audit M13 the rotor was ideal.
- **Motor model.** `I = Q/Kt + I0`, `Kt = 60/(2π·Kv)`, at the rotor's own
  RPM, and `P_in = (rpm/Kv + I·Rm)·I`, i.e. shaft power plus idle and copper
  loss. The motor's current and power ratings are Status checks, not clips
  on the power. With no Kv the motor is a lumped 85%.
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
  sag worse late in a flight. Set `--battery_soc_model linear` to replace the
  curve with a straight line from full charge to the cutoff (it used to hold
  full-charge voltage for the whole flight; audit C5). Mission runs
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
Version** — the current VTOL is **v1.11.0**, and the current multicopter and
fixed-wing are **v2.42.0**. The version also appears in the window title bar
and at the top of the Output pane on startup.

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

The VTOL builds the aircraft through one function for all three paths as of
v1.11, and its CLI takes a GUI-saved file with `--config`. The multicopter
CLI, before this release, evaluated **hover** at whatever `--speed` was given
while the GUI evaluated it stationary; both now hold hover at 0 m/s and say
so in the output.

### Efficiency or endurance changed after updating

Several physics corrections have landed. Results from older versions are not
comparable — this is expected, not a regression. The
[release notes](#release-notes) say what changed in each version.

**VTOL endurance dropped by about 10% in v1.11.** That is the new motor
model charging the motors' own losses, which earlier versions assumed away.
If you entered a combined motor-and-propeller efficiency in **Cruise prop
efficiency**, divide it by about 0.9 — the field is now the propeller alone.
Setting both motor Kv values to 0 reproduces the old numbers exactly.

### `UnicodeEncodeError: 'charmap' codec can't encode character` (Windows)

Fixed in this release. On Windows, output that goes to a pipe or a file is
encoded with the ANSI code page (cp1252), which has no μ, Ω or →, so a run
whose output was redirected — `... > results.txt`, the batch driver, or the
test suite — crashed partway through its report. Every simulator now prints
`?` for a symbol the console cannot show instead. Setting
`PYTHONIOENCODING=utf-8` gets the real symbols.

### Tests fail with `FileNotFoundError: ...tests/data/...`

`tests/data/` is missing from your copy. It holds the two bench tables and
the discharge curve the table tests read, and it was not in the repository
before this release. See `tests/README.md` for what goes in it.

### Best endurance or best range reports a speed at the edge of the range

Fixed in v2.4.0. Before that, the multicopter power curve rose monotonically
with speed, so there was no minimum to find and the optimiser returned
whichever bound it started from.

### A mission leg reports "Invalid" but the same speed flies fine single-point

Fixed in v2.13.2 for the wind case. A leg was judged on its groundspeed at
the instant it started accelerating, rather than at its commanded speed.

If you still see it, check that the commanded airspeed for that leg genuinely
exceeds the headwind component — the status message now says which case it is.

### The window says "not responding" while running

Fixed in v2.11.1 for the propeller-table case. If you still see it, the likely
cause is a very high **Max speed for plot**: the performance charts evaluate
the model at 201 points across that range, so a large value multiplies the
work. Reduce it, or run without a prop table to confirm.

### CSV table not parsing

Confirm it contains **Thrust** and **Power** columns. Extra header rows are
fine. Open the file and check the values are numeric.

---

## Release notes

**VTOL v1.11.0** — a motor model, and the multicopter's and fixed-wing's inputs and outputs
- **Kv and Rm now do something.** They were stored and never read; motor
  losses were assumed folded into the ESC efficiency. A Kv / Rm / I0 motor
  model now gives RPM, current, back-EMF, throttle and the motor's own losses
  for every rotor group. **ESC efficiency is now the ESC alone and Cruise prop
  efficiency the propeller alone**; the shipped examples were converted, and
  Kv = 0 reproduces the old chain exactly. Hover and cruise cost about 10%
  more on the reference aircraft.
- **Current ratings are checked.** Lift and cruise motor current and ESC
  continuous and burst current now have Status rows; before, the fields
  existed and nothing read them. ESC resistance, likewise inert, now splits
  the ESC loss into conduction and switching.
- **The duplicate Avionics power and ESC efficiency fields** on
  Mission/Environment are gone. Configs saved with them still load them onto
  the Avionics and ESC tabs.
- **Reserve percent** is an input: the reserve on a fixed-speed run, and an
  override of a mission file's.
- **One builder for GUI, CLI and batch.** Every flag defaults to blank, the
  CLI takes a GUI-saved file with `--config`, and the batch driver reads the
  VTOL's field map from the simulator. Its own copy had no CD0, CL_max or CL
  cap, so batch runs of VTOL configs had been flying the CLI's default drag.
- **About 90 inputs** carried over from the multicopter and fixed-wing, and
  the Simple view now shows every field whose counterpart is Simple in either
  (91 of 161).
- **Outputs** carried over from both: Metrics sections for thrust and power,
  climb and glide, turns, runway, the motors, propellers, thermal and
  environment; the matching Status checks; four more Fixed Speed panels and
  the motor operating-point figure; about 30 more mission variables;
  Sensitivity and Compare extended; a mission now fills the motor, ESC and
  rotor groups on Status with their worst values.
- **The Trinity F90+ example hovers on three rotors.** It had two, which the
  motor model exposed: 90% hover throttle and no hover at the published
  4500 m ceiling. It now flies 90.0 min against the published 90.
- Regen during a mission was over-credited: recovered energy was divided by
  the drive efficiency instead of multiplied by it.
- **Tests and tooling (all simulators).** Every CLI now survives a Windows
  console that cannot encode μ or Ω; `tests/data/` and `pytest.ini`, which the
  suite always needed and the repository never had, are now included; the
  golden snapshot gained 694 VTOL values; 26 new VTOL tests. The multicopter
  CLI now evaluates hover stationary, as its GUI does, and both say which
  orientation ran.
- **Inputs a dropdown makes irrelevant are greyed out**, as on the multicopter
  and fixed-wing: in cell mode the pack fields (cells in series and parallel
  per pack, pack capacity and weight), in pack mode the cell fields; whichever
  mass the Mass Entry Mode calculates; the cruise motor and propeller on the
  vectored types; coaxial spacing on a flat layout; the drag fields the drag
  mode does not use; and the SoC curve on the linear model. Greying out
  changes no result.
- **"derive structure" / "enter structure" are now "derive airframe" /
  "enter airframe"**, the weight field reads "All-up weight without payload"
  and the structure field "Airframe mass". Configs saved with the old names,
  and the old `--mass_mode` values, still load.
- **Wiring shares the multicopter's and fixed-wing's model**: the new wire
  temperature limit and connector voltage inputs, the same Status rows (voltage
  drop, wire temperature, connector current and voltage, at hover), and the
  motor throttle judged at the voltage left after the lead's drop.

**VTOL v1.10.0** — the other simulators' tabs and inputs
*Reconstructed from the tests added with it; the release's own notes were
lost when that commit overwrote this README.*
- Avionics rail editor with regulator losses (rails replace the flat figure;
  peripheral current adds to them), and an ESC tab.
- Battery pack-entry mode, amp limits that outrank the C-rate, charge time,
  energy density, an SoC-model dropdown and hand-entered breakpoints.
- Weight and Power Budget share charts; Status split into four sub-tables;
  Metrics grouped into collapsible sections with an explanation column, and
  speeds in km/h and knots.
- Mission Plots covering the whole model; the Airframe Diagram's per-rotor
  loading table and dimension annotations; Sensitivity showing every
  perturbation level; a Plot Settings tab.
- Structure-mass entry mode, a View menu (window scale, plot size, fonts,
  presets), scrollable input tabs, and tab and button order matching the other
  two simulators.
- Wind-aware fixed-speed sweeps and station keeping, a pressure field, and a
  2x2 sweep figure.

**VTOL v1.0.0** — fully time-stepped mission model
The VTOL now integrates its missions in time, as the multicopter and
fixed-wing do. Every phase runs through ONE loop that steps 0.25 s at a
time, ramps speed toward the target within the acceleration limit, moves
altitude at the commanded rate, and charges power for the speed and climb
actually flown.
- **Special cases removed, not added to.** The phase-level model evaluated
  each phase at a single point with exceptions layered on: sub-steps for
  transitions, a lead-in for cruise legs, a separate branch for hovering in
  wind. Those disagreed with each other — a transition paid for accelerating
  while a cruise leg did not. One loop removes the category.
- **Legs land exactly on their goal.** The final step is truncated and
  clamped, so a 5000 m leg covers 5000 m rather than 5001.5, and a survey's
  legs sum to what was asked for.
- **A leg can now END before reaching its speed** instead of overshooting its
  own distance to get there. The status reports the speed it actually ended
  at. That is the real behaviour of an unflyable pattern.
- **Pack I2R is charged against the cells, per step, scaled by state of
  charge.** It was previously shown in the Power Budget but never billed. On
  an 18 km cruise the draw now climbs from 349 W to 364 W as the pack empties
  and its resistance rises — 3441 series points where the old model had
  about ten.
- **A test premise turned out to be wrong, and the physics is the
  interesting part.** "Transients cost energy" is not generally true: once
  the ramp is PART of the leg rather than charged on top of it, the aircraft
  spends that time at a lower speed, and a VTOL's power rises steeply with
  speed — 138 W at 14 m/s against 430 W at 24. On a four-leg pattern
  ramping at 1.5 m/s2 costs 16.90 Wh against 17.26 Wh flown instantly, while
  taking longer. The honest invariants are the track and the time.

**VTOL v0.10.0** — the transition pays for its own acceleration
Checking whether the transition was fully characterised turned up two faults.
- **The thrust hand-over was already right.** Across the transition the
  rotors give up the lifting and the wing takes it, continuously: on the
  reference aircraft rotor thrust runs 61.2 N -> 0 while wing lift runs
  0 -> 58.8 N, with a cruise thrust appearing to beat drag. That part needed
  no change.
- **The acceleration was missing.** The transition integrated power across a
  sweep of SPEEDS with no kinetic term, as though the aircraft arrived at
  each speed for free. It does not: reaching 22 m/s costs 1452 J, which is
  **31% of the whole transition phase** and was simply absent. The cruise
  legs had paid it since v0.9.0; the transition had not.
  It is charged as mechanical power through the ESC, like any other shaft
  power, rather than added straight to the pack draw.
- **Both directions used to sweep 0 -> v_end**, so the landing transition was
  modelled as another acceleration and the kinetic cost was charged TWICE per
  round trip instead of once out and released on the way back. Transitions
  now integrate from the speed the aircraft is actually at: outbound
  411 W / 1.371 Wh against 216 W / 0.719 Wh inbound, and regen applies only
  to the deceleration, where there is something to recover.
- **Four new tests**, including one that checks the power gap between an 8 s
  and a 20 s transition matches the kinetic energy spread over those times —
  so the term cannot silently drift.
- **VTOL tests: 96 -> 100.**

**VTOL v0.9.0** — transients and the measured SoC curve
The last two items. The VTOL now has everything the other two simulators do.
- **Measured SoC curve (CSV).** I said last release this needed the mission
  loop rebuilt. That was wrong: `configure_battery_soc_model` already took a
  `curve_csv` and the VTOL was passing `None`. It needed wiring, not
  rebuilding. A measured curve outranks the chemistry preset, so the sag near
  the end of the pack comes from your cells — 3.63 V/cell at 10% SoC against
  the preset's 3.50.
- **Transients**: acceleration and deceleration limits with optional regen.
  Accelerating costs power on top of steady drag, and a survey flown as short
  legs with a speed change at each end pays it repeatedly — on a 4-leg
  pattern, 19.15 Wh becomes 23.81 Wh at 1 m/s². Regen recovers a little
  (23.24 Wh at 30%); the default recovers nothing, which is honest for a
  fixed-pitch propeller. Left blank, every number is bit-for-bit what it was.
- **Two faults caught while building it.** The lead-in first added its
  distance ON TOP of the leg, so a 400 m leg flew 400 m *after* accelerating
  and the mission grew from 2.76 to 5.14 km. And a leg too short to reach its
  speed — 60 m to slow from 24 to 14 m/s at 1 m/s², which needs about 190 —
  now reports the overshoot in its phase status rather than quietly clamping,
  because an unflyable pattern is worth knowing about.
- **A missing SoC curve now raises.** The core resolver falls back to the
  chemistry preset when a curve cannot be read, which would give preset
  numbers wearing measured clothes. It fails loudly instead, as the
  propeller tables already did.
- **VTOL tests: 88 -> 96.**

**VTOL v0.8.0** — hover download applied properly to lift+cruise
Clearing the debt left in v0.7.0, where the download was removed rather than
fixed because applying it to hover alone put a step at zero airspeed.
- **Download now scales with the rotors' share of the weight** through the
  lift+cruise transition, exactly as the vectored model already did. It
  exists because rotor wash strikes structure below, and there is less wash
  to strike with the less the rotors are lifting — so it fades as the wing
  takes over. Hover (share 1, full download) and wing-borne cruise (share 0,
  none) become the two ends of one continuous curve instead of two branches
  that disagree where they meet.
- **Verified continuous**: 619.7963 W at V=0 and at V=1e-9, identical, and
  the worst sampled step falls from 6.50% at 0.25 m/s to 0.84% at 0.031 m/s —
  the proportional shrink that distinguishes a steep slope from a jump.
- Lift+cruise hover power rises from 585 W to 620 W on the reference
  aircraft. That is the correction, not a regression: the previous figure
  omitted a real force.
- **Four tests updated, and two of them were the thing that was wrong.**
  Hover thrust is weight x (1 + download), not weight. And climb work uses
  the FULL hover thrust: I first wrote weight x rate, reasoning that download
  is a thrust penalty rather than extra mass, but download is a real downward
  force on the structure and a climbing aircraft moves up against it, so that
  force does work too. The model was right both times.

**VTOL v0.7.0** — Simple/Advanced, CLI parity, and a download that was never applied
This closes the port: the VTOL now has every feature the other two simulators
have, apart from transients and the SoC curve CSV, which its phase-level
mission model cannot use without being rebuilt first.
- **Simple / Advanced input toggle.** Simple shows 35 of the 65 inputs — the
  weight, wing, pack, rotor and propeller sizes and the environment. It hides
  figure-of-merit tuning, stopped-rotor drag area, bench tables, connector
  ratings and the download override, all of which have sensible defaults and
  exist to be refined later. A view setting only: hidden fields keep their
  values, and a test asserts the computed answer is identical in both modes.
- **CLI parity, 42 -> 59 arguments.** The wiring, connector, rating, prop-mass
  and download inputs were GUI-only, so a config saved from the GUI described
  an aircraft the CLI could not express.
- **Found while testing the new CLI flag: the hover download was never
  applied to lift+cruise.** `HOVER_DOWNLOAD_FRACTION` listed 0.04 for it from
  the start, and only the vectored types ever used it, so the entry read as
  modelled when it was not and `--hover_download` was silently a no-op there.
  Applying it turned out to break something real: lift+cruise computes hover
  and transition in separate branches, so a download on one alone puts a STEP
  at zero airspeed — 1.04x weight at V=0 against 1.00x just above it. The
  continuity test caught it in the same run.
  Doing it properly means scaling download by rotor lift share across the
  lift+cruise transition, as the vectored model already does. Until then the
  entry is REMOVED rather than left listing a number that is never applied,
  and the help text, the CLI flag and a test all say the flag applies to the
  vectored types only.
- **VTOL tests: 85 -> 88.**

**VTOL v0.6.0** — exports, menu bar and Airframe Diagram
- **Menu bar**: File / View / Help, matching the other two simulators. File
  carries Load and Save Config plus the three exports; View has Window Scale;
  Help has About / Version.
- **Export CSV, Export Excel and Generate PDF Report.** The VTOL could not
  produce a deliverable at all before. The exporters are shared in
  `rotorworks_core.py` — CSV stacks titled tables in the layout the other
  simulators already produce, so existing scripts read it unchanged; Excel
  gives each table its own sheet.
- **Exports are built by reading the tables on screen**, so a file cannot
  disagree with what the user is looking at — there is only one source. They
  also follow the run type: a single point exports the Power Budget and a
  speed sweep, a mission exports its phase table and omits both, because a
  mission has no single operating point and exporting one would export a
  number the screen deliberately refuses to show.
- **Airframe Diagram tab**, a plan view to scale drawn from the entered
  numbers, nose up. Lift+cruise draws booms fore and aft of the wing with a
  separate nose propeller; the vectored types mount their rotors along the
  wing and draw no cruise propeller, because they have none. It reports the
  tip-to-tip gap between adjacent discs and says so in red when they overlap.
  Only span, wing area and propeller diameters are real inputs — boom
  positions are a reasonable arrangement, not a claim about a specific
  airframe, so it is for checking proportions and clearances rather than a
  layout to build from.
  Unlike the budgets and mission plots, the diagram survives both run types:
  it describes the aircraft, not a flight.
- **VTOL tests: 77 -> 85.**

**VTOL v0.5.0** — measured propeller tables, wind, batch and matrix coverage
The three gaps that mattered most, in the order they change answers.
- **Measured propeller tables.** The VTOL ran entirely on figure-of-merit and
  propeller-efficiency GUESSES. It now loads a thrust/power bench CSV for the
  lift rotors and the cruise prop and uses the efficiency the hardware
  actually achieved. On a 16 kg quad-rotor VTOL the shipped 22 in table gives
  a figure of merit of 0.628 against the 0.650 estimate — 3.5% more hover
  power than assumed.
  Outside the thrust range the table covers, the estimate is used again and
  Status says which is in force, because a bench test says nothing about
  thrusts it never produced. A bad path raises immediately: a table you
  think is loaded but is not is worse than no table.
  The loader, power lookup and efficiency derivation live in
  `rotorworks_core.py` so the VTOL is not a third copy.
- **Wind.** The VTOL had none — not one line. Power follows AIRSPEED but
  progress follows GROUNDSPEED, so a leg measured over the ground takes
  longer into a headwind: on the survey mission a 5 m/s headwind costs +23%
  energy over the identical 24.2 km track, and the same wind behind saves
  21%. Wind reaches the GUI, the CLI (`--wind`, `--wind_direction`) and the
  sensitivity re-fly.
  A test asserting that station-keeping in wind costs MORE turned out to be
  wrong — the model was right. Holding station into 8 m/s costs 203 W against
  585 W hovering, because the wing is already carrying much of the weight.
  That is translational lift, and the test now asserts it.
- **Batch driver and coverage matrix.** `rotorworks-batch.py` accepts
  `--sim vtol`, with its own GUI-key-to-CLI map and output parsing — a sweep
  that runs but reports `nan` looks like it worked, so the matrix test checks
  for that specifically. The matrix gained 11 VTOL cases: every configuration
  with and without a table, every mission calm and windy, and a batch sweep.
- **VTOL tests: 68 -> 77; matrix: 83 -> 94.**

**VTOL v0.4.0** — the multicopter and fixed-wing feature set, ported
The VTOL tool had two display tabs against the other simulators' ten. It now
has the same set, answering the same questions the same way.
- **Seven new display tabs**: Status, Mission Plots, Weight Budget, Power
  Budget, Mission Diagram, Sensitivity and Compare, plus a **Wiring** input
  tab. "Plots" is renamed **Fixed Speed Plots** to match.
- **Status is VTOL-shaped.** It checks hover as well as cruise, because hover
  is the heaviest steady load a VTOL carries and the case that sizes the
  battery, the motors and the connectors. It also checks the transition/stall
  margin, the wing's lift share at the cruise speed (flagging a cruise speed
  set below the transition speed), and rotor tilt for the vectored types.
- **A mission now produces a time series**, so Mission Plots, the altitude
  trace and worst-case Status all have data. Status shows the WORST value
  each check reached; Metrics shows the LAST instant flown; both say which.
- **Optional detail inputs**, all blank by default: battery C-ratings, lift
  and cruise motor power ratings, component masses, a hover download
  override, and a main wire run with connector ratings. Wiring loss is solved
  with the pack current and appears as its own Power Budget row.
- **Every fault this project shipped elsewhere is now a test here**: that an
  input actually reaches the model (the multicopter's wiring never did), that
  Compare moves between two mission runs (the fixed-wing refreshed it before
  storing the result), that a mission clears what only a single point can
  answer, and that Sensitivity clears when it goes stale.
- **VTOL tests: 27 -> 68.**

**VTOL v0.3.0** — tiltrotor, tiltwing and tailsitter
All four configurations now run. The three new ones previously raised
`NotImplementedError` on purpose, rather than fall back to lift+cruise physics
that would have been confidently wrong.
- **Vectored thrust.** Lift+cruise carries two propulsion systems — rotors
  that only lift, a propeller that only pushes. The other three carry ONE set
  of rotors that does both by pointing its thrust. Two force balances fix the
  thrust and its direction at every speed:
  `T·cos(tilt) + L_wing = W·(1+download)` and `T·sin(tilt) = D`.
  Tilt sweeps from 0 deg in hover to 90 deg in wing-borne cruise.
- **Hover download** separates them from each other. A tiltrotor's rotors
  blow onto a wing lying flat beneath them, costing about a tenth of the
  weight (the V-22 runs 10-12%); a tiltwing and a tailsitter turn the wing
  edge-on to the wash and escape most of it. The per-type figures are typical
  published values and every one can be overridden with measured data.
- **The central trade shows up correctly.** Tiltrotor pays the most to hover
  (673 W against 602 for the others on the reference aircraft), and every
  vectored type beats lift+cruise in cruise (151 W against 214 at 18 m/s)
  because lift+cruise carries stopped rotors dragging through the whole leg.
- **Tiltwing and tailsitter are identical in still air, deliberately.** In
  steady flight they genuinely are close. What separates them is dynamics —
  a tailsitter weathervanes broadside in a crosswind, a tiltwing keeps its
  fuselage level, a tiltwing's wing gets slipstream lift in transition — none
  of which a steady power model sees. Quantifying slipstream lift needs a
  coverage factor there is no data to set, so both default to the same
  download rather than being forced apart by an invented coefficient.
- **28 new VTOL tests (27 -> 55)**, including force balance on both axes at
  every speed, monotonic tilt, download vanishing once the wing takes over,
  and a mission flown by every type. Continuity is tested by halving the
  sample step and requiring the worst change to shrink proportionally — which
  distinguishes a steep slope from a genuine jump, something a fixed
  threshold cannot do.

**VTOL v0.1.0** — new file, versioned separately
- `vtol-power-sim-gui.py` — **lift+cruise only**. Reuses the rotor model from
  the multicopter and the wing model from the fixed-wing, both through
  `rotorworks_core`, and adds the part neither has: the transition, where the
  wing and the rotors share the lift.
  - The lift split is set by what the wing can carry:
    `L_wing = min(q*S*CL_cap, W)` and `T_rotor = W - L_wing`. Both propulsion
    systems draw at once through the transition, which is why it is the most
    power-hungry part of the flight.
  - `CL_cap` during transition defaults below `CL_max`, because transitioning
    at the stall boundary leaves no gust margin.
  - Stopped lift rotors are carried as drag in cruise, phased in with the
    wing's lift share so the transition and cruise models agree at the
    boundary rather than jumping.
  - Mission output reports the **energy split across hover, transition and
    cruise**, which is where a VTOL's endurance is won or lost.
- **A configuration dropdown offers tiltrotor, tiltwing and tailsitter**, so
  the input set is defined and configurations saved today stay readable. They
  are **refused with a clear message**, not approximated as lift+cruise: their
  transition physics differs enough that a fallback would be confidently wrong.
- The Glauert forward-flight inflow solver moved from the multicopter into
  `rotorworks_core`, since the VTOL lift rotors need the same physics.

**v2.42.0** — wiring losses and connector ratings
- **New Wiring tab** in both simulators, all inputs optional. Left blank the
  model behaves exactly as before.
- **Wire run**: one-way length plus either an AWG gauge or a measured Ω/m.
  Length is one-way but current has to come back, so the resistance counts
  BOTH conductors — forgetting the return path halves every loss and is the
  easiest way to make wiring look harmless.
  The loss is solved INSIDE the power fixed-point loop, because more loss
  draws more current which costs more loss. The ESC and motor also see the
  pack voltage minus `I·R`, so a long thin lead costs thrust as well as watts:
  the ESC loss, the motor current and the throttle are all worked out at the
  voltage that actually reaches the ESC.
- **Wire loss gets its own Power Budget row**, not folded into ESC losses: it
  is the one loss a user can halve with a thicker cable, so it has to be
  visible to be actionable. On a 1.8 kg 4S quad with 2 m of 16 AWG it is
  3.5 W at hover, 2.6% of the total.
- **Wire temperature** is estimated on Status: the steady temperature of the
  lead in still air, with copper's resistance rising as it heats. Amber within
  20 °C of the insulation limit (default 150 °C, editable), red above it; a
  lead that cannot shed its own heat reports "above 400 °C" rather than a
  number. Still air is the conservative case — a lead in the prop wash runs
  cooler.
- **Voltage drop** is checked too: green up to 3% of the full pack voltage,
  amber to 5%, red beyond.
- **Connector ratings** for the battery, ESC and motor links. Twelve types
  (XT30/60/90, AS150, EC3/EC5, Deans, bullets 3.5-8 mm) pre-fill typical
  continuous and burst figures, both editable — burst ratings vary too much
  between manufacturers to hardcode.
- **Status checks each connector against the current through IT**, which is
  not the same number in each position: the battery connector carries the
  whole pack current, an ESC connector one motor's share, and the motor
  connector phase current (about 1.15x the DC current). Green under the
  continuous rating, amber between, red above the burst — the same dual-limit
  treatment as the battery C-rate rows.
- **Connector voltage ratings**: each connector also takes a rated voltage,
  checked against the pack's FULL-charge voltage. Picking an Amass type (XT30,
  XT60, XT90, AS150) fills in its published 500 V DC; the others publish none,
  so the field stays blank and the check reads "Not Specified".
- **Metrics** gains a Wiring section (lead resistance, loss, drop, ESC input
  voltage and temperature), and the Output pane and CLI print a "Main lead"
  line whenever a lead is entered.
- The wiring model lives in `rotorworks_core`, and the GUI and CLI build it
  through the same function, so they give the same answer. The VTOL uses it
  too. The CLI flags are the VTOL's: `--wire_length`, `--wire_awg`,
  `--wire_ohm_per_m`, `--wire_temp_limit` and `--connector_{batt,esc,motor}_{cont,max,volt}`;
  the batch driver maps the saved GUI fields onto them.

**v2.41.0** — Mission Diagram tab
- **New Mission Diagram tab** in both simulators, drawn after a mission run.
  Missions are written as legs — a heading, a distance, an altitude — never as
  coordinates, so the SHAPE of the route was not stated anywhere. Integrating
  the legs recovers it, which is the only way to see whether a pattern closes,
  overlaps itself, or drifts.
- **Ground track** with waypoints numbered in flight order, takeoff marked
  with a green triangle and landing with a red one, and an **arrow at each
  waypoint showing where the airframe is pointing**. That last part matters
  for a multirotor: a square flown with the nose fixed and one flown by yawing
  at each corner trace the identical path, and only the arrows tell them
  apart.
  North is up and east is right, so the map reads as a map.
- **Altitude profile** beside it, waypoints numbered to match.
- The tab clears with a note after a fixed speed sweep, which has no route.
- **Power Budget pie chart removed** from both. A dozen slivers with a legend
  longer than the chart conveyed less than the table beside it, and the
  smallest rails were unreadable at any size — the percentage column already
  carries the same information.

**v2.40.0**
- **Sensitivity results clear when a new run makes them stale**, in both
  simulators and on both run paths. A sensitivity sweep is tied to the run it
  was computed from; leaving the previous rankings on screen after the design
  or the mission changes is worse than showing nothing, because the numbers
  look current and there is no way to tell they are not. The table now says
  which kind of run invalidated it and to press Run Sensitivity again.
- **Removed a leftover sentence from the About window** in both simulators.
  An earlier edit deleted the opening lines of that paragraph and left its
  tail behind, so the box ended mid-sentence with "field, you are running an
  older copy of this script."

**v2.39.1** — fixed-wing mission comparison actually compares
You were right that it did not work. **Two** ordering faults, one symptom.
- **The mission path never called `refresh_comparison()`.** Both calls lived
  in `run_single_point`, so a pinned baseline sat unchanged while the run
  behind it moved. The multicopter had this exact fault and it was fixed in
  v2.16.0; the fix was never carried across.
- **Adding the call was not enough**: it ran BEFORE `_last_run["metrics"]` was
  stored, so the tab compared the new baseline against the PREVIOUS run's
  numbers — still +0.00 everywhere. Storing first fixes it.
- Verified: with a 25% weight increase between two mission runs, **21 of 25
  rows move**, in the right directions (flight time -15.7 min, total power
  +24.4 W).
- **No automated test for this.** The GUI harness cannot drive a mission run —
  it has no way to set the Mission JSON field and cannot tell the several
  "Browse" buttons apart. That gap is now documented in `tests/test_gui.py`,
  along with what it would unlock: mission coverage for Status, Metrics and
  Power Budget, none of which are exercised on the mission path either.

**v2.39.0** — the multicopter mission corrections applied to the fixed-wing
- **Status and Metrics now say what they show.** After a mission, Status is
  the WORST value each check reached and Metrics is the LAST evaluated
  instant. Metrics was previously fed the worst-case dict, which presents an
  operating point the aircraft never flew — each field's worst moment happens
  at a different time.
- **Fixed Speed Plots and Power Budget clear on a mission run.** The mission
  cleared the plots and then rebuilt the sweep a few lines later, so the
  placeholder never survived — the same fault the multicopter had. The run
  now lands on Mission Plots.
- **The altitude trace ramps.** It recorded the phase TARGET, so a climb
  plotted as an instant jump. The fixed-wing evaluates each phase once rather
  than stepping through time, so the fix differs from the multicopter's: it
  records where the aircraft STARTS a phase and where it ENDS it, and the plot
  draws the ramp between. A 0-120 m climb now reads 0 at t=0 and 120 at t=90.
- **Stall speed now rises with bank.** In a turn the wing carries n times the
  weight, so stall goes as `sqrt(n)` — at 45 deg that is 10.87 m/s against the
  9.14 being reported. The mission loop already used the corrected value
  internally; only the displayed metric was wrong, so the margin the user read
  was not the margin they had.
- **Bank already fed the power calculation** and needed no change: 84.0 W
  level against 93.8 W at 45 deg on the 2 m survey aircraft.
- Compare already gated mixed runs ("Baseline is a mission run; current is a
  single-point run") and needed no change.

**v2.38.0** — peripheral current was being ignored, not just mis-displayed
Chasing why the fixed-wing Power Budget omitted peripheral current turned up
a deeper fault in BOTH simulators.
- **Regulated rails and direct-from-pack peripherals are independent loads
  and now ADD.** The help has always said so — "use this for devices wired
  straight to pack voltage; use the Avionics tab for anything on a regulated
  rail... never enter the same device in both" — but the model used
  peripheral current only as a FALLBACK for "no rails defined". So a payload
  wired straight to the pack drew **nothing at all** as soon as a single BEC
  rail existed, and the number typed into the box did nothing.
  Fixed at all six sites across the two simulators: the metrics core, the
  mission path, the plotting path and the CSV export.
- **This also corrects v2.35.0.** That release hid the peripheral row from the
  Power Budget to suppress a negative "Unaccounted" equal to the peripheral
  draw. The residual was real, but it was the MODEL ignoring the load, not the
  table over-reporting it — so the fix was in the wrong direction. The row is
  back and the budget closes.
- **Peripheral power is valued at NOMINAL pack voltage in the budget**,
  matching how the model charges it. Using the loaded voltage disagreed by the
  sag and left a -3.6 W residual.
- Two new tests: that the two loads add exactly, and that typing a peripheral
  current changes the answer when rails are present.

**v2.37.1**
- **Cruise speed marked on the fixed-wing performance plots**, matching the
  multicopter. Five panels already drew a red stall line but nothing showed
  where the aircraft is actually being asked to fly, so a design sitting on
  the wrong side of a knee looked fine. A grey dash-dot line now marks cruise
  on Flight Time & Range, Thrust, Power, Rate of Climb and Drag. The Drag
  Polar is left alone — its x-axis is CD, not airspeed.

**v2.37.0** — per-leg translation direction, and the altitude trace
- **Mission phases can set their own `translation_direction_deg`.** It was a
  config-level value, one for the whole flight, so a square flown with the
  nose FIXED — north, then sideways east, then backwards south — could not be
  expressed as a mission at all.
- **The mission altitude trace was recording the phase TARGET, not the actual
  height.** A 30 s climb from 0 to 60 m plotted as an instant jump to 60 and
  stayed there through the landing. The climb ENERGY was always correct: the
  potential-power term is integrated per step and matches `m*g*h` to 0.7%
  (2.72 Wh measured against 2.70 ideal for 16.5 kg lifted 60 m). Only the
  trace disagreed with the physics behind it, which is why it went unnoticed.
  Altitude is now integrated step by step for every phase type. The first fix
  only covered distance-based phases, which left a duration-based takeoff
  climb flat at zero — the trace now ramps correctly both up and down.
- **Three new example missions** flying the same 400 m square three ways:
  `mc_07a` fixed yaw (translating in four directions), `mc_07b` yawing at each
  corner, `mc_07c` rounded corners flown nose-first throughout.

**v2.36.0** — the hover/translating inconsistency, and the vocabulary gap
- **Root cause found for the 4% hover discrepancy.** The coaxial interference
  penalty was discounted a flat 30% whenever `orientation != "hover"`, at ANY
  airspeed. So a stationary aircraft described as "translating at 0 m/s" got a
  forward-flight benefit that requires a freestream it does not have.
  The relief is real — the freestream sweeps the upper rotor's wake clear of
  the lower one — but it depends on HOW FAST you are going, not on which word
  describes the flight mode. It now blends on `V / (V + v_hover)`: no relief
  at rest, half at `V = v_hover`, approaching the full 30% at speed. Hover and
  translating-at-zero now agree to 1 part in 10^12.
- **Swept for the same class of bug.** Power is continuous across 0-25 m/s
  (worst step 0.94%), thrust equals weight exactly at rest, and hover matches
  translating-at-zero for both flat and coaxial layouts. Five new physics
  tests lock these in.
- **Golden snapshot regenerated.** 17 values moved across 4 keys, all coaxial
  X8; flat, fixed-wing, battery and atmosphere untouched. Largest shift
  +2.34% at 5 m/s, smallest at high speed — the expected shape, since the old
  flat discount was most wrong where there was least freestream. `hover@0.0`
  did not move, because hover always paid the full penalty.
- **The CLI never learned the word "translating"** — in two places, the
  argparse `choices` list and a second validator inside `main()`. The batch
  driver therefore could not run a config the GUI had just written. Two new
  CLI tests assert every entry point accepts the same vocabulary, and that an
  invalid one still fails.
- **A batch test routed configs by filename prefix**, so `Jaguar-quad.json`
  was fed to the fixed-wing simulator. It now routes by the config's declared
  `schema`, which is the thing that actually says what a config is.

**v2.35.0** — power-budget, hover and diagram corrections
- **Peripheral current was double-counted in the Power Budget.** The model
  draws avionics power from the RAILS when any are defined and falls back to
  peripheral current only when none are — either, never both. The budget
  listed both, which showed up as a negative "Unaccounted" row exactly equal
  to the peripheral draw. The budget now closes with no residual.
- **A hover run reported hover power but cruise-speed endurance.**
  `compute_operating_metrics` forces hover to 0 m/s; `estimate_flight_time_minutes`
  did not, so the same run mixed two conditions — hovering at 17 m/s in the
  speed box gave 47.99 min against the correct 23.80 min.
- **The plan view is drawn nose-up.** It plotted +x (forward) to the RIGHT, so
  0 deg pointed right and 90 deg pointed LEFT — the opposite of both the help
  text and normal plan-view convention. Now 0 deg is up, 90 deg is starboard,
  180 aft, 270 port, all verified.
- **Rated Voltage removed** from the Motor and ESC tabs in both simulators,
  superseded by the min/max rating range added in 2.20.0.
- **Payload row shows its mass fraction**; battery mass fraction, drive mass
  fraction, motor configuration and energy density removed from Environment &
  Design, where they duplicated the Airframe section.
- **Power Budget clears after a mission**, like the plot tabs, since a mission
  has no single operating point to break down.

**v2.34.0** — sensitivity works on missions
Sensitivity only ever perturbed a single operating point, so the question it
could answer was "what does this change do at cruise" — never "what does it do
to the flight I actually intend to make".
- **After a mission, each perturbation re-flies the WHOLE mission.** That
  matters because the two answers differ: a change that improves cruise can
  still fail a mission on its hover legs, and only re-flying it shows that.
- **The output list switches with the run type**, since the questions are not
  interchangeable. Mission mode offers energy, time, distance, reserve margin,
  minimum state of charge, peak pack current and peak motor temperature;
  "hover efficiency" has no meaning for a mission and "reserve margin" none
  for a sweep. A note on the tab says which mode is active.
- Conditions are preserved: the perturbed runs use the same wind, temperature,
  pressure and orientation as the mission you ran, so only the lever changes.
- Cost is real — 16 levers x 4 factors is 64 full mission simulations, about
  15-20 s. Worth it for the answer, but not instant.
- Both simulators. On the 2 m survey lawnmower, reserve margin is dominated by
  battery capacity at 53% swing, then propeller efficiency at 10%.

**v2.33.2**
- **Documented why a steady tilt alone does not unbalance the rotors.**
  Holding a constant tilt at constant speed needs no net moment: weight acts
  at the CG, and equal thrust on a symmetric airframe also acts through the
  CG. Differential thrust is what CHANGES attitude; once the aircraft is at
  the angle it wants, that difference disappears. Drag acting at a different
  height from the CG is what unbalances them continuously.
- **Named what is NOT modelled**, in the help text and on the Per-Rotor
  Loading panel: rear rotors flying in the front rotors' wake need more POWER
  for the same thrust, and blade flapping adds a nose-up moment in forward
  flight. Both are real, both make the rear rotors work harder than this
  model shows, and neither is captured.

**v2.33.1**
- **Fixed-wing Cruise Speed moved from Airframe to Mission/Environment**,
  matching where the multicopter's went in v2.19.0. That release moved the
  multicopter's and missed the fixed-wing's, so the two simulators disagreed
  about where the same setting lives. Cruise speed describes how an aircraft
  is FLOWN, not what it is.

**v2.33.0** — what Status and Metrics mean after a mission
Both tabs already behaved differently after a mission than after a sweep, and
neither said so. "Pack current 33 A" reads like a steady value when it is
actually the worst instant of an entire flight.
- **Status now states its scope.** After a mission every row is the WORST
  value reached at any point — highest current, power, thrust and temperature,
  LOWEST pack voltage, state of charge and reserve margin — so a row passes
  only if it passed throughout. After a fixed speed run it says that instead.
  (The worst-case tracking already existed; it was simply unlabelled.)
- **Metrics now shows the mission's LAST evaluated instant**, and says so.
  Previously it was not repopulated after a mission at all, so it kept
  displaying whatever fixed-speed run came before — stale numbers presented as
  current ones. Populating it from the worst-case dict would have been worse
  still: each field's worst moment happens at a different time, so that
  combination describes a point the aircraft never flew.
- **Fixed Speed Plots really do clear after a mission now.** The 2.23.0 change
  called the clearing helper and then redrew the sweep three lines later, so
  the placeholder never survived. A mission produces no fixed-speed sweep;
  drawing one from the config implied it did. The run now lands on Mission
  Plots instead.

**v2.32.0** — per-rotor loading on the Airframe Diagram
- **New Per-Rotor Loading panel** beside the plan view. One row per rotor:
  position, thrust in newtons and grams, share against the mean, and margin
  to the propeller's rated maximum with the percentage used. The
  hardest-working rotor is highlighted amber, and any rotor over its rating
  red — because the rotor that saturates first is what limits the aircraft,
  and an average cannot show which one that is.
- A summary row gives the load **spread** (max/min) and the worst rotor's
  excess over the mean.
- **The diagram now numbers each rotor** and draws an arrow for the direction
  of travel, so the table can be read against the picture without guessing
  which rotor is which.
- Coaxial layouts are handled: a pair shares a position, so the pair's thrust
  is halved per rotor and the row is marked `(x2)`.
- Worked example — 450 survey quad travelling 45° off the nose at 14 m/s with
  a 6 cm drag offset: rotor 3 (aft-left) carries **+6.8%** above the mean
  while rotor 1 unloads by the same amount, a spread of 1.147:1. With the drag
  offset left blank the load is even and the panel says so.
- Also added the **Drag Height above CG** input row, which 2.31.0 defined but
  never placed on a tab.

**v2.31.0** — translation direction as an input, and per-rotor loading
- **Translation Direction is now a GUI field** on Mission/Environment. It
  existed in the config from 2.26.0 but was never given a widget, so the only
  way to set it was hand-editing JSON.
  It is deliberately separate from Course Heading: course is where you are
  going relative to the **wind**; translation direction is which way the
  airframe is being pushed relative to **itself** — 0° straight ahead,
  90° straight right, 45° toward the front-right motor. It sets the drag
  silhouette and the pitch/roll split.
- **Max Pitch and Max Roll** also got fields, each falling back to Max Tilt.
- **Per-rotor load sharing.** Drag acting above the centre of gravity makes a
  pitching moment `M = D x h`, and attitude is only held if the rotors counter
  it with differential thrust — so the trailing rotors work harder than the
  leading ones. Which rotor saturates first is decided by that imbalance, not
  by the average, and the model previously assumed every rotor carried
  exactly `T/n`.
  New **Drag Height above CG** input; leave it blank and the load stays even,
  exactly as before. The split uses the same rotor geometry the Airframe
  Diagram draws, so the loading corresponds to the layout on screen.
  New metrics: `rotor_thrusts_N`, `rotor_thrust_max_N`, `rotor_thrust_min_N`,
  `rotor_load_spread`, `rotor_imbalance_pct`.
- Removed the stale "if you do not see the Simple / Advanced selector..."
  paragraph from the About window in both simulators.

**v2.30.1** — battery mass counted correctly; entered airframe mass honoured
Both faults shared one cause: the mass arithmetic read the RAW pack-weight
field, which is the weight of a SINGLE pack. The aircraft carries
series x parallel of them, so a 2S1P arrangement of 2100 g packs weighs
4200 g — exactly what the Weight Budget was already showing.
- **The mass validation understated the battery** and let impossible designs
  through. On the heavy-lift X8 it compared 7370 g of components against a
  9000 g all-up weight and passed, while the true total is 9470 g.
- **The heavy-lift example was therefore still wrong.** Corrected again, from
  9000 g to **11500 g**, which leaves 2030 g of structure. The first
  correction in 2.29.0 used the same faulty arithmetic, so it moved the number
  without fixing the problem.
- **An entered airframe mass never reached the Weight Budget.** In "enter
  airframe" mode the structure mass is an INPUT, but the budget still derived
  it as a residual — showing 0.0 g while the field said 1000 g. It now shows
  what was entered, so the table cannot disagree with the input driving it.
- Both simulators fixed; all 10 configs re-verified against the corrected
  arithmetic.

**v2.30.0** — example configs refreshed, VTOL config save/load
- **All 10 example configs updated** to carry the full current input set:
  avionics mass, mass entry mode, per-axis pitch and roll limits, motor/ESC/
  battery temperature limits, time-at-maximum allowances and voltage rating
  ranges. Fixed-wing configs gained field lengths and a minimum climb rate,
  which is what their take-off, landing and thrust-margin checks now judge
  against. Dead inputs (frontal area, RTH and diversion reserves) removed.
- **New `vtol_2m4_lift_cruise_survey`**, and the VTOL simulator gained
  **config save/load** — it had none, so a VTOL design could not be kept or
  shared.
- **The multicopter orientation dropdown still said "forward".** The 2.26.0
  translation work changed the physics and the metrics but never reached the
  widget or its validator, so a config saved with `orientation: translating`
  was rejected by the very GUI that would have produced it. Found by loading
  the refreshed configs. The dropdown now offers **translating / hover**, and
  "forward" still loads from older files.
- **The Jaguar example pointed at an absolute path** on one machine's Desktop
  for its propeller table. Cleared, with a note to load your own.
- Every config leaves a positive airframe mass after components are
  subtracted, so none is refused by the 2.29.0 validation.

**v2.29.0** — mass entry, validation, and text wrapping
- **Mass Entry Mode is a dropdown**, and whichever mass is being CALCULATED is
  greyed out. Leaving both editable allowed entering two numbers that
  contradict, with nothing to say which the simulation used.
- **Impossible mass combinations are refused.** In "derive airframe" mode the
  airframe mass is the residual after components are subtracted; if the
  components already exceed the all-up weight the residual is negative, which
  is not a slightly-wrong answer but an impossible aircraft. The run stops
  with a message naming both figures and the three ways to fix it.
- **This immediately caught a bad example config.** The heavy-lift X8 declared
  6500 g all-up without payload while carrying 7020 g of motors, ESCs, props
  and battery — an implied airframe of **-520 g**. Corrected to 9000 g, which
  leaves a realistic ~1980 g of structure for a 22 in X8.
- **Report status notes now wrap.** The status table was the one table still
  unwrapped, so its notes — the longest text in the report — ran off the page.
  The report also gained the `edge` colour so it matches the GUI.
- **Status notes are readable in full in the GUI.** `ttk.Treeview` has no
  option to wrap text inside a cell, so rather than truncating invisibly the
  full note for the selected row is shown in a wrapping strip beneath the
  table.

**v2.28.0** — Sensitivity and Compare broadened
Both tabs were written before much of the model existed and had fallen behind
what the simulators compute.
- **Multicopter sensitivity: 8 to 16 levers.** Added payload mass (the one
  most users vary first, previously reachable only through all-up weight),
  prop pitch, air density (standing in for altitude and temperature — a
  multirotor is far more sensitive to it than a wing, since hover power goes
  as 1/sqrt(rho)), figure of merit, battery and ESC resistance, profile area,
  and translation direction, which became a real design variable in 2.26.0.
- **Multicopter sensitivity outputs: 3 to 7.** Hover endurance (distinct from
  cruise), pack current, motor temperature and hover efficiency.
- **Fixed-wing sensitivity: 8 to 15 levers**, including wing span, which at
  fixed area changes aspect ratio and is the single biggest lever on induced
  drag. Outputs gained stall speed, L/D, rate of climb and take-off distance.
- **Compare: 11 to 24 rows (multicopter) and 12 to 25 (fixed-wing).** Now
  covers attitude (pitch, roll, total tilt), the itemised losses — two designs
  can draw identical total power while wasting very different amounts as heat
  — thermal margins, and the performance margins a change is usually bought
  for: thrust available, best climb, take-off distance, glide ratio, and the
  optimal speeds.

**v2.27.0** — multicopter turn model
Mission phases accept an optional **`turn_radius_m`**. The fixed-wing already
modelled bank through its load factor; the multicopter did not model turns at
all, so a mission of tight circuits cost the same as flying the distance
straight.
- A turning multirotor holds its weight AND supplies centripetal force, so
  thrust rises by `1/cos(bank)` where `tan(bank) = V^2 / (R*g)` — the same
  relation as for an aeroplane, since both must tilt their lift vector
  sideways. `turn_thrust_N` treats weight, drag and centripetal force as the
  mutually perpendicular vectors they are:
  `T = sqrt(W^2 + D^2 + Fc^2)`.
- The cost is substantial. On the reference quad at 12 m/s:
  a 50 m radius costs **+4.5%** power, 25 m **+18%**, 15 m **+48%**, and a
  10 m turn **doubles** it.
- Phases without a turn radius are untouched, so existing missions behave
  exactly as before — asserted by a test.
- Endurance and range remain steady-state estimates with no turn applied;
  turns are a mission-phase property, not a fixed-speed one.

**v2.26.0** — translation direction (multicopter)
A multirotor does not have a single "forward": it can translate any way
without yawing, and which way it goes changes both the silhouette it presents
and how the required tilt splits between pitch and roll. The old model treated
every translation as nose-first and hid both effects.
- **Orientation "forward" is now "translating"**, with a new **translation
  direction** input measured from the nose (0 deg ahead, 90 deg right).
  "forward" is still accepted and means translating at 0 deg, so existing
  configs, missions and CLI calls are unchanged — asserted by a test that
  compares every metric between the two names.
- **Drag depends on direction.** The presented area interpolates between the
  frontal and side silhouettes as
  `A(psi) = A_front*cos^2(psi) + A_side*sin^2(psi)`, exact for a rectangular
  prism. On the reference quad at 12 m/s that is 2.24 N nose-first against
  3.72 N sideways — 66% more drag and 16% more power, which the old model
  could not express at all.
- **Tilt is reported as pitch and roll**, split by the direction of travel:
  `tan(pitch) = tan(tilt)*cos(psi)`, `tan(roll) = tan(tilt)*sin(psi)`.
  Forward translation is pure pitch, sideways is pure roll, a diagonal splits
  evenly. Same case above: 7.2 deg of pitch flying forward, 11.9 deg of roll
  flying sideways.
- **Separate pitch and roll limits**, each falling back to the old single
  tilt limit. Long-armed airframes rarely have equal authority in both axes,
  and one number could not say so.

**v2.25.1** — multicopter status thresholds given a basis
The same treatment the fixed-wing limits got. Measured against all five
example aircraft first, then changed only where the data said the limit was
wrong.
- **Disk loading is no longer pass/fail.** It is a design choice — a cinewhoop
  runs high disk loading on purpose. What it does fix is the best hover
  efficiency physically available, `ideal g/W = 1000 / (g0 * sqrt(DL/2rho))`,
  so the row now reports that ceiling instead of judging the number.
- **Hover efficiency is judged against that ceiling**, not a flat 5 g/W. The
  old limit measured disc SIZE more than design quality: trivially easy on a
  heavy-lift with 22 in discs, near impossible on a 3 in cinewhoop. The
  cinewhoop now reads "44% of the 11.5 g/W ideal for this disk loading".
- **Hover drive efficiency scales with rotor size.** The multicopter reports
  ideal hover power over electrical power: rotor, motor, ESC and wiring
  together, so it is not a rotor figure of merit and is no longer called one.
  Small propellers run at low Reynolds number and cannot reach what a large
  rotor can, so the target is the rotor figure of merit in use (entered, or
  the default that falls with diameter) times about 0.80 for the motor and
  ESC. Above 1.0 is flagged as impossible rather than capped.
- **Prop solidity scales with blade count.** Solidity rises with blade number
  almost by definition, so one 0.05-0.15 window judged 3-blade propellers
  against a 2-blade expectation and flagged normal designs as suspect.
- **Tip Mach uses the local speed of sound**, matching the fixed-wing fix, and
  now shows the tip speed in m/s alongside.

**v2.25.0** — Power Budget tab (#57, both simulators)
- **New Power Budget tab**, beside Weight Budget. The weight budget answers
  "what is this aircraft made of"; this answers "what is the battery actually
  paying for". A design can be light and still lose a large share of its
  energy to heat.
- Every row is either **delivered** (green) or **lost** (red), with a voltage
  and current column. Anything drawing straight from the pack reports
  "Battery" for voltage, because its voltage is whatever the pack happens to
  be at rather than a designed value. Each avionics rail contributes two rows:
  the power delivered at rail voltage, and the regulator loss getting there.
- Subtotals for total delivered and total losses, then the grand total, plus a
  share diagram colour-keyed the same way.
- **The total is taken at the CELLS, not the pack terminals.** Terminal power
  is already measured at the sagged voltage, so counting the pack's own I2R
  loss inside it charges that loss twice — which showed up immediately as a
  negative "Unaccounted" row exactly equal to the I2R. The budget uses
  `P_cells = P_terminals + I2R` so every joule is counted once.
- An **Unaccounted** row appears whenever the itemised rows genuinely do not
  close, rather than absorbing the difference into the percentages. That is
  what caught the double-count above.

**v2.24.0** — propeller coefficients, and a correction to the C_T fit
- **TConst (C_T) and PConst (C_P) now appear in the Propeller metrics** of
  both simulators (#46). With a table loaded they are **measured** from it —
  `T = C_T x rho x n^2 x D^4`, `P = C_P x rho x n^3 x D^5` — and the note
  reports how many points were used and how tightly C_T holds across them.
  Without a table they fall back to the geometry estimate and say so.
- **The geometry C_T fit was about twice too high and has been recalibrated.**
  Deriving coefficients from the two measured tables gave C_T 0.062 for the
  22x6.6 and 0.079 for the 18x8, against roughly 0.13 and 0.14 from the old
  `0.10 + 0.10*p/D`. That fit was invented, not measured. Since RPM goes as
  `1/sqrt(C_T)` it understated propeller speed by about 40%.
  It had passed an earlier check only because the resulting hover tip speeds
  landed in a plausible band — which is not evidence. The fit is now
  `0.022 + 0.129*p/D`, within 4% of both measurements, and hover tip speeds
  move from 60-63 m/s to 70-91 m/s, which matches real multirotors better.
  Two propellers is still a thin basis; supply TConst or a table when it
  matters.
- **The voltage-rating unit is a dropdown** (S or V) rather than free text
  (#12), in both simulators and for both motor and ESC. A two-valued choice
  typed by hand invites "volts" or a typo that silently reads as the wrong
  unit.

**v2.23.1** — the same plot and export work applied to the fixed-wing
- Mission Plots take up to 4 variables with two y-axes a side and colour-keyed
  axes, and the x-axis switches between mission time and distance.
- "Plots" renamed **Fixed Speed Plots**, and each plot tab is cleared with an
  explanatory note when the other kind of run happens.
- The power panel now draws **mechanical alongside electrical** required
  power, so the drivetrain loss is the gap between them, with power available
  as a third trace.
- Exports follow the run: the sweep CSV gained Power Electrical, Power
  Mechanical, Thrust Required and Thrust Available so its columns match the
  curves; a mission run exports its own history instead of a stale sweep.
- **Not ported, deliberately:** the multicopter's thrust-component and drag
  panels (#22, #23). A fixed-wing has no "hover attitude" trace to remove, and
  it already plots induced against parasitic drag and thrust required against
  available — the equivalent information in the form that suits a wing.

**v2.23.0** — mission plots and exports (multicopter)
- **Mission Plots take up to 4 variables** (#4), two y-axes on the left and
  two on the right, each axis and its curve sharing a distinct colour so a
  trace is identifiable without reading the legend. The panel says the limit
  is 4; selecting more plots the first four and explains why — beyond that the
  axes crowd each other and nothing is readable.
- **Mission plot x-axis can be time or distance** (#19). Distance is the more
  useful axis on a survey pattern, where what matters is where along the route
  something happened.
- **"Plots" is now "Fixed Speed Plots"** (#17), and each plot tab belongs to
  one kind of run. Running a mission clears the fixed-speed plots with a note
  saying so, and running a sweep clears the mission plots (#18). A stale chart
  from the other kind of run is worse than an empty tab, because the axes look
  perfectly plausible.
- **Exports follow whatever was plotted** (#21). After a fixed speed sweep the
  CSV/Excel carries speed, flight time, range, mechanical and electrical
  power, the three thrust components and the three drag curves — the columns
  mirror the curves on screen, in the same order. After a mission it carries
  the mission history instead of a stale sweep.

**v2.22.0** — fixed-speed plots reworked (multicopter)
- **"Run Single-Point" is now "Run Fixed Speed Sweep"**, and the output header
  reads "Fixed Speed Run". The button always swept speed to build the curves;
  the old name understated it.
- **Power panel shows mechanical AND electrical** instead of the ambiguous
  "hover attitude" trace. The gap between the two curves is the drivetrain
  loss, which is far more useful than a line that was neither hover nor the
  flight being simulated.
- **Thrust panel resolves into components**: total, the horizontal part that
  beats drag, and the vertical part that holds the weight (constant, equal to
  weight). Hover attitude removed.
- **New drag panel** replaces the single-point power breakdown, which belongs
  on its own tab rather than inside a speed sweep.
  Profile and parasitic drag are labelled as **alternative silhouettes, not
  additive components** — a multirotor pitches nose-down in forward flight and
  presents its frontal area, so total drag equals the parasitic curve. The
  profile curve is shown for comparison because it is what the same aircraft
  would suffer translating level, and it is usually the larger of the two.

**v2.21.1** — fixed-wing status thresholds given a basis
Every hard-coded fixed-wing limit either became an input or acquired a
derivation. None of them are constants chosen by nobody any more.
- **Take-off run and landing distance are Mission/Environment inputs.** How
  much runway is enough depends on your field: 100 m is generous for a
  hand-launch and impossible off a short strip. With no field entered the
  distance is reported without a verdict, rather than judged against a number
  the user never chose.
- **Thrust margin is judged by the climb it buys.** A percentage alone does
  not say whether the aircraft climbs; excess thrust becomes climb rate
  directly through `RC = (T - D) x V / W`. The row now shows that climb rate
  and checks it against a **minimum climb rate** input.
- **Prop tip speed is a Mach limit, not a speed limit.** Compressibility at
  the tip is what costs efficiency and makes noise, and the speed of sound
  falls with temperature — so the old fixed 200 m/s was several percent wrong
  on a cold day. Now `<= Mach 0.60`, evaluated against the local speed of
  sound. On the 2 m survey example this reclassifies 199.4 m/s from a clean
  pass to `edge` at Mach 0.58, which is the honest reading.
- **Reynolds number keeps its bands but states the physics**: below ~70 k the
  laminar separation bubble does not reattach, above ~200 k ordinary
  published polars apply, and between the two the airfoil simply has to be
  chosen for low Re. The mean chord it was computed at is now shown.

**v2.21.0** — mass entry, and the avionics-mass fix
- **Avionics Mass never reached the model.** The field was collected and the
  config class accepted it, but nothing connected the two in `build_config`,
  so a non-zero value vanished before it could appear in the Weight Budget or
  the metrics. Wired in both simulators.
- **"Base Weight" renamed "All Up Weight without Payload"**, with help saying
  what it is used for: the airframe mass is derived from it by subtracting
  battery, motors, ESCs, propellers and avionics, and that residual is what
  the Weight Budget shows as "Airframe / Structure".
- **New Mass Entry Mode.** `derive airframe` behaves as before. `enter
  airframe` reverses it: give the bare structure mass and the all-up weight is
  built up from the components — better when designing from a parts list than
  when weighing a finished aircraft.

**v2.20.2**
- **The fixed-wing thrust-to-weight check used a rotorcraft criterion.** It
  demanded `>= 1.2:1` and called anything less "marginal climb performance",
  which flagged the 2 m survey example red at 0.54:1 — while that aircraft
  climbs at 494 m/min, holds an 80% thrust margin at cruise and takes off in
  19 m. Almost no real fixed-wing could have passed it.
  A wing carries the weight, so thrust only has to beat drag: level flight
  needs `T/W > 1/(L/D)`, which is 0.11 for that survey aircraft and 0.05 for
  the 3 m glider. The threshold now scales with the aircraft's own L/D, and
  the note says what the number means — T/W above 1 is needed only to climb
  vertically.
  The thrust figure feeding it was correct throughout; only the limit was
  wrong.

**v2.20.1** — the same status pass applied to the fixed-wing
- Tag-colour rendering fix, the `edge` band, `_classify` and
  `_dual_limit_row` all ported, so both simulators now colour and classify
  identically.
- Pack current, discharge C-rate, motor current and motor power carry both
  ratings on one row.
- Motor, ESC and battery thermal limits, time-at-maximum allowances, and the
  motor voltage-rating check are all present here too.

**v2.20.0** — status system (multicopter)
- **Status colours now actually render in the GUI.** The tags were always
  configured; Tk 8.6.9 and later silently drop Treeview tag backgrounds
  unless the style's state map is stripped first. That is why the PDF was
  coloured and the window was not.
- **Four states instead of three.** A new `edge` band marks a value within 5%
  of its limit: passing, but with no margin for a gust, a hot day or a tired
  battery.
- **Dual-limit rows.** Pack current, discharge C-rate, motor current, motor
  power and ESC current each carry BOTH ratings on one row — green under the
  continuous rating, amber between the two, red above the maximum. Two
  separate "vs cont" and "vs max" rows made a design sitting between them
  look like one pass and one fail, when it is really a time-limited condition.
- **Time-at-maximum inputs** for the motor, the ESC and the battery. When
  given, exceeding the continuous rating reports how long it may be held.
  When blank, the note says duration is unchecked rather than implying it is
  fine.
- **Thermal limits are inputs**, replacing hard-coded 100/90/55 °C for motor,
  ESC and battery.
- **Motor voltage rating check**, alongside the existing ESC one. Ratings can
  be entered as volts or as an S-count range, since 4-6S motors are specified
  both ways.

**v2.19.0** — inputs reorganised (both simulators)
- **Non-motor loads live together.** Peripheral Current moved from Airframe
  to the top of the Avionics tab, above the rail table. Splitting them across
  tabs made it easy to enter the same device twice — once as a raw pack draw
  and once as a regulated rail.
- **New Avionics Mass input**, beside it, and it now appears in the Weight
  Budget. It previously vanished into the airframe residual, so a user could
  not see it.
- **Cruise Speed and Max Tilt moved to Mission/Environment.** Both describe
  how the aircraft is flown, not what it is.
- **New Plot Settings tab** holding the plot speed range. That is a display
  choice, not a physical input.
- **Frontal Area removed.** Forward flight uses parasite area and
  hover-attitude uses profile area; frontal area survived only as a fallback,
  which now points at parasite area instead.
- **RTH and Diversion reserve inputs removed.** One reserve percentage now
  covers all of it, and its help says so.
- **The reserve help states what the percentage is OF**: usable energy, not
  pack energy. A 100 Wh pack at 80% usable gives 80 Wh, so 20% reserve holds
  back 16 Wh of that — not 20 Wh.
- **SoC curve is a file picker**, not a typed path.
- **Altitude, temperature and pressure default to ISA sea level** rather than
  blanks, so a new user starts from a defined atmosphere.

**v2.18.1** — the same metrics pass applied to the fixed-wing
- **"What it means" column**, matching the multicopter. The PDF report folds
  the note in beside the value.
- **Aircraft section gains mass fractions**: all-up weight, payload, battery
  mass fraction and drive mass fraction. The battery note differs from the
  multicopter's on purpose — a fixed-wing tolerates a higher battery fraction
  because cruise power rises only slowly with weight, where a multirotor pays
  for every gram in hover.
- **Environment reports its inputs**: altitude, cruise altitude, temperature,
  wind speed, and the head/cross split — not only the derived air density.
- **Losses sit with the component that produces them**: pack I2R in Battery,
  motor copper loss in Thrust & Power, ESC and battery headroom in Thermal
  & Losses.
- New metric keys: `altitude_m`, `ambient_temp_C`, `wind_mps`.
  (`battery_loss_W` and `motor_copper_loss_W` already existed.)

**v2.18.0** — Metrics restructuring (multicopter)
- **A "What it means" column.** Every metric can now carry a one-line
  explanation. A table of bare numbers assumes the reader already knows which
  of two same-unit figures is which; several here (figure of merit, the two
  g/W rows, propulsion power) genuinely need saying. The PDF report folds the
  note in beside the value.
- **New Airframe section**, first in the list: all-up weight, payload,
  battery mass fraction, drive mass fraction and motor configuration — what
  the vehicle IS, before how it performs.
- **Environment & Design now reports its inputs**, not just the derived air
  density: altitude, temperature, pressure, wind speed and direction, and the
  head/cross wind split.
- **Losses sit with the thing that produces them.** Pack I2R moved to
  Battery, motor copper loss to Motor @ Operating Point. Thermal Estimates
  gained ESC and battery headroom.
- **Total Drive & Power** replaces the vague "Total Current" with total motor
  current, current into avionics, peripheral current, avionics conversion
  loss, and total power losses.
- New metric keys behind these: `altitude_m`, `ambient_temp_C`,
  `pressure_Pa`, `wind_mps`, `headwind_mps`, `crosswind_mps`,
  `battery_i2r_loss_W`.

**v2.17.0** — clarity pass over the Metrics tab and the PDF report
- **Hover meant "hover attitude at the cruise speed box"**, so hover power
  fell as you raised a speed the aircraft was not flying at, and a hover run
  reported a flight distance. Hover now forces 0 m/s and zero groundspeed.
- **The report's Metrics page showed only section headings.** Once Metrics
  became collapsible, `get_children()` on the root returned the headings and
  nothing under them. It now walks the tree.
- **Report tables wrap** instead of running past the column edge or over the
  next column.
- **Propulsion power excluded motor copper loss.** It was `P_in` less ESC and
  avionics only, so winding heat counted as useful output and flattered
  system efficiency. Renamed **Propulsion Power**, and system efficiency is
  now labelled `P_propulsion / P_in` with a note that avionics power is real
  output that simply does not make lift.
- **"Specific thrust" was a misnomer** — in aerodynamics that is thrust per
  unit mass flow. Renamed **Thrust per Watt (this point)**, and it and Hover
  Efficiency now carry their equations, since they share units.
- **Ideal Hover Power and Actual Induced Power** now state their definitions;
  they differ by exactly the figure of merit.
- Undefined limits read **"Not Specified"** rather than a bare dash.
- The reserve-margin note spells out the arithmetic: usable Wh, minus reserve
  held back, minus what the flight consumes.
- Rows removed as duplicated or not meaningful in a fixed-speed run:
  Cont C-rate limit, Energy Density, Thermal Headroom, Power split, Tilt
  Limit, Acceleration, Reserve Margin, and the duplicate Battery Flight Time.
  Reserve Target renamed **Reserve Battery Amount**.

**v2.16.0**
- **The Compare tab ignored mission runs.** Only the single-point path
  refreshed it, so after Run Mission the deltas were whatever the last
  single-point run produced — a changed parameter looked like it had no
  effect. Mission runs now feed the comparison from their worst-case point.
- **The comparison refuses to mix run types.** A mission's worst-case point
  and a single-point run measure different things, so comparing them would
  show differences that are just a change of run type. Pinning records which
  kind it was, and a mismatch says so instead of reporting a bogus delta.
- **PDF reports now contain the whole analysis**: inputs, metrics, status
  checks, terminal output, every plot, the weight budget, the **airframe
  diagram**, the **sensitivity sweep**, and the **comparison against a pinned
  baseline**. Previously the diagram was missed (it has its own canvas) and
  the two analysis tabs were absent entirely. Sections with no data are
  omitted rather than printed empty.

**v2.15.1**
- **The Status tab kept its own copy of thrust-to-weight** and was missed when
  the Metrics tab was corrected in 2.15.0, so it went on reporting `1.00:1`
  against a `>= 1.5:1` limit — a check that could never fail, on a number that
  was never a margin. It now reads the same available-thrust figure as the
  Metrics tab (6.61:1 on the heavy-lift example), with payload shown at both
  TWR 1.0 and TWR 2.0. The old ratio is retained as "Thrust required /
  weight", labelled a trim check rather than a margin.
  The fixed-wing was already correct — it used available thrust throughout.

**v2.15.0**
- **Hover-attitude thrust added drag linearly instead of in quadrature.**
  Drag is horizontal and weight is vertical, so they combine as a vector
  magnitude. On a 16.5 kg X8 at 18 m/s the old `weight + drag` gave 187 N
  where the true magnitude is 164 N. At 0 m/s both forms agree, which is why
  hover-only checks never caught it.
- **The "Hover" plot series is relabelled "Hover attitude (no yaw into
  wind)".** It was never thrust at 0 m/s — it is the aircraft holding station
  against wind, or translating without turning to face the direction of
  travel, evaluated across airspeed. The old label invited exactly the
  question "why does hover vary with speed".
- **Thrust-to-weight is now a design metric.** It compared REQUIRED thrust
  with weight, which is ~1 by definition in steady flight — the motors are
  carrying the aircraft, so of course it balances — and told a designer
  nothing. It now compares MAXIMUM AVAILABLE thrust with all-up weight: the
  heavy-lift example goes from a meaningless 1.001 to 6.61 : 1.
  Also added: Thrust Available (total), and payload capacity both at TWR 1.0
  and at TWR 2.0, the usual minimum for control authority. The old ratio is
  still shown, labelled "Thrust Required / Weight" with a note that it sits
  at 1 by construction.

**v2.14.0**
- **RPM is now estimated when no propeller table or TConst is supplied**, so
  Back-EMF, Mechanical Power, Motor efficiency, Throttle and Tip Mach read as
  numbers instead of NaN.
  Thrust alone does not determine RPM — two props of the same diameter making
  the same thrust turn at different speeds depending on pitch and blade area —
  so this is a genuine estimate from `T = C_T * rho * n^2 * D^4`, with C_T
  fitted to pitch/diameter and blade count. Expect **+/-30% on C_T**, which is
  **+/-15% on RPM**. A measured table or an explicit `TConst` both take
  priority and are far more accurate. The metrics carry
  `prop_rpm_is_estimated` so the two cases are distinguishable.
  Sanity check: across 5in to 22in props the estimate puts hover tip speed at
  60-63 m/s, which is where real multirotors sit.
- **Motor efficiency was badly wrong and is now correct.** The display mixed
  the two sides of the ESC: electrical power used pack-side current x pack
  voltage, while mechanical power used that same pack-side current with the
  winding back-EMF. The ESC chops the pack voltage, so the winding sees much
  less voltage and much more current. On a 12S heavy-lift this read **13.4%**
  motor efficiency; it now reads **89.0%**, which is what a large low-Kv motor
  at 22% throttle should do.
  Winding current is solved from conservation of electrical power across the
  ESC, and both currents are now shown separately.

**v2.13.2**
- **Wind no longer aborts flyable multicopter mission legs.** A distance leg
  starting from a hover spends the first second or two of its acceleration
  ramp below the headwind, so its *instantaneous* groundspeed is zero. The
  guard judged the phase on that instant and reported
  `Invalid: zero groundspeed with distance phase`, killing a 10 m/s leg into a
  3 m/s headwind — a 7 m/s steady groundspeed, entirely flyable.
  The check now asks whether the leg can make progress at its **commanded**
  airspeed, with a 120 s stall timeout so a genuine deadlock still terminates.
  A headwind above the commanded airspeed is still rejected, now with a
  message naming the real cause.
  Single-point runs were never affected (no ramp), and neither was the
  fixed-wing.

**v2.13.1**
- **Fixed the matplotlib figure leak.** Embedded plots were built with
  `pyplot.subplots()`, which keeps every figure alive in a module-level
  registry until something closes it. Nothing ever did, so a GUI session
  accumulated a figure per run for its whole life and matplotlib warned
  "More than 20 figures have been opened".
  All 10 embedded-figure sites now build from `matplotlib.figure.Figure`
  directly via `core.make_figure()`, so a figure is freed with its canvas.
  Verified at zero retained figures after 80 builds; the full test matrix went
  from 12 matplotlib warnings to none.
  Interactive CLI plotting still uses pyplot, where the registry is doing its
  job — `plt.show()` needs to find the figures.

**v2.13.0**
- **New `tests/test_matrix.py` closes the coverage gap that let two bugs
  ship.** It runs the product of the dimensions that actually vary: both
  simulators x every example config x every example mission x GUI and CLI x
  with and without a measured propeller table. 67 cases.
  Both escaped bugs were of the form "works in one interface, or one table
  state, and not the other" — the `name 'm' is not defined` crash, and a
  fixed-wing table loader that carried a known fault for six releases because
  nothing exercised it.
  The matrix also asserts the table *changes the answer*, so a table that is
  accepted but silently ignored cannot pass.
- **Unknown hover wind resistance prints `n/a`** instead of a bare `nan`. The
  NaN was correct — it means no reference area was given — but it read like a
  failure. Found by the new matrix.

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

**Earlier**
- Pack capacity no longer scales with the series count (it never should have).
- Fixed-wing cruise power uses forward-flight rather than static momentum
  theory — this alone changed endurance by about 5x.
- Multicopter arm drag is no longer counted twice in forward flight.

---

## Testing

A pytest suite lives in `tests/` — 752 tests covering physics, the shared
core, the CLI, the GUI, the batch driver, the VTOL, and the drag calculator.
The full suite was last run clean on Windows. See `tests/README.md`.

```bash
pip install pytest
pytest                                # everything, ~30 minutes
pytest -m "not slow and not gui"      # no subprocesses or windows, ~2 minutes
xvfb-run -a pytest                    # headless machines (GUI tests need a display)
```

| Suite | Tests | Covers |
|---|---|---|
| `test_golden.py` | 1 | 1042 stored numeric outputs across all three simulators; fails if any value drifts |
| `test_core.py` | 90 | The shared core, plus checks that the simulators really delegate to it |
| `test_physics.py` | 136 | Battery topology, atmosphere, rotor inflow, prop efficiency, SoC, drag, landing, drag calculator |
| `test_cli.py` | 70 | Subprocess runs of every argument path and example mission, plus edge cases and malformed input |
| `test_gui.py` | 46 | Real Tk window: hover events, mode toggle, config load, missions, exports |
| `test_batch.py` | 20 | Sweeps, sizing, mode enforcement, and GUI↔CLI consistency |
| `test_matrix.py` | 108 | Every config and mission, GUI **and** CLI, with **and** without a propeller table — the VTOL through its CLI |
| `test_vtol.py` | 250 | All four VTOL types: vectored-thrust force balance, tilt, download, transition, missions, wind, bench tables, the motor model, GUI/CLI/batch parity, and every GUI tab |
| `test_drag_calculator.py` | 31 | Shoelace geometry, self-intersection detection, pixel scaling, ISA density, ArduPilot BCOEF and MCOEF |

Almost every test corresponds to a bug that was shipped at some point; the
docstrings name the symptom. Two are worth knowing about: the tooltip test
fires real hover events (an earlier version counted widgets and passed while
every tooltip crashed), and the consistency test asserts that the same config
file produces identical numbers through the GUI and through the batch driver.
