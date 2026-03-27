# RotorWorks Multicopter Power Simulator (GUI + CLI)
UASforge
dronefoundry

A cross-platform multicopter performance simulator.  
It supports **single-point analysis**, **mission simulation**, configurable **battery / motor / ESC / prop / avionics rails**, **status limit checks**, and **plots** including mission time-series.

## Features

- **GUI (Tkinter)**
  - Tabs for Drone, Battery, Motor, ESC, Avionics, Prop, Mission/Env
  - **Metrics tab** (single-point output summary)
  - **Status tab** with color-coded limit checks (green/yellow/red)
  - **Mission Plots tab** to plot selected variables vs mission time with multi-unit y-axes
  - Save/Load configuration to JSON (so you don’t retype values)
  - View menu to change UI scaling dynamically

- **CLI**
  - Run single-point calculations headless
  - Run mission simulations using a JSON mission file
  - Provide avionics rails via a voltage-tree argument

- **Motor/Prop test-table support**
  - Load a **CSV** with columns such as `Thrust (g)`, `Power (W)`, `RPM`, `Current (A)`, `Voltage (V)`, `Throttle`, etc.
  - Robust parsing handles “extra header rows” and normalizes columns automatically
  - Uses `Throttle` from the table for more accurate single-point throttle reporting (falls back to analytic log/linear estimation otherwise)

- **Coaxial vs flat configurations**
  - `motor_configuration`: `flat` or `coaxial`
  - `coaxial_spacing_m` affects coaxial interference penalty (power multiplier decreases as spacing increases)
  - Geometry-based drag fallback uses body box + square tube arms when drag inputs are not provided

## Requirements

- Python **3.9+** recommended
- Dependencies:

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
````

`requirements.txt`:

```txt
numpy>=1.21
scipy>=1.8
pandas>=1.4
matplotlib>=3.6
```

> `tkinter` is part of the Python standard library and does **not** need pip installation.

---

## Quick Start (GUI)

Run the simulator:

```bash
python multicopter-power-sim-gui_table_esc_status_fixed_save_load_view_scale_vehicle_params_armwidth_tilt_statusrow_coaxial_spacing_mission_plots_metrics_tableparse_final.py
```

### Configure tabs

1. **Drone**

   * Total weight
   * Motor configuration (flat / coaxial)
   * Body dimensions, arm length & width
   * Max tilt angle
2. **Battery**

   * Cell / pack configuration
   * Capacity, resistance, discharge limits
3. **Motor**

   * KV / limits **or** motor-prop CSV
4. **ESC**

   * Continuous & max current
   * Resistance (for power loss)
5. **Avionics**

   * Add voltage rails (V, A, efficiency)
6. **Prop**

   * Diameter, pitch, limits
7. **Mission / Env**

   * Speed, wind, altitude
   * Or load a mission JSON

### Run

* **Single point:** `Run single-point + plot`
* **Mission:** `Run mission (JSON) + plot`

---

## Quick Start (CLI)

Single-point example:

```bash
python simulator.py \
  --speed_mps 10 \
  --weight_kg 2.5 \
  --num_motors 4 \
  --series_cells 6 \
  --parallel_cells 1 \
  --cell_capacity_mah 5000 \
  --avionics_voltage_tree "5.0:(2,0.9),12.0:(1,0.85)"
```

Mission run:

```bash
python simulator.py --mission_json mission.json
```

List all CLI options:

```bash
python simulator.py --help
```

---

## Motor-Prop CSV Table Format

The simulator accepts **CSV exports**.

### Required columns (any naming variant is accepted):

* **Thrust (g)**
* **Power (W)**

### Optional (used when present):

* `RPM`
* `Current (A)`
* `Voltage (V)`
* `Throttle` (e.g. `40%`)
* `Efficiency (g/W)`
* `Operating Temperature (℃)`
* `Torque (N*m)`

### Notes

* Header row **does not need to be first line**
* Column names are normalized automatically
* Power interpolation is **thrust-based**
* Throttle column overrides analytic throttle estimation in single-point mode

---

## Mission JSON Format

Mission simulation uses a JSON file with ordered phases.

### Example

```json
{
  "phases": [
    {
      "name": "takeoff",
      "duration_s": 15,
      "airspeed_mps": 0,
      "altitude_m": 10
    },
    {
      "name": "cruise",
      "duration_s": 300,
      "airspeed_mps": 12,
      "altitude_m": 50
    },
    {
      "name": "loiter",
      "duration_s": 120,
      "airspeed_mps": 6,
      "altitude_m": 50
    },
    {
      "name": "landing",
      "duration_s": 20,
      "airspeed_mps": 0,
      "altitude_m": 0
    }
  ]
}
```

### Outputs

* Time-series data for:

  * Speed, altitude, tilt
  * Battery voltage/current/capacity
  * Motor power, RPM, thrust
* Worst-case values used for **Status** checks

---

## Metrics Tab (Single-Point Output)

### Battery

* Load (C)
* Loaded voltage
* Rated voltage
* Energy (Wh)
* Total & usable capacity
* Flight time at this operating point
* Battery weight

### Motor @ Operating Point

* Current, voltage, RPM
* Electric & mechanical power
* Throttle (log & linear)
* Power-to-weight (W/kg)
* Efficiency
* Resistance
* Specific thrust (g/W)
* Estimated temperature

### Total Drive

* Total drivetrain weight
* Thrust-to-weight ratio
* Total current
* P(in), P(out)
* Overall efficiency

### Multicopter

* All-up weight
* Required tilt vs max tilt
* Speed (km/h & mph)
* Estimated range (m & mi)
* Total disk area (cm² & in²)
* Maximum additional payload

---

## Modeling Notes / Assumptions

* Forward-flight tilt:

  ```
  tilt = atan(D / W)
  ```
* `max_tilt_deg` is **enforced**
* Coaxial interference penalty depends on spacing ratio `s / D`
* ESC losses:

  ```
  P_loss = I²R + I_idle·V
  ```
* Avionics power:

  ```
  I_pack = Σ(V·I / eff) / V_pack
  ```
* Geometry drag fallback:

  * Box-shaped body
  * Square-tube arms
  * Rotor disk drag is **not assumed** unless included via CdA

This is a **performance-level model**, not CFD or transient motor dynamics.

---

## Troubleshooting

### UI text is too small

Use **View → UI Scale** (150–200%).

### Tkinter not found (Linux)

```bash
sudo apt install python3-tk
```

### CSV not parsing

* Ensure it contains **Thrust** and **Power** columns
* Extra header rows are OK
* Open CSV and verify numeric units

### “Cannot maintain speed”

* Required tilt exceeds `max_tilt_deg`
* Reduce speed, drag, or weight

---
