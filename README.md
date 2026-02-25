# rotorworks
UASforge
dronefoundry

Multicopter performance simulator with three modeling modes:
1. Motor test table (CSV input with thrust vs power).
2. Motor electrical model (using KV, idle current, resistance, limits).
3. Theoretical induced velocity model (simplified).

Features:
- Power consumption (hover + forward flight)
- Flight time & distance
- Best endurance & best range speeds
- Plotting of performance curves
- Mission profile simulation (JSON)
- NEW: Optional Tkinter GUI for entering inputs and viewing plots in one window.
      CLI interface is preserved.

Examples (CLI):
    python multicopter-power-sim.py --num_motors 4 --weight 1.5 --area 0.05 \
        --battery_operating_voltage_min 3.0 --battery_operating_voltage_max 4.2 \
        --battery_capacity 5000 --battery_weight 400 --battery_energy_density 200 \
        --battery_charge_current_max 5 --battery_discharge_cont 60 --battery_resistance_cell 20 \
        --battery_cell_count 4 --battery_chemistry LiIon \
        --motor_kv 650 --motor_idle_current 0.5 --motor_resistance 0.2 --motor_max_current 20 --motor_max_power 200 \
        --prop_diameter 12 --prop_pitch 6 \
        --speed 10 --plot

    # Use motor/prop test table for power interpolation:
    python multicopter-power-sim.py ... --prop_table motor_data.csv --plot

    # Run GUI:
    python multicopter-power-sim.py --gui
