"""
Shared fixtures for the RotorWorks test suite.

The simulators are single-file scripts rather than an installed package, and
they import tkinter lazily inside launch_gui() so that CLI use works on
machines without tk.  These fixtures load them by path, with matplotlib forced
to a headless backend.

Run everything:      pytest
Skip the slow ones:  pytest -m "not slow"
GUI tests only:      pytest -m gui
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

# Repository root = parent of tests/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MULTICOPTER = os.path.join(ROOT, "multicopter-power-sim-gui.py")
FIXEDWING = os.path.join(ROOT, "fixedwing-power-sim-gui.py")
ROTORWORKS = os.path.join(ROOT, "rotorworks-batch.py")
DRAGCALC = os.path.join(ROOT, "drag_coefficient_calculator.py")
EXAMPLES = os.path.join(ROOT, "examples")


def _load(path: str, name: str):
    """Import a simulator script by file path, headless."""
    import matplotlib

    matplotlib.use("Agg")

    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_tkinter():
    """
    Insert placeholder tkinter modules for tests that never open a window.

    The simulators import tkinter inside launch_gui(), so module-level import
    succeeds without it — but some environments have no tk at all, and we want
    the physics tests to run there too.
    """
    for name in (
        "tkinter",
        "tkinter.ttk",
        "tkinter.messagebox",
        "tkinter.filedialog",
        "tkinter.simpledialog",
        "tkinter.font",
        "tkinter.scrolledtext",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))


@pytest.fixture(scope="session")
def mc():
    """The multicopter simulator module."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        _stub_tkinter()
    return _load(MULTICOPTER, "rw_multicopter")


@pytest.fixture(scope="session")
def fw():
    """The fixed-wing simulator module."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        _stub_tkinter()
    return _load(FIXEDWING, "rw_fixedwing")


@pytest.fixture(scope="session")
def rw():
    """The rotorworks batch driver module."""
    return _load(ROTORWORKS, "rw_batch")


@pytest.fixture(scope="session")
def dragcalc():
    """The drag coefficient calculator module (physics only; no window built)."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        _stub_tkinter()
    return _load(DRAGCALC, "rw_dragcalc")


@pytest.fixture(scope="session")
def paths():
    """Absolute paths to the scripts and example data."""
    return {
        "root": ROOT,
        "multicopter": MULTICOPTER,
        "fixedwing": FIXEDWING,
        "rotorworks": ROTORWORKS,
        "dragcalc": DRAGCALC,
        "examples": EXAMPLES,
        "configs": os.path.join(EXAMPLES, "configs"),
        "missions": os.path.join(EXAMPLES, "missions"),
    }


# ----------------------------------------------------------------------
# Reference aircraft used across the physics tests.
#
# These are deliberately plain, fully-specified builds. Do not "improve" the
# numbers: several tests assert against values derived from them by hand.
# ----------------------------------------------------------------------

@pytest.fixture
def mc_quad(mc):
    """A 1.8 kg quad on a 4S 5200 mAh pack with 10x4.5 props."""
    batt = mc.BatteryConfig(
        chemistry="LiPo",
        operating_voltage_min=3.3, operating_voltage_nominal=3.7,
        operating_voltage_max=4.2,
        unit_mode="pack", pack_capacity_mAh=5200, pack_weight_g=520,
        series_units=1, parallel_units=1, cells_series_per_unit=4,
        discharge_percent=80, resistance_cell_mOhm=4.0, discharge_c_cont=25,
    )
    drone = mc.DroneConfig(
        num_motors=4, battery=batt,
        motor=mc.MotorConfig(
            kv=920, idle_current=0.5, idle_voltage=10, rated_voltage=4,
            resistance=0.115, max_current=18, max_power=250, weight_g=56),
        propeller=mc.PropellerConfig(
            diameter_in=10, pitch_in=4.5, max_rpm=0, max_thrust_g=1100,
            blades=2, weight_g=13),
        drone_weight_g=1800,
        profile_drag_coefficient=None, profile_area=None,
        parasite_drag_coefficient=None, parasite_area=None, frontal_area=None,
        cruise_speed=10.0, periph_current=0.0,
        body_length_m=0.22, body_width_m=0.16, body_height_m=0.09,
        arm_length_m=0.225, arm_width_m=0.016,
    )
    drone.air_density = mc.compute_air_density(0)
    drone.derive_drag_from_geometry_if_missing()
    return drone


@pytest.fixture
def fw_plane(fw):
    """A 2.6 kg, 2 m span survey aircraft on a 4S 10000 mAh pack."""
    batt = fw.BatteryConfig(
        chemistry="LiPo",
        operating_voltage_min=3.3, operating_voltage_nominal=3.7,
        operating_voltage_max=4.2,
        unit_mode="pack", pack_capacity_mAh=10000, pack_weight_g=880,
        series_units=1, parallel_units=1, cells_series_per_unit=4,
        discharge_percent=80, resistance_cell_mOhm=3.5,
    )
    airframe = fw.AirframeConfig(
        wing_span_m=2.0, wing_area_m2=0.46, CD0=0.028, oswald=0.87,
        CL_max=1.25, prop_efficiency=0.76, mu_roll=0.05, mu_brake=0.30,
    )
    return fw.FixedWingConfig(
        aircraft_weight_g=2600, airframe=airframe, battery=batt,
        motor=fw.MotorConfig(
            kv=750, idle_current=0.8, idle_voltage=10, rated_voltage=4,
            resistance=0.06, max_current=40, max_power=600, weight_g=160),
        propeller=fw.PropellerConfig(
            diameter_in=12, pitch_in=8, blades=2, weight_g=22),
        cruise_speed_mps=19.0, air_density=1.225,
        reference_altitude_m=120,
    )
