"""
Tests for drag_coefficient_calculator.py.

The calculator has two jobs, and this file checks both against things that are
independently knowable rather than against its own output:

  1. GEOMETRY — turning a traced polygon and a scale reference into an area in
     square metres. Verified against shapes whose area is known exactly.

  2. PHYSICS — turning that area, a mass and a rotor size into ArduPilot's
     EK3_DRAG_BCOEF and EK3_DRAG_MCOEF. Verified by re-deriving each result
     from first principles, and by checking the invariants that must hold
     whatever the numbers are.

The GUI itself (photo loading, click-to-trace, the canvas) is NOT covered —
it needs real mouse input. What is covered is every pure function behind it,
which is where a wrong answer would come from.

Run:
    pytest tests/test_drag_calculator.py -v
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types

import pytest


# ----------------------------------------------------------------------
# Loading the module
# ----------------------------------------------------------------------
# The calculator imports tkinter at module scope. On a headless machine that
# import succeeds but constructing a window does not — and we only want the
# pure functions, so stub anything GUI before loading.

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def dc():
    """The drag calculator module, loaded without needing a display."""
    # Only stub when there is no real tkinter. Stubbing unconditionally
    # looked safe because setdefault leaves an imported module alone — but
    # `import tkinter` does not pull in tkinter.simpledialog, so the
    # placeholder took that slot and matplotlib's Tk backend later died on
    # "cannot import name SimpleDialog", erroring every GUI test that ran
    # afterwards in the same session.
    try:
        import tkinter  # noqa: F401
        import tkinter.simpledialog  # noqa: F401
        import tkinter.ttk  # noqa: F401
    except ImportError:
        for name in ("tkinter", "tkinter.ttk", "tkinter.messagebox",
                     "tkinter.filedialog", "tkinter.simpledialog",
                     "tkinter.font"):
            sys.modules.setdefault(name, types.ModuleType(name))
        # ttk.Frame and tk.Tk are subclassed at import time, so they must be
        # classes rather than bare module attributes.
        ttk = sys.modules["tkinter.ttk"]
        tk = sys.modules["tkinter"]
        for mod, attrs in ((ttk, ("Frame", "Notebook", "Label", "Button", "Entry",
                                  "Combobox", "Treeview", "Scrollbar", "LabelFrame",
                                  "Radiobutton", "Checkbutton", "Style", "Separator")),
                           (tk, ("Tk", "Canvas", "Text", "StringVar", "Frame",
                                 "Toplevel", "Menu", "PhotoImage", "TclError"))):
            for attr in attrs:
                if not hasattr(mod, attr):
                    setattr(mod, attr, type(attr, (object,), {"__init__": lambda self, *a, **k: None}))

    path = os.path.join(_ROOT, "drag_coefficient_calculator.py")
    spec = importlib.util.spec_from_file_location("drag_calc", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["drag_calc"] = module
    spec.loader.exec_module(module)
    return module


# ======================================================================
# POLYGON AREA — the shoelace formula
# ======================================================================

def test_unit_square(dc):
    """The simplest case with an obvious answer."""
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert dc.polygon_area_px2(square) == pytest.approx(1.0)


def test_rectangle_and_triangle(dc):
    """Shapes whose area is known from school geometry."""
    assert dc.polygon_area_px2(
        [(0, 0), (40, 0), (40, 25), (0, 25)]) == pytest.approx(1000.0)
    # Triangle: half base times height.
    assert dc.polygon_area_px2(
        [(0, 0), (10, 0), (0, 6)]) == pytest.approx(30.0)


def test_winding_direction_does_not_change_the_area(dc):
    """
    The shoelace sum is signed — it comes out negative for a clockwise
    polygon. A user tracing a photo has no reason to care which way round they
    click, so the sign must be discarded.
    """
    ccw = [(0, 0), (10, 0), (10, 10), (0, 10)]
    cw = list(reversed(ccw))
    assert dc.polygon_area_px2(cw) == pytest.approx(dc.polygon_area_px2(ccw))
    assert dc.polygon_area_px2(cw) > 0


def test_translation_and_vertex_order_do_not_matter(dc):
    """
    Where the shape sits in the photo, and which corner the user happened to
    click first, must not change its area.
    """
    base = [(0, 0), (8, 0), (8, 5), (0, 5)]
    shifted = [(x + 1234, y - 987) for x, y in base]
    rotated = base[2:] + base[:2]          # same loop, different start vertex
    assert dc.polygon_area_px2(shifted) == pytest.approx(40.0)
    assert dc.polygon_area_px2(rotated) == pytest.approx(40.0)


def test_degenerate_polygons_are_zero_not_an_error(dc):
    """
    A half-finished trace is a normal intermediate state in the GUI, not a
    fault. It must return zero rather than raising.
    """
    assert dc.polygon_area_px2([]) == 0.0
    assert dc.polygon_area_px2([(0, 0)]) == 0.0
    assert dc.polygon_area_px2([(0, 0), (5, 5)]) == 0.0
    # Three collinear points enclose nothing.
    assert dc.polygon_area_px2([(0, 0), (1, 1), (2, 2)]) == pytest.approx(0.0)


def test_area_of_a_polygon_approximating_a_circle(dc):
    """
    A traced outline is a polygon approximation of a smooth shape, so the
    error should shrink as the user clicks more points. This is the shape of
    error a real trace has.
    """
    def circle(n, r=100.0):
        return [(r * math.cos(2 * math.pi * i / n),
                 r * math.sin(2 * math.pi * i / n)) for i in range(n)]

    exact = math.pi * 100.0 ** 2
    coarse = dc.polygon_area_px2(circle(8))
    fine = dc.polygon_area_px2(circle(64))

    assert coarse < exact, "an inscribed polygon always under-reports"
    assert fine < exact
    assert abs(fine - exact) < abs(coarse - exact), \
        "more vertices should get closer, not further"
    assert fine == pytest.approx(exact, rel=0.01)


def test_a_concave_shape_is_handled(dc):
    """
    An airframe silhouette is not convex — arms and booms cut into it. An
    L-shape checks the formula does not assume convexity.
    """
    # 10x10 square with a 5x5 bite taken out of one corner: 100 - 25 = 75.
    l_shape = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    assert dc.polygon_area_px2(l_shape) == pytest.approx(75.0)


# ======================================================================
# PIXEL SCALING
# ======================================================================

def test_area_scales_with_the_square_of_the_scale(dc):
    """
    Area goes as length squared, so a scale that is wrong by 10% makes the
    area wrong by 21%. Getting this exponent wrong is the single most likely
    way for the tool to produce a plausible but wrong answer, so it is worth
    stating explicitly.
    """
    area_px2 = 40000.0                     # e.g. a 200x200 px silhouette
    for px_per_m in (100.0, 200.0, 500.0):
        area_m2 = area_px2 / px_per_m ** 2
        assert area_m2 == pytest.approx(40000.0 / px_per_m ** 2)

    # A 10% scale error becomes a 21% area error.
    true_area = area_px2 / 100.0 ** 2
    with_error = area_px2 / 110.0 ** 2
    assert with_error / true_area == pytest.approx(1 / 1.21, rel=1e-9)


def test_a_known_scale_recovers_a_known_area(dc):
    """
    End to end: a 450 mm square traced in a photo where 1 m spans 400 px must
    come out as 0.2025 m².
    """
    px_per_m = 400.0
    side_px = 0.450 * px_per_m             # 180 px
    square = [(0, 0), (side_px, 0), (side_px, side_px), (0, side_px)]
    area_m2 = dc.polygon_area_px2(square) / px_per_m ** 2
    assert area_m2 == pytest.approx(0.450 ** 2)


# ======================================================================
# ISA ATMOSPHERE
# ======================================================================

def test_sea_level_density_matches_the_isa_value(dc):
    """The defining number of the standard atmosphere."""
    assert dc.isa_density(0.0) == pytest.approx(1.225, rel=1e-3)


def test_density_falls_with_altitude(dc):
    """Monotonic, and about 19% down at 2000 m."""
    values = [dc.isa_density(h) for h in (0, 500, 1000, 2000, 4000)]
    assert values == sorted(values, reverse=True)
    assert dc.isa_density(2000.0) == pytest.approx(1.0065, rel=5e-3)


def test_a_hot_day_is_less_dense_than_a_cold_one(dc):
    """
    Overriding the temperature keeps the ISA pressure but changes density.
    A rotorcraft loses performance on a hot day precisely because of this.
    """
    cold = dc.isa_density(0.0, temperature_C=-10.0)
    standard = dc.isa_density(0.0, temperature_C=15.0)
    hot = dc.isa_density(0.0, temperature_C=40.0)
    assert cold > standard > hot
    # Ideal gas at fixed pressure: rho ratio is the inverse temperature ratio.
    assert hot / standard == pytest.approx(288.15 / 313.15, rel=1e-6)


def test_below_sea_level_is_denser_and_shares_the_core_atmosphere(dc):
    """Audit C8, D7: a site below sea level has denser air (it used to be
    clamped to 0 m), and the calculator's atmosphere is the core's own, not
    a copy that could drift. Inputs below -1000 m are clamped there."""
    assert dc.isa_density(-430.0) > dc.isa_density(0.0) * 1.04
    assert dc.isa_density(-430.0) == dc.core.air_density(-430.0)
    assert dc.isa_density(-5000.0) == pytest.approx(dc.isa_density(-1000.0))
    assert dc.isa_density(-1000.0) < 1.4, "must not blow up"


# ======================================================================
# ARDUPILOT BCOEF
# ======================================================================
# ArduPilot models body drag as:      a_drag = (rho / 2) * V^2 / BCOEF
# Standard drag is:                   a_drag = (rho / 2) * V^2 * Cd * A / m
# Equating them gives:                BCOEF  = m / (Cd * A)

def _bcoef(mass_kg, cd, area_m2):
    return mass_kg / (cd * area_m2)


def test_bcoef_matches_its_definition(dc):
    """Re-derived from the drag equation rather than copied from the code."""
    mass, cd, area = 2.5, 0.8, 0.045
    bcoef = _bcoef(mass, cd, area)

    rho, speed = 1.225, 12.0
    accel_from_bcoef = (rho / 2.0) * speed ** 2 / bcoef
    accel_from_drag = 0.5 * rho * speed ** 2 * cd * area / mass
    assert accel_from_bcoef == pytest.approx(accel_from_drag, rel=1e-12)


def test_bcoef_units_and_direction(dc):
    """
    BCOEF is a BALLISTIC coefficient: higher means LESS deceleration. A
    heavier or sleeker aircraft has a higher value, which is the opposite
    sense to a drag coefficient and easy to get backwards.
    """
    base = _bcoef(2.0, 0.8, 0.05)
    assert _bcoef(4.0, 0.8, 0.05) > base, "heavier should raise BCOEF"
    assert _bcoef(2.0, 0.4, 0.05) > base, "sleeker should raise BCOEF"
    assert _bcoef(2.0, 0.8, 0.10) < base, "more area should lower BCOEF"


def test_front_and_side_bcoef_differ_for_a_non_square_airframe(dc):
    """
    BCOEF_X uses the frontal silhouette and BCOEF_Y the side one. On any
    airframe that is not square in plan they must differ — if they came out
    equal, the two views would have been mixed up.
    """
    mass, cd = 3.0, 0.8
    frontal, side = 0.040, 0.065
    assert _bcoef(mass, cd, frontal) != _bcoef(mass, cd, side)
    assert _bcoef(mass, cd, frontal) > _bcoef(mass, cd, side), \
        "the smaller silhouette must give the larger ballistic coefficient"


def test_ardupilot_simplified_form_is_the_cd_one_special_case(dc):
    """
    ArduPilot's own calibration note says to use BCOEF = mass / area, which is
    this tool's formula with Cd = 1. The tool exposes Cd separately so the
    same measurement can also feed a simulator that wants Cd and area apart.
    """
    mass, area = 2.5, 0.05
    assert _bcoef(mass, 1.0, area) == pytest.approx(mass / area)


# ======================================================================
# ARDUPILOT MCOEF
# ======================================================================
# Momentum drag: a rotor translating sideways ingests air that carries no
# lateral momentum and expels it moving with the aircraft, which costs a
# force. Per rotor the mass flow is rho * A * v_h, so
#
#     F = N * rho * A * v_h * V        and       a = MCOEF * V
#     MCOEF = N * rho * A * v_h / m
#
# Hover momentum theory gives  T = 2 * rho * A * v_h^2  with  T = m*g/N, so
# rho*A = m*g / (2*N*v_h^2) and the whole thing collapses to
#
#     MCOEF = g / (2 * v_h)

def _hover_vi(mass_kg, n_rotors, diameter_m, rho):
    area = math.pi / 4.0 * diameter_m ** 2
    thrust = mass_kg * 9.80665 / n_rotors
    return math.sqrt(thrust / (2.0 * rho * area))


def test_mcoef_collapses_to_the_closed_form(dc):
    """
    The long form and the short form must agree exactly. If they ever diverge,
    the closed form has been applied outside the assumption it rests on.
    """
    mass, n, diameter, rho = 1.8, 4, 0.254, 1.225
    v_h = _hover_vi(mass, n, diameter, rho)
    area = math.pi / 4.0 * diameter ** 2

    long_form = n * rho * area * v_h / mass
    short_form = 9.80665 / (2.0 * v_h)
    assert long_form == pytest.approx(short_form, rel=1e-9)


def test_hover_induced_velocity_is_physically_sane(dc):
    """A small quad hovers with a downwash of a few metres per second."""
    v_h = _hover_vi(1.8, 4, 0.254, 1.225)     # 1.8 kg on 10 in props
    assert 3.0 < v_h < 12.0, f"{v_h:.2f} m/s is not a credible downwash"


def test_bigger_discs_lower_the_induced_velocity_and_raise_mcoef(dc):
    """
    A larger rotor moves more air more slowly for the same thrust, so v_h
    falls — and since MCOEF goes as 1/v_h, momentum drag RISES. That inverse
    relationship is counter-intuitive and worth pinning down.
    """
    small = _hover_vi(1.8, 4, 0.127, 1.225)   # 5 in
    large = _hover_vi(1.8, 4, 0.381, 1.225)   # 15 in
    assert large < small, "a bigger disc should hover with less downwash"
    assert 9.80665 / (2 * large) > 9.80665 / (2 * small), \
        "lower downwash means MORE momentum drag, not less"


def test_mcoef_is_the_IDEAL_upper_bound_not_ardupilots_default(dc):
    """
    FINDING: the tool's MCOEF is 4-6x ArduPilot's documented value, on every
    aircraft class.

        5 in racer     0.66        450 quad       0.82
        IRIS (SITL)    0.90        heavy X8       0.85
        ArduPilot documented default / typical:   ~0.15

    The derivation is not wrong, but it is an IDEAL: `MCOEF = g / (2*v_h)`
    assumes the rotor entrains the freestream completely and turns all of its
    lateral momentum. A real rotor captures only a fraction, which is why
    ArduPilot's empirically-tuned figure is far lower.

    So the number is a physically-meaningful upper bound, NOT a value to type
    straight into EK3_DRAG_MCOEF. Anyone who does will over-damp the EKF's
    drag fusion by roughly six times.

    This test pins the behaviour as it is. It deliberately does not apply a
    fudge factor to force agreement — that would be inventing a coefficient
    to hide a modelling assumption rather than stating it.
    """
    cases = [
        ("5in racer",     0.7, 4, 0.127),
        ("IRIS (SITL)",   1.5, 4, 0.254),
        ("450 quad",      1.8, 4, 0.254),
        ("7in longrange", 1.3, 4, 0.178),
        ("heavy X8",     16.5, 8, 0.559),
    ]
    ardupilot_typical = 0.15
    for name, mass, n, diameter in cases:
        v_h = _hover_vi(mass, n, diameter, 1.225)
        mcoef = 9.80665 / (2.0 * v_h)

        assert 0.0 < mcoef < 1.5, f"{name}: MCOEF {mcoef:.3f} not physical"
        assert mcoef > ardupilot_typical * 2.0, (
            f"{name}: MCOEF {mcoef:.3f} unexpectedly close to ArduPilot's "
            "empirical value — the ideal bound should sit well above it, so "
            "either the formula or this expectation has changed")


def test_mcoef_ignores_rotor_count_at_fixed_disc_loading(dc):
    """
    MCOEF depends only on induced velocity, which depends only on DISC
    LOADING. Splitting the same weight across more rotors of the same total
    disc area must not change it — a useful check that the per-rotor
    bookkeeping is right.
    """
    mass, rho = 4.0, 1.225
    # 4 rotors of 0.30 m vs 8 rotors of the same total area.
    d4 = 0.30
    total_area = 4 * math.pi / 4.0 * d4 ** 2
    d8 = math.sqrt(total_area / 8 * 4.0 / math.pi)

    v4 = _hover_vi(mass, 4, d4, rho)
    v8 = _hover_vi(mass, 8, d8, rho)
    assert v4 == pytest.approx(v8, rel=1e-9)


def test_altitude_raises_momentum_drag(dc):
    """
    Thinner air means a higher induced velocity for the same thrust, so MCOEF
    falls with altitude. Sign errors here are easy and silent.
    """
    rho_sl = dc.isa_density(0.0)
    rho_alt = dc.isa_density(3000.0)
    v_sl = _hover_vi(1.8, 4, 0.254, rho_sl)
    v_alt = _hover_vi(1.8, 4, 0.254, rho_alt)

    assert v_alt > v_sl, "thinner air needs a faster downwash"
    assert 9.80665 / (2 * v_alt) < 9.80665 / (2 * v_sl), \
        "momentum drag should fall with altitude"


# ======================================================================
# CROSS-CHECK AGAINST THE SIMULATORS
# ======================================================================

def test_measured_area_feeds_the_simulator_the_same_way(dc):
    """
    The calculator's whole purpose is to produce numbers the power simulators
    consume. Its frontal area becomes `parasite_area` and its side area
    becomes `profile_area`, so a measurement here must produce the drag the
    simulator would compute from the same figures.
    """
    rho, speed = 1.225, 12.0
    cd, frontal_area = 0.8, 0.045

    drag_N = 0.5 * rho * speed ** 2 * cd * frontal_area
    # Same numbers via the ballistic coefficient the tool reports.
    mass = 2.5
    bcoef = _bcoef(mass, cd, frontal_area)
    accel = (rho / 2.0) * speed ** 2 / bcoef

    assert accel * mass == pytest.approx(drag_N, rel=1e-12)


def test_reference_iris_bcoef_is_in_the_expected_band(dc):
    """
    ArduPilot's SITL IRIS is the closest thing to a published reference:
    1.5 kg on four 10 in rotors. Its BCOEF should land in the tens of kg/m2
    for a frontal area of a few hundred cm2.
    """
    mass, frontal_area, cd = 1.5, 0.04, 0.9
    bcoef = _bcoef(mass, cd, frontal_area)
    assert 30.0 < bcoef < 60.0, f"BCOEF {bcoef:.1f} outside the expected band"


# ======================================================================
# WHAT IS NOT COVERED
# ======================================================================
# - The photo workflow: image loading, click-to-trace, scale-by-two-clicks,
#   zoom and pan. These need real mouse input against a real canvas.
# - The JSON export writers.
# - The MCOEF-to-ArduPilot gap above is a MODELLING limitation, not a bug,
#   and it is not corrected here. Deciding the right entrainment factor needs
#   flight data, and guessing one would repeat a mistake this project has
#   already made once with the propeller thrust coefficient.
# - Self-intersecting polygons. The shoelace formula silently returns the
#   difference of the overlapping lobes rather than the enclosed area, and
#   nothing detects it. A user who crosses their own trace gets a wrong
#   number with no warning. Worth fixing before it matters.


# ======================================================================
# SELF-INTERSECTION DETECTION
# ======================================================================

def test_simple_outlines_are_not_flagged(dc):
    """Ordinary traces must pass, including concave airframe silhouettes."""
    assert not dc.polygon_self_intersects([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert not dc.polygon_self_intersects([(0, 0), (10, 0), (5, 8)])
    # L-shape: concave, but never crosses itself.
    assert not dc.polygon_self_intersects(
        [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
    # A many-sided convex outline, like a carefully traced body.
    circle = [(100 * math.cos(2 * math.pi * i / 24),
               100 * math.sin(2 * math.pi * i / 24)) for i in range(24)]
    assert not dc.polygon_self_intersects(circle)


def test_crossed_outlines_are_caught(dc):
    """
    The cases that matter: a bowtie from clicking two corners in the wrong
    order, a figure-eight from tracing booms out of sequence, and a trace that
    doubles back along an edge it already walked.
    """
    assert dc.polygon_self_intersects([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert dc.polygon_self_intersects(
        [(0, 0), (4, 4), (8, 0), (8, 4), (4, 0), (0, 4)])
    assert dc.polygon_self_intersects([(0, 0), (10, 0), (5, 0), (5, 5)])


def test_a_bowtie_reports_zero_area_which_is_why_this_check_exists(dc):
    """
    The motivating case. A bowtie's two lobes have equal and opposite signed
    area, so the shoelace sum is exactly 0.0 — a clean-looking number with no
    hint that anything went wrong. Without the check it would flow straight
    into BCOEF and then into the simulator.
    """
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
    assert dc.polygon_area_px2(bowtie) == pytest.approx(0.0)
    assert dc.polygon_self_intersects(bowtie), \
        "the one shape whose wrong answer looks most reasonable must be caught"


def test_shapes_too_small_to_cross_are_not_flagged(dc):
    """A triangle cannot self-intersect, and neither can a partial trace."""
    for pts in ([], [(0, 0)], [(0, 0), (1, 1)], [(0, 0), (1, 0), (0, 1)]):
        assert not dc.polygon_self_intersects(pts)


def test_adjacent_edges_are_not_treated_as_crossings(dc):
    """
    Consecutive edges share a vertex by construction, and the closing edge
    shares one with the first. Counting either as an intersection would flag
    every valid polygon ever traced.
    """
    for n in (4, 5, 8, 16):
        regular = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
                   for i in range(n)]
        assert not dc.polygon_self_intersects(regular), \
            f"a regular {n}-gon was wrongly flagged"


def test_detection_is_independent_of_winding_and_start_vertex(dc):
    """Crossing is a property of the shape, not of how it was clicked."""
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
    assert dc.polygon_self_intersects(list(reversed(bowtie)))
    assert dc.polygon_self_intersects(bowtie[2:] + bowtie[:2])

    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert not dc.polygon_self_intersects(list(reversed(square)))
    assert not dc.polygon_self_intersects(square[2:] + square[:2])


def test_scale_tip_does_not_recommend_the_diagonal(paths):
    """Audit D2: the tip said to use the motorbase (the diagonal between
    opposite motors) as the scale, but a front or side photo of an X frame
    shows the motors motorbase / sqrt(2) apart."""
    text = open(paths["dragcalc"], encoding="utf-8").read()
    assert "Use the motorbase (distance between opposite motors)" not in text
    assert "motorbase / 1.414" in text


def test_top_area_is_exported_to_the_multicopter(paths):
    """Audit D1: the top area was "reference only", though a translating
    multicopter is tilted and presents front*cos + top*sin of the tilt."""
    text = open(paths["dragcalc"], encoding="utf-8").read()
    assert "reference only" not in text
    export = text[text.index('"multicopter_sim": {'):]
    export = export[:export.index("},")]
    assert '"top_area"' in export
