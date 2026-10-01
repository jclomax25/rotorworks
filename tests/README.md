# Test suite

790 tests covering the shared core, physics, the CLI, the GUI, the batch
driver, the VTOL simulator, and a full config/mission coverage matrix. The
whole suite was last run clean on Windows (790 passed).

Almost every test here corresponds to a bug that was actually shipped. The
docstrings say which one, so a future failure reads as "the pack capacity
regression is back" rather than "test_foo failed".

---

## Running

```bash
pip install pytest
pytest                                  # everything (~30 minutes)
```

Run it from the repository root: `pytest.ini` there registers the marks and
points pytest at `tests/`.

Faster subsets:

```bash
pytest -m "not slow and not gui"        # no subprocesses, no windows, ~2 min
pytest tests/test_physics.py            # multicopter/fixed-wing physics, ~5 s
pytest tests/test_golden.py             # the numeric snapshot, ~3 s
pytest tests/test_cli.py                # subprocess runs, ~5 min
pytest -m gui                           # GUI only
pytest -k battery                       # anything matching "battery"
```

On a headless machine (CI, a server, WSL without an X server) the GUI tests
need a virtual display:

```bash
xvfb-run -a pytest                      # Linux
```

Without a display they skip themselves rather than fail, so `pytest` is always
safe to run.

---

## Layout

| File | Tests | Speed | What it covers |
|---|---|---|---|
| `conftest.py` | — | — | Loads the simulators by path, provides reference aircraft |
| `test_golden.py` | 1 | ~3 s | 1042 stored numeric outputs across all three simulators |
| `test_core.py` | 90 | ~1 s | `rotorworks_core`: SoC, wind, inflow, sensitivity, comparison, propeller coefficients, power budget, airframe geometry |
| `test_physics.py` | 136 | ~5 s | Battery topology, atmosphere, rotor inflow, prop efficiency, drag, turns, translation direction, thresholds, figure leaks |
| `test_vtol.py` | 257 | ~6 min | All four VTOL types: vectored-thrust force balance, tilt, download, continuity, missions, the motor electrical model, the ported GUI tabs, window-chrome parity, output-tab presentation, the battery and peripheral-load model, unit conversions, plot panels, input sections and tab order, and GUI/CLI/batch parity of the config builder |
| `test_cli.py` | 70 | ~5 min | Real subprocess runs: every argument path, every example mission, edge cases, malformed input |
| `test_gui.py` | 50 | ~4 min | Real Tk window: hover events, mode toggle, config load, missions, exports, diagram, sensitivity, comparison, the Wiring tab |
| `test_matrix.py` | 108 | ~8 min | Every config and mission x GUI and CLI x with and without a propeller table; the VTOL through its CLI |
| `test_wiring.py` | 27 | ~40 s | The shared wiring model: lead resistance, loss, drop and temperature; connector current and voltage checks; each simulator paying for the lead, and a blank Wiring tab changing nothing; one set of CLI flags and field names everywhere |
| `test_drag_calculator.py` | 31 | ~1 s | Shoelace geometry, self-intersection detection, pixel scaling, ISA density, ArduPilot BCOEF and MCOEF |
| `test_batch.py` | 20 | ~2.5 min | Sweeps, sizing, mode enforcement, GUI-config translation, GUI↔CLI consistency |

### Marks

- `slow` — spawns subprocesses; minutes rather than seconds
- `gui` — builds a real Tk window; needs a display

Both are registered in `pytest.ini` at the repository root, and
`--strict-markers` is on so a typo in a mark name is an error rather than a
silent no-op.

### Test data

`tests/data/` holds the measured inputs the table tests read. They are the
same files as the examples, copied so the tests do not change when someone
edits an example:

| File | What it is | Same as |
|---|---|---|
| `motor_prop_table.csv` | T-Motor MN6007 II KV160 on a 22x6.6 propeller, eCalc-style export with a title row | `examples/motor-propeller-csv/mc_motor_prop_csv_example.csv` |
| `fw_motor_prop_table.csv` | AT5220 KV380 on an APC 18x8 | `examples/motor-propeller-csv/fw_motor_prop_csv_example.csv` |
| `soc_curve_lipo.csv` | A LiPo discharge curve: SoC, OCV per cell, resistance scale | `examples/soc_curve_lipo.csv` |

The two propeller tables are also what the shared core's thrust-coefficient
estimate is calibrated against, so `test_geometry_estimate_matches_measured_propellers`
checks the estimate against the data it was fitted to.

These files were missing from the repository until v1.11 — the tests that
read them were committed and the data was not — so on a fresh clone about 65
table and curve tests failed with `FileNotFoundError`. If you see that again,
this folder is what is missing.

### Windows and captured output

The CLI tests run each simulator as a subprocess and capture its output. On
Windows, captured output is encoded with the ANSI code page (cp1252), which
cannot represent the μ, Ω, → and similar symbols the reports print, and until
v1.11 every CLI and batch test failed there with `UnicodeEncodeError`. Every
simulator now calls `rotorworks_core.make_console_safe()` first, which prints
`?` for a symbol the console cannot show instead of crashing. Linux and macOS
use UTF-8 and never saw the problem, which is how it survived: the suite was
only ever run on them.

---

## The golden snapshot

`test_golden.py` stores 1042 computed values — power, thrust, endurance, range,
battery arithmetic, atmosphere — from fixed configurations, and fails if any
of them moves by more than 1 part in 10^9:

| Cases | Count | Values | What |
|---|---|---|---|
| `mc/…`, `fw/…`, battery, atmosphere | 32 | 348 | The original snapshot, unchanged since the shared-core extraction |
| `vtol/<type>/@<speed>` | 16 | 576 | All four VTOL types at 0, 8, 14 and 22 m/s: power chain, motor model, lift handover, climb, runway, thermal |
| `vtol/example/<config>` | 3 | 108 | The three example VTOLs at their own cruise speed |
| `vtol/mission/vtol_01_wind4` | 1 | 10 | A whole time-stepped mission in wind with an acceleration limit |

The VTOL cases were added in v1.11, after its motor model changed every VTOL
power figure, so the next refactor of that code has the same protection the
other two simulators have had.

It exists for refactoring. During the shared-core extraction it caught two
signature mismatches within seconds of them being introduced, and named the
exact values that moved. Regenerate it **only** when a physics change is
intended:

```bash
python tests/test_golden.py --update
```

Then read the diff before committing. A snapshot regenerated without reading
the diff is worse than no snapshot.

## The four tests worth understanding

**`test_hovering_every_help_marker_shows_a_tooltip`** fires real `<Enter>`
events at every `?` marker. An earlier version of this test counted the
markers instead, found all 89, and passed — while every one of them raised
`NameError` the moment a user hovered it. The tooltip helper was defined at
module level, but `tkinter` is imported lazily inside `launch_gui()`, so `tk`
was out of scope. Constructing a widget is not the same as exercising it.

**`test_simulators_share_the_same_function_objects`** asserts that
`mc.wind_components_mps is fw.wind_components_mps is core.wind_components_mps`
— literally the same object, not merely equivalent behaviour. If someone
pastes a local copy back into one simulator, the duplication returns silently
unless a test checks identity. This one does.

**`test_gui_and_batch_agree_on_the_same_config`** loads the same config file
through the GUI and through `rotorworks-batch.py`, then compares flight time.
The fixed-wing CLI had no ESC arguments for the project's whole history, so
the batch path silently dropped losses the GUI applied — about 1% on a typical
airframe, invisible unless you compared the two directly.

**`test_matrix.py` as a whole** exists because two bugs shipped through the
same hole: `name 'm' is not defined`, which fired only when a propeller table
was loaded, and a fixed-wing table loader that carried a known fault for six
releases because nothing exercised it. Both were of the form "works in one
interface, or one table state, and not the other". Individual tests could not
close that; only the product of the dimensions could.

Note its display skip is deliberately **not** module-level. The CLI half needs
no display, and skipping the whole file on a headless box would silently drop
half the coverage while still reporting green.

---

## Adding a test

Use the `mc` / `fw` / `vtol` / `rw` fixtures to get a simulator module, and
`mc_quad` / `fw_plane` for a ready-built reference aircraft:

```python
def test_something(fw, fw_plane):
    m = fw.compute_metrics(fw_plane, 19.0)
    assert m["flight_time_min"] > 0
```

Two conventions worth keeping:

1. **Say what broke.** If the test guards a real bug, describe the symptom in
   the docstring — the wrong number, not just the wrong behaviour.
2. **Assert on behaviour, not structure.** Check that hovering produces a
   tooltip, not that a tooltip widget exists.

The reference aircraft in `conftest.py` are deliberately plain and fully
specified. Several tests assert against values derived from them by hand, so
changing those fixtures will break tests that look unrelated.

---

## What is not covered

Worth knowing before you rely on a green run:

- **The multicopter and fixed-wing View menus** (window scale, plot size, UI
  and plot font size, presets) are untested. The VTOL's is not: `test_vtol.py`
  invokes every entry and asserts the figure actually resizes, because that
  menu was ported after the other two and the port needed proving.
- **The VTOL is in the matrix through its CLI only** — every type with and
  without a bench table, every mission in still air and in wind, and a batch
  sweep. Its GUI is covered by `test_vtol.py` rather than by the matrix.
- **Plot *contents*** — the new mission-plot axes, the drag and thrust
  component panels, and the Power Budget share diagram are confirmed to render
  without error, not to be correct. The one exception is the multicopter power
  bucket.
- **The Avionics tab's add / remove / clear rail buttons** are exercised only
  indirectly, through configs that already contain rails.
- **Plot contents** are not inspected. The tests confirm that plotting runs
  without error and that export files are non-trivial in size; they do not
  check that a curve has the right shape, except for the multicopter power
  bucket in `test_physics.py`.
- **The drag coefficient calculator's photo workflow** (image loading, scale
  setting, polygon drawing by click) is untested — it needs real mouse input.
  Its physics *is* covered in `test_physics.py`: shoelace area against known
  polygons, BCOEF against the ArduPilot IRIS reference, and MCOEF against the
  documented range.
