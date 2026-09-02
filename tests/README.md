# Test suite

219 tests covering the shared core, physics, the CLI, the GUI, and the
batch driver.

Almost every test here corresponds to a bug that was actually shipped. The
docstrings say which one, so a future failure reads as "the pack capacity
regression is back" rather than "test_foo failed".

---

## Running

```bash
pip install pytest
pytest                                  # everything (~7 minutes)
```

Faster subsets:

```bash
pytest -m "not slow"                    # physics only, ~4 seconds
pytest tests/test_physics.py            # same thing, explicitly
pytest tests/test_cli.py                # subprocess runs, ~2.5 min
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
| `test_golden.py` | 1 | ~1 s | 348 stored numeric outputs across both simulators |
| `test_core.py` | 56 | ~1 s | `rotorworks_core`, plus checks that both simulators delegate to it |
| `test_physics.py` | 57 | ~4 s | Battery topology, atmosphere, rotor inflow, prop efficiency, SoC, drag, landing, metrics sanity, drag calculator |
| `test_cli.py` | 64 | ~2.5 min | Real subprocess runs: every argument path, every example mission, edge cases, malformed input |
| `test_gui.py` | 22 | ~2 min | Real Tk window: hover events, mode toggle, config load, missions, exports |
| `test_batch.py` | 20 | ~2 min | Sweeps, sizing, mode enforcement, GUI-config translation, GUI↔CLI consistency |

### Marks

- `slow` — spawns subprocesses; minutes rather than seconds
- `gui` — builds a real Tk window; needs a display

Both are registered in `pytest.ini`, and `--strict-markers` is on so a typo in
a mark name is an error rather than a silent no-op.

---

## The golden snapshot

`test_golden.py` stores 348 computed values — power, thrust, endurance, range,
battery arithmetic, atmosphere — from fixed configurations, and fails if any
of them moves by more than 1 part in 10^9.

It exists for refactoring. During the shared-core extraction it caught two
signature mismatches within seconds of them being introduced, and named the
exact values that moved. Regenerate it **only** when a physics change is
intended:

```bash
python tests/test_golden.py --update
```

Then read the diff before committing. A snapshot regenerated without reading
the diff is worse than no snapshot.

## The three tests worth understanding

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

---

## Adding a test

Use the `mc` / `fw` / `rw` fixtures to get a simulator module, and `mc_quad` /
`fw_plane` for a ready-built reference aircraft:

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

- **The View menu** (window scale, plot size, UI font size) is untested.
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
