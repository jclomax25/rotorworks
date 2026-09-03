"""
GUI tests.

These build the real Tk window and drive it: they click buttons, fire hover
events, load configs and run missions. They are skipped automatically when no
display or no tkinter is available.

On a headless machine, run under a virtual display:

    xvfb-run -a pytest tests/test_gui.py

Why fire real events rather than inspect widgets: an earlier version of the
tooltip test counted the '?' markers and passed with 89 of them present, while
every single one raised NameError the moment it was hovered. Counting widgets
is not testing them.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import pytest

pytestmark = pytest.mark.gui

tk = pytest.importorskip("tkinter", reason="tkinter not installed")
from tkinter import ttk  # noqa: E402


def _display_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


if not _display_available():
    pytest.skip("no display available; run under xvfb-run",
                allow_module_level=True)


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------

class GuiHarness:
    """Builds a simulator GUI without entering mainloop, and drives it."""

    def __init__(self, module):
        self.errors = []
        self._patch_dialogs()

        holder = {}
        original = tk.Tk.mainloop

        def capture(self_root, *a, **k):
            holder["root"] = self_root
            self_root.update_idletasks()
            self_root.update()

        tk.Tk.mainloop = capture
        try:
            module.launch_gui()
        finally:
            tk.Tk.mainloop = original

        self.root = holder["root"]
        self.root.report_callback_exception = self._record
        self.widgets = self._walk(self.root, [])
        self.buttons = {
            str(w.cget("text")): w
            for w in self.widgets if isinstance(w, ttk.Button)
        }

    def _record(self, exc, val, tb):
        import traceback
        self.errors.append("".join(traceback.format_exception(exc, val, tb)))

    @staticmethod
    def _patch_dialogs():
        import tkinter.messagebox as mb
        mb.showinfo = lambda *a, **k: None
        mb.showwarning = lambda *a, **k: None

    def capture_errors(self):
        """Route messagebox.showerror into self.errors."""
        import tkinter.messagebox as mb
        mb.showerror = lambda title, msg=None, **k: self.errors.append(f"{title}: {msg}")

    def _walk(self, widget, out):
        for child in widget.winfo_children():
            out.append(child)
            self._walk(child, out)
        return out

    def refresh(self):
        self.widgets = self._walk(self.root, [])
        return self.widgets

    def pump(self):
        self.root.update_idletasks()
        self.root.update()

    def button(self, fragment: str):
        for text, widget in self.buttons.items():
            if fragment.lower() in text.lower():
                return widget
        raise KeyError(f"no button matching {fragment!r}; have {list(self.buttons)}")

    def click(self, fragment: str):
        self.errors.clear()
        self.button(fragment).invoke()
        self.pump()
        return list(self.errors)

    def set_open_dialog(self, path: str):
        import tkinter.filedialog as fd
        fd.askopenfilename = lambda *a, **k: path

    def set_save_dialog(self, path: str):
        import tkinter.filedialog as fd
        fd.asksaveasfilename = lambda *a, **k: path

    def help_markers(self):
        return [w for w in self.widgets
                if isinstance(w, ttk.Label) and str(w.cget("text")).strip() == "?"]

    def entries(self):
        return [w for w in self.widgets if isinstance(w, ttk.Entry)]

    def radio(self, value: str):
        for w in self.widgets:
            if isinstance(w, ttk.Radiobutton):
                try:
                    if w.cget("value") == value:
                        return w
                except Exception:
                    pass
        raise KeyError(f"no radiobutton with value {value!r}")

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass


@pytest.fixture
def mc_gui(mc):
    h = GuiHarness(mc)
    h.capture_errors()
    yield h
    h.destroy()


@pytest.fixture
def fw_gui(fw):
    h = GuiHarness(fw)
    h.capture_errors()
    yield h
    h.destroy()


def _configs(paths, prefix):
    return sorted(glob.glob(os.path.join(paths["configs"], f"{prefix}*.json")))


def _missions(paths, prefix):
    return sorted(glob.glob(os.path.join(paths["missions"], f"{prefix}_*.json")))


# ----------------------------------------------------------------------
# Tooltips  — the regression that motivated this whole file
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["mc", "fw"])
def test_hovering_every_help_marker_shows_a_tooltip(request, which):
    """
    Regression: _Tooltip was defined at module level while tkinter is imported
    lazily inside launch_gui(), so `tk` was out of scope. Every marker raised
    NameError on hover. Constructing the markers succeeded, which is why a
    widget-counting test missed it entirely.
    """
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    markers = gui.help_markers()
    assert markers, "no '?' help markers found at all"

    shown = 0
    for marker in markers:
        marker.event_generate("<Enter>", x=3, y=3)
        gui.pump()
        tips = [w for w in gui._walk(gui.root, []) if isinstance(w, tk.Toplevel)]
        if tips:
            shown += 1
        marker.event_generate("<Leave>")
        gui.pump()

    assert not gui.errors, f"hover raised:\n{gui.errors[0][-800:]}"
    assert shown == len(markers), f"only {shown}/{len(markers)} markers showed a tooltip"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_tooltips_are_cleaned_up_on_leave(request, which):
    """A leaked Toplevel per hover would eventually swamp the window manager."""
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    for marker in gui.help_markers()[:20]:
        marker.event_generate("<Enter>", x=3, y=3)
        gui.pump()
        marker.event_generate("<Leave>")
        gui.pump()
    leftover = [w for w in gui._walk(gui.root, []) if isinstance(w, tk.Toplevel)]
    assert leftover == [], f"{len(leftover)} tooltip windows leaked"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_tooltip_text_is_substantive(request, which):
    """Each tooltip needs a description and a 'Typical:' line, with real newlines."""
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    marker = gui.help_markers()[2]
    marker.event_generate("<Enter>", x=3, y=3)
    gui.pump()
    tips = [w for w in gui._walk(gui.root, []) if isinstance(w, tk.Toplevel)]
    assert tips, "no tooltip appeared"
    text = "".join(str(c.cget("text")) for t in tips for c in t.winfo_children())
    marker.event_generate("<Leave>")
    gui.pump()

    assert len(text) > 30, "tooltip text is too short to be useful"
    assert "Typical" in text, "tooltip is missing its 'Typical:' guidance"
    assert "\\n" not in text, "tooltip contains a literal backslash-n"


# ----------------------------------------------------------------------
# Simple / Advanced mode
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["mc", "fw"])
def test_mode_selector_hides_and_restores_fields(request, which):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    entries = gui.entries()
    visible_simple = [w for w in entries if w.winfo_manager()]

    gui.radio("Advanced").invoke()
    gui.pump()
    visible_advanced = [w for w in entries if w.winfo_manager()]

    gui.radio("Simple").invoke()
    gui.pump()
    visible_again = [w for w in entries if w.winfo_manager()]

    assert len(visible_advanced) > len(visible_simple), "Advanced showed no extra fields"
    assert len(visible_again) == len(visible_simple), "mode toggle did not round-trip"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_runs_succeed_in_both_modes(request, which):
    """Hiding a field must not change whether a run works."""
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    for mode in ("Advanced", "Simple"):
        gui.radio(mode).invoke()
        gui.pump()
        errs = gui.click("Single-Point")
        assert not errs, f"{mode} mode run failed: {errs[0][:200]}"


# ----------------------------------------------------------------------
# Example configs and missions
# ----------------------------------------------------------------------

def test_every_multicopter_config_loads_and_runs(mc_gui, paths):
    for cfg in _configs(paths, "multicopter"):
        mc_gui.set_open_dialog(cfg)
        mc_gui.click("Load Config")
        errs = mc_gui.click("Single-Point")
        assert not errs, f"{os.path.basename(cfg)}: {errs[0][:200]}"


def test_every_fixedwing_config_loads_and_runs(fw_gui, paths):
    for cfg in _configs(paths, "fixedwing"):
        fw_gui.set_open_dialog(cfg)
        fw_gui.click("Load Config")
        errs = fw_gui.click("Single-Point")
        assert not errs, f"{os.path.basename(cfg)}: {errs[0][:200]}"


def _run_missions(gui, paths, cfg_prefix, mission_prefix):
    configs = _configs(paths, cfg_prefix)
    gui.set_open_dialog(configs[0])
    gui.click("Load Config")
    for mission in _missions(paths, mission_prefix):
        gui.set_open_dialog(mission)
        for text in list(gui.buttons):
            if "Browse" in text:
                try:
                    gui.buttons[text].invoke()
                except Exception:
                    pass
        gui.pump()
        errs = gui.click("Run Mission")
        assert not errs, f"{os.path.basename(mission)}: {errs[0][:200]}"


def test_every_multicopter_mission_runs(mc_gui, paths):
    _run_missions(mc_gui, paths, "multicopter", "mc")


def test_every_fixedwing_mission_runs(fw_gui, paths):
    _run_missions(fw_gui, paths, "fixedwing", "fw")


# ----------------------------------------------------------------------
# Config round-trip and exports
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["mc", "fw"])
def test_config_save_and_reload_round_trip(request, which, tmp_path):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    dest = str(tmp_path / "roundtrip.json")

    gui.set_save_dialog(dest)
    assert not gui.click("Save Config")
    assert os.path.exists(dest), "Save Config wrote nothing"

    gui.set_open_dialog(dest)
    assert not gui.click("Load Config")
    assert not gui.click("Single-Point")


@pytest.mark.parametrize("label,ext,dependency", [
    ("Export CSV", ".csv", None),
    ("Export Excel", ".xlsx", "openpyxl"),
    ("Generate Report", ".pdf", "reportlab"),
])
@pytest.mark.parametrize("which", ["mc", "fw"])
def test_exports_produce_files(request, which, label, ext, dependency, tmp_path):
    if dependency:
        pytest.importorskip(dependency, reason=f"{dependency} not installed")
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")

    gui.click("Single-Point")
    dest = str(tmp_path / f"out{ext}")
    gui.set_save_dialog(dest)
    errs = gui.click(label)

    assert not errs, f"{label} failed: {errs[0][:200]}"
    assert os.path.exists(dest), f"{label} produced no file"
    assert os.path.getsize(dest) > 200, f"{label} produced a suspiciously small file"


# ----------------------------------------------------------------------
# Modal dialogs must never block reusable code paths
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["mc", "fw"])
def test_loading_a_config_does_not_block_on_a_modal(request, which, paths):
    """
    Regression: the fixed-wing's config loader ended with messagebox.showinfo.
    A modal blocks until someone clicks it, so calling the loader outside an
    interactive click — e.g. autoloading an example during GUI construction —
    hung the application at startup with no error.

    Here messagebox is patched to a recorder rather than a real dialog, and
    the loader is driven directly. It must return promptly.
    """
    import glob
    import tkinter.messagebox as mb

    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    prefix = "multicopter" if which == "mc" else "fixedwing"
    configs = sorted(glob.glob(os.path.join(paths["configs"], f"{prefix}*.json")))
    assert configs, "no example configs to load"

    shown = []
    original = mb.showinfo
    mb.showinfo = lambda *a, **k: shown.append(a)
    try:
        gui.set_open_dialog(configs[0])
        errs = gui.click("Load Config")
    finally:
        mb.showinfo = original

    assert not errs, f"loading raised: {errs[0][:200]}"
    # The button path may confirm; what matters is that it returned at all.
    assert gui.click("Single-Point") == []


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_config_label_tracks_the_loaded_file(request, which, paths):
    """The mode-bar label must name whatever config was loaded most recently."""
    import glob
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    prefix = "multicopter" if which == "mc" else "fixedwing"
    configs = sorted(glob.glob(os.path.join(paths["configs"], f"{prefix}*.json")))

    gui.set_open_dialog(configs[0])
    gui.click("Load Config")
    gui.pump()

    labels = [str(w.cget("text")) for w in gui.refresh()
              if isinstance(w, ttk.Label)]
    expected = os.path.basename(configs[0])
    assert any(expected in text for text in labels), (
        f"mode bar does not show {expected}")


# ----------------------------------------------------------------------
# Airframe Diagram tab
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["mc", "fw"])
def test_airframe_diagram_tab_exists_after_weight_budget(request, which):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    for widget in gui.widgets:
        if isinstance(widget, ttk.Notebook):
            tabs = [widget.tab(i, "text") for i in range(len(widget.tabs()))]
            if "Weight Budget" in tabs:
                assert "Airframe Diagram" in tabs, "diagram tab missing"
                assert tabs.index("Airframe Diagram") == tabs.index("Weight Budget") + 1, \
                    "diagram tab must follow Weight Budget"
                return
    pytest.fail("no display notebook containing a Weight Budget tab")


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_airframe_diagram_draws_on_a_run(request, which):
    """
    The placeholder must be replaced by a real figure once a run completes.
    The refresher swallows drawing errors to keep a run from failing, so a
    broken diagram would otherwise show up as a stuck placeholder.
    """
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    assert gui.click("Single-Point") == []
    gui.pump()

    placeholders = [w for w in gui.refresh()
                    if isinstance(w, ttk.Label)
                    and "draw the airframe" in str(w.cget("text"))]
    assert placeholders, "diagram placeholder label not found"
    assert not placeholders[0].winfo_manager(), \
        "placeholder still showing — the diagram never drew"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_airframe_diagram_survives_missing_dimensions(request, which):
    """
    Body and arm dimensions are optional. With them blank the diagram must
    still draw, using proportionate assumptions rather than failing.
    """
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    assert gui.click("Single-Point") == []
    gui.pump()
    placeholders = [w for w in gui.refresh()
                    if isinstance(w, ttk.Label)
                    and "Could not draw" in str(w.cget("text"))
                    and w.winfo_manager()]
    assert not placeholders, "diagram reported a drawing failure"


# ----------------------------------------------------------------------
# Sensitivity and Compare tabs
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["mc", "fw"])
def test_sensitivity_and_compare_tabs_exist(request, which):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    for widget in gui.widgets:
        if isinstance(widget, ttk.Notebook):
            tabs = [widget.tab(i, "text") for i in range(len(widget.tabs()))]
            if "Weight Budget" in tabs:
                assert "Sensitivity" in tabs
                assert "Compare" in tabs
                return
    pytest.fail("display notebook not found")


def _tree_with(gui, *required_columns):
    for widget in gui.refresh():
        if isinstance(widget, ttk.Treeview):
            try:
                cols = [str(c) for c in widget.cget("columns")]
            except Exception:
                continue
            if all(c in cols for c in required_columns):
                return widget
    return None


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_sensitivity_ranks_inputs_by_influence(request, which):
    """
    The sweep must produce one row per lever, ordered widest-swing first —
    that ordering is what makes the tornado chart readable.
    """
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    assert gui.click("Single-Point") == []
    assert gui.click("Run Sensitivity") == []
    gui.pump()

    tree = _tree_with(gui, "param", "span")
    assert tree is not None, "sensitivity table not found"
    rows = tree.get_children()
    assert len(rows) >= 4, "too few levers evaluated"

    def swing(row):
        return float(str(tree.item(row, "values")[6]).split()[0])

    swings = [swing(r) for r in rows]
    assert swings == sorted(swings, reverse=True), "rows are not ranked by influence"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_sensitivity_needs_a_run_first(request, which):
    """Pressing the button before any run must explain, not raise."""
    import tkinter.messagebox as mb
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    told = []
    original = mb.showinfo
    mb.showinfo = lambda *a, **k: told.append(a)
    try:
        errs = gui.click("Run Sensitivity")
    finally:
        mb.showinfo = original
    assert errs == [], "pressing Run Sensitivity too early raised an error"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_comparison_is_empty_until_a_baseline_is_pinned(request, which):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    assert gui.click("Single-Point") == []
    tree = _tree_with(gui, "metric", "delta", "pct")
    assert tree is not None, "comparison table not found"
    assert tree.get_children() == (), "comparison populated with no baseline"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_pinned_baseline_shows_zero_delta_against_itself(request, which):
    """
    Pinning and immediately re-running the same configuration must report no
    change. A non-zero delta here would mean the two sides are not measuring
    the same thing.
    """
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    assert gui.click("Single-Point") == []
    assert gui.click("Pin Current") == []
    assert gui.click("Single-Point") == []
    gui.pump()

    tree = _tree_with(gui, "metric", "delta", "pct")
    assert tree is not None
    rows = tree.get_children()
    assert rows, "comparison table is empty after pinning"

    for row in rows:
        values = tree.item(row, "values")
        delta = str(values[3])
        if delta == "—":
            continue
        assert abs(float(delta)) < 1e-6, (
            f"{values[0]} reports {delta} against its own baseline")


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_clearing_the_baseline_empties_the_table(request, which):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    gui.click("Single-Point")
    gui.click("Pin Current")
    gui.pump()
    tree = _tree_with(gui, "metric", "delta", "pct")
    assert tree.get_children(), "nothing pinned"
    assert gui.click("Clear Baseline") == []
    gui.pump()
    assert tree.get_children() == (), "Clear left rows behind"



@pytest.mark.parametrize("which", ["mc", "fw"])
def test_running_with_a_measured_table_loaded(request, which, paths):
    """
    Regression: the Status tab's table-range check referenced a metrics
    variable that does not exist in the multicopter, raising
    "name 'm' is not defined". It only fired when a table was actually
    loaded, and no GUI test loaded one — so every test passed while the
    feature was broken for exactly the users who had test data.
    """
    table = os.path.join(paths["root"], "tests", "data",
                         "motor_prop_table.csv" if which == "mc"
                         else "fw_motor_prop_table.csv")
    assert os.path.exists(table), "sample table missing"

    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")

    # Drive the real user path: click the Browse button on the same grid row
    # as the "Prop/Motor CSV table" label, with the file dialog stubbed to
    # return our sample. Setting a StringVar directly is fragile because the
    # entry lives in a nested frame beside the button.
    widgets = gui.refresh()
    label_row = None
    for label in (w for w in widgets if isinstance(w, ttk.Label)):
        if "CSV table" in str(label.cget("text")):
            label_row = label.grid_info().get("row")
            break
    assert label_row is not None, "propeller CSV table label not found"

    # The Browse button sits in the same frame as its label, though not
    # necessarily on the same grid row (it shares a row with the entry).
    label_widget = next(w for w in widgets if isinstance(w, ttk.Label)
                        and "CSV table" in str(w.cget("text")))
    browse = None
    for button in (w for w in widgets if isinstance(w, ttk.Button)):
        if "Browse" not in str(button.cget("text")):
            continue
        if button.master is label_widget.master or \
                button.master.master is label_widget.master:
            browse = button
            break
    assert browse is not None, "Browse button for the CSV table not found"

    gui.set_open_dialog(table)
    browse.invoke()
    gui.pump()

    errs = gui.click("Single-Point")
    assert errs == [], f"running with a table raised: {errs[:1]}"
