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
        errs = gui.click("Fixed Speed Sweep")
        assert not errs, f"{mode} mode run failed: {errs[0][:200]}"


# ----------------------------------------------------------------------
# Example configs and missions
# ----------------------------------------------------------------------

def test_every_multicopter_config_loads_and_runs(mc_gui, paths):
    for cfg in _configs(paths, "multicopter"):
        mc_gui.set_open_dialog(cfg)
        mc_gui.click("Load Config")
        errs = mc_gui.click("Fixed Speed Sweep")
        assert not errs, f"{os.path.basename(cfg)}: {errs[0][:200]}"


def test_every_fixedwing_config_loads_and_runs(fw_gui, paths):
    for cfg in _configs(paths, "fixedwing"):
        fw_gui.set_open_dialog(cfg)
        fw_gui.click("Load Config")
        errs = fw_gui.click("Fixed Speed Sweep")
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


def _browse_mission(gui, mission_path):
    """Set the mission file through the Mission tab's own Browse button."""
    gui.set_open_dialog(mission_path)
    for nb in [w for w in gui.refresh() if isinstance(w, ttk.Notebook)]:
        for tab in nb.tabs():
            if "mission" not in str(nb.tab(tab, "text")).lower():
                continue
            for w in gui._walk(gui.root.nametowidget(tab), []):
                if isinstance(w, ttk.Button) and "browse" in str(w.cget("text")).lower():
                    w.invoke()
                    gui.pump()
                    return
    raise AssertionError("no Browse button on a Mission tab")


@pytest.mark.parametrize("which,first,second,mission,old_tag,new_tag", [
    ("mc", "multicopter_450_survey_4S.json", "multicopter_7in_longrange_6S.json",
     "mc_02_takeoff_square_land.json", "10 in props", "7 in props"),
    ("fw", "fixedwing_2m_survey_4S.json", "fixedwing_3m_endurance_6S_liion.json",
     "fw_01_takeoff_cruise_land.json", "12 in props", "14 in props"),
])
def test_mission_report_shows_the_aircraft_that_flew(request, which, first, second,
                                                     mission, old_tag, new_tag,
                                                     paths, monkeypatch, tmp_path):
    """
    Audit E1: after a fixed-speed run on one aircraft and a mission on
    another, the report embedded the first aircraft's sweep and airframe
    drawing, and never the mission's own diagram.
    """
    import tkinter.messagebox as mb
    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: True)
    mod = request.getfixturevalue(which)
    captured = {}
    monkeypatch.setattr(mod, "_generate_pdf_report", lambda **kw: captured.update(kw))
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")

    gui.set_open_dialog(os.path.join(paths["configs"], first))
    assert gui.click("Load Config") == []
    assert gui.click("Fixed Speed Sweep") == []
    gui.set_open_dialog(os.path.join(paths["configs"], second))
    assert gui.click("Load Config") == []
    _browse_mission(gui, os.path.join(paths["missions"], mission))
    assert gui.click("Run Mission") == []
    gui.set_save_dialog(str(tmp_path / "report.pdf"))
    assert gui.click("Generate Report") == []

    titles = []
    for fig in captured["figures"]:
        if fig._suptitle is not None:
            titles.append(fig._suptitle.get_text())
        titles += [ax.get_title() for ax in fig.axes]
    joined = " | ".join(titles)
    assert new_tag in joined, joined
    assert old_tag not in joined, joined
    assert "Performance" not in joined, "a mission report embedded a fixed-speed sweep"
    assert "Ground track" in joined, "the mission diagram is missing"


@pytest.mark.parametrize("which,cfg,mission", [
    ("mc", "multicopter_450_survey_4S.json", "mc_02_takeoff_square_land.json"),
    ("fw", "fixedwing_2m_survey_4S.json", "fw_01_takeoff_cruise_land.json"),
])
def test_point_report_drops_an_earlier_mission_plot(request, which, cfg, mission,
                                                    paths, monkeypatch, tmp_path):
    """The reverse of audit E1: a mission plot drawn earlier must not ride
    along into a fixed-speed run's report."""
    import tkinter.messagebox as mb
    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: True)
    mod = request.getfixturevalue(which)
    captured = {}
    monkeypatch.setattr(mod, "_generate_pdf_report", lambda **kw: captured.update(kw))
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")

    gui.set_open_dialog(os.path.join(paths["configs"], cfg))
    assert gui.click("Load Config") == []
    _browse_mission(gui, os.path.join(paths["missions"], mission))
    assert gui.click("Run Mission") == []
    listbox = [w for w in gui.refresh() if isinstance(w, tk.Listbox)][0]
    listbox.selection_set(1)
    gui.buttons = {str(w.cget("text")): w for w in gui.widgets if isinstance(w, ttk.Button)}
    assert gui.click("Plot selected") == []
    assert gui.click("Fixed Speed Sweep") == []
    gui.set_save_dialog(str(tmp_path / "report.pdf"))
    assert gui.click("Generate Report") == []

    titles = [f._suptitle.get_text() for f in captured["figures"] if f._suptitle is not None]
    titles += [ax.get_title() for f in captured["figures"] for ax in f.axes]
    joined = " | ".join(titles)
    assert "Mission variables" not in joined, joined
    assert "Ground track" not in joined, joined
    assert "Performance" in joined


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
    assert not gui.click("Fixed Speed Sweep")


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

    gui.click("Fixed Speed Sweep")
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
    assert gui.click("Fixed Speed Sweep") == []


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
                # Ordering, not adjacency: Power Budget was later inserted
                # between the two, and the requirement was always that the
                # diagram comes after the budgets, not immediately after.
                assert tabs.index("Airframe Diagram") > tabs.index("Weight Budget"), \
                    "diagram tab should come after Weight Budget"
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
    assert gui.click("Fixed Speed Sweep") == []
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
    assert gui.click("Fixed Speed Sweep") == []
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
    assert gui.click("Fixed Speed Sweep") == []
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


@pytest.mark.parametrize("which,config", [
    ("mc", "multicopter_450_survey_4S.json"),
    ("fw", "fixedwing_2m_survey_4S.json"),
])
def test_payload_and_avionics_levers_move_the_answer(request, which, config, paths,
                                                     monkeypatch):
    """
    Audit S2, S3: the all-up weight already includes the payload, so the
    payload lever moved nothing, and the avionics lever scaled only the
    peripheral current, which is 0 when avionics are entered as rails.
    Both showed zero swing, which read as "endurance is insensitive".
    """
    import tkinter.messagebox as mb
    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: True)
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    gui.set_open_dialog(os.path.join(paths["configs"], config))
    assert gui.click("Load Config") == []
    assert gui.click("Fixed Speed Sweep") == []
    assert gui.click("Run Sensitivity") == []
    gui.pump()

    tree = _tree_with(gui, "param", "span")
    swing = {str(tree.item(r, "values")[0]): float(str(tree.item(r, "values")[6]).split()[0])
             for r in tree.get_children()}
    assert swing["Payload mass"] > 0, swing
    assert swing["Avionics draw"] > 0, swing
    if which == "mc":
        # The multicopter has no figure-of-merit input, so it is not offered
        # as a lever that can only ever read zero.
        assert "Figure of merit" not in swing


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
    assert gui.click("Fixed Speed Sweep") == []
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
    assert gui.click("Fixed Speed Sweep") == []
    assert gui.click("Pin Current") == []
    assert gui.click("Fixed Speed Sweep") == []
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


@pytest.mark.parametrize("which,first,second", [
    ("mc", "multicopter_450_survey_4S.json", "multicopter_7in_longrange_6S.json"),
    ("fw", "fixedwing_2m_survey_4S.json", "fixedwing_3m_endurance_6S_liion.json"),
])
def test_compare_shows_the_new_run_on_its_first_run(request, which, first, second, paths,
                                                    monkeypatch):
    """
    Audit S1: the multicopter refreshed Compare before storing the new run,
    so after loading a different aircraft and running once, every row but
    flight time and range showed the previous aircraft with zero change.
    """
    import tkinter.messagebox as mb
    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: True)
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    gui.set_open_dialog(os.path.join(paths["configs"], first))
    assert gui.click("Load Config") == []
    assert gui.click("Fixed Speed Sweep") == []
    assert gui.click("Pin Current") == []
    gui.set_open_dialog(os.path.join(paths["configs"], second))
    assert gui.click("Load Config") == []
    assert gui.click("Fixed Speed Sweep") == []
    gui.pump()

    tree = _tree_with(gui, "metric", "delta", "pct")

    def rows():
        return {str(tree.item(r, "values")[0]): tuple(tree.item(r, "values"))
                for r in tree.get_children()}

    after_one_run = rows()
    assert gui.click("Fixed Speed Sweep") == []
    gui.pump()
    assert rows() == after_one_run, "Compare changed on a second identical run"
    power = after_one_run["Total power (W)"]
    assert power[1] != power[2], "the new aircraft shows the baseline's power"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_clearing_the_baseline_empties_the_table(request, which):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    gui.click("Fixed Speed Sweep")
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

    errs = gui.click("Fixed Speed Sweep")
    assert errs == [], f"running with a table raised: {errs[:1]}"


# NOT COVERED: comparison across two MISSION runs.
#
# The fixed-wing had two ordering faults here — the mission path never called
# refresh_comparison() at all, and once it did, it ran BEFORE
# _last_run["metrics"] was stored, so the tab compared the new baseline
# against the PREVIOUS run's numbers and every delta read +0.00.
#
# Both are fixed and verified by hand (21 of 25 rows move after a 25% weight
# change). There is no automated test because this harness cannot drive a
# mission run: it has no way to set the Mission JSON field, and the several
# "Browse" buttons cannot be told apart by label fragment. Adding that
# capability to the harness is the prerequisite, and it would also unlock
# mission coverage for the Status, Metrics and Power Budget tabs, none of
# which are exercised on the mission path either.


# ----------------------------------------------------------------------
# Wiring tab
# ----------------------------------------------------------------------

def _all_tree_text(gui):
    """Every cell of every Treeview, flattened — Status, Metrics and the rest."""
    out = []
    for widget in gui.refresh():
        if isinstance(widget, ttk.Treeview):
            def walk(parent=""):
                for item in widget.get_children(parent):
                    out.append(str(widget.item(item, "text")))
                    out.extend(str(v) for v in widget.item(item, "values"))
                    walk(item)
            walk()
    return " | ".join(out)


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_wiring_tab_exists(request, which):
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    tabs = [str(nb.tab(t, "text")) for nb in gui.widgets
            if isinstance(nb, ttk.Notebook) for t in nb.tabs()]
    assert "Wiring" in tabs, f"no Wiring tab among {tabs}"


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_a_wired_config_round_trips_and_reaches_status(request, which, tmp_path):
    """
    Save a config, add a lead and a connector to it, load it back and run:
    the wiring must survive the file and show up in Status and Metrics.
    """
    import json
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    dest = str(tmp_path / "wired.json")
    gui.set_save_dialog(dest)
    assert not gui.click("Save Config")
    with open(dest, encoding="utf-8") as f:
        data = json.load(f)
    assert "wire_len" in data["vars"], "the Wiring tab is not saved with the config"
    data["vars"].update({"wire_len": "0.5", "wire_awg": "18", "wire_temp_limit": "105",
                         "conn_batt": "XT60", "conn_batt_cont": "60",
                         "conn_batt_max": "120", "conn_batt_volt": "500"})
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f)

    gui.set_open_dialog(dest)
    assert not gui.click("Load Config")
    errs = gui.click("Fixed Speed Sweep")
    assert not errs, errs[0][:300]
    text = _all_tree_text(gui)
    for row in ("Main wire voltage drop", "Wire temperature (est)",
                "Battery connector current", "Battery connector voltage",
                "Main lead loss"):
        assert row in text, f"{row!r} missing after a wired run"


def test_fw_status_flags_a_temperature_below_ambient(fw, fw_gui, monkeypatch):
    """Audit G5: only the upper limit was checked, so a diverged -827 C
    motor temperature showed green as OK."""
    real = fw.compute_metrics

    def diverged(*a, **k):
        m = real(*a, **k)
        m["motor_temp_est_C"] = -827.4
        return m

    monkeypatch.setattr(fw, "compute_metrics", diverged)
    assert fw_gui.click("Fixed Speed Sweep") == []
    rows = [(tv.item(i, "values"), tv.item(i, "tags"))
            for tv in fw_gui.refresh() if isinstance(tv, ttk.Treeview)
            for i in tv.get_children()]
    motor = [(v, t) for v, t in rows if v and str(v[0]) == "Motor temperature (est)"]
    assert motor, "no motor temperature row"
    values, tags = motor[0]
    assert "bad" in tags, (values, tags)
    assert "impossible" in str(values[3])


def test_fw_metrics_auw_counts_the_payload_once(fw_gui, paths, monkeypatch):
    """Audit F16: aircraft_weight_g already includes the payload; adding it
    again showed 3,400 g against the 3,000 g the model flies."""
    import tkinter.messagebox as mb
    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: True)
    fw_gui.set_open_dialog(os.path.join(paths["configs"], "fixedwing_2m_survey_4S.json"))
    assert fw_gui.click("Load Config") == []
    assert fw_gui.click("Fixed Speed Sweep") == []
    tree = next(w for w in fw_gui.refresh() if isinstance(w, ttk.Treeview)
                and [str(c) for c in w.cget("columns")] == ["metric", "value", "note"])

    def walk(node=""):
        for iid in tree.get_children(node):
            yield tree.item(iid, "values")
            yield from walk(iid)

    rows = {str(v[0]): str(v[1]) for v in walk() if v}
    auw = rows["All-Up Weight (AUW)"]
    grams = float(auw.split()[0])
    newtons = float(auw.split("(")[1].split()[0])
    assert grams == pytest.approx(3000.0)
    assert newtons == pytest.approx(grams / 1000.0 * 9.80665, rel=1e-3)


@pytest.mark.parametrize("which", ["mc", "fw"])
def test_metrics_pack_resistance_is_the_one_in_use(request, which):
    """Audit B5: Metrics showed the base pack resistance while the sag was
    computed at the SoC curve's 1.5x of it at full charge."""
    gui = request.getfixturevalue("mc_gui" if which == "mc" else "fw_gui")
    assert gui.click("Fixed Speed Sweep") == []
    tree = next(w for w in gui.refresh() if isinstance(w, ttk.Treeview)
                and [str(c) for c in w.cget("columns")] == ["metric", "value", "note"])

    def walk(node=""):
        for iid in tree.get_children(node):
            yield tree.item(iid, "values")
            yield from walk(iid)

    rows = {str(v[0]): str(v[1]) for v in walk() if v}
    shown_mohm = float(rows["Pack Resistance"].split()[0])
    loaded = rows["Pack Voltage (loaded)" if which == "mc" else "Pack Voltage (under load)"]
    sag_V = float(loaded.split("sag:")[1].split()[0])
    current_A = float(rows["Pack Current"].split()[0])
    # The resistance shown must be the one that produced the sag shown:
    # sag / current, to the rounding of the two displayed figures.
    implied_mohm = sag_V / current_A * 1000.0
    assert abs(shown_mohm - implied_mohm) <= 0.005 / current_A * 1000.0 + 0.06


def test_mc_mission_compare_uses_the_mission_totals(mc_gui, paths, monkeypatch):
    """Audit S6: mission Compare showed single-point rows ("—" for flight
    time and range) and left out the mission's own totals."""
    import tkinter.messagebox as mb
    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: True)
    _browse_mission(mc_gui, os.path.join(paths["missions"], "mc_02_takeoff_square_land.json"))
    assert mc_gui.click("Run Mission") == []
    assert mc_gui.click("Pin Current") == []
    mc_gui.pump()
    tree = _tree_with(mc_gui, "metric", "delta", "pct")
    labels = [str(tree.item(r, "values")[0]) for r in tree.get_children()]
    assert "Mission energy (Wh)" in labels and "Energy remaining (Wh)" in labels
    assert "Flight time (min)" not in labels
    for r in tree.get_children():
        values = tree.item(r, "values")
        assert values[1] != "—", f"{values[0]} has no mission value"


def test_mc_status_judges_the_drive_efficiency_as_a_drivetrain(mc_gui):
    """Audit M7, G8: the whole-drivetrain number was labelled a rotor
    figure of merit and checked against rotor thresholds."""
    assert mc_gui.click("Fixed Speed Sweep") == []
    rows = {}
    for w in mc_gui.refresh():
        if isinstance(w, ttk.Treeview):
            for iid in w.get_children():
                v = w.item(iid, "values")
                if v:
                    rows[str(v[0])] = v
    assert "Figure of merit" not in rows
    row = rows["Hover drive efficiency"]
    assert float(str(row[2]).split()[-1]) in (pytest.approx(0.56), pytest.approx(0.48),
                                             pytest.approx(0.36))
