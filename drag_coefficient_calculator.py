#!/usr/bin/env python3
"""
drag_coefficient_calculator.py
================================
UAV Drag Coefficient Calculator

Two tools in one application:

  Tab 1 — Body Drag (BCOEF)
    Load front, side, and top-view photographs of your vehicle.
    Click to outline the vehicle silhouette as a polygon.  Enter a
    reference dimension (e.g. motor-to-motor distance in cm) so the
    tool can convert pixel area to real m².  Computes:

      • EK3_DRAG_BCOEF_X  [kg/m²]  – ArduPilot EKF frontal ballistic coeff
      • EK3_DRAG_BCOEF_Y  [kg/m²]  – ArduPilot EKF side ballistic coeff
      • parasite_area / parasite_drag_coefficient  – for the multicopter sim
        (forward-flight drag, driven by frontal cross-section)
      • profile_area  / profile_drag_coefficient   – for the multicopter sim
        (hover lateral drag, driven by top cross-section)

  Tab 2 — Propeller Drag (MCOEF)
    Physics-based estimate of EK3_DRAG_MCOEF from actuator-disk theory.
    Inputs: mass, number of motors, prop diameter, altitude, temperature.
    Output: hover induced velocity, MCOEF, interpretation notes.

Relationship between ArduPilot BCOEF and the multicopter sim
-------------------------------------------------------------
ArduPilot's EKF uses:
    a_drag = (ρ / 2) × V² / BCOEF
which expands to:
    F_drag = m × a_drag = (ρ / 2) × V² × A × Cd

So:   BCOEF = mass / (Cd × Area)

ArduPilot's own calibration guide tells you to use   BCOEF = mass / Area,
implicitly assuming Cd = 1.0 and trusting that measured projected area
from a photograph is accurate enough.

The multicopter power sim uses separate inputs:
    F_parasite = ½ρV² × parasite_area × parasite_drag_coefficient
    F_profile  = ½ρV² × profile_area  × profile_drag_coefficient

This tool lets you choose Cd explicitly (default 1.0 to match ArduPilot)
and outputs both the combined BCOEF and the split Cd / Area for the sim.

Usage:
    python drag_coefficient_calculator.py

Requirements:
    pip install Pillow
"""

from __future__ import annotations

import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pillow is needed only for image loading; all other features work without it.
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================
# PHYSICS CONSTANTS
# ============================================================
G0   = 9.80665   # m/s²  – standard gravity
RHO0 = 1.225     # kg/m³ – ISA sea-level air density
T0   = 288.15    # K     – sea-level ISA temperature
P0   = 101325.0  # Pa    – sea-level ISA pressure
L    = 0.0065    # K/m   – tropospheric temperature lapse rate
R    = 287.05    # J/kg/K – specific gas constant for dry air


# ============================================================
# ISA AIR DENSITY
# ============================================================
def isa_density(altitude_m: float,
                temperature_C: Optional[float] = None) -> float:
    """
    Compute air density [kg/m³] using the International Standard Atmosphere.

    If temperature_C is provided it overrides the ISA lapse-rate temperature,
    allowing a hot-day or cold-day density to be used while keeping the ISA
    pressure profile.
    """
    h     = max(float(altitude_m), 0.0)
    T_isa = T0 - L * h                                # ISA temperature [K]
    P_isa = P0 * (T_isa / T0) ** (G0 / (R * L))      # ISA pressure [Pa]
    T_K   = T_isa if temperature_C is None else float(temperature_C) + 273.15
    return P_isa / (R * max(T_K, 1.0))


# ============================================================
# POLYGON AREA  (Shoelace / Gauss formula)
# ============================================================
def polygon_area_px2(points: List[Tuple[float, float]]) -> float:
    """
    Compute the area of a simple (non-self-intersecting) polygon defined by a
    list of (x, y) pixel-coordinate vertices using the shoelace formula.

    Returns area in px².  The sign is discarded (i.e. always positive).
    """
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


# ============================================================
# VIEW CANVAS  — one photo + polygon editor per vehicle view
# ============================================================
class ViewCanvas(ttk.Frame):
    """
    Self-contained widget that manages one vehicle-view photograph and the
    corresponding polygon outline drawn by the user.

    Workflow
    --------
    1. Load an image with the Load Image button.
    2. Click Set Scale → click two points over a feature of known real-world
       length (e.g. motor-to-motor distance), then enter that distance in cm.
    3. Click Draw Outline → click to place polygon vertices around the vehicle
       body (exclude propeller blades).
       - Right-click  : remove the last vertex
       - Double-click : close the polygon
       - Click near vertex 1 when ≥ 3 points exist : auto-close (snap)
    4. Read the computed area via get_area_m2().
    """

    # Canvas dimensions in pixels — adjust if your display is very small/large.
    CANVAS_W = 580
    CANVAS_H = 430

    # Visual constants
    VERTEX_R      = 5        # radius of vertex handle circles [px]
    SNAP_RADIUS   = 14       # snap-to-close distance [px]

    # Colour palette
    C_POLYGON  = "#2E75B6"   # polygon edges and open-mode vertex handles
    C_CLOSED_V = "#1F5490"   # vertex handles when polygon is closed
    C_FILL     = "#2E75B6"   # translucent polygon fill (stippled)
    C_SCALE    = "#E05252"   # scale-reference line and points
    C_RUBBER   = "#AAAAAA"   # rubber-band edge (in-progress)

    def __init__(self, parent: tk.Widget, view_label: str, **kwargs):
        super().__init__(parent, **kwargs)
        self.view_label = view_label   # "Front", "Side", or "Top"

        # ---- Image state ----
        self._pil_image:  Optional[Image.Image]        = None
        self._tk_image:   Optional[ImageTk.PhotoImage] = None
        self._img_scale:    float = 1.0   # canvas px / image px
        self._img_offset_x: float = 0.0  # horizontal letterbox offset
        self._img_offset_y: float = 0.0  # vertical letterbox offset

        # ---- Scale reference state ----
        self._scale_mode: bool = False
        self._scale_pts: List[Tuple[float, float]] = []   # in image coords
        self._px_per_m:  Optional[float] = None
        self._scale_desc: str = "not set"

        # ---- Polygon state ----
        self._draw_mode:     bool = False
        self._vertices: List[Tuple[float, float]] = []    # in image coords
        self._polygon_closed: bool = False
        self._mouse_canvas: Optional[Tuple[int, int]] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        pane = ttk.Frame(self)
        pane.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        pane.columnconfigure(0, weight=1)
        pane.columnconfigure(1, weight=0)
        pane.rowconfigure(0, weight=1)

        # ---- Canvas frame ----
        cf = ttk.LabelFrame(pane, text=f"{self.view_label} View", padding=4)
        cf.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            cf,
            width=self.CANVAS_W, height=self.CANVAS_H,
            bg="#1A1A2E", cursor="crosshair",
            relief="sunken", bd=1,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._draw_placeholder()

        # Canvas event bindings
        self.canvas.bind("<ButtonPress-1>",   self._on_left_click)
        self.canvas.bind("<ButtonPress-3>",   self._on_right_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Motion>",          self._on_motion)

        # ---- Controls panel ----
        ctrl = ttk.Frame(pane)
        ctrl.grid(row=0, column=1, sticky="ns")

        def btn(text, cmd, **kw):
            b = ttk.Button(ctrl, text=text, command=cmd, **kw)
            return b

        self._btn_load  = btn("📂  Load Image",    self._load_image)
        self._btn_scale = btn("📏  Set Scale…",    self._enter_scale_mode, state="disabled")
        self._btn_draw  = btn("✏️  Draw Outline",   self._enter_draw_mode,  state="disabled")
        self._btn_undo  = btn("↩  Undo Point",     self._undo_last_vertex)
        self._btn_close = btn("✔  Close Polygon",  self._close_polygon)
        self._btn_clear = btn("🗑  Clear All",      self.clear)

        for i, b in enumerate([self._btn_load, self._btn_scale, self._btn_draw,
                                self._btn_undo, self._btn_close]):
            b.grid(row=i, column=0, sticky="ew", pady=(0, 4))
        self._btn_clear.grid(row=5, column=0, sticky="ew", pady=(0, 12))

        ttk.Separator(ctrl, orient="horizontal").grid(
            row=6, column=0, sticky="ew", pady=4)

        # ---- Status area ----
        sf = ttk.LabelFrame(ctrl, text="Status", padding=6)
        sf.grid(row=7, column=0, sticky="nsew")

        def stat_pair(label: str) -> ttk.Label:
            """Helper: add a labelled status value row and return the value label."""
            ttk.Label(sf, text=label, foreground="#888888",
                      font=("TkDefaultFont", 8)).pack(anchor="w")
            v = ttk.Label(sf, text="—", font=("TkDefaultFont", 9, "bold"),
                          wraplength=140, justify="left")
            v.pack(anchor="w", padx=(8, 0), pady=(0, 4))
            return v

        self._st_mode  = stat_pair("Mode")
        self._st_scale = stat_pair("Scale")
        self._st_pts   = stat_pair("Vertices")
        self._st_area  = stat_pair("Area")

        self._set_status("Load image to begin", "—", "0", "—")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _draw_placeholder(self) -> None:
        """Draw the 'no image loaded' placeholder text."""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.CANVAS_W // 2, self.CANVAS_H // 2,
            text=f"Load a {self.view_label.lower()}-view\nimage to begin",
            fill="#555577", font=("TkDefaultFont", 13), justify="center",
            tags="placeholder",
        )

    def _set_status(self, mode: str, scale: str, pts: str, area: str) -> None:
        self._st_mode.configure(text=mode)
        self._st_scale.configure(text=scale)
        self._st_pts.configure(text=pts)
        self._st_area.configure(text=area)

    def _area_str(self) -> str:
        """Format the current polygon area for display."""
        a = self.get_area_m2()
        if a is None:
            return "—"
        return f"{a:.5f} m²\n= {a * 1e4:.1f} cm²"

    # ------------------------------------------------------------------
    # COORDINATE TRANSFORMS  (image ↔ canvas)
    # ------------------------------------------------------------------
    def _to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        """Image-pixel coordinates → canvas-pixel coordinates."""
        return (ix * self._img_scale + self._img_offset_x,
                iy * self._img_scale + self._img_offset_y)

    def _to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        """Canvas-pixel coordinates → image-pixel coordinates."""
        s = max(self._img_scale, 1e-12)
        return ((cx - self._img_offset_x) / s,
                (cy - self._img_offset_y) / s)

    # ------------------------------------------------------------------
    # IMAGE LOADING
    # ------------------------------------------------------------------
    def _load_image(self) -> None:
        if not HAS_PIL:
            messagebox.showerror(
                "Pillow not installed",
                "Image loading requires Pillow.\n\n"
                "Install with:   pip install Pillow")
            return

        path = filedialog.askopenfilename(
            title=f"Load {self.view_label} View Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("All files", "*.*"),
            ])
        if not path:
            return

        self.clear()

        try:
            self._pil_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image error", f"Could not open image:\n{exc}")
            return

        self._render_image()
        self._btn_scale.configure(state="normal")
        self._set_status(
            "Image loaded — click Set Scale",
            "not set", "0", "—")

    def _render_image(self) -> None:
        """Scale the PIL image to fit the canvas and display it."""
        if self._pil_image is None:
            return
        iw, ih = self._pil_image.size
        cw, ch  = self.CANVAS_W, self.CANVAS_H

        # Fit within canvas while preserving aspect ratio
        scale = min(cw / iw, ch / ih)
        self._img_scale    = scale
        self._img_offset_x = (cw - iw * scale) / 2.0
        self._img_offset_y = (ch - ih * scale) / 2.0

        dw = max(1, int(iw * scale))
        dh = max(1, int(ih * scale))
        resized = self._pil_image.resize((dw, dh), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(
            self._img_offset_x, self._img_offset_y,
            anchor="nw", image=self._tk_image, tags="image")

    # ------------------------------------------------------------------
    # SCALE REFERENCE
    # ------------------------------------------------------------------
    def _enter_scale_mode(self) -> None:
        """Switch to scale-reference mode.  User must click two points."""
        if self._pil_image is None:
            return
        self._scale_mode = True
        self._draw_mode  = False
        self._scale_pts  = []
        self._redraw_overlay()
        self._set_status(
            "SCALE: click point 1 of known distance",
            self._scale_desc, str(len(self._vertices)), self._area_str())

    def _finish_scale(self) -> None:
        """Called after the user has clicked both scale reference points."""
        if len(self._scale_pts) < 2:
            return

        # Distance in image pixels between the two reference points
        x1, y1 = self._scale_pts[0]
        x2, y2 = self._scale_pts[1]
        ref_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if ref_px < 2.0:
            messagebox.showwarning("Scale",
                "The two points are too close together. Please try again.")
            self._scale_pts = []
            return

        # Ask user for the real-world distance
        raw = simpledialog.askstring(
            "Scale Reference",
            "Enter the real-world distance between the\n"
            "two clicked points (in centimetres):",
            parent=self)
        if raw is None:
            self._scale_pts = []
            return
        try:
            dist_cm = float(raw.strip())
            if dist_cm <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input",
                "Please enter a positive number for the distance in cm.")
            self._scale_pts = []
            return

        dist_m = dist_cm / 100.0
        self._px_per_m  = ref_px / dist_m
        self._scale_desc = (f"{dist_cm:.1f} cm = {ref_px:.0f} px\n"
                            f"({self._px_per_m:.1f} px/m)")
        self._scale_mode = False

        # Enable the Draw Outline button now that we have a scale
        self._btn_draw.configure(state="normal")
        self._redraw_overlay()
        self._set_status(
            "Scale set — click Draw Outline",
            self._scale_desc,
            str(len(self._vertices)),
            self._area_str())

    # ------------------------------------------------------------------
    # POLYGON DRAWING
    # ------------------------------------------------------------------
    def _enter_draw_mode(self) -> None:
        """Switch to polygon drawing mode."""
        if self._px_per_m is None:
            messagebox.showinfo("Set scale first",
                "Please set the scale reference before drawing the outline.")
            return
        self._draw_mode = True
        self._scale_mode = False
        if self._polygon_closed:
            # Re-open to allow editing
            self._polygon_closed = False
        self._set_status(
            "DRAW: click to add vertices\n"
            "Right-click = undo  |  Dbl-click = close",
            self._scale_desc,
            str(len(self._vertices)),
            self._area_str())

    def _close_polygon(self) -> None:
        """Close the polygon, computing the final area."""
        if len(self._vertices) < 3:
            messagebox.showinfo("Not enough points",
                "You need at least 3 vertices to close the polygon.")
            return
        self._polygon_closed = True
        self._draw_mode = False
        self._redraw_overlay()
        self._set_status(
            "Polygon closed ✔",
            self._scale_desc,
            str(len(self._vertices)),
            self._area_str())

    def _undo_last_vertex(self) -> None:
        """Remove the most recently added polygon vertex."""
        if self._polygon_closed:
            self._polygon_closed = False
            self._draw_mode = True
        if self._vertices:
            self._vertices.pop()
        self._redraw_overlay()
        self._set_status(
            "Point removed",
            self._scale_desc,
            str(len(self._vertices)),
            self._area_str())

    # ------------------------------------------------------------------
    # CANVAS EVENT HANDLERS
    # ------------------------------------------------------------------
    def _on_left_click(self, event: tk.Event) -> None:
        if self._pil_image is None:
            return
        cx, cy = float(event.x), float(event.y)
        ix, iy = self._to_image(cx, cy)

        if self._scale_mode:
            # Accumulate scale reference points
            self._scale_pts.append((ix, iy))
            self._redraw_overlay()
            if len(self._scale_pts) == 1:
                self._set_status(
                    "SCALE: click point 2",
                    self._scale_desc,
                    str(len(self._vertices)),
                    self._area_str())
            elif len(self._scale_pts) >= 2:
                self._finish_scale()
            return

        if self._draw_mode and not self._polygon_closed:
            # Snap-to-start: if we're close to the first vertex, auto-close
            if len(self._vertices) >= 3:
                fx, fy = self._to_canvas(*self._vertices[0])
                dist_to_first = math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
                if dist_to_first < self.SNAP_RADIUS:
                    self._close_polygon()
                    return

            self._vertices.append((ix, iy))
            self._redraw_overlay()
            self._set_status(
                "DRAW: click to add vertices\n"
                "Right-click = undo  |  Dbl-click = close",
                self._scale_desc,
                str(len(self._vertices)),
                self._area_str())

    def _on_right_click(self, _event: tk.Event) -> None:
        """Right-click removes the last polygon vertex."""
        self._undo_last_vertex()

    def _on_double_click(self, _event: tk.Event) -> None:
        """Double-click closes the polygon."""
        if self._draw_mode and len(self._vertices) >= 3:
            self._close_polygon()

    def _on_motion(self, event: tk.Event) -> None:
        """Track mouse for rubber-band line and snap indicator."""
        self._mouse_canvas = (event.x, event.y)
        if (self._draw_mode and not self._polygon_closed and self._vertices):
            self._redraw_overlay()

    # ------------------------------------------------------------------
    # OVERLAY RENDERING
    # ------------------------------------------------------------------
    def _redraw_overlay(self) -> None:
        """Delete and redraw all overlay elements (scale line, polygon)."""
        self.canvas.delete("overlay")
        if self._scale_pts:
            self._draw_scale_line()
        if self._vertices:
            self._draw_polygon_overlay()

    def _draw_scale_line(self) -> None:
        """Render the scale reference line and its endpoint markers."""
        pts_c = [self._to_canvas(ix, iy) for ix, iy in self._scale_pts]

        # Endpoint circles
        for cx, cy in pts_c:
            r = 6
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=self.C_SCALE, outline="white", width=1,
                tags="overlay")

        if len(pts_c) >= 2:
            # Solid line between the two points
            self.canvas.create_line(
                pts_c[0][0], pts_c[0][1], pts_c[1][0], pts_c[1][1],
                fill=self.C_SCALE, width=2, dash=(6, 3), tags="overlay")
            # Label at midpoint
            mx = (pts_c[0][0] + pts_c[1][0]) / 2
            my = (pts_c[0][1] + pts_c[1][1]) / 2
            self.canvas.create_text(mx, my - 12,
                text="scale ref", fill=self.C_SCALE,
                font=("TkDefaultFont", 8, "bold"), tags="overlay")

        elif len(pts_c) == 1 and self._mouse_canvas and self._scale_mode:
            # Rubber-band from first scale point to mouse
            mx, my = self._mouse_canvas
            self.canvas.create_line(
                pts_c[0][0], pts_c[0][1], mx, my,
                fill=self.C_SCALE, width=1, dash=(4, 4), tags="overlay")

    def _draw_polygon_overlay(self) -> None:
        """Render polygon edges, vertex handles, rubber-band, and fill."""
        pts_c = [self._to_canvas(ix, iy) for ix, iy in self._vertices]

        # Draw edges between consecutive vertices
        for i in range(len(pts_c) - 1):
            self.canvas.create_line(
                pts_c[i][0], pts_c[i][1],
                pts_c[i + 1][0], pts_c[i + 1][1],
                fill=self.C_POLYGON, width=2, tags="overlay")

        if self._polygon_closed and len(pts_c) >= 3:
            # Closing edge
            self.canvas.create_line(
                pts_c[-1][0], pts_c[-1][1],
                pts_c[0][0],  pts_c[0][1],
                fill=self.C_POLYGON, width=2, tags="overlay")
            # Translucent fill (Tkinter stipple simulates transparency)
            flat = [coord for pt in pts_c for coord in pt]
            self.canvas.create_polygon(
                flat,
                fill=self.C_FILL, stipple="gray25",
                outline=self.C_POLYGON, width=2,
                tags="overlay")

        # Rubber-band line from last vertex to mouse cursor
        if (self._draw_mode and not self._polygon_closed
                and self._mouse_canvas and pts_c):
            mx, my = self._mouse_canvas
            self.canvas.create_line(
                pts_c[-1][0], pts_c[-1][1], mx, my,
                fill=self.C_RUBBER, width=1, dash=(5, 3), tags="overlay")

            # Snap-to-start indicator (yellow ring around first vertex)
            if len(pts_c) >= 3:
                fx, fy = pts_c[0]
                if math.sqrt((mx - fx) ** 2 + (my - fy) ** 2) < self.SNAP_RADIUS:
                    r = self.VERTEX_R + 5
                    self.canvas.create_oval(
                        fx - r, fy - r, fx + r, fy + r,
                        outline="yellow", width=2, tags="overlay")

        # Vertex handle circles
        vc = self.C_CLOSED_V if self._polygon_closed else self.C_POLYGON
        for i, (cx, cy) in enumerate(pts_c):
            r = self.VERTEX_R
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=vc, outline="white", width=1, tags="overlay")
            # Mark the first vertex with a label so user knows the snap target
            if i == 0:
                self.canvas.create_text(
                    cx + 10, cy - 9,
                    text="1", fill="yellow",
                    font=("TkDefaultFont", 8, "bold"), tags="overlay")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def get_area_m2(self) -> Optional[float]:
        """
        Return the measured polygon area in m².

        Returns None if:
          - The polygon is not yet closed, or
          - Fewer than 3 vertices have been placed, or
          - The scale reference has not been set.
        """
        if not self._polygon_closed:
            return None
        if len(self._vertices) < 3:
            return None
        if self._px_per_m is None or self._px_per_m <= 0:
            return None
        area_px2 = polygon_area_px2(self._vertices)
        return area_px2 / (self._px_per_m ** 2)

    def clear(self) -> None:
        """Reset the widget to its initial blank state."""
        self._pil_image      = None
        self._tk_image       = None
        self._img_scale      = 1.0
        self._img_offset_x   = 0.0
        self._img_offset_y   = 0.0
        self._scale_mode     = False
        self._scale_pts      = []
        self._px_per_m       = None
        self._scale_desc     = "not set"
        self._draw_mode      = False
        self._vertices       = []
        self._polygon_closed = False
        self._mouse_canvas   = None

        self._draw_placeholder()
        self._btn_scale.configure(state="disabled")
        self._btn_draw.configure(state="disabled")
        self._set_status("Load image to begin", "—", "0", "—")

    def get_summary(self) -> dict:
        """Return a dict describing this view's current measurements."""
        area = self.get_area_m2()
        return {
            "view":       self.view_label,
            "area_m2":    area,
            "area_cm2":   area * 1e4 if area is not None else None,
            "n_vertices": len(self._vertices),
            "px_per_m":   self._px_per_m,
        }


# ============================================================
# BODY DRAG TAB
# ============================================================
class BodyDragTab(ttk.Frame):
    """
    Tab 1 — Body Drag (BCOEF / parasite area).

    Contains three ViewCanvas instances (front, side, top) plus a results
    panel that computes ArduPilot EK3 parameters and multicopter sim inputs.

    Aerodynamic Cd note
    -------------------
    ArduPilot's own calibration guide tells the user to compute
        BCOEF = mass / projected_area
    This implicitly assumes Cd = 1.0 (absorbed into the ballistic coefficient).
    We expose Cd as an adjustable parameter (default 1.0) so users who have
    wind-tunnel or CFD data can use a more accurate value.  Either way:
        BCOEF = mass / (Cd × area)

    Multicopter sim mapping
    -----------------------
    Forward-flight drag model uses parasite_area × parasite_drag_coefficient.
    Hover lateral drag uses profile_area × profile_drag_coefficient.
    Frontal area → parasite  (dominant in forward flight)
    Top area     → profile   (dominant in hover lateral motion)
    Side area    → BCOEF_Y only (ArduPilot EKF Y-axis)
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)
        self._last_results: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)

        # ---- View sub-notebook (front / side / top) ----
        view_nb = ttk.Notebook(self)
        view_nb.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 0))

        self.front = ViewCanvas(view_nb, "Front")
        self.side  = ViewCanvas(view_nb, "Side")
        self.top   = ViewCanvas(view_nb, "Top")

        view_nb.add(self.front, text="  Front View  (BCOEF_X / parasite)  ")
        view_nb.add(self.side,  text="  Side View   (BCOEF_Y)              ")
        view_nb.add(self.top,   text="  Top View    (profile / hover)       ")

        # ---- Results panel ----
        res_outer = ttk.LabelFrame(self, text="Results & Outputs", padding=10)
        res_outer.grid(row=1, column=0, sticky="ew", padx=4, pady=(4, 4))
        res_outer.columnconfigure(0, weight=0)
        res_outer.columnconfigure(1, weight=1)
        res_outer.columnconfigure(2, weight=1)

        # Left: inputs
        inp = ttk.Frame(res_outer)
        inp.grid(row=0, column=0, sticky="nw", padx=(0, 20))

        def add_inp(row, label, var, hint=None):
            ttk.Label(inp, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(inp, textvariable=var, width=10).grid(
                row=row, column=1, sticky="w", padx=(6, 0))
            if hint:
                ttk.Label(inp, text=hint, foreground="#888888",
                          font=("TkDefaultFont", 8)).grid(
                    row=row + 1, column=0, columnspan=2, sticky="w")

        self.v_mass = tk.StringVar(value="1.5")
        self.v_cd   = tk.StringVar(value="1.0")

        r = 0
        add_inp(r, "Vehicle mass (kg):", self.v_mass); r += 1
        add_inp(r, "Body Cd:", self.v_cd,
                hint="1.0 = ArduPilot default; bluff body 0.9–1.3"); r += 2

        ttk.Button(inp, text="⚡  Calculate",
                   command=self._calculate).grid(
            row=r, column=0, columnspan=2, pady=(10, 0), sticky="w")
        ttk.Button(inp, text="💾  Export JSON…",
                   command=self._export_json).grid(
            row=r + 1, column=0, columnspan=2, pady=(4, 0), sticky="w")

        # Centre: ArduPilot parameters
        ap = ttk.LabelFrame(res_outer, text="ArduPilot (EK3) Parameters", padding=8)
        ap.grid(row=0, column=1, sticky="nsew", padx=(0, 8))

        def res_row(parent, label):
            f = ttk.Frame(parent)
            f.pack(fill="x", pady=2)
            ttk.Label(f, text=label, width=22, anchor="w",
                      foreground="#555555").pack(side="left")
            val = ttk.Label(f, text="—", font=("TkDefaultFont", 10, "bold"),
                            foreground="#1F3864")
            val.pack(side="left", padx=4)
            return val

        self._lbl_bcoef_x  = res_row(ap, "EK3_DRAG_BCOEF_X:")
        self._lbl_bcoef_y  = res_row(ap, "EK3_DRAG_BCOEF_Y:")
        self._lbl_front_a  = res_row(ap, "Frontal area:")
        self._lbl_side_a   = res_row(ap, "Side area:")
        self._lbl_top_a    = res_row(ap, "Top area:")

        # Right: multicopter sim parameters
        sim = ttk.LabelFrame(res_outer, text="Multicopter Sim Inputs", padding=8)
        sim.grid(row=0, column=2, sticky="nsew")

        self._lbl_par_area  = res_row(sim, "parasite_area:")
        self._lbl_par_cd    = res_row(sim, "parasite_drag_coeff:")
        self._lbl_prof_area = res_row(sim, "profile_area:")
        self._lbl_prof_cd   = res_row(sim, "profile_drag_coeff:")

        # Notes
        note = (
            "Notes:\n"
            "• parasite_area = frontal area (forward flight drag)\n"
            "• profile_area  = top area  (hover lateral drag)\n"
            "• Both use the same body Cd entered above\n"
            "• Side area feeds BCOEF_Y only (ArduPilot EKF)"
        )
        ttk.Label(res_outer, text=note, foreground="#666666",
                  font=("TkDefaultFont", 8), justify="left").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

    # ------------------------------------------------------------------
    def _calculate(self) -> None:
        """Parse inputs, validate, and compute all drag outputs."""
        # --- Validate inputs ---
        try:
            mass_kg = float(self.v_mass.get().strip())
            if mass_kg <= 0:
                raise ValueError("Mass must be > 0")
        except ValueError as exc:
            messagebox.showerror("Input error", f"Invalid mass: {exc}")
            return

        try:
            cd = float(self.v_cd.get().strip())
            if not (0.05 <= cd <= 5.0):
                raise ValueError("Cd should be between 0.05 and 5.0")
        except ValueError as exc:
            messagebox.showerror("Input error", f"Invalid Cd: {exc}")
            return

        front_a = self.front.get_area_m2()
        side_a  = self.side.get_area_m2()
        top_a   = self.top.get_area_m2()     # optional

        # Front and side are required; top is optional (produces profile_area)
        missing = []
        if front_a is None:
            missing.append("Front view: complete scale + polygon")
        if side_a is None:
            missing.append("Side view: complete scale + polygon")
        if missing:
            messagebox.showwarning("Incomplete data",
                "Cannot calculate — please complete:\n"
                + "\n".join(f"  • {m}" for m in missing))
            return

        # --- Compute ---
        # ArduPilot:  BCOEF = mass / (Cd × Area)
        bcoef_x = mass_kg / (cd * front_a)
        bcoef_y = mass_kg / (cd * side_a)

        # Multicopter sim:
        #   parasite  → forward-flight drag → driven by frontal area
        #   profile   → hover lateral drag  → driven by top area
        par_area  = front_a
        prof_area = top_a    # may be None

        def fmt_a(v: Optional[float]) -> str:
            if v is None:
                return "not measured"
            return f"{v:.5f} m²  ({v * 1e4:.1f} cm²)"

        def fmt_b(v: float) -> str:
            return f"{v:.2f} kg/m²"

        self._lbl_bcoef_x.configure(text=fmt_b(bcoef_x))
        self._lbl_bcoef_y.configure(text=fmt_b(bcoef_y))
        self._lbl_front_a.configure(text=fmt_a(front_a))
        self._lbl_side_a.configure(text=fmt_a(side_a))
        self._lbl_top_a.configure(text=fmt_a(top_a))

        self._lbl_par_area.configure(text=fmt_a(par_area))
        self._lbl_par_cd.configure(text=f"{cd:.2f}")
        self._lbl_prof_area.configure(
            text=fmt_a(prof_area) if prof_area else "measure top view first")
        self._lbl_prof_cd.configure(
            text=f"{cd:.2f}" if prof_area else "—")

        self._last_results = {
            "mass_kg":           mass_kg,
            "body_Cd":           cd,
            "frontal_area_m2":   front_a,
            "side_area_m2":      side_a,
            "top_area_m2":       top_a,
            "ardupilot": {
                "EK3_DRAG_BCOEF_X": round(bcoef_x, 2),
                "EK3_DRAG_BCOEF_Y": round(bcoef_y, 2),
            },
            "multicopter_sim": {
                "parasite_area":             round(par_area, 6),
                "parasite_drag_coefficient": round(cd, 4),
                "profile_area":              round(prof_area, 6) if prof_area else None,
                "profile_drag_coefficient":  round(cd, 4) if prof_area else None,
            },
        }

    def _export_json(self) -> None:
        if not self._last_results:
            messagebox.showinfo("No results", "Run Calculate first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Body Drag Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._last_results, fh, indent=2)
        messagebox.showinfo("Saved", f"Results written to:\n{path}")


# ============================================================
# PROPELLER DRAG TAB
# ============================================================
class PropDragTab(ttk.Frame):
    """
    Tab 2 — Propeller Momentum Drag (MCOEF).

    Derives EK3_DRAG_MCOEF from actuator-disk theory.

    Theory
    ------
    In hover each rotor pushes air downward at the induced velocity v_h:

        T_per_motor = W / N_motors                       [N]
        v_h = sqrt( T_per_motor / (2 · ρ · A_disk) )    [m/s]

    When the vehicle moves forward at airspeed V, the rotor wake (momentum
    flow) creates an additional 'momentum drag' force.  Summing over all
    N rotors and assuming the hover-condition induced velocity throughout:

        F_momentum = N · ρ · A_disk · v_h · V
                   = (W / (2 · v_h)) · V                [N]

    The resulting deceleration is:

        a_drag = F / mass = (g / (2 · v_h)) · V          [m/s²]

    The ArduPilot EKF models propeller momentum drag as:

        a_drag = MCOEF · V

    Therefore:

        MCOEF = g / (2 · v_h)

    Typical values: 0.1 – 1.0.
    Small props on a heavy vehicle → large v_h → small MCOEF.
    Large props on a light vehicle → small v_h → large MCOEF.

    Assumptions
    -----------
    • Ideal actuator-disk (no blade profile drag contribution).
    • Hover-condition induced velocity v_h used at all forward speeds
      (conservative; the real v_h drops slightly in forward flight).
    • MCOEF is for propeller drag only, separate from body BCOEF.
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)
        self._last_results: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # ---- Input panel ----
        inp_frame = ttk.LabelFrame(self, text="Vehicle & Propulsion Inputs", padding=10)
        inp_frame.grid(row=0, column=0, sticky="nw", padx=(4, 4), pady=4)
        inp_frame.columnconfigure(2, weight=1)

        def inp(row, label, var, hint="", req=True):
            marker = "" if req else " (opt.)"
            ttk.Label(inp_frame, text=label + marker).grid(
                row=row, column=0, sticky="w", pady=2)
            ttk.Entry(inp_frame, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", padx=6)
            if hint:
                ttk.Label(inp_frame, text=hint, foreground="#888888",
                          font=("TkDefaultFont", 8)).grid(
                    row=row, column=2, sticky="w")

        r = 0
        self.v_mass   = tk.StringVar(value="1.5")
        self.v_nm     = tk.StringVar(value="4")
        self.v_diam   = tk.StringVar(value="10")
        self.v_alt    = tk.StringVar(value="0")
        self.v_temp   = tk.StringVar(value="")
        self.v_kv     = tk.StringVar(value="")
        self.v_batt   = tk.StringVar(value="")
        self.v_thr    = tk.StringVar(value="")

        inp(r, "Vehicle mass (kg):",      self.v_mass, "AUW including battery"); r += 1
        inp(r, "Number of motors:",        self.v_nm,   "e.g. 4, 6, 8");         r += 1
        inp(r, "Prop diameter (inches):",  self.v_diam, "e.g. 10");              r += 1
        inp(r, "Altitude (m):",            self.v_alt,  "0 = sea level");        r += 1
        inp(r, "Temperature (°C):",        self.v_temp, "blank = ISA", req=False); r += 1

        ttk.Separator(inp_frame, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=6); r += 1
        ttk.Label(inp_frame, text="Optional — RPM cross-check",
                  foreground="#888888").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        inp(r, "Motor KV (rpm/V):",        self.v_kv,   "", req=False); r += 1
        inp(r, "Battery voltage (V):",     self.v_batt, "", req=False); r += 1
        inp(r, "Hover throttle (0–1):",    self.v_thr,  "typical ~0.45–0.55", req=False); r += 1

        ttk.Button(inp_frame, text="⚡  Calculate MCOEF",
                   command=self._calculate).grid(
            row=r, column=0, columnspan=3, pady=(10, 0), sticky="w")

        # ---- Theory panel ----
        theory_frame = ttk.LabelFrame(self, text="Physics", padding=10)
        theory_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=4)
        theory_frame.columnconfigure(0, weight=1)
        theory_frame.rowconfigure(0, weight=1)

        theory_text = (
            "ACTUATOR-DISK MOMENTUM DRAG\n"
            "═══════════════════════════\n\n"
            "Hover thrust per motor\n"
            "  T = W / N_motors\n\n"
            "Hover induced velocity (per rotor)\n"
            "  v_h = √[ T / (2 · ρ · A_disk) ]\n"
            "  where A_disk = π/4 · D²\n\n"
            "Forward-flight momentum drag force\n"
            "  (N rotors, airspeed V)\n"
            "  F = (W / 2·v_h) · V\n\n"
            "Deceleration\n"
            "  a = F / m = (g / 2·v_h) · V\n\n"
            "ArduPilot EKF model\n"
            "  a_prop_drag = MCOEF · V\n\n"
            "Therefore\n"
            "  MCOEF = g / (2 · v_h)\n\n"
            "Typical range: 0.1 – 1.0\n\n"
            "MCOEF is a linear (speed-proportional) drag\n"
            "coefficient distinct from body BCOEF which is\n"
            "quadratic (speed²-proportional).\n\n"
            "For best accuracy, validate MCOEF with a flight\n"
            "test following ArduPilot's EK3_DRAG_MCOEF guide."
        )
        txt = tk.Text(theory_frame, wrap="word", width=42, height=26,
                      font=("Courier", 9), relief="flat", state="normal",
                      bg="#F4F6F8", padx=4, pady=4)
        txt.insert("1.0", theory_text)
        txt.configure(state="disabled")
        txt.grid(row=0, column=0, sticky="nsew")

        # ---- Results panel ----
        res_frame = ttk.LabelFrame(self, text="Results", padding=10)
        res_frame.grid(row=1, column=0, columnspan=2, sticky="nsew",
                       padx=4, pady=(0, 4))
        res_frame.columnconfigure(0, weight=1)
        res_frame.columnconfigure(1, weight=1)
        res_frame.columnconfigure(2, weight=1)
        res_frame.rowconfigure(1, weight=1)

        def big_result(col, label) -> ttk.Label:
            f = ttk.Frame(res_frame)
            f.grid(row=0, column=col, sticky="nsew", padx=16, pady=(0, 8))
            ttk.Label(f, text=label, foreground="#666666",
                      font=("TkDefaultFont", 9)).pack(anchor="w")
            lbl = ttk.Label(f, text="—",
                            font=("TkDefaultFont", 16, "bold"),
                            foreground="#1F3864")
            lbl.pack(anchor="w")
            return lbl

        self._lbl_vh    = big_result(0, "Hover induced velocity  v_h")
        self._lbl_mcoef = big_result(1, "EK3_DRAG_MCOEF  (ArduPilot)")
        self._lbl_interp= big_result(2, "Interpretation")

        # Detail text area
        # ttk widgets don't expose "background" via cget on all platforms
        # (raises TclError on Windows / Python 3.9).  Fall back gracefully.
        try:
            _detail_bg = res_frame.cget("background")
        except tk.TclError:
            _detail_bg = "white"
        self._detail = tk.Text(res_frame, wrap="word", height=6,
                               font=("TkDefaultFont", 9), relief="flat",
                               state="disabled", bg=_detail_bg)
        self._detail.grid(row=1, column=0, columnspan=3, sticky="nsew")

        ttk.Button(res_frame, text="💾  Export JSON…",
                   command=self._export_json).grid(
            row=2, column=0, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------------
    def _calculate(self) -> None:
        """Validate inputs and compute MCOEF from actuator-disk theory."""
        errors: List[str] = []

        def get_f(var: tk.StringVar, name: str,
                  lo: Optional[float] = None,
                  hi: Optional[float] = None,
                  required: bool = True) -> Optional[float]:
            s = var.get().strip()
            if not s:
                if required:
                    errors.append(f"{name} is required")
                return None
            try:
                v = float(s)
            except ValueError:
                errors.append(f"{name}: not a valid number")
                return None
            if lo is not None and v < lo:
                errors.append(f"{name}: must be ≥ {lo}")
            if hi is not None and v > hi:
                errors.append(f"{name}: must be ≤ {hi}")
            return v

        mass_kg  = get_f(self.v_mass, "Mass (kg)",           lo=0.001)
        n_motors = get_f(self.v_nm,   "Number of motors",    lo=1, hi=64)
        diam_in  = get_f(self.v_diam, "Prop diameter (in)",  lo=0.5, hi=120)
        alt_m    = get_f(self.v_alt,  "Altitude (m)",        lo=0, hi=8000)
        temp_C   = get_f(self.v_temp, "Temperature (°C)",    required=False)
        kv       = get_f(self.v_kv,   "Motor KV",            lo=0, required=False)
        batt_v   = get_f(self.v_batt, "Battery voltage (V)", lo=0, required=False)
        thr      = get_f(self.v_thr,  "Hover throttle",      lo=0, hi=1.0, required=False)

        if errors:
            messagebox.showerror("Input errors", "\n".join(f"• {e}" for e in errors))
            return

        N   = max(1, int(round(n_motors)))
        rho = isa_density(alt_m or 0.0, temperature_C=temp_C)

        # Single-rotor disk area
        D_m    = diam_in * 0.0254
        A_disk = math.pi / 4.0 * D_m ** 2

        # Weight and hover thrust per motor
        W            = mass_kg * G0
        T_per_motor  = W / N

        # Hover induced velocity from ideal actuator-disk theory:
        #   T = 2 · ρ · A · v_h²  →  v_h = sqrt( T / (2·ρ·A) )
        v_h = math.sqrt(T_per_motor / max(2.0 * rho * A_disk, 1e-12))

        # MCOEF from the momentum drag argument (see class docstring)
        mcoef = G0 / (2.0 * v_h)

        # Disk loading  [N/m²]
        disk_loading = W / max(N * A_disk, 1e-12)

        # Optional RPM cross-check from KV and battery voltage
        rpm_estimate: Optional[float] = None
        if kv and batt_v:
            # Effective voltage at hover throttle
            thr_eff  = thr if thr else 0.50
            # RPM ≈ KV × V_batt × sqrt(throttle) (crude estimate)
            # A more accurate model would use BEMF, but this gives a ballpark.
            rpm_estimate = kv * batt_v * math.sqrt(max(thr_eff, 0.0))

        # Interpretation
        if mcoef < 0.20:
            interp = "Low\n(large / heavy rotors)"
        elif mcoef < 0.45:
            interp = "Moderate"
        elif mcoef < 0.70:
            interp = "Typical for\nsmall UAVs"
        else:
            interp = "High\n(small props / light craft)"

        # Update result labels
        self._lbl_vh.configure(text=f"{v_h:.3f} m/s")
        self._lbl_mcoef.configure(text=f"{mcoef:.4f}")
        self._lbl_interp.configure(text=interp)

        # Detail text
        alt_str  = f"{alt_m or 0:.0f} m"
        temp_str = f"{temp_C:.1f} °C" if temp_C is not None else "ISA"
        rpm_str  = (f"\nEst. hover RPM (KV·V·√thr)  ≈  {rpm_estimate:.0f} rpm"
                    if rpm_estimate else "")

        detail = (
            f"Air density  ρ = {rho:.4f} kg/m³  "
            f"(altitude {alt_str}, temperature {temp_str})\n"
            f"Prop disk area  A = π/4 × ({D_m*100:.1f} cm)² = {A_disk*1e4:.2f} cm²  per rotor\n"
            f"Hover T per motor  = W/N = {T_per_motor:.3f} N  "
            f"({T_per_motor / G0 * 1000:.0f} g)\n"
            f"Disk loading  DL = W / (N·A) = {disk_loading:.1f} N/m²  "
            f"({disk_loading / G0 * 1000 / 1e4:.2f} g/cm²)\n"
            f"Induced velocity  v_h = √({T_per_motor:.3f} / (2×{rho:.4f}×{A_disk*1e4:.2f} cm²)) "
            f"= {v_h:.3f} m/s"
            + rpm_str
        )

        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("1.0", detail)
        self._detail.configure(state="disabled")

        self._last_results = {
            "inputs": {
                "mass_kg":          mass_kg,
                "num_motors":       N,
                "prop_diameter_in": diam_in,
                "altitude_m":       alt_m or 0.0,
                "temperature_C":    temp_C,
                "motor_kv":         kv,
                "battery_voltage_V": batt_v,
                "hover_throttle":   thr,
            },
            "computed": {
                "air_density_kg_m3":             round(rho, 5),
                "disk_area_m2_per_rotor":        round(A_disk, 6),
                "hover_thrust_N_per_motor":      round(T_per_motor, 4),
                "disk_loading_N_m2":             round(disk_loading, 3),
                "hover_induced_velocity_mps":    round(v_h, 4),
                "estimated_hover_rpm":           (round(rpm_estimate) if rpm_estimate else None),
            },
            "ardupilot": {
                "EK3_DRAG_MCOEF": round(mcoef, 4),
            },
        }

    def _export_json(self) -> None:
        if not self._last_results:
            messagebox.showinfo("No results", "Run Calculate first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save MCOEF Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._last_results, fh, indent=2)
        messagebox.showinfo("Saved", f"Results written to:\n{path}")


# ============================================================
# MAIN APPLICATION
# ============================================================
class App(tk.Tk):
    """Root window for the UAV Drag Coefficient Calculator."""

    def __init__(self):
        super().__init__()
        self.title("UAV Drag Coefficient Calculator")
        self.geometry("1140x820")
        self.minsize(920, 680)

        if not HAS_PIL:
            messagebox.showwarning(
                "Pillow not installed",
                "Image loading requires Pillow (PIL).\n\n"
                "The Body Drag photo-measurement features will not be\n"
                "available until you install it:\n\n"
                "    pip install Pillow\n\n"
                "The Propeller Drag tab works without Pillow.")

        self._build_menu()
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.configure(menu=menubar)

        file_m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_m)
        file_m.add_command(
            label="Export Body Drag JSON…",
            command=lambda: self.body_tab._export_json())
        file_m.add_command(
            label="Export MCOEF JSON…",
            command=lambda: self.prop_tab._export_json())
        file_m.add_separator()
        file_m.add_command(label="Quit", command=self.quit)

        help_m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_m)
        help_m.add_command(label="Workflow Guide",  command=self._show_workflow)
        help_m.add_command(label="Parameter Notes", command=self._show_param_notes)
        help_m.add_command(label="About",           command=self._show_about)

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self.body_tab = BodyDragTab(nb)
        self.prop_tab = PropDragTab(nb)

        nb.add(self.body_tab, text="  Body Drag  (BCOEF)  ")
        nb.add(self.prop_tab, text="  Propeller Drag  (MCOEF)  ")

    # ------------------------------------------------------------------
    def _show_workflow(self) -> None:
        messagebox.showinfo("Workflow Guide", """\
BODY DRAG WORKFLOW
==================

1. Photograph your vehicle from three directions:
   • Front view  – looking directly at the nose/front
   • Side view   – looking at the right side (or left)
   • Top view    – looking straight down from above
   Use a plain background and photograph from several
   metres away to minimise perspective distortion.

2. For EACH view (open the correct sub-tab):
   a. Click [Load Image]
   b. Click [Set Scale…], then click TWO POINTS on
      the image that correspond to a known real distance
      (e.g. motor centre to motor centre, or arm span).
      Enter the real distance in cm when prompted.
   c. Click [Draw Outline], then click around the
      vehicle outline — EXCLUDE propeller blades.
      Right-click = undo last point.
      Double-click (or click near vertex 1) = close.

3. Enter mass and body Cd, then click Calculate.

4. Copy the ArduPilot parameter values into your
   autopilot, and the sim input values into the
   multicopter power simulator.

TIPS
----
• Use the motorbase (distance between opposite motors)
  as the scale reference — it's easy to measure accurately.
• Include landing legs, camera mounts, and payload in
  the outline; exclude rotor blades.
• A slightly over-estimated area is safer than under.
""")

    def _show_param_notes(self) -> None:
        messagebox.showinfo("Parameter Notes", """\
BCOEF vs Cd — what is different?
==================================
ArduPilot's EK3_DRAG_BCOEF is a BALLISTIC COEFFICIENT
(units: kg/m²), NOT a dimensionless drag coefficient.

    BCOEF = mass / (Cd × projected_area)

ArduPilot's own guide uses Cd = 1.0 implicitly:
    BCOEF = mass / area

The multicopter power sim uses separate inputs:
    F = ½·ρ·V²·parasite_area·parasite_drag_coefficient
    F = ½·ρ·V²·profile_area·profile_drag_coefficient

With Cd = 1.0 the two models are numerically identical.
Setting Cd > 1.0 makes the sim's drag higher (more
conservative) for the same measured area.

Typical body Cd values
  Streamlined body     0.3 – 0.5
  Box / bluff body     0.8 – 1.3  ← typical multicopter
  Flat plate           ~1.28

MCOEF vs BCOEF
==============
BCOEF produces drag ∝ V²  (quadratic — body drag)
MCOEF produces drag ∝ V   (linear — propeller momentum drag)
Both are needed for accurate ArduPilot wind estimation.
""")

    def _show_about(self) -> None:
        messagebox.showinfo("About", """\
UAV Drag Coefficient Calculator
================================
Version 1.0

Computes:
  • EK3_DRAG_BCOEF_X / BCOEF_Y  (body drag, photo method)
  • EK3_DRAG_MCOEF               (prop momentum drag, physics)
  • Multicopter sim parasite/profile drag inputs

Body drag method:
  ArduPilot EK3_DRAG_BCOEF documentation
  (ardupilot.org/copter/docs/airspeed-estimation.html)

Propeller drag method:
  Actuator-disk momentum theory
  Johnson, "Helicopter Theory" Ch. 2

Dependencies: Python 3.8+, tkinter (stdlib), Pillow (image I/O)
""")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
