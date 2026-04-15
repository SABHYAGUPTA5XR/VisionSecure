"""
VisionSecure — Control Panel (tkinter Dashboard)
Full interactive dashboard for configuring the redaction pipeline.
Runs on the main thread (tkinter requirement).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from typing import Callable, Optional


# All supported PII types
PII_TYPES = [
    "EMAIL", "PHONE", "IP_ADDRESS", "CREDIT_CARD",
    "CVV", "API_KEY", "AADHAAR", "PAN_CARD", "PASSWORD", "CUSTOM"
]

# Colour labels for display (matches output_window colours)
PII_DISPLAY_COLORS = {
    "EMAIL":         "#FF0000",
    "PHONE":         "#FF8C00",
    "IP_ADDRESS":    "#FFD700",
    "CREDIT_CARD":   "#B400B4",
    "CVV":           "#C8C8C8",
    "API_KEY":       "#0064FF",
    "AADHAAR":       "#00FFFF",
    "PAN_CARD":      "#FF00B4",
    "PASSWORD":      "#C8C8C8",
    "CUSTOM":        "#00FF00",
}


class ControlPanel:
    """
    Tkinter-based control panel for VisionSecure.
    Provides controls for redaction modes, PII toggles, sliders, and stats.
    """

    def __init__(
        self,
        root: tk.Tk,
        config: dict,
        on_start: Optional[Callable] = None,
        on_pause: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
        on_add_custom_regex: Optional[Callable] = None,
        on_record_toggle: Optional[Callable] = None,
    ):
        self.root = root
        self.config = config
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_stop = on_stop
        self.on_add_custom_regex = on_add_custom_regex
        self.on_record_toggle = on_record_toggle

        # Internal state
        self._status = "Stopped"
        self._recording = False

        # Tkinter variables
        self._pii_vars = {}          # PII type → BooleanVar
        self._mode_vars = {}         # PII type → StringVar ("blur" / "ghost_mask")
        self._blur_kernel_var = tk.IntVar(value=51)
        self._confidence_var = tk.IntVar(value=40)
        self._fps_target_var = tk.IntVar(value=20)
        self._show_boxes_var = tk.BooleanVar(value=True)
        self._custom_regex_var = tk.StringVar()
        self._custom_label_var = tk.StringVar()

        # Capture region variables (avoid hall-of-mirrors)
        self._use_region_var = tk.BooleanVar(value=True)
        self._region_x_var = tk.StringVar(value="0")
        self._region_y_var = tk.StringVar(value="0")
        self._region_w_var = tk.StringVar(value="960")
        self._region_h_var = tk.StringVar(value="1080")

        # Stats display
        self._fps_display = tk.StringVar(value="0")
        self._frame_redactions = tk.StringVar(value="0")
        self._session_total = tk.StringVar(value="0")
        self._status_var = tk.StringVar(value=">> Stopped")

        self._build_ui()
        self._apply_dark_theme()
        self._sync_config()

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self):
        """Build the entire control panel UI."""
        self.root.title("VisionSecure - Control Panel")
        self.root.geometry("480x900")
        self.root.minsize(460, 700)
        self.root.resizable(True, True)

        # Position control panel at the RIGHT edge of screen so it's
        # outside the default capture region (left half of screen).
        try:
            screen_w = self.root.winfo_screenwidth()
            self.root.geometry(f"480x900+{screen_w - 500}+50")
        except Exception:
            pass

        # Main scrollable canvas
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.main_frame = ttk.Frame(canvas)

        self.main_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Build sections
        self._build_header()
        self._build_controls()
        self._build_capture_region()
        self._build_pii_toggles()
        self._build_redaction_modes()
        self._build_sliders()
        self._build_custom_regex()
        self._build_annotations_toggle()
        self._build_record_toggle()
        self._build_stats()

    # ------------------------------------------------------------------
    # Header section
    # ------------------------------------------------------------------

    def _build_header(self):
        """App title and status indicator."""
        header = ttk.Frame(self.main_frame)
        header.pack(fill="x", padx=15, pady=(15, 5))

        title = ttk.Label(
            header, text="VisionSecure",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(side="left")

        status = ttk.Label(
            header, textvariable=self._status_var,
            font=("Segoe UI", 11),
        )
        status.pack(side="right")

        # Subtitle
        subtitle = ttk.Label(
            self.main_frame,
            text="Real-Time PII Redaction Agent",
            font=("Segoe UI", 10),
        )
        subtitle.pack(padx=15, anchor="w")

        ttk.Separator(self.main_frame, orient="horizontal").pack(
            fill="x", padx=15, pady=8
        )

    # ------------------------------------------------------------------
    # Start / Pause / Stop controls
    # ------------------------------------------------------------------

    def _build_controls(self):
        """Start, Pause, Stop buttons."""
        section = ttk.LabelFrame(self.main_frame, text="  Pipeline Controls  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        btn_frame = ttk.Frame(section)
        btn_frame.pack(fill="x")

        self.start_btn = ttk.Button(
            btn_frame, text="> Start", command=self._on_start, width=12
        )
        self.start_btn.pack(side="left", padx=(0, 5))

        self.pause_btn = ttk.Button(
            btn_frame, text="|| Pause", command=self._on_pause, width=12,
            state="disabled"
        )
        self.pause_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="[] Stop", command=self._on_stop, width=12,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(5, 0))

    # ------------------------------------------------------------------
    # Capture Region (avoids hall-of-mirrors recursive capture)
    # ------------------------------------------------------------------

    def _build_capture_region(self):
        """Capture region selector to avoid recursive screen capture."""
        section = ttk.LabelFrame(self.main_frame, text="  Capture Region  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        ttk.Label(
            section,
            text="Define the screen area to capture. Keep the Output\n"
                 "window OUTSIDE this region to avoid recursion.",
            font=("Segoe UI", 8),
            wraplength=400,
        ).pack(anchor="w", pady=(0, 5))

        ttk.Checkbutton(
            section, text="Use Custom Region (recommended)",
            variable=self._use_region_var,
            command=self._sync_config,
        ).pack(anchor="w", pady=(0, 5))

        coord_frame = ttk.Frame(section)
        coord_frame.pack(fill="x")

        for col, (label, var) in enumerate([
            ("X:", self._region_x_var),
            ("Y:", self._region_y_var),
            ("W:", self._region_w_var),
            ("H:", self._region_h_var),
        ]):
            ttk.Label(coord_frame, text=label, width=3).grid(row=0, column=col*2, padx=2)
            entry = ttk.Entry(coord_frame, textvariable=var, width=7)
            entry.grid(row=0, column=col*2+1, padx=2)
            entry.bind("<FocusOut>", lambda e: self._sync_config())

        # Quick presets
        preset_frame = ttk.Frame(section)
        preset_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(
            preset_frame, text="Left Half",
            command=lambda: self._set_region_preset("left_half"), width=10
        ).pack(side="left", padx=2)
        ttk.Button(
            preset_frame, text="Top Half",
            command=lambda: self._set_region_preset("top_half"), width=10
        ).pack(side="left", padx=2)
        ttk.Button(
            preset_frame, text="Full Screen",
            command=lambda: self._set_region_preset("full"), width=10
        ).pack(side="left", padx=2)

    # ------------------------------------------------------------------
    # PII Type toggles
    # ------------------------------------------------------------------

    def _build_pii_toggles(self):
        """Individual ON/OFF checkboxes for each PII category."""
        section = ttk.LabelFrame(self.main_frame, text="  PII Detection Toggles  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        for pii_type in PII_TYPES:
            var = tk.BooleanVar(value=True)
            self._pii_vars[pii_type] = var

            row = ttk.Frame(section)
            row.pack(fill="x", pady=1)

            cb = ttk.Checkbutton(
                row, text=pii_type.replace("_", " ").title(),
                variable=var,
                command=self._sync_config,
            )
            cb.pack(side="left")

            # Colour indicator
            color = PII_DISPLAY_COLORS.get(pii_type, "#FFFFFF")
            indicator = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            indicator.create_oval(2, 2, 12, 12, fill=color, outline=color)
            indicator.pack(side="right", padx=5)

    # ------------------------------------------------------------------
    # Redaction mode selectors (per PII type)
    # ------------------------------------------------------------------

    def _build_redaction_modes(self):
        """Per-PII-type radio buttons: Blur vs Ghost Mask."""
        section = ttk.LabelFrame(self.main_frame, text="  Redaction Mode (per PII type)  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        header = ttk.Frame(section)
        header.pack(fill="x", pady=(0, 5))
        ttk.Label(header, text="PII Type", font=("Segoe UI", 9, "bold"), width=18).pack(side="left")
        ttk.Label(header, text="Blur", font=("Segoe UI", 9, "bold"), width=8).pack(side="left")
        ttk.Label(header, text="Ghost Mask", font=("Segoe UI", 9, "bold")).pack(side="left")

        for pii_type in PII_TYPES:
            var = tk.StringVar(value="blur")
            self._mode_vars[pii_type] = var

            row = ttk.Frame(section)
            row.pack(fill="x", pady=1)

            ttk.Label(row, text=pii_type.replace("_", " ").title(), width=18).pack(side="left")

            ttk.Radiobutton(
                row, text="", variable=var, value="blur",
                command=self._sync_config, width=6
            ).pack(side="left")

            ttk.Radiobutton(
                row, text="", variable=var, value="ghost_mask",
                command=self._sync_config
            ).pack(side="left")

    # ------------------------------------------------------------------
    # Parameter sliders
    # ------------------------------------------------------------------

    def _build_sliders(self):
        """Blur kernel, confidence threshold, FPS target sliders."""
        section = ttk.LabelFrame(self.main_frame, text="  Parameters  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        # Blur kernel size (11–101, odd steps)
        ttk.Label(section, text="Blur Kernel Size:").pack(anchor="w")
        blur_frame = ttk.Frame(section)
        blur_frame.pack(fill="x", pady=(0, 8))
        blur_label = ttk.Label(blur_frame, textvariable=self._blur_kernel_var, width=4)
        blur_label.pack(side="right")
        blur_slider = ttk.Scale(
            blur_frame, from_=11, to=101,
            variable=self._blur_kernel_var,
            command=lambda v: self._on_blur_change(v),
        )
        blur_slider.pack(fill="x", side="left", expand=True)

        # Confidence threshold (50–95%)
        ttk.Label(section, text="OCR Confidence Threshold (%):").pack(anchor="w")
        conf_frame = ttk.Frame(section)
        conf_frame.pack(fill="x", pady=(0, 8))
        conf_label = ttk.Label(conf_frame, textvariable=self._confidence_var, width=4)
        conf_label.pack(side="right")
        ttk.Scale(
            conf_frame, from_=50, to=95,
            variable=self._confidence_var,
            command=lambda v: self._sync_config(),
        ).pack(fill="x", side="left", expand=True)

        # FPS target
        ttk.Label(section, text="Target FPS:").pack(anchor="w")
        fps_frame = ttk.Frame(section)
        fps_frame.pack(fill="x")
        fps_label = ttk.Label(fps_frame, textvariable=self._fps_target_var, width=4)
        fps_label.pack(side="right")
        ttk.Scale(
            fps_frame, from_=5, to=60,
            variable=self._fps_target_var,
            command=lambda v: self._sync_config(),
        ).pack(fill="x", side="left", expand=True)

    # ------------------------------------------------------------------
    # Custom regex input
    # ------------------------------------------------------------------

    def _build_custom_regex(self):
        """Text field + label + Add button for user-defined patterns."""
        section = ttk.LabelFrame(self.main_frame, text="  Custom Regex Pattern  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        ttk.Label(section, text="Label:").pack(anchor="w")
        ttk.Entry(section, textvariable=self._custom_label_var).pack(fill="x", pady=(0, 5))

        ttk.Label(section, text="Regex Pattern:").pack(anchor="w")
        ttk.Entry(section, textvariable=self._custom_regex_var).pack(fill="x", pady=(0, 5))

        ttk.Button(
            section, text="+ Add Pattern",
            command=self._on_add_custom_regex
        ).pack(anchor="w")

    # ------------------------------------------------------------------
    # Bounding box annotations toggle
    # ------------------------------------------------------------------

    def _build_annotations_toggle(self):
        """Show/hide coloured detection boxes on output."""
        section = ttk.LabelFrame(self.main_frame, text="  Display Options  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        ttk.Checkbutton(
            section, text="Show Bounding Box Annotations",
            variable=self._show_boxes_var,
            command=self._sync_config,
        ).pack(anchor="w")

    # ------------------------------------------------------------------
    # Record output toggle
    # ------------------------------------------------------------------

    def _build_record_toggle(self):
        """Record output to disk toggle."""
        section = ttk.LabelFrame(self.main_frame, text="  Recording  ", padding=10)
        section.pack(fill="x", padx=15, pady=5)

        self.record_btn = ttk.Button(
            section, text="(o) Start Recording",
            command=self._on_record_toggle
        )
        self.record_btn.pack(anchor="w")

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    def _build_stats(self):
        """Session statistics: total redactions, FPS, breakdown."""
        section = ttk.LabelFrame(self.main_frame, text="  Session Statistics  ", padding=10)
        section.pack(fill="x", padx=15, pady=(5, 15))

        stats_grid = ttk.Frame(section)
        stats_grid.pack(fill="x")

        # FPS
        ttk.Label(stats_grid, text="Current FPS:", font=("Segoe UI", 9)).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Label(
            stats_grid, textvariable=self._fps_display,
            font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=1, sticky="e", pady=2)

        # Frame redactions
        ttk.Label(stats_grid, text="Frame Redactions:", font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Label(
            stats_grid, textvariable=self._frame_redactions,
            font=("Segoe UI", 9, "bold")
        ).grid(row=1, column=1, sticky="e", pady=2)

        # Session total
        ttk.Label(stats_grid, text="Session Total:", font=("Segoe UI", 9)).grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Label(
            stats_grid, textvariable=self._session_total,
            font=("Segoe UI", 9, "bold")
        ).grid(row=2, column=1, sticky="e", pady=2)

        stats_grid.columnconfigure(1, weight=1)

    # ==================================================================
    # Dark Theme
    # ==================================================================

    def _apply_dark_theme(self):
        """Apply a modern dark theme to the control panel."""
        self.root.configure(bg="#1e1e2e")

        style = ttk.Style()
        style.theme_use("clam")

        # Colours
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        surface = "#313244"
        border = "#45475a"

        style.configure(".", background=bg, foreground=fg, fieldbackground=surface,
                        bordercolor=border, troughcolor=surface)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TLabelframe", background=bg, foreground=accent,
                        bordercolor=border)
        style.configure("TLabelframe.Label", background=bg, foreground=accent,
                        font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background=surface, foreground=fg,
                        bordercolor=border, padding=(8, 4))
        style.map("TButton",
                  background=[("active", accent), ("pressed", "#74c7ec")],
                  foreground=[("active", "#1e1e2e")])
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.map("TCheckbutton", background=[("active", bg)])
        style.configure("TRadiobutton", background=bg, foreground=fg)
        style.map("TRadiobutton", background=[("active", bg)])
        style.configure("TScale", background=bg, troughcolor=surface)
        style.configure("TEntry", fieldbackground=surface, foreground=fg,
                        bordercolor=border)
        style.configure("TSeparator", background=border)
        style.configure("Vertical.TScrollbar", background=surface,
                        troughcolor=bg, bordercolor=bg, arrowcolor=fg)

    # ==================================================================
    # Event handlers
    # ==================================================================

    def _on_start(self):
        self._sync_config()  # Ensure capture region is set before starting
        self._status = "Running"
        self._status_var.set(">> Running")
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        if self.on_start:
            self.on_start()

    def _on_pause(self):
        if self._status == "Running":
            self._status = "Paused"
            self._status_var.set(">> Paused")
            self.pause_btn.configure(text="> Resume")
        else:
            self._status = "Running"
            self._status_var.set(">> Running")
            self.pause_btn.configure(text="|| Pause")
        if self.on_pause:
            self.on_pause()

    def _on_stop(self):
        self._status = "Stopped"
        self._status_var.set(">> Stopped")
        self.start_btn.configure(state="normal")
        self.pause_btn.configure(state="disabled", text="|| Pause")
        self.stop_btn.configure(state="disabled")
        if self.on_stop:
            self.on_stop()

    def _set_region_preset(self, preset: str):
        """Set capture region to a preset value."""
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
        except Exception:
            sw, sh = 1920, 1080

        if preset == "left_half":
            self._region_x_var.set("0")
            self._region_y_var.set("0")
            self._region_w_var.set(str(sw // 2))
            self._region_h_var.set(str(sh))
        elif preset == "top_half":
            self._region_x_var.set("0")
            self._region_y_var.set("0")
            self._region_w_var.set(str(sw))
            self._region_h_var.set(str(sh // 2))
        elif preset == "full":
            self._use_region_var.set(False)

        self._sync_config()

    def _on_blur_change(self, value):
        """Ensure blur kernel is always odd."""
        v = int(float(value))
        if v % 2 == 0:
            v += 1
        self._blur_kernel_var.set(v)
        self._sync_config()

    def _on_add_custom_regex(self):
        label = self._custom_label_var.get().strip()
        pattern = self._custom_regex_var.get().strip()
        if not label or not pattern:
            messagebox.showwarning(
                "VisionSecure",
                "Please provide both a label and a regex pattern."
            )
            return
        if self.on_add_custom_regex:
            self.on_add_custom_regex(label, pattern)
        self._custom_label_var.set("")
        self._custom_regex_var.set("")

    def _on_record_toggle(self):
        self._recording = not self._recording
        if self._recording:
            self.record_btn.configure(text="[] Stop Recording")
        else:
            self.record_btn.configure(text="(o) Start Recording")
        if self.on_record_toggle:
            self.on_record_toggle(self._recording)

    # ==================================================================
    # Sync configuration to shared dict
    # ==================================================================

    def _sync_config(self):
        """Push all UI values into the shared config dict."""
        # PII toggles
        toggles = {}
        for pii_type, var in self._pii_vars.items():
            toggles[pii_type] = var.get()
        self.config["pii_toggles"] = toggles

        # Redaction modes
        modes = {}
        for pii_type, var in self._mode_vars.items():
            modes[pii_type] = var.get()
        self.config["redaction_modes"] = modes

        # Parameters
        kernel = self._blur_kernel_var.get()
        if kernel % 2 == 0:
            kernel += 1
        self.config["blur_kernel_size"] = kernel
        self.config["blur_sigma"] = max(kernel // 3, 1)
        self.config["confidence_threshold"] = self._confidence_var.get()
        self.config["target_fps"] = self._fps_target_var.get()
        self.config["show_bounding_boxes"] = self._show_boxes_var.get()

        # Capture region (to avoid hall-of-mirrors)
        if self._use_region_var.get():
            try:
                self.config["capture_region"] = {
                    "left": int(self._region_x_var.get()),
                    "top": int(self._region_y_var.get()),
                    "width": max(int(self._region_w_var.get()), 100),
                    "height": max(int(self._region_h_var.get()), 100),
                }
            except (ValueError, TypeError):
                self.config["capture_region"] = None
        else:
            self.config["capture_region"] = None

    # ==================================================================
    # External stats update (called from main loop)
    # ==================================================================

    def update_stats(self, fps: float, frame_redactions: int, session_total: int):
        """Update the stats display (call from main thread via root.after)."""
        try:
            self._fps_display.set(f"{fps:.1f}")
            self._frame_redactions.set(str(frame_redactions))
            self._session_total.set(str(session_total))
        except tk.TclError:
            pass  # Widget destroyed
