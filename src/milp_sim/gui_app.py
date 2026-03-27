from __future__ import annotations

import math
import os
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import Point

from .config import DEFAULT_CONFIG, SimulationConfig
from .session import OfflineSession, OnlineSession


class _BaseGuiApp:
    def __init__(
        self,
        session,
        cfg: SimulationConfig = DEFAULT_CONFIG,
        *,
        window_title: str,
        enable_online_runtime: bool,
        enable_offline_tools: bool,
    ) -> None:
        self.cfg = cfg
        self.session = session
        self.enable_online_runtime = enable_online_runtime
        self.enable_offline_tools = enable_offline_tools
        self.obstacle_draw_mode = False
        self.obstacle_remove_click_mode = False
        self.obstacle_points: list[tuple[float, float]] = []
        self.task_click_mode = False
        self.dragging_task_id: int | None = None
        self.drag_origin_pos: tuple[float, float] | None = None
        self.drag_preview_pos: tuple[float, float] | None = None
        self._map_view_state: dict[str, tuple[float, float]] | None = None

        self.root = tk.Tk()
        self.root.title(window_title)
        self._configure_ui_style()
        self._configure_window_geometry()

        self.refresh_interval_ms = 500
        self.render_refresh_interval_ms = 40
        self.log_count_var = tk.StringVar(value="8")
        self.add_demand_var = tk.StringVar(value="2")
        self.add_task_id_var = tk.StringVar(value="")
        self.add_x_var = tk.StringVar(value="")
        self.add_y_var = tk.StringVar(value="")
        self.random_demand_var = tk.StringVar(value="")
        self.cancel_task_id_var = tk.StringVar(value="")
        self.obstacle_point_count_var = tk.StringVar(value="0")
        self.obstacle_mode_var = tk.StringVar(value="OFF")
        self.obstacle_remove_mode_var = tk.StringVar(value="OFF")
        self.task_mode_var = tk.StringVar(value="OFF")
        self.last_action_var = tk.StringVar(value="Ready")
        self.online_state_var = tk.StringVar(value="OFF")
        self.sim_time_var = tk.StringVar(value="0.00")
        self.next_event_var = tk.StringVar(value="-")
        self.replan_reason_var = tk.StringVar(value="-")
        self.sim_speed_var = tk.StringVar(value="1x")
        self.obstacle_remove_var = tk.StringVar(value="")
        self._last_status_text = ""
        self._last_logs_text = ""
        self._last_tasks_text = ""
        self._last_drag_refresh_ts = 0.0
        self.drag_refresh_interval_s = 1.0 / 30.0
        self._last_online_tick_wall_ts: float | None = None
        self._last_text_refresh_wall_ts = 0.0
        self._prev_online_render_state: dict[int, dict[str, object]] = {}
        self._curr_online_render_state: dict[int, dict[str, object]] = {}
        self.left_canvas: tk.Canvas | None = None
        self.left_canvas_window: int | None = None
        self.left_controls_frame: ttk.Frame | None = None
        self.obstacle_remove_combo = None
        self.status_text: ScrolledText | None = None
        self.tasks_text: ScrolledText | None = None
        self.logs_text: ScrolledText | None = None
        self.comparison_figure = None
        self.comparison_canvas = None
        self.compare_ax_with = None
        self.compare_ax_without = None
        self.compare_ax_without_verify_cost = None
        self.secondary_ax = None
        self._scenario_json_path: Path | None = None
        if self.cfg.scenario_file:
            source_path = Path(self.cfg.scenario_file)
            if source_path.suffix.lower() == ".json":
                self._scenario_json_path = source_path

        self._build_layout()
        self._refresh_all()
        self._schedule_refresh()

    def _configure_ui_style(self) -> None:
        self.root.configure(bg="#e9edf2")
        # Improve readability on HiDPI/WSLg displays.
        screen_height = self.root.winfo_screenheight()
        ui_scale = 1.25
        base_font_size = 10
        mono_font_size = 10
        if screen_height <= 900:
            ui_scale = 1.1
        if screen_height <= 800:
            ui_scale = 1.0
            base_font_size = 9
            mono_font_size = 9
        self.root.tk.call("tk", "scaling", ui_scale)

        style = ttk.Style(self.root)
        available_themes = set(style.theme_names())
        preferred_theme = "clam" if "clam" in available_themes else style.theme_use()
        style.theme_use(preferred_theme)

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=base_font_size)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family="Segoe UI", size=base_font_size)
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(family="Cascadia Mono", size=mono_font_size)

        self.root.option_add("*Font", default_font)

        style.configure("TFrame", background="#e9edf2")
        style.configure("TLabelframe", background="#e9edf2")
        style.configure("TLabelframe.Label", font=("Segoe UI Semibold", base_font_size), foreground="#1f2937")
        style.configure("TLabel", background="#e9edf2", foreground="#111827")
        style.configure("TButton", padding=(10, 4), font=("Segoe UI", base_font_size))
        style.configure("TEntry", fieldbackground="#ffffff")

    def _configure_window_geometry(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width_limit = max(screen_width - 32, 640)
        height_limit = max(screen_height - 72, 520)
        width = min(1600, width_limit)
        height = min(920, height_limit)
        pos_x = max((screen_width - width) // 2, 0)
        pos_y = max((screen_height - height) // 2, 0)
        self.root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self.root.minsize(min(width, 1120), min(height, 700))

    def _build_layout(self) -> None:
        if self.enable_online_runtime:
            self.root.columnconfigure(0, weight=0)
            self.root.columnconfigure(1, weight=3)
            self.root.columnconfigure(2, weight=2)
            self.root.rowconfigure(0, weight=1)

            left = ttk.Frame(self.root, padding=(10, 10, 4, 10))
            center = ttk.Frame(self.root, padding=8)
            right = ttk.Frame(self.root, padding=10)

            left.grid(row=0, column=0, sticky="ns")
            center.grid(row=0, column=1, sticky="nsew")
            right.grid(row=0, column=2, sticky="nsew")

            left.rowconfigure(0, weight=1)
            left.columnconfigure(0, weight=1)
            center.rowconfigure(0, weight=1)
            center.columnconfigure(0, weight=1)

            right.rowconfigure(0, weight=1)
            right.rowconfigure(1, weight=1)
            right.rowconfigure(2, weight=1)
            right.columnconfigure(0, weight=1)

            left_canvas = tk.Canvas(
                left,
                background="#e9edf2",
                borderwidth=0,
                highlightthickness=0,
                width=320,
            )
            left_scrollbar = ttk.Scrollbar(left, orient="vertical", command=left_canvas.yview)
            left_controls = ttk.Frame(left_canvas)

            left_canvas.grid(row=0, column=0, sticky="nsew")
            left_scrollbar.grid(row=0, column=1, sticky="ns")
            left_canvas.configure(yscrollcommand=left_scrollbar.set)

            self.left_canvas = left_canvas
            self.left_controls_frame = left_controls
            self.left_canvas_window = left_canvas.create_window((0, 0), window=left_controls, anchor="nw")

            left_controls.bind("<Configure>", self._on_left_controls_configure)
            left_canvas.bind("<Configure>", self._on_left_canvas_configure)
            self.root.bind_all("<MouseWheel>", self._on_left_panel_mousewheel, add="+")
            self.root.bind_all("<Button-4>", self._on_left_panel_mousewheel, add="+")
            self.root.bind_all("<Button-5>", self._on_left_panel_mousewheel, add="+")

            self._build_left_controls(left_controls)
            self._build_center_plot(center)
            self._build_right_panels(right)
            return

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=(10, 10, 4, 10))
        center = ttk.Frame(self.root, padding=8)

        left.grid(row=0, column=0, sticky="ns")
        center.grid(row=0, column=1, sticky="nsew")

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        left_canvas = tk.Canvas(
            left,
            background="#e9edf2",
            borderwidth=0,
            highlightthickness=0,
            width=320,
        )
        left_scrollbar = ttk.Scrollbar(left, orient="vertical", command=left_canvas.yview)
        left_controls = ttk.Frame(left_canvas)

        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar.grid(row=0, column=1, sticky="ns")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        self.left_canvas = left_canvas
        self.left_controls_frame = left_controls
        self.left_canvas_window = left_canvas.create_window((0, 0), window=left_controls, anchor="nw")

        left_controls.bind("<Configure>", self._on_left_controls_configure)
        left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_left_panel_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_left_panel_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_left_panel_mousewheel, add="+")

        self._build_left_controls(left_controls)
        self._build_center_plot(center)

    def _on_left_controls_configure(self, _event) -> None:
        if self.left_canvas is None:
            return
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event) -> None:
        if self.left_canvas is None or self.left_canvas_window is None:
            return
        self.left_canvas.itemconfigure(self.left_canvas_window, width=event.width)

    def _is_left_panel_widget(self, widget: tk.Misc | None) -> bool:
        if widget is None:
            return False
        current = widget
        while current is not None:
            if current == self.left_canvas or current == self.left_controls_frame:
                return True
            parent_name = current.winfo_parent()
            if not parent_name:
                break
            current = current.nametowidget(parent_name)
        return False

    def _on_left_panel_mousewheel(self, event) -> None:
        if self.left_canvas is None or not self._is_left_panel_widget(event.widget):
            return
        if getattr(event, "delta", 0):
            step = -int(event.delta / 120)
        elif getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = 0
        if step:
            self.left_canvas.yview_scroll(step, "units")

    def _build_left_controls(self, parent) -> None:
        top = ttk.LabelFrame(parent, text="Top Controls", padding=8)
        top.pack(fill="x", pady=(0, 8))

        ttk.Button(top, text="Initialize/Reset", command=self._on_reset).pack(fill="x", pady=3)
        if self.enable_offline_tools:
            ttk.Button(top, text="Reset + Replay Last Ops", command=self._on_reset_replay).pack(fill="x", pady=3)
            ttk.Button(top, text="Undo Last Action", command=self._on_undo).pack(fill="x", pady=3)
            ttk.Button(top, text="Re-auction Now", command=self._on_reallocate_offline).pack(fill="x", pady=3)
        ttk.Button(top, text="Save Snapshot", command=self._on_save_snapshot).pack(fill="x", pady=3)
        if self.enable_offline_tools:
            ttk.Button(top, text="Save Scenario JSON", command=self._on_save_scenario_json).pack(fill="x", pady=3)
            ttk.Button(top, text="Save Scenario As...", command=self._on_save_scenario_json_as).pack(fill="x", pady=3)
        ttk.Button(top, text="Export Logs", command=self._on_export_logs).pack(fill="x", pady=3)
        if self.enable_offline_tools:
            ttk.Button(top, text="Export Path Quality", command=self._on_export_path_quality).pack(fill="x", pady=3)
        ttk.Button(top, text="Export Task Ops JSON", command=self._on_export_task_ops_json).pack(fill="x", pady=3)
        ttk.Button(top, text="Load Task Ops JSON", command=self._on_replay_task_ops_json).pack(fill="x", pady=3)

        if self.enable_online_runtime:
            online_frame = ttk.LabelFrame(parent, text="Online Runtime", padding=8)
            online_frame.pack(fill="x", pady=8)
            ttk.Label(online_frame, text="state").grid(row=0, column=0, sticky="w")
            ttk.Label(online_frame, textvariable=self.online_state_var).grid(row=0, column=1, sticky="w")
            ttk.Label(online_frame, text="sim time").grid(row=1, column=0, sticky="w")
            ttk.Label(online_frame, textvariable=self.sim_time_var).grid(row=1, column=1, sticky="w")
            ttk.Label(online_frame, text="next event").grid(row=2, column=0, sticky="w")
            ttk.Label(online_frame, textvariable=self.next_event_var).grid(row=2, column=1, sticky="w")
            ttk.Label(online_frame, text="replan").grid(row=3, column=0, sticky="w")
            ttk.Label(online_frame, textvariable=self.replan_reason_var, wraplength=190).grid(
                row=3, column=1, sticky="w"
            )
            ttk.Button(online_frame, text="Start Online", command=self._on_start_online).grid(
                row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0)
            )
            ttk.Button(online_frame, text="Pause/Resume", command=self._on_toggle_online_pause).grid(
                row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0)
            )
            ttk.Button(online_frame, text="Step x1", command=lambda: self._on_online_step(1)).grid(
                row=6, column=0, sticky="ew", pady=(5, 0)
            )
            ttk.Button(online_frame, text="Next Frame", command=self._on_frame_next).grid(
                row=6, column=1, sticky="ew", pady=(5, 0)
            )
            ttk.Button(online_frame, text="Prev Frame", command=self._on_frame_prev).grid(
                row=7, column=0, sticky="ew", pady=(5, 0)
            )
            ttk.Label(online_frame, text="speed").grid(row=8, column=0, sticky="w", pady=(6, 0))
            speed_box = ttk.Combobox(
                online_frame,
                textvariable=self.sim_speed_var,
                values=("1x",),
                state="readonly",
                width=8,
            )
            speed_box.grid(row=8, column=1, sticky="ew", pady=(6, 0))
            online_frame.columnconfigure(0, weight=1)
            online_frame.columnconfigure(1, weight=1)

        obstacle_frame = ttk.LabelFrame(parent, text="Draw Obstacle Polygon", padding=8)
        obstacle_frame.pack(fill="x", pady=8)
        ttk.Label(obstacle_frame, text="draw mode").grid(row=0, column=0, sticky="w")
        ttk.Label(obstacle_frame, textvariable=self.obstacle_mode_var).grid(row=0, column=1, sticky="w")
        ttk.Label(obstacle_frame, text="points").grid(row=1, column=0, sticky="w")
        ttk.Label(obstacle_frame, textvariable=self.obstacle_point_count_var).grid(row=1, column=1, sticky="w")

        ttk.Button(
            obstacle_frame,
            text="Start/Stop Draw",
            command=self._on_toggle_obstacle_draw,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(obstacle_frame, text="Undo Point", command=self._on_obstacle_undo).grid(
            row=3, column=0, sticky="ew", pady=(5, 0)
        )
        ttk.Button(obstacle_frame, text="Clear Points", command=self._on_obstacle_clear).grid(
            row=3, column=1, sticky="ew", pady=(5, 0)
        )
        ttk.Label(obstacle_frame, text="click remove").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Label(obstacle_frame, textvariable=self.obstacle_remove_mode_var).grid(row=4, column=1, sticky="w", pady=(6, 0))
        if self.enable_offline_tools:
            ttk.Button(obstacle_frame, text="Start/Stop Click Remove", command=self._on_toggle_obstacle_remove_click).grid(
                row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0)
            )
        ttk.Button(
            obstacle_frame,
            text="Apply Obstacle",
            command=self._on_apply_obstacle,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Label(obstacle_frame, text="remove idx").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.obstacle_remove_combo = ttk.Combobox(
            obstacle_frame,
            textvariable=self.obstacle_remove_var,
            values=(),
            state="readonly",
            width=12,
        )
        self.obstacle_remove_combo.grid(row=7, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(obstacle_frame, text="Remove Obstacle", command=self._on_remove_obstacle).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        if self.enable_offline_tools:
            ttk.Button(obstacle_frame, text="Replace Obstacle", command=self._on_replace_obstacle).grid(
                row=9, column=0, columnspan=2, sticky="ew", pady=(5, 0)
            )
        obstacle_frame.columnconfigure(0, weight=1)
        obstacle_frame.columnconfigure(1, weight=1)

        add_frame = ttk.LabelFrame(parent, text="Add Task", padding=8)
        add_frame.pack(fill="x", pady=8)

        ttk.Label(add_frame, text="click mode").grid(row=0, column=0, sticky="w")
        ttk.Label(add_frame, textvariable=self.task_mode_var).grid(row=0, column=1, sticky="w")

        ttk.Label(add_frame, text="demand").grid(row=1, column=0, sticky="w")
        ttk.Entry(add_frame, textvariable=self.add_demand_var, width=12).grid(row=1, column=1, sticky="ew")

        ttk.Label(add_frame, text="task_id(opt)").grid(row=2, column=0, sticky="w")
        ttk.Entry(add_frame, textvariable=self.add_task_id_var, width=12).grid(row=2, column=1, sticky="ew")

        ttk.Label(add_frame, text="x").grid(row=3, column=0, sticky="w")
        ttk.Entry(add_frame, textvariable=self.add_x_var, width=12).grid(row=3, column=1, sticky="ew")

        ttk.Label(add_frame, text="y").grid(row=4, column=0, sticky="w")
        ttk.Entry(add_frame, textvariable=self.add_y_var, width=12).grid(row=4, column=1, sticky="ew")

        ttk.Button(add_frame, text="Start/Stop Click Add", command=self._on_toggle_task_click).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        ttk.Button(
            add_frame,
            text="Add At Coord + Replan" if self.enable_online_runtime else "Add At Coord",
            command=self._on_add_at_coord,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        add_frame.columnconfigure(1, weight=1)

        random_frame = ttk.LabelFrame(parent, text="Add Random Task", padding=8)
        random_frame.pack(fill="x", pady=8)

        ttk.Label(random_frame, text="demand(opt)").grid(row=0, column=0, sticky="w")
        ttk.Entry(random_frame, textvariable=self.random_demand_var, width=12).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(
            random_frame,
            text="Add Random + Replan" if self.enable_online_runtime else "Add Random Task",
            command=self._on_add_random,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        random_frame.columnconfigure(1, weight=1)

        cancel_frame = ttk.LabelFrame(parent, text="Cancel Task", padding=8)
        cancel_frame.pack(fill="x", pady=8)

        ttk.Label(cancel_frame, text="task_id").grid(row=0, column=0, sticky="w")
        ttk.Entry(cancel_frame, textvariable=self.cancel_task_id_var, width=12).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(
            cancel_frame,
            text="Cancel + Replan" if self.enable_online_runtime else "Cancel Task",
            command=self._on_cancel_task,
        ).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        cancel_frame.columnconfigure(1, weight=1)

        misc = ttk.LabelFrame(parent, text="View", padding=8)
        misc.pack(fill="x", pady=8)

        ttk.Label(misc, text="log lines").grid(row=0, column=0, sticky="w")
        ttk.Entry(misc, textvariable=self.log_count_var, width=10).grid(row=0, column=1, sticky="ew")
        ttk.Label(misc, text="last action").grid(row=1, column=0, sticky="w")
        ttk.Label(misc, textvariable=self.last_action_var, wraplength=220).grid(row=1, column=1, sticky="w")
        ttk.Button(misc, text="Refresh Now", command=self._refresh_all).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        misc.columnconfigure(1, weight=1)

    def _build_center_plot(self, parent) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        if self.enable_online_runtime:
            self.figure = plt.Figure(figsize=(8, 8), dpi=130)
            self.ax = self.figure.add_subplot(111)
            self.secondary_ax = None
            self.compare_ax_without_verify_cost = None
            self.figure.subplots_adjust(left=0.05, right=0.985, bottom=0.05, top=0.96)
        else:
            self.figure = plt.Figure(figsize=(12.6, 9.2), dpi=130)
            grid = self.figure.add_gridspec(2, 2, height_ratios=(1.0, 1.12), wspace=0.08, hspace=0.16)
            self.ax = self.figure.add_subplot(grid[0, 0])
            self.secondary_ax = self.figure.add_subplot(grid[0, 1])
            self.compare_ax_without_verify_cost = self.figure.add_subplot(grid[1, :])
            self.figure.subplots_adjust(left=0.04, right=0.985, bottom=0.045, top=0.955)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

    def _capture_map_view_state(self) -> dict[str, tuple[float, float]] | None:
        if not self.ax.has_data():
            return None
        return {
            "xlim": tuple(float(v) for v in self.ax.get_xlim()),
            "ylim": tuple(float(v) for v in self.ax.get_ylim()),
        }

    def _restore_map_view_state(self, view_state: dict[str, tuple[float, float]] | None) -> None:
        if not view_state:
            return
        xlim = view_state.get("xlim")
        ylim = view_state.get("ylim")
        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)

    def _plot_axes(self) -> list:
        axes = [self.ax]
        if self.secondary_ax is not None:
            axes.append(self.secondary_ax)
        if self.compare_ax_without_verify_cost is not None:
            axes.append(self.compare_ax_without_verify_cost)
        return axes

    def _event_axis(self, event):
        for axis in self._plot_axes():
            if event.inaxes == axis:
                return axis
        return None

    def _build_right_panels(self, parent) -> None:
        status_title = "Online Status" if self.enable_online_runtime else "Status"
        status_frame = ttk.LabelFrame(parent, text=status_title, padding=6)
        status_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        status_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)

        tasks_frame = ttk.LabelFrame(parent, text="Tasks", padding=6)
        tasks_frame.grid(row=1 if not self.enable_online_runtime else 2, column=0, sticky="nsew")
        tasks_frame.rowconfigure(0, weight=1)
        tasks_frame.columnconfigure(0, weight=1)

        self.status_text = ScrolledText(
            status_frame,
            wrap="word",
            font=("Cascadia Mono", 10),
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief="flat",
            borderwidth=0,
        )
        self.status_text.grid(row=0, column=0, sticky="nsew")

        if self.enable_online_runtime:
            middle_frame = ttk.LabelFrame(parent, text="Recent Logs", padding=6)
            middle_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
            middle_frame.rowconfigure(0, weight=1)
            middle_frame.columnconfigure(0, weight=1)
            self.logs_text = ScrolledText(
                middle_frame,
                wrap="word",
                font=("Cascadia Mono", 10),
                bg="#f8fafc",
                fg="#0f172a",
                insertbackground="#0f172a",
                relief="flat",
                borderwidth=0,
            )
            self.logs_text.grid(row=0, column=0, sticky="nsew")

        self.tasks_text = ScrolledText(
            tasks_frame,
            wrap="word",
            font=("Cascadia Mono", 10),
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief="flat",
            borderwidth=0,
        )
        self.tasks_text.grid(row=0, column=0, sticky="nsew")

    @staticmethod
    def _set_text(widget: ScrolledText, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, value)
        widget.configure(state="disabled")

    @staticmethod
    def _parse_optional_int(value: str) -> int | None:
        v = value.strip()
        if not v:
            return None
        return int(v)

    @staticmethod
    def _parse_required_int(value: str, name: str) -> int:
        v = value.strip()
        if not v:
            raise ValueError(f"{name} is required")
        return int(v)

    @staticmethod
    def _parse_required_float(value: str, name: str) -> float:
        v = value.strip()
        if not v:
            raise ValueError(f"{name} is required")
        return float(v)

    def _refresh_map(self) -> None:
        if not self.enable_online_runtime:
            assert self.secondary_ax is not None
            self.session.draw_comparison_on_axes(
                self.ax,
                self.secondary_ax,
                self.compare_ax_without_verify_cost,
            )
            axes = self._plot_axes()
            if self.dragging_task_id is not None and self.drag_preview_pos is not None:
                x, y = self.drag_preview_pos
                for axis in axes:
                    axis.scatter([x], [y], s=80, marker="x", color="#be123c", linewidth=2.0, zorder=15)
                    axis.text(
                        x,
                        y,
                        f"  T{self.dragging_task_id}",
                        fontsize=9,
                        color="#9f1239",
                        ha="left",
                        va="bottom",
                        zorder=16,
                    )
            if self.obstacle_points:
                xs = [p[0] for p in self.obstacle_points]
                ys = [p[1] for p in self.obstacle_points]
                for axis in axes:
                    axis.scatter(xs, ys, s=35, color="#0ea5e9", edgecolor="white", linewidth=0.5, zorder=8)
                    axis.plot(xs, ys, color="#0ea5e9", linewidth=1.5, linestyle="--", zorder=7)
                    if len(self.obstacle_points) >= 3:
                        axis.plot(
                            [self.obstacle_points[-1][0], self.obstacle_points[0][0]],
                            [self.obstacle_points[-1][1], self.obstacle_points[0][1]],
                            color="#0ea5e9",
                            linewidth=1.5,
                            linestyle="--",
                            zorder=7,
                        )
            if self.obstacle_draw_mode:
                for axis in axes:
                    axis.text(
                        0.01,
                        0.98,
                        "Obstacle Draw Mode: ON (click map to add points)",
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        color="#0f172a",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#dbeafe", alpha=0.8, edgecolor="none"),
                        zorder=20,
                    )
            if self.obstacle_remove_click_mode:
                for axis in axes:
                    axis.text(
                        0.01,
                        0.86,
                        "Obstacle Remove Mode: ON (click obstacle to remove)",
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        color="#0f172a",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", alpha=0.9, edgecolor="none"),
                        zorder=20,
                    )
            if self.task_click_mode:
                for axis in axes:
                    axis.text(
                        0.01,
                        0.92,
                        "Task Click Mode: ON (click map to add task)",
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        color="#0f172a",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#dcfce7", alpha=0.85, edgecolor="none"),
                        zorder=20,
                    )
            self.canvas.draw_idle()
            return

        prev_view = self._capture_map_view_state()
        render_state = self._interpolated_online_render_state() if self.enable_online_runtime else None
        self.session.draw_on_axis(self.ax, render_state=render_state)
        self._restore_map_view_state(prev_view or self._map_view_state)
        if self.dragging_task_id is not None and self.drag_preview_pos is not None:
            x, y = self.drag_preview_pos
            self.ax.scatter([x], [y], s=80, marker="x", color="#be123c", linewidth=2.0, zorder=15)
            self.ax.text(
                x,
                y,
                f"  T{self.dragging_task_id}",
                fontsize=9,
                color="#9f1239",
                ha="left",
                va="bottom",
                zorder=16,
            )
        if self.obstacle_points:
            xs = [p[0] for p in self.obstacle_points]
            ys = [p[1] for p in self.obstacle_points]
            self.ax.scatter(xs, ys, s=35, color="#0ea5e9", edgecolor="white", linewidth=0.5, zorder=8)
            self.ax.plot(xs, ys, color="#0ea5e9", linewidth=1.5, linestyle="--", zorder=7)
            if len(self.obstacle_points) >= 3:
                self.ax.plot(
                    [self.obstacle_points[-1][0], self.obstacle_points[0][0]],
                    [self.obstacle_points[-1][1], self.obstacle_points[0][1]],
                    color="#0ea5e9",
                    linewidth=1.5,
                    linestyle="--",
                    zorder=7,
                )
        if self.obstacle_draw_mode:
            self.ax.text(
                0.01,
                0.98,
                "Obstacle Draw Mode: ON (click map to add points)",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#dbeafe", alpha=0.8, edgecolor="none"),
                zorder=20,
            )
        if self.obstacle_remove_click_mode:
            self.ax.text(
                0.01,
                0.86,
                "Obstacle Remove Mode: ON (click obstacle to remove)",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", alpha=0.9, edgecolor="none"),
                zorder=20,
            )
        if self.task_click_mode:
            self.ax.text(
                0.01,
                0.92,
                "Task Click Mode: ON (click map to add task)",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#dcfce7", alpha=0.85, edgecolor="none"),
                zorder=20,
            )
        self._map_view_state = self._capture_map_view_state()
        self.canvas.draw_idle()

    def _pick_task_id_near(self, x: float, y: float, axis=None, ratio: float = 0.03) -> int | None:
        plot_axis = axis or self.ax
        x_min, x_max = plot_axis.get_xlim()
        y_min, y_max = plot_axis.get_ylim()
        tol = ratio * max(abs(x_max - x_min), abs(y_max - y_min))
        tol2 = tol * tol

        nearest_task_id: int | None = None
        nearest_d2 = float("inf")

        for task in self.session.list_tasks():
            if task.status == "canceled":
                continue
            dx = task.position[0] - x
            dy = task.position[1] - y
            d2 = dx * dx + dy * dy
            if d2 <= tol2 and d2 < nearest_d2:
                nearest_d2 = d2
                nearest_task_id = task.id
        return nearest_task_id

    def _refresh_text_panels(self) -> None:
        log_n = self._parse_optional_int(self.log_count_var.get()) or 8
        status_text = self.session.format_status_text()
        if self.status_text is not None and status_text != self._last_status_text:
            self._set_text(self.status_text, status_text)
            self._last_status_text = status_text

        if self.enable_online_runtime and self.logs_text is not None:
            logs_text = self.session.format_logs_text(n=log_n)
            if logs_text != self._last_logs_text:
                self._set_text(self.logs_text, logs_text)
                self._last_logs_text = logs_text

        tasks_text = self.session.format_tasks_text(limit=80)
        if self.tasks_text is not None and tasks_text != self._last_tasks_text:
            self._set_text(self.tasks_text, tasks_text)
            self._last_tasks_text = tasks_text

        if self.enable_online_runtime:
            snap = self.session.runtime_snapshot()
            self.online_state_var.set("RUN" if snap.online_running else ("IDLE" if self.session.online_enabled else "OFF"))
            self.sim_time_var.set(f"{snap.sim_time:.2f}s")
            if snap.pending_events:
                nxt = snap.pending_events[0]
                self.next_event_var.set(f"{nxt.time_s:.2f}s:{nxt.event_type}")
            else:
                self.next_event_var.set("-")
            self.replan_reason_var.set(snap.last_replan_reason or "-")
        obstacles = self.session.list_obstacles()
        values = [str(idx) for idx, _ in obstacles]
        if self.obstacle_remove_combo is not None:
            self.obstacle_remove_combo.configure(values=values)
            if values and self.obstacle_remove_var.get() not in values:
                self.obstacle_remove_var.set(values[0])
            if not values:
                self.obstacle_remove_var.set("")

    def _refresh_offline_comparison(self) -> None:
        if self.enable_online_runtime or self.comparison_canvas is None:
            return
        try:
            self.session.draw_comparison_on_axes(
                self.compare_ax_with,
                self.compare_ax_without,
                self.compare_ax_without_verify_cost,
            )
            self.comparison_canvas.draw_idle()
        except Exception:
            self.compare_ax_with.clear()
            self.compare_ax_without.clear()
            self.compare_ax_with.set_axis_off()
            self.compare_ax_without.set_axis_off()
            if self.compare_ax_without_verify_cost is not None:
                self.compare_ax_without_verify_cost.clear()
                self.compare_ax_without_verify_cost.set_axis_off()
            self.compare_ax_with.text(
                0.5,
                0.5,
                "Comparison unavailable",
                ha="center",
                va="center",
                fontsize=11,
                color="#991b1b",
                transform=self.compare_ax_with.transAxes,
            )
            self.compare_ax_without.text(
                0.5,
                0.5,
                "Check status panel for details",
                ha="center",
                va="center",
                fontsize=10,
                color="#374151",
                transform=self.compare_ax_without.transAxes,
            )
            if self.compare_ax_without_verify_cost is not None:
                self.compare_ax_without_verify_cost.text(
                    0.5,
                    0.5,
                    "Comparison unavailable",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#374151",
                    transform=self.compare_ax_without_verify_cost.transAxes,
                )
            self.comparison_canvas.draw_idle()

    def _refresh_all(self) -> None:
        try:
            self._refresh_map()
            self._refresh_text_panels()
            self._refresh_offline_comparison()
        except Exception as exc:
            messagebox.showerror("Refresh Error", str(exc))

    def _capture_online_render_state(self) -> dict[int, dict[str, object]]:
        if self.session.engine is None:
            return {}
        return {
            v.id: {
                "current_pos": (float(v.current_pos[0]), float(v.current_pos[1])),
                "current_heading": float(v.current_heading),
                "is_moving": bool(v.is_moving),
            }
            for v in self.session.engine.vehicles
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        while angle > math.pi:
            angle -= 2.0 * math.pi
        return angle

    def _interpolated_online_render_state(self) -> dict[int, dict[str, object]] | None:
        if not (self.session.online_enabled and self.session.online_running):
            return None
        if not self._prev_online_render_state or not self._curr_online_render_state:
            return None
        if self._last_online_tick_wall_ts is None:
            return self._curr_online_render_state

        step_s = max(float(self.session.online_dt), 1e-6)
        alpha = min(max((time.perf_counter() - self._last_online_tick_wall_ts) / step_s, 0.0), 1.0)
        render_state: dict[int, dict[str, object]] = {}
        for vid, curr in self._curr_online_render_state.items():
            prev = self._prev_online_render_state.get(vid, curr)
            prev_pos = prev["current_pos"]
            curr_pos = curr["current_pos"]
            prev_heading = float(prev["current_heading"])
            curr_heading = float(curr["current_heading"])
            d_heading = self._wrap_angle(curr_heading - prev_heading)
            render_state[vid] = {
                "current_pos": (
                    float(prev_pos[0]) + (float(curr_pos[0]) - float(prev_pos[0])) * alpha,
                    float(prev_pos[1]) + (float(curr_pos[1]) - float(prev_pos[1])) * alpha,
                ),
                "current_heading": prev_heading + d_heading * alpha,
                "is_moving": bool(curr["is_moving"] or prev["is_moving"]),
            }
        return render_state

    def _reset_online_render_clock(self) -> None:
        self._last_online_tick_wall_ts = None
        self._prev_online_render_state = {}
        self._curr_online_render_state = {}

    def _advance_online_clock(self, now: float) -> None:
        step_s = max(float(self.session.online_dt), 1e-6)
        if self._last_online_tick_wall_ts is None:
            state = self._capture_online_render_state()
            self._prev_online_render_state = state
            self._curr_online_render_state = state
            self._last_online_tick_wall_ts = now
            return

        while now - self._last_online_tick_wall_ts >= step_s - 1e-9:
            prev_state = self._capture_online_render_state()
            self.session.tick(n=1)
            curr_state = self._capture_online_render_state()
            self._prev_online_render_state = prev_state
            self._curr_online_render_state = curr_state
            self._last_online_tick_wall_ts += step_s

    def _schedule_refresh(self) -> None:
        now = time.perf_counter()
        if self.enable_online_runtime:
            try:
                if self.session.online_enabled and self.session.online_running:
                    self._advance_online_clock(now)
                elif not self.session.online_running:
                    self._reset_online_render_clock()
            except Exception as exc:
                self.last_action_var.set(f"tick error: {exc}")
        try:
            self._refresh_map()
            if (
                not self.enable_online_runtime
                or not self.session.online_enabled
                or not self.session.online_running
                or now - self._last_text_refresh_wall_ts >= 0.2
            ):
                self._refresh_text_panels()
                self._refresh_offline_comparison()
                self._last_text_refresh_wall_ts = now
        except Exception as exc:
            messagebox.showerror("Refresh Error", str(exc))
        next_delay = (
            self.render_refresh_interval_ms
            if self.enable_online_runtime and self.session.online_enabled and self.session.online_running
            else self.refresh_interval_ms
        )
        self.root.after(next_delay, self._schedule_refresh)

    def _on_reset(self) -> None:
        try:
            self.session.reset()
            self._map_view_state = None
            self.obstacle_points = []
            self.obstacle_draw_mode = False
            self.obstacle_remove_click_mode = False
            self.task_click_mode = False
            self.dragging_task_id = None
            self.drag_origin_pos = None
            self.drag_preview_pos = None
            self.obstacle_mode_var.set("OFF")
            self.obstacle_remove_mode_var.set("OFF")
            self.obstacle_point_count_var.set("0")
            self.task_mode_var.set("OFF")
            self.last_action_var.set("Scenario reset")
            self._refresh_all()
            messagebox.showinfo("Reset", "Scenario initialized from seed.")
        except Exception as exc:
            messagebox.showerror("Reset Error", str(exc))

    def _on_reset_replay(self) -> None:
        try:
            self.session.reset(replay_last_actions=True)
            self._map_view_state = None
            self.obstacle_points = []
            self.obstacle_draw_mode = False
            self.obstacle_remove_click_mode = False
            self.task_click_mode = False
            self.dragging_task_id = None
            self.drag_origin_pos = None
            self.drag_preview_pos = None
            self.obstacle_mode_var.set("OFF")
            self.obstacle_remove_mode_var.set("OFF")
            self.obstacle_point_count_var.set("0")
            self.task_mode_var.set("OFF")
            self.last_action_var.set("Scenario reset and replayed")
            self._refresh_all()
            messagebox.showinfo("Reset + Replay", "Scenario reset and replayed from last recorded actions.")
        except Exception as exc:
            messagebox.showerror("Reset Replay Error", str(exc))

    def _on_undo(self) -> None:
        try:
            self.session.undo()
            self.obstacle_points = []
            self.dragging_task_id = None
            self.drag_origin_pos = None
            self.drag_preview_pos = None
            self.obstacle_point_count_var.set("0")
            self.last_action_var.set("Undo done")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Undo Error", str(exc))

    def _on_save_snapshot(self) -> None:
        try:
            path = self.session.save_snapshot()
            self._refresh_all()
            self.last_action_var.set(f"Snapshot saved: {path.name}")
            messagebox.showinfo("Snapshot Saved", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Snapshot Error", str(exc))

    def _on_save_scenario_json(self) -> None:
        saver = getattr(self.session, "save_scenario_json", None)
        if not callable(saver):
            messagebox.showerror("Save Scenario Error", "Current session does not support scenario JSON export.")
            return

        if self._scenario_json_path is None:
            self._on_save_scenario_json_as()
            return

        try:
            path = saver(path=self._scenario_json_path)
            self._scenario_json_path = Path(path)
            self._refresh_all()
            self.last_action_var.set(f"Scenario saved: {path.name}")
            messagebox.showinfo("Scenario Saved", f"Saved scenario JSON to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Scenario Error", str(exc))

    def _on_save_scenario_json_as(self) -> None:
        saver = getattr(self.session, "save_scenario_json", None)
        if not callable(saver):
            messagebox.showerror("Save Scenario Error", "Current session does not support scenario JSON export.")
            return

        if self._scenario_json_path is not None:
            initial_dir = str(self._scenario_json_path.parent)
            initial_name = self._scenario_json_path.name
        elif self.cfg.scenario_file:
            src = Path(self.cfg.scenario_file)
            initial_dir = str(src.parent if str(src.parent) else Path.cwd())
            initial_name = f"{src.stem}.json"
        else:
            initial_dir = "examples"
            initial_name = "scenario_offline_saved.json"

        dest = filedialog.asksaveasfilename(
            title="Save Scenario JSON As",
            initialdir=initial_dir,
            initialfile=initial_name,
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not dest:
            return

        try:
            path = saver(path=dest)
            self._scenario_json_path = Path(path)
            self._refresh_all()
            self.last_action_var.set(f"Scenario saved as: {path.name}")
            messagebox.showinfo("Scenario Saved", f"Saved scenario JSON to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Scenario Error", str(exc))

    def _on_export_logs(self) -> None:
        try:
            coord, verify, big = self.session.export_logs()
            self._refresh_all()
            self.last_action_var.set("Logs exported")
            messagebox.showinfo(
                "Logs Exported",
                f"Coordination:\n{coord}\n\nVerification:\n{verify}\n\nAuction Big Log:\n{big}",
            )
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _on_export_path_quality(self) -> None:
        try:
            if not hasattr(self.session, "export_path_quality_report"):
                raise RuntimeError("Current session does not support path quality export.")
            report, summary = self.session.export_path_quality_report()
            self._refresh_all()
            self.last_action_var.set(f"Path quality exported: {summary.name}")
            messagebox.showinfo(
                "Path Quality Exported",
                f"Per-vehicle:\n{report}\n\nSummary:\n{summary}",
            )
        except Exception as exc:
            messagebox.showerror("Path Quality Export Error", str(exc))

    def _on_export_task_ops_json(self) -> None:
        try:
            path = self.session.export_task_ops_json()
            self._refresh_all()
            self.last_action_var.set(f"Task ops exported: {path.name}")
            messagebox.showinfo("Task Ops Exported", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Task Ops Export Error", str(exc))

    def _on_replay_task_ops_json(self) -> None:
        source = filedialog.askopenfilename(
            title="Select Task Ops JSON",
            initialdir="outputs",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not source:
            return
        try:
            replayed = self.session.replay_task_ops_json(path=source, reset_first=True)
            self._map_view_state = None
            self.obstacle_points = []
            self.obstacle_draw_mode = False
            self.obstacle_remove_click_mode = False
            self.task_click_mode = False
            self.dragging_task_id = None
            self.drag_origin_pos = None
            self.drag_preview_pos = None
            self.obstacle_mode_var.set("OFF")
            self.obstacle_remove_mode_var.set("OFF")
            self.obstacle_point_count_var.set("0")
            self.task_mode_var.set("OFF")
            if self.enable_online_runtime:
                self._reset_online_render_clock()
            self.last_action_var.set(f"Task ops loaded: {replayed}")
            self._refresh_all()
            messagebox.showinfo("Task Ops Loaded", f"Loaded {replayed} operations for playback from:\n{source}")
        except Exception as exc:
            messagebox.showerror("Task Ops Replay Error", str(exc))

    def _pick_obstacle_idx_near(self, x: float, y: float, axis=None, ratio: float = 0.02) -> int | None:
        artifacts = getattr(self.session, "artifacts", None)
        if artifacts is None:
            return None
        obstacles = list(getattr(artifacts.world, "obstacles", []))
        if not obstacles:
            return None

        plot_axis = axis or self.ax
        x_min, x_max = plot_axis.get_xlim()
        y_min, y_max = plot_axis.get_ylim()
        tol = ratio * max(abs(x_max - x_min), abs(y_max - y_min))
        pt = Point(float(x), float(y))

        best: tuple[float, float, int] | None = None
        for idx, obs in enumerate(obstacles):
            dist = float(obs.distance(pt))
            inside = bool(obs.buffer(1e-9).contains(pt))
            if (not inside) and dist > tol:
                continue
            score = 0.0 if inside else dist
            candidate = (score, float(obs.area), idx)
            if best is None or candidate < best:
                best = candidate
        return None if best is None else int(best[2])

    def _has_pending_offline_reallocation(self) -> bool:
        checker = getattr(self.session, "has_pending_reallocation", None)
        return bool(callable(checker) and checker())

    def _on_reallocate_offline(self) -> None:
        if self.enable_online_runtime:
            return
        try:
            action = getattr(self.session, "reallocate_now", None)
            if not callable(action):
                raise RuntimeError("offline session does not support manual re-auction")
            result = action()
            self._refresh_all()
            self.last_action_var.set("Offline re-auction completed")
            messagebox.showinfo(
                "Re-auction Completed",
                f"System total time: {result.system_total_time:.3f}",
            )
        except Exception as exc:
            messagebox.showerror("Re-auction Error", str(exc))

    def _on_start_online(self) -> None:
        try:
            self.session.start_online()
            self._reset_online_render_clock()
            self.last_action_var.set("Online runtime started")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Online Start Error", str(exc))

    def _on_toggle_online_pause(self) -> None:
        try:
            if not self.session.online_enabled:
                self.session.start_online()
                self._reset_online_render_clock()
                self.last_action_var.set("Online runtime started")
            elif self.session.online_running:
                self.session.pause_online()
                self._reset_online_render_clock()
                self.last_action_var.set("Online runtime paused")
            else:
                self.session.resume_online()
                self._reset_online_render_clock()
                self.last_action_var.set("Online runtime resumed")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Online Pause/Resume Error", str(exc))

    def _on_online_step(self, n: int) -> None:
        try:
            if not self.session.online_enabled:
                self.session.start_online()
            self._reset_online_render_clock()
            self.session.tick(n=int(n))
            self.last_action_var.set(f"Online stepped x{n}")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Online Step Error", str(exc))

    def _on_frame_prev(self) -> None:
        try:
            if not self.session.online_enabled:
                self.session.start_online()
            self.session.pause_online()
            self._reset_online_render_clock()
            self.session.frame_prev()
            self.last_action_var.set("Moved to previous frame")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Prev Frame Error", str(exc))

    def _on_frame_next(self) -> None:
        try:
            if not self.session.online_enabled:
                self.session.start_online()
            self.session.pause_online()
            self._reset_online_render_clock()
            self.session.frame_next()
            self.last_action_var.set("Moved to next frame")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Next Frame Error", str(exc))

    def _on_add_random(self) -> None:
        try:
            demand = self._parse_optional_int(self.random_demand_var.get())
            task = self.session.add_random_task(demand=demand)
            self._refresh_all()
            if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                self.last_action_var.set(f"Random task added: T{task.id} (pending re-auction)")
            else:
                self.last_action_var.set(f"Random task added: T{task.id}")
            messagebox.showinfo(
                "Random Task Added",
                f"Added T{task.id} at ({task.position[0]:.2f}, {task.position[1]:.2f}) demand={task.demand}",
            )
        except Exception as exc:
            messagebox.showerror("Add Random Error", str(exc))

    def _on_add_at_coord(self) -> None:
        try:
            x = self._parse_required_float(self.add_x_var.get(), "x")
            y = self._parse_required_float(self.add_y_var.get(), "y")
            demand = self._parse_required_int(self.add_demand_var.get(), "demand")
            task_id = self._parse_optional_int(self.add_task_id_var.get())
            task = self.session.add_task(x=x, y=y, demand=demand, task_id=task_id)
            if task_id is not None:
                self.add_task_id_var.set("")
            self._refresh_all()
            if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                self.last_action_var.set(f"Task added by coord: T{task.id} (pending re-auction)")
            else:
                self.last_action_var.set(f"Task added by coord: T{task.id}")
            messagebox.showinfo(
                "Task Added",
                f"Added T{task.id} at ({task.position[0]:.2f}, {task.position[1]:.2f}) demand={task.demand}",
            )
        except Exception as exc:
            messagebox.showerror("Add Task Error", str(exc))

    def _on_cancel_task(self) -> None:
        try:
            task_id = self._parse_required_int(self.cancel_task_id_var.get(), "task_id")
            self.session.cancel_task(task_id=task_id)
            self._refresh_all()
            if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                self.last_action_var.set(f"Task canceled: T{task_id} (pending re-auction)")
            else:
                self.last_action_var.set(f"Task canceled: T{task_id}")
            messagebox.showinfo("Task Canceled", f"Canceled T{task_id}")
        except Exception as exc:
            messagebox.showerror("Cancel Error", str(exc))

    def _on_canvas_click(self, event) -> None:
        axis = self._event_axis(event)
        if axis is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        if self.obstacle_draw_mode:
            self.obstacle_points.append((x, y))
            self.obstacle_point_count_var.set(str(len(self.obstacle_points)))
            self.last_action_var.set(f"Obstacle point added ({x:.2f}, {y:.2f})")
            self._refresh_map()
            return

        if self.obstacle_remove_click_mode:
            idx = self._pick_obstacle_idx_near(x, y, axis=axis)
            if idx is None:
                self.last_action_var.set("No obstacle near click point")
                self._refresh_map()
                return
            try:
                self.session.remove_obstacle(obstacle_idx=idx)
                if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                    self.last_action_var.set(f"Obstacle removed by click: idx={idx} (pending re-auction)")
                else:
                    self.last_action_var.set(f"Obstacle removed by click: idx={idx}")
                self._refresh_all()
            except Exception as exc:
                messagebox.showerror("Obstacle Click Remove Error", str(exc))
            return

        if self.task_click_mode:
            try:
                demand = self._parse_required_int(self.add_demand_var.get(), "demand")
                task_id = self._parse_optional_int(self.add_task_id_var.get())
                task = self.session.add_task(x=x, y=y, demand=demand, task_id=task_id)
                if task_id is not None:
                    self.add_task_id_var.set("")
                label = f"Task added by click: T{task.id} ({task.position[0]:.2f}, {task.position[1]:.2f})"
                if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                    label += " (pending re-auction)"
                self.last_action_var.set(label)
                self._refresh_all()
            except Exception as exc:
                messagebox.showerror("Task Click Add Error", str(exc))

    def _on_canvas_press(self, event) -> None:
        axis = self._event_axis(event)
        if axis is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        button = int(getattr(event, "button", 0) or 0)

        if button == 3:
            if self.obstacle_draw_mode or self.task_click_mode:
                return
            task_id = self._pick_task_id_near(x, y, axis=axis)
            if task_id is None:
                return
            try:
                self.session.cancel_task(task_id=task_id)
                if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                    self.last_action_var.set(f"Task deleted by right click: T{task_id} (pending re-auction)")
                else:
                    self.last_action_var.set(f"Task deleted by right click: T{task_id}")
                self._refresh_all()
            except Exception as exc:
                messagebox.showerror("Task Delete Error", str(exc))
            return

        if button != 1:
            return

        if self.obstacle_draw_mode or self.obstacle_remove_click_mode or self.task_click_mode:
            self._on_canvas_click(event)
            return

        task_id = self._pick_task_id_near(x, y, axis=axis)
        if task_id is None:
            return

        task = next((t for t in self.session.list_tasks() if t.id == task_id), None)
        if task is None:
            return

        self.dragging_task_id = task_id
        self.drag_origin_pos = (task.position[0], task.position[1])
        self.drag_preview_pos = (x, y)
        self._last_drag_refresh_ts = 0.0
        self.last_action_var.set(f"Dragging T{task_id}...")
        self._refresh_map()

    def _on_canvas_motion(self, event) -> None:
        if self.dragging_task_id is None:
            return
        if self._event_axis(event) is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.drag_preview_pos = (float(event.xdata), float(event.ydata))
        now = time.perf_counter()
        if (now - self._last_drag_refresh_ts) < self.drag_refresh_interval_s:
            return
        self._last_drag_refresh_ts = now
        self._refresh_map()

    def _on_canvas_release(self, event) -> None:
        button = int(getattr(event, "button", 0) or 0)
        if button != 1:
            return
        if self.dragging_task_id is None:
            return

        task_id = self.dragging_task_id
        origin = self.drag_origin_pos
        target = self.drag_preview_pos
        self.dragging_task_id = None
        self.drag_origin_pos = None
        self.drag_preview_pos = None

        if origin is None or target is None:
            self._refresh_map()
            return

        if abs(origin[0] - target[0]) < 1e-6 and abs(origin[1] - target[1]) < 1e-6:
            self.last_action_var.set(f"Drag canceled: T{task_id}")
            self._refresh_map()
            return

        try:
            moved = self.session.move_task(task_id=task_id, x=target[0], y=target[1])
            label = f"Task moved: T{task_id} -> ({moved.position[0]:.2f}, {moved.position[1]:.2f})"
            if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                label += " (pending re-auction)"
            self.last_action_var.set(label)
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Task Move Error", str(exc))
            self._refresh_all()

    def _on_toggle_obstacle_draw(self) -> None:
        if not self.obstacle_draw_mode and self.task_click_mode:
            self.task_click_mode = False
            self.task_mode_var.set("OFF")
        if not self.obstacle_draw_mode and self.obstacle_remove_click_mode:
            self.obstacle_remove_click_mode = False
            self.obstacle_remove_mode_var.set("OFF")
        self.obstacle_draw_mode = not self.obstacle_draw_mode
        self.obstacle_mode_var.set("ON" if self.obstacle_draw_mode else "OFF")
        self.last_action_var.set(f"Obstacle draw mode: {self.obstacle_mode_var.get()}")
        self._refresh_map()

    def _on_toggle_task_click(self) -> None:
        if not self.task_click_mode and self.obstacle_draw_mode:
            self.obstacle_draw_mode = False
            self.obstacle_mode_var.set("OFF")
        if not self.task_click_mode and self.obstacle_remove_click_mode:
            self.obstacle_remove_click_mode = False
            self.obstacle_remove_mode_var.set("OFF")
        self.task_click_mode = not self.task_click_mode
        self.task_mode_var.set("ON" if self.task_click_mode else "OFF")
        self.last_action_var.set(f"Task click mode: {self.task_mode_var.get()}")
        self._refresh_map()

    def _on_toggle_obstacle_remove_click(self) -> None:
        if not self.obstacle_remove_click_mode and self.obstacle_draw_mode:
            self.obstacle_draw_mode = False
            self.obstacle_mode_var.set("OFF")
        if not self.obstacle_remove_click_mode and self.task_click_mode:
            self.task_click_mode = False
            self.task_mode_var.set("OFF")
        self.obstacle_remove_click_mode = not self.obstacle_remove_click_mode
        self.obstacle_remove_mode_var.set("ON" if self.obstacle_remove_click_mode else "OFF")
        self.last_action_var.set(f"Obstacle click remove mode: {self.obstacle_remove_mode_var.get()}")
        self._refresh_map()

    def _on_obstacle_undo(self) -> None:
        if self.obstacle_points:
            self.obstacle_points.pop()
            self.obstacle_point_count_var.set(str(len(self.obstacle_points)))
            self._refresh_map()

    def _on_obstacle_clear(self) -> None:
        self.obstacle_points = []
        self.obstacle_point_count_var.set("0")
        self.last_action_var.set("Obstacle points cleared")
        self._refresh_map()

    def _on_apply_obstacle(self) -> None:
        try:
            polygon = self.session.add_obstacle_polygon(self.obstacle_points)
            self.obstacle_points = []
            self.obstacle_draw_mode = False
            self.obstacle_mode_var.set("OFF")
            self.obstacle_point_count_var.set("0")
            if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                self.last_action_var.set(f"Obstacle applied (area={polygon.area:.2f}, pending re-auction)")
            else:
                self.last_action_var.set(f"Obstacle applied (area={polygon.area:.2f})")
            self._refresh_all()
            messagebox.showinfo(
                "Obstacle Added",
                (
                    f"Added obstacle polygon, area={polygon.area:.2f}. "
                    + ("Click Re-auction Now to recompute allocation." if not self.enable_online_runtime else "")
                ),
            )
        except Exception as exc:
            messagebox.showerror("Obstacle Error", str(exc))

    def _on_remove_obstacle(self) -> None:
        try:
            v = self.obstacle_remove_var.get().strip()
            if not v:
                raise ValueError("no obstacle index selected")
            idx = int(v)
            self.session.remove_obstacle(obstacle_idx=idx)
            if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                self.last_action_var.set(f"Obstacle removed: idx={idx} (pending re-auction)")
            else:
                self.last_action_var.set(f"Obstacle removed: idx={idx}")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Obstacle Remove Error", str(exc))

    def _on_replace_obstacle(self) -> None:
        try:
            v = self.obstacle_remove_var.get().strip()
            if not v:
                raise ValueError("no obstacle index selected")
            if len(self.obstacle_points) < 3:
                raise ValueError("draw at least 3 points for replacement obstacle")
            idx = int(v)
            replacer = getattr(self.session, "replace_obstacle_polygon", None)
            if not callable(replacer):
                raise RuntimeError("current session does not support obstacle replacement")
            poly = replacer(obstacle_idx=idx, points=self.obstacle_points)
            self.obstacle_points = []
            self.obstacle_draw_mode = False
            self.obstacle_mode_var.set("OFF")
            self.obstacle_point_count_var.set("0")
            if not self.enable_online_runtime and self._has_pending_offline_reallocation():
                self.last_action_var.set(f"Obstacle replaced: idx={idx} (pending re-auction)")
            else:
                self.last_action_var.set(f"Obstacle replaced: idx={idx}")
            self._refresh_all()
            messagebox.showinfo(
                "Obstacle Replaced",
                (
                    f"Replaced obstacle idx={idx}, area={poly.area:.2f}. "
                    + ("Click Re-auction Now to recompute allocation." if not self.enable_online_runtime else "")
                ),
            )
        except Exception as exc:
            messagebox.showerror("Obstacle Replace Error", str(exc))

    def run(self) -> None:
        self.root.mainloop()


class OfflineGuiApp(_BaseGuiApp):
    def __init__(self, cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
        super().__init__(
            session=OfflineSession(cfg),
            cfg=cfg,
            window_title="MILP Static Allocation GUI",
            enable_online_runtime=False,
            enable_offline_tools=True,
        )


class OnlineGuiApp(_BaseGuiApp):
    def __init__(self, cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
        super().__init__(
            session=OnlineSession(cfg),
            cfg=cfg,
            window_title="MILP Online Allocation GUI",
            enable_online_runtime=True,
            enable_offline_tools=False,
        )


MilpGuiApp = OfflineGuiApp


def run_offline_gui(cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
    try:
        app = OfflineGuiApp(cfg=cfg)
    except tk.TclError as exc:
        raise RuntimeError(
            "Failed to start GUI. A graphical display is required (X11/Wayland/macOS window server)."
        ) from exc
    app.run()


def run_online_gui(cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
    try:
        app = OnlineGuiApp(cfg=cfg)
    except tk.TclError as exc:
        raise RuntimeError(
            "Failed to start GUI. A graphical display is required (X11/Wayland/macOS window server)."
        ) from exc
    app.run()


def run_gui(cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
    run_offline_gui(cfg)
