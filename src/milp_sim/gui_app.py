from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .config import DEFAULT_CONFIG, SimulationConfig
from .session import SimulationSession


class MilpGuiApp:
    def __init__(self, cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self.session = SimulationSession(cfg)
        self.obstacle_draw_mode = False
        self.obstacle_points: list[tuple[float, float]] = []
        self.task_click_mode = False
        self.dragging_task_id: int | None = None
        self.drag_origin_pos: tuple[float, float] | None = None
        self.drag_preview_pos: tuple[float, float] | None = None
        self._map_view_state: dict[str, tuple[float, float]] | None = None

        self.root = tk.Tk()
        self.root.title("MILP Dynamic Task Allocation GUI")
        self.root.geometry("1600x920")
        self._configure_ui_style()

        self.refresh_interval_ms = 500
        self.log_count_var = tk.StringVar(value="8")
        self.obstacle_point_count_var = tk.StringVar(value="0")
        self.obstacle_mode_var = tk.StringVar(value="OFF")
        self.task_mode_var = tk.StringVar(value="OFF")
        self.last_action_var = tk.StringVar(value="Ready")
        self.online_state_var = tk.StringVar(value="OFF")
        self.sim_time_var = tk.StringVar(value="0.00")
        self.next_event_var = tk.StringVar(value="-")
        self.replan_reason_var = tk.StringVar(value="-")
        self.sim_speed_var = tk.StringVar(value="1x")
        self.obstacle_remove_var = tk.StringVar(value="")

        self._build_layout()
        self._refresh_all()
        self._schedule_refresh()

    def _configure_ui_style(self) -> None:
        self.root.configure(bg="#e9edf2")
        # Improve readability on HiDPI/WSLg displays.
        self.root.tk.call("tk", "scaling", 1.25)

        style = ttk.Style(self.root)
        available_themes = set(style.theme_names())
        preferred_theme = "clam" if "clam" in available_themes else style.theme_use()
        style.theme_use(preferred_theme)

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=10)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family="Segoe UI", size=10)
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(family="Cascadia Mono", size=10)

        self.root.option_add("*Font", default_font)

        style.configure("TFrame", background="#e9edf2")
        style.configure("TLabelframe", background="#e9edf2")
        style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 10), foreground="#1f2937")
        style.configure("TLabel", background="#e9edf2", foreground="#111827")
        style.configure("TButton", padding=(10, 4), font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="#ffffff")

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=3)
        self.root.columnconfigure(2, weight=2)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        center = ttk.Frame(self.root, padding=8)
        right = ttk.Frame(self.root, padding=10)

        left.grid(row=0, column=0, sticky="ns")
        center.grid(row=0, column=1, sticky="nsew")
        right.grid(row=0, column=2, sticky="nsew")

        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_left_controls(left)
        self._build_center_plot(center)
        self._build_right_panels(right)

    def _build_left_controls(self, parent) -> None:
        top = ttk.LabelFrame(parent, text="Top Controls", padding=8)
        top.pack(fill="x", pady=(0, 8))

        ttk.Button(top, text="Initialize/Reset", command=self._on_reset).pack(fill="x", pady=3)
        ttk.Button(top, text="Reset + Replay Last Ops", command=self._on_reset_replay).pack(fill="x", pady=3)
        ttk.Button(top, text="Undo Last Action", command=self._on_undo).pack(fill="x", pady=3)
        ttk.Button(top, text="Save Snapshot", command=self._on_save_snapshot).pack(fill="x", pady=3)
        ttk.Button(top, text="Export Logs", command=self._on_export_logs).pack(fill="x", pady=3)

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
        ttk.Button(
            obstacle_frame,
            text="Apply Obstacle",
            command=self._on_apply_obstacle,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Label(obstacle_frame, text="remove idx").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.obstacle_remove_combo = ttk.Combobox(
            obstacle_frame,
            textvariable=self.obstacle_remove_var,
            values=(),
            state="readonly",
            width=12,
        )
        self.obstacle_remove_combo.grid(row=5, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(obstacle_frame, text="Remove Obstacle", command=self._on_remove_obstacle).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        obstacle_frame.columnconfigure(0, weight=1)
        obstacle_frame.columnconfigure(1, weight=1)

        add_frame = ttk.LabelFrame(parent, text="Add Task", padding=8)
        add_frame.pack(fill="x", pady=8)

        self.add_demand_var = tk.StringVar(value="2")
        self.add_task_id_var = tk.StringVar(value="")

        ttk.Label(add_frame, text="click mode").grid(row=0, column=0, sticky="w")
        ttk.Label(add_frame, textvariable=self.task_mode_var).grid(row=0, column=1, sticky="w")

        ttk.Label(add_frame, text="demand").grid(row=1, column=0, sticky="w")
        ttk.Entry(add_frame, textvariable=self.add_demand_var, width=12).grid(row=1, column=1, sticky="ew")

        ttk.Label(add_frame, text="task_id(opt)").grid(row=2, column=0, sticky="w")
        ttk.Entry(add_frame, textvariable=self.add_task_id_var, width=12).grid(row=2, column=1, sticky="ew")

        ttk.Button(add_frame, text="Start/Stop Click Add", command=self._on_toggle_task_click).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        add_frame.columnconfigure(1, weight=1)

        random_frame = ttk.LabelFrame(parent, text="Add Random Task", padding=8)
        random_frame.pack(fill="x", pady=8)

        self.random_demand_var = tk.StringVar(value="")
        ttk.Label(random_frame, text="demand(opt)").grid(row=0, column=0, sticky="w")
        ttk.Entry(random_frame, textvariable=self.random_demand_var, width=12).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(
            random_frame,
            text="Add Random + Re-auction",
            command=self._on_add_random,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        random_frame.columnconfigure(1, weight=1)

        cancel_frame = ttk.LabelFrame(parent, text="Cancel Task", padding=8)
        cancel_frame.pack(fill="x", pady=8)

        self.cancel_task_id_var = tk.StringVar(value="")
        ttk.Label(cancel_frame, text="task_id").grid(row=0, column=0, sticky="w")
        ttk.Entry(cancel_frame, textvariable=self.cancel_task_id_var, width=12).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(cancel_frame, text="Cancel + Re-auction", command=self._on_cancel_task).grid(
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
        self.figure = plt.Figure(figsize=(8, 8), dpi=130)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.05, right=0.985, bottom=0.05, top=0.96)
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

    def _build_right_panels(self, parent) -> None:
        status_frame = ttk.LabelFrame(parent, text="Status", padding=6)
        status_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        status_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)

        logs_frame = ttk.LabelFrame(parent, text="Recent Logs", padding=6)
        logs_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        logs_frame.rowconfigure(0, weight=1)
        logs_frame.columnconfigure(0, weight=1)

        tasks_frame = ttk.LabelFrame(parent, text="Tasks", padding=6)
        tasks_frame.grid(row=2, column=0, sticky="nsew")
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

        self.logs_text = ScrolledText(
            logs_frame,
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

    def _refresh_map(self) -> None:
        prev_view = self._capture_map_view_state()
        self.session.draw_on_axis(self.ax)
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

    def _pick_task_id_near(self, x: float, y: float, ratio: float = 0.03) -> int | None:
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
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
        self._set_text(self.status_text, self.session.format_status_text())
        self._set_text(self.logs_text, self.session.format_logs_text(n=log_n))
        self._set_text(self.tasks_text, self.session.format_tasks_text(limit=80))
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
        self.obstacle_remove_combo.configure(values=values)
        if values and self.obstacle_remove_var.get() not in values:
            self.obstacle_remove_var.set(values[0])
        if not values:
            self.obstacle_remove_var.set("")

    def _refresh_all(self) -> None:
        try:
            self._refresh_map()
            self._refresh_text_panels()
        except Exception as exc:
            messagebox.showerror("Refresh Error", str(exc))

    def _schedule_refresh(self) -> None:
        try:
            if self.session.online_enabled and self.session.online_running:
                self.session.tick(n=1)
        except Exception as exc:
            self.last_action_var.set(f"tick error: {exc}")
        self._refresh_all()
        self.root.after(self.refresh_interval_ms, self._schedule_refresh)

    def _on_reset(self) -> None:
        try:
            self.session.reset()
            self._map_view_state = None
            self.obstacle_points = []
            self.obstacle_draw_mode = False
            self.task_click_mode = False
            self.dragging_task_id = None
            self.drag_origin_pos = None
            self.drag_preview_pos = None
            self.obstacle_mode_var.set("OFF")
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
            self.task_click_mode = False
            self.dragging_task_id = None
            self.drag_origin_pos = None
            self.drag_preview_pos = None
            self.obstacle_mode_var.set("OFF")
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

    def _on_export_logs(self) -> None:
        try:
            coord, verify = self.session.export_logs()
            self._refresh_all()
            self.last_action_var.set("Logs exported")
            messagebox.showinfo("Logs Exported", f"Coordination:\n{coord}\n\nVerification:\n{verify}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _on_start_online(self) -> None:
        try:
            self.session.start_online()
            self.last_action_var.set("Online runtime started")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Online Start Error", str(exc))

    def _on_toggle_online_pause(self) -> None:
        try:
            if not self.session.online_enabled:
                self.session.start_online()
                self.last_action_var.set("Online runtime started")
            elif self.session.online_running:
                self.session.pause_online()
                self.last_action_var.set("Online runtime paused")
            else:
                self.session.resume_online()
                self.last_action_var.set("Online runtime resumed")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Online Pause/Resume Error", str(exc))

    def _on_online_step(self, n: int) -> None:
        try:
            if not self.session.online_enabled:
                self.session.start_online()
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
            self.last_action_var.set(f"Random task added: T{task.id}")
            messagebox.showinfo(
                "Random Task Added",
                f"Added T{task.id} at ({task.position[0]:.2f}, {task.position[1]:.2f}) demand={task.demand}",
            )
        except Exception as exc:
            messagebox.showerror("Add Random Error", str(exc))

    def _on_cancel_task(self) -> None:
        try:
            task_id = self._parse_required_int(self.cancel_task_id_var.get(), "task_id")
            self.session.cancel_task(task_id=task_id)
            self._refresh_all()
            self.last_action_var.set(f"Task canceled: T{task_id}")
            messagebox.showinfo("Task Canceled", f"Canceled T{task_id}")
        except Exception as exc:
            messagebox.showerror("Cancel Error", str(exc))

    def _on_canvas_click(self, event) -> None:
        if event.inaxes != self.ax:
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

        if self.task_click_mode:
            try:
                demand = self._parse_required_int(self.add_demand_var.get(), "demand")
                task_id = self._parse_optional_int(self.add_task_id_var.get())
                task = self.session.add_task(x=x, y=y, demand=demand, task_id=task_id)
                if task_id is not None:
                    self.add_task_id_var.set("")
                self.last_action_var.set(
                    f"Task added by click: T{task.id} ({task.position[0]:.2f}, {task.position[1]:.2f})"
                )
                self._refresh_all()
            except Exception as exc:
                messagebox.showerror("Task Click Add Error", str(exc))

    def _on_canvas_press(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        button = int(getattr(event, "button", 0) or 0)

        if button == 3:
            if self.obstacle_draw_mode or self.task_click_mode:
                return
            task_id = self._pick_task_id_near(x, y)
            if task_id is None:
                return
            try:
                self.session.cancel_task(task_id=task_id)
                self.last_action_var.set(f"Task deleted by right click: T{task_id}")
                self._refresh_all()
            except Exception as exc:
                messagebox.showerror("Task Delete Error", str(exc))
            return

        if button != 1:
            return

        if self.obstacle_draw_mode or self.task_click_mode:
            self._on_canvas_click(event)
            return

        task_id = self._pick_task_id_near(x, y)
        if task_id is None:
            return

        task = next((t for t in self.session.list_tasks() if t.id == task_id), None)
        if task is None:
            return

        self.dragging_task_id = task_id
        self.drag_origin_pos = (task.position[0], task.position[1])
        self.drag_preview_pos = (x, y)
        self.last_action_var.set(f"Dragging T{task_id}...")
        self._refresh_map()

    def _on_canvas_motion(self, event) -> None:
        if self.dragging_task_id is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.drag_preview_pos = (float(event.xdata), float(event.ydata))
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
            self.last_action_var.set(
                f"Task moved: T{task_id} -> ({moved.position[0]:.2f}, {moved.position[1]:.2f})"
            )
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Task Move Error", str(exc))
            self._refresh_all()

    def _on_toggle_obstacle_draw(self) -> None:
        if not self.obstacle_draw_mode and self.task_click_mode:
            self.task_click_mode = False
            self.task_mode_var.set("OFF")
        self.obstacle_draw_mode = not self.obstacle_draw_mode
        self.obstacle_mode_var.set("ON" if self.obstacle_draw_mode else "OFF")
        self.last_action_var.set(f"Obstacle draw mode: {self.obstacle_mode_var.get()}")
        self._refresh_map()

    def _on_toggle_task_click(self) -> None:
        if not self.task_click_mode and self.obstacle_draw_mode:
            self.obstacle_draw_mode = False
            self.obstacle_mode_var.set("OFF")
        self.task_click_mode = not self.task_click_mode
        self.task_mode_var.set("ON" if self.task_click_mode else "OFF")
        self.last_action_var.set(f"Task click mode: {self.task_mode_var.get()}")
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
            self.last_action_var.set(f"Obstacle applied (area={polygon.area:.2f})")
            self._refresh_all()
            messagebox.showinfo(
                "Obstacle Added",
                f"Added obstacle polygon, area={polygon.area:.2f}. Planner rebuilt.",
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
            self.last_action_var.set(f"Obstacle removed: idx={idx}")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Obstacle Remove Error", str(exc))

    def run(self) -> None:
        self.root.mainloop()


def run_gui(cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
    try:
        app = MilpGuiApp(cfg=cfg)
    except tk.TclError as exc:
        raise RuntimeError(
            "Failed to start GUI. A graphical display is required (X11/Wayland/macOS window server)."
        ) from exc
    app.run()
