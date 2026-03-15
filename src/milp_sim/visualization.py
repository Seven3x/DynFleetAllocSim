from __future__ import annotations

import os
from pathlib import Path
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from .entities import Task, Vehicle
from .map_utils import WorldMap


def draw_world(ax, world: WorldMap) -> None:
    bx, by = world.boundary_polygon.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.2)

    dx, dy = world.depot_polygon.exterior.xy
    ax.fill(dx, dy, color="#d9f2d9", alpha=0.85, label="depot")

    for obs in world.obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color="#6b7280", alpha=0.85)

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def _vehicle_colors(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def draw_initial_scene_on_axis(
    ax,
    world: WorldMap,
    vehicles: List[Vehicle],
    tasks: List[Task],
    title: str = "Initial Scene",
) -> None:
    ax.clear()
    draw_world(ax, world)

    colors = _vehicle_colors(len(vehicles))
    for v in vehicles:
        color = colors[v.id]
        ax.scatter(v.start_pos[0], v.start_pos[1], s=80, color=color, edgecolor="black", zorder=4)
        ax.text(v.start_pos[0] + 0.8, v.start_pos[1] + 0.8, f"V{v.id}", fontsize=8, color=color)

    for t in tasks:
        ax.scatter(t.position[0], t.position[1], s=38, color="#ef4444", edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(t.position[0] + 0.5, t.position[1] + 0.5, f"T{t.id}", fontsize=7, color="#991b1b")

    ax.set_title(title)


def draw_final_scene_on_axis(
    ax,
    world: WorldMap,
    vehicles: List[Vehicle],
    tasks: List[Task],
    title: str = "Final Allocation and Planned Trajectories",
    show_task_meta: bool = True,
    show_vehicle_sequences: bool = True,
    task_font_size: int = 6,
    vehicle_font_size: int = 8,
    label_box: bool = False,
    curve_width: float = 2.0,
) -> None:
    ax.clear()
    draw_world(ax, world)

    colors = _vehicle_colors(len(vehicles))

    color_by_status = {
        "locked": "#ef4444",
        "in_progress": "#f97316",
        "completed": "#10b981",
        "canceled": "#111827",
        "withdrawn": "#f59e0b",
        "unassigned": "#9ca3af",
        "tentative": "#ec4899",
        "verifying": "#8b5cf6",
    }
    marker_by_status = {
        "canceled": "x",
    }

    for t in tasks:
        c = color_by_status.get(t.status, "#fca5a5")
        mk = marker_by_status.get(t.status, "o")
        if mk == "o":
            ax.scatter(
                t.position[0],
                t.position[1],
                s=36,
                color=c,
                marker=mk,
                edgecolor="white",
                linewidth=0.4,
            )
        else:
            ax.scatter(
                t.position[0],
                t.position[1],
                s=36,
                color=c,
                marker=mk,
                linewidth=0.8,
            )
        label = f"T{t.id}"
        if show_task_meta:
            if t.assigned_vehicle is not None:
                label += f"/V{t.assigned_vehicle}"
            label += f"/{t.status[:3]}"

        dx = 0.25 + ((t.id % 3) - 1) * 0.18
        dy = 0.25 + (((t.id // 3) % 3) - 1) * 0.14
        bbox = (
            dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.72, edgecolor="none")
            if label_box
            else None
        )
        ax.text(
            t.position[0] + dx,
            t.position[1] + dy,
            label,
            fontsize=task_font_size,
            bbox=bbox,
        )

    for v in vehicles:
        color = colors[v.id]
        ax.scatter(v.start_pos[0], v.start_pos[1], s=90, color=color, edgecolor="black", zorder=4)
        ax.text(v.start_pos[0] + 0.8, v.start_pos[1] + 0.8, f"V{v.id}", fontsize=vehicle_font_size, color=color)

        pts = v.route_points
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=curve_width, alpha=0.95)

        if show_vehicle_sequences and v.task_sequence:
            seq_text = "->".join(f"T{tid}" for tid in v.task_sequence)
            end_pt = pts[-1]
            ax.text(
                end_pt[0] + 0.8,
                end_pt[1] - 0.8,
                seq_text,
                fontsize=max(6, vehicle_font_size - 1),
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.68, edgecolor="none"),
            )

    ax.set_title(title)


def plot_initial_scene(
    world: WorldMap,
    vehicles: List[Vehicle],
    tasks: List[Task],
    save_path: Path,
    dpi: int,
    fig_size: tuple,
) -> None:
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    draw_initial_scene_on_axis(ax=ax, world=world, vehicles=vehicles, tasks=tasks, title="Initial Scene")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_final_scene(
    world: WorldMap,
    vehicles: List[Vehicle],
    tasks: List[Task],
    save_path: Path,
    dpi: int,
    fig_size: tuple,
) -> None:
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    draw_final_scene_on_axis(
        ax=ax,
        world=world,
        vehicles=vehicles,
        tasks=tasks,
        title="Final Allocation and Planned Trajectories",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
