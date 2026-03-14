from __future__ import annotations

import shlex

from .config import DEFAULT_CONFIG, SimulationConfig
from .session import SimulationSession


def _print_help() -> None:
    print("Commands:")
    print("  help                              Show this help")
    print("  status                            Print fleet/task summary")
    print("  tasks [status]                    List tasks (optionally filter by status)")
    print("  add x y demand [task_id]          Add task at position and re-auction")
    print("  add_random [demand]               Add random valid task and re-auction")
    print("  cancel task_id                    Cancel task and re-auction")
    print("  reset                             Rebuild a fresh scenario from seed")
    print("  undo                              Undo last mutating action")
    print("  plot [filename]                   Save current figure to outputs/")
    print("  export_logs [prefix]              Export coordination/verification logs")
    print("  logs [n]                          Show last n verification/coordination logs")
    print("  quit / exit                       Exit console")


class InteractiveConsole:
    def __init__(self, cfg: SimulationConfig) -> None:
        self.session = SimulationSession(cfg)

    def _print_status(self) -> None:
        print(self.session.format_status_text())

    def _print_tasks(self, status_filter: str | None) -> None:
        text = self.session.format_tasks_text(status_filter=status_filter, limit=5000)
        print(text if text else "(no tasks)")

    def _add_task(self, x: float, y: float, demand: int, task_id: int | None) -> None:
        task = self.session.add_task(x=x, y=y, demand=demand, task_id=task_id)
        print(f"added T{task.id} at ({task.position[0]:.2f}, {task.position[1]:.2f}) demand={task.demand}")

    def _add_random(self, demand: int | None) -> None:
        task = self.session.add_random_task(demand=demand)
        print(
            f"added random T{task.id} at ({task.position[0]:.2f}, {task.position[1]:.2f}) "
            f"demand={task.demand}"
        )

    def _cancel_task(self, task_id: int) -> None:
        self.session.cancel_task(task_id=task_id)
        print(f"canceled T{task_id}")

    def _plot(self, filename: str | None) -> None:
        path = self.session.save_snapshot(filename=filename)
        print(f"saved plot: {path}")

    def _export_logs(self, prefix: str | None) -> None:
        coord_path, verify_path = self.session.export_logs(prefix=prefix)
        print(f"saved logs: {coord_path} , {verify_path}")

    def _logs(self, n: int) -> None:
        print(self.session.format_logs_text(n=n))

    def handle_command(self, line: str) -> bool:
        tokens = shlex.split(line)
        if not tokens:
            return True

        cmd = tokens[0].lower()

        try:
            if cmd in {"quit", "exit"}:
                return False
            if cmd == "help":
                _print_help()
            elif cmd == "status":
                self._print_status()
            elif cmd == "tasks":
                self._print_tasks(tokens[1] if len(tokens) > 1 else None)
            elif cmd == "add":
                if len(tokens) not in {4, 5}:
                    raise ValueError("usage: add x y demand [task_id]")
                x = float(tokens[1])
                y = float(tokens[2])
                demand = int(tokens[3])
                task_id = int(tokens[4]) if len(tokens) == 5 else None
                self._add_task(x=x, y=y, demand=demand, task_id=task_id)
            elif cmd == "add_random":
                demand = int(tokens[1]) if len(tokens) > 1 else None
                self._add_random(demand=demand)
            elif cmd == "cancel":
                if len(tokens) != 2:
                    raise ValueError("usage: cancel task_id")
                self._cancel_task(task_id=int(tokens[1]))
            elif cmd == "reset":
                self.session.reset()
                print("scenario reset done")
            elif cmd == "undo":
                self.session.undo()
                print("undo done")
            elif cmd == "plot":
                self._plot(filename=tokens[1] if len(tokens) > 1 else None)
            elif cmd == "export_logs":
                self._export_logs(prefix=tokens[1] if len(tokens) > 1 else None)
            elif cmd == "logs":
                n = int(tokens[1]) if len(tokens) > 1 else 5
                self._logs(n=n)
            else:
                print(f"unknown command: {cmd}")
                _print_help()
        except Exception as exc:
            print(f"error: {exc}")

        return True


def run_interactive(cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
    console = InteractiveConsole(cfg)
    print("Interactive console started. Type `help` for commands.")
    console._print_status()

    while True:
        try:
            line = input("milp> ")
        except EOFError:
            print("exit")
            break
        if not console.handle_command(line):
            break
