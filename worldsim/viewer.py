"""A lightweight Tkinter visualization for controlled Phase 3 actions."""

import tkinter as tk
from typing import Any

from .actions import Action
from .engine import Simulation
from .models import Observation, WorldState


class WorldViewer:
    """Displays the grid, entity, clock, and pause/resume controls."""

    cell_size = 36
    tick_interval_ms = 350

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.root = tk.Tk()
        self.root.title("WorldSim - Phase 3")
        self._job: str | None = None

        self.status = tk.StringVar()
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg="#f7f8fa")
        self.canvas.pack(padx=16, pady=(16, 8))

        controls = tk.Frame(self.root)
        controls.pack(pady=(0, 16))
        tk.Label(controls, textvariable=self.status, width=32, anchor="w").pack(side=tk.LEFT)
        self.pause_button = tk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=(12, 0))

        movement = tk.Frame(self.root)
        movement.pack(pady=(0, 16))
        tk.Button(movement, text="Up", width=7, command=lambda: self.apply_action(Action.UP)).grid(row=0, column=1)
        tk.Button(movement, text="Left", width=7, command=lambda: self.apply_action(Action.LEFT)).grid(row=1, column=0)
        tk.Button(movement, text="Recharge", width=7, command=lambda: self.apply_action(Action.RECHARGE)).grid(row=1, column=1)
        tk.Button(movement, text="Right", width=7, command=lambda: self.apply_action(Action.RIGHT)).grid(row=1, column=2)
        tk.Button(movement, text="Down", width=7, command=lambda: self.apply_action(Action.DOWN)).grid(row=2, column=1)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Up>", lambda _event: self.apply_action(Action.UP))
        self.root.bind("<Down>", lambda _event: self.apply_action(Action.DOWN))
        self.root.bind("<Left>", lambda _event: self.apply_action(Action.LEFT))
        self.root.bind("<Right>", lambda _event: self.apply_action(Action.RIGHT))
        self.root.bind("<r>", lambda _event: self.apply_action(Action.RECHARGE))
        self.root.bind("<space>", lambda _event: self.apply_action(Action.RECHARGE))
        self.render(self.simulation.state)

    def render(self, state: WorldState) -> None:
        size = self.cell_size
        self.canvas.configure(width=state.width * size, height=state.height * size)
        self.canvas.delete("all")
        self._draw_perception(self.simulation.observe())
        for x in range(state.width + 1):
            self.canvas.create_line(x * size, 0, x * size, state.height * size, fill="#d9dde3")
        for y in range(state.height + 1):
            self.canvas.create_line(0, y * size, state.width * size, y * size, fill="#d9dde3")
        # Draw resources
        for resource in state.resources:
            rx, ry = resource.position.x, resource.position.y
            r_pad = 10
            self.canvas.create_oval(
                rx * size + r_pad, ry * size + r_pad,
                (rx + 1) * size - r_pad, (ry + 1) * size - r_pad,
                fill="#10b981", outline="",
            )
        p = state.entity.position
        padding = 7
        self.canvas.create_oval(
            p.x * size + padding, p.y * size + padding,
            (p.x + 1) * size - padding, (p.y + 1) * size - padding,
            fill="#2463eb", outline="",
        )
        mode = "paused" if self.simulation.paused else "running"
        self.status.set(f"Tick {state.tick}  |  Energy {state.entity.energy}  |  {mode}")
        self.pause_button.configure(text="Resume" if self.simulation.paused else "Pause")


    def _draw_perception(self, observation: Observation) -> None:
        """Tint exactly the cells exposed by the simulation observation interface."""
        size = self.cell_size
        for cell in observation.cells:
            self.canvas.create_rectangle(
                cell.x * size,
                cell.y * size,
                (cell.x + 1) * size,
                (cell.y + 1) * size,
                fill="#dce8ff",
                outline="",
            )

    def toggle_pause(self) -> None:
        if self.simulation.paused:
            self.simulation.resume()
        else:
            self.simulation.pause()
        self.render(self.simulation.state)

    def apply_action(self, action: Action) -> None:
        """Apply a user-supplied action exclusively through Simulation.step."""
        result = self.simulation.step(action)
        self.render(result.state)

    def _advance(self) -> None:
        self.render(self.simulation.tick())
        self._job = self.root.after(self.tick_interval_ms, self._advance)

    def run(self) -> None:
        self._job = self.root.after(self.tick_interval_ms, self._advance)
        self.root.mainloop()

    def close(self) -> None:
        if self._job is not None:
            self.root.after_cancel(self._job)
        self.root.destroy()


class MultiAgentWorldViewer:
    """Displays multi-agent grid, agent positions, energy, targets, and conflicts."""

    cell_size = 36
    tick_interval_ms = 400
    agent_colors = (
        "#2563eb",  # Blue
        "#d97706",  # Amber
        "#7c3aed",  # Purple
        "#dc2626",  # Red
        "#0d9488",  # Teal
        "#db2777",  # Pink
        "#4b5563",  # Slate
        "#ca8a04",  # Yellow
    )

    def __init__(self, simulation: Any, policies: dict[str, Any] | None = None) -> None:
        self.simulation = simulation
        self.policies = policies or {}
        self.root = tk.Tk()
        self.root.title("WorldSim - Phase 13 Multi-Agent")
        self._job: str | None = None

        self.status = tk.StringVar()
        self._last_result: Any | None = None
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg="#f7f8fa")
        self.canvas.pack(padx=16, pady=(16, 8))

        controls = tk.Frame(self.root)
        controls.pack(pady=(0, 16))
        tk.Label(controls, textvariable=self.status, width=55, anchor="w").pack(side=tk.LEFT)
        self.pause_button = tk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=(12, 0))
        self.step_button = tk.Button(controls, text="Step Tick", command=self.step_tick)
        self.step_button.pack(side=tk.LEFT, padx=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.render(self.simulation.state)

    def render(self, state: Any) -> None:
        size = self.cell_size
        self.canvas.configure(width=state.width * size, height=state.height * size)
        self.canvas.delete("all")

        # 1. Grid lines
        for x in range(state.width + 1):
            self.canvas.create_line(x * size, 0, x * size, state.height * size, fill="#d9dde3")
        for y in range(state.height + 1):
            self.canvas.create_line(0, y * size, state.width * size, y * size, fill="#d9dde3")

        # 2. Resources
        for resource in state.resources:
            rx, ry = resource.position.x, resource.position.y
            r_pad = 10
            fill_col = "#10b981" if not resource.consumable else "#34d399"
            self.canvas.create_oval(
                rx * size + r_pad, ry * size + r_pad,
                (rx + 1) * size - r_pad, (ry + 1) * size - r_pad,
                fill=fill_col, outline="",
            )
            self.canvas.create_text(
                (rx + 0.5) * size, (ry + 0.5) * size,
                text=f"+{resource.recharge_amount}", fill="#064e3b", font=("Arial", 8, "bold")
            )

        # 3. Targets
        for idx, (aid, agent) in enumerate(state.agents.items()):
            col = self.agent_colors[idx % len(self.agent_colors)]
            tgt = agent.target or state.shared_target
            if tgt is not None:
                tx, ty = tgt.x, tgt.y
                self.canvas.create_rectangle(
                    tx * size + 4, ty * size + 4,
                    (tx + 1) * size - 4, (ty + 1) * size - 4,
                    outline=col, width=2
                )

        # 4. Highlight deadlocks and blocked agents
        if self._last_result is not None:
            for dl in getattr(self._last_result, "deadlock_events", ()):
                loc = dl.location
                self.canvas.create_rectangle(
                    loc.x * size + 2, loc.y * size + 2,
                    (loc.x + 1) * size - 2, (loc.y + 1) * size - 2,
                    outline="#ef4444", width=2, dash=(3, 3)
                )

        # 5. Agents
        blocked_set = set(getattr(self._last_result, "blocked_agents", ())) if self._last_result is not None else set()
        for idx, (aid, agent) in enumerate(state.agents.items()):
            col = self.agent_colors[idx % len(self.agent_colors)]
            p = agent.position
            padding = 6
            fill_col = col if agent.active else "#9ca3af"
            outline_col = "#ef4444" if aid in blocked_set else "#ffffff"
            outline_width = 3.0 if aid in blocked_set else 1.5

            self.canvas.create_oval(
                p.x * size + padding, p.y * size + padding,
                (p.x + 1) * size - padding, (p.y + 1) * size - padding,
                fill=fill_col, outline=outline_col, width=outline_width
            )
            # Label agent id & energy
            status_tag = " [B]" if aid in blocked_set else ""
            self.canvas.create_text(
                (p.x + 0.5) * size, (p.y + 0.5) * size,
                text=f"{aid}{status_tag}\nE:{agent.energy}", fill="#ffffff", font=("Arial", 7, "bold"), justify=tk.CENTER
            )

        mode = "paused" if self.simulation.paused else "running"
        active_count = sum(1 for a in state.agents.values() if a.active)
        reached_count = sum(1 for a in state.agents.values() if a.status == "reached_target")
        cols = (
            self.simulation.cumulative_vertex_collisions
            + self.simulation.cumulative_swap_collisions
            + self.simulation.cumulative_occupied_collisions
        )
        dl_count = len(getattr(self.simulation, "deadlock_events", []))
        self.status.set(f"Tick {state.tick} | Active {active_count}/{len(state.agents)} | Goals {reached_count} | Cols {cols} | DL {dl_count} | {mode}")
        self.pause_button.configure(text="Resume" if self.simulation.paused else "Pause")

    def toggle_pause(self) -> None:
        if self.simulation.paused:
            self.simulation.resume()
        else:
            self.simulation.pause()
        self.render(self.simulation.state)

    def step_tick(self) -> None:
        """Step one tick using registered policies."""
        actions: dict[str, Action] = {}
        for aid, agent in self.simulation.state.agents.items():
            if not agent.active:
                actions[aid] = Action.UP
                continue
            obs = self.simulation.observe_agent(aid)
            pol = self.policies.get(aid)
            if pol is not None and hasattr(pol, "choose_action"):
                actions[aid] = pol.choose_action(obs)
            elif pol is not None and hasattr(pol, "select_action"):
                actions[aid] = pol.select_action(obs, explore=False)
            else:
                actions[aid] = Action.UP

        result = self.simulation.step(actions)
        self._last_result = result
        self.render(result.state)

    def close(self) -> None:
        if self._job is not None:
            self.root.after_cancel(self._job)
        self.root.destroy()
