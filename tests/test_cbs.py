"""Unit and Integration Tests for Conflict-Based Search (CBS) MAPF Solver."""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.cbs import (
    CBSSolver,
    CBSPolicy,
    Conflict,
    EdgeConstraint,
    SpaceTimeAStar,
    VertexConstraint,
)
from worldsim.models import Position
from worldsim.multi_agent import AgentConfig, MultiAgentObservation, MultiAgentWorldConfig


class TestConflictBasedSearch(unittest.TestCase):
    """Test suite verifying CBS low-level, high-level, and policy execution."""

    def test_spacetime_astar_single_agent(self) -> None:
        """Verify space-time A* finds optimal shortest path without constraints."""
        solver = SpaceTimeAStar(width=5, height=5, walls=set())
        start = Position(0, 0)
        goal = Position(3, 3)
        path = solver.plan("agent_1", start, goal, constraints=set())
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertEqual(len(path) - 1, 6)  # Manhattan distance |3-0| + |3-0| = 6

    def test_spacetime_astar_vertex_constraint(self) -> None:
        """Verify space-time A* detours around a forbidden space-time vertex."""
        solver = SpaceTimeAStar(width=5, height=5, walls=set())
        start = Position(0, 0)
        goal = Position(2, 0)
        # Forbid (1, 0) at t=1
        constraint = VertexConstraint(agent_id="agent_1", position=Position(1, 0), time_step=1)
        path = solver.plan("agent_1", start, goal, constraints={constraint})
        self.assertIsNotNone(path)
        # Must not be at (1, 0) at t=1
        self.assertNotEqual(path[1], Position(1, 0))
        self.assertEqual(path[-1], goal)

    def test_cbs_two_agent_vertex_conflict(self) -> None:
        """Verify CBS resolves a vertex conflict where two agents cross paths at (1, 1)."""
        # Agent 1 moves (0, 1) -> (2, 1)
        # Agent 2 moves (1, 0) -> (1, 2)
        # Both would reach (1, 1) at t=1
        solver = CBSSolver(width=3, height=3)
        starts = {"a1": Position(0, 1), "a2": Position(1, 0)}
        goals = {"a1": Position(2, 1), "a2": Position(1, 2)}

        solution = solver.solve(starts, goals)
        self.assertTrue(solution.success)
        self.assertEqual(solution.status, "OPTIMAL")

        # Verify no vertex conflict at any time step
        p1 = solution.paths["a1"]
        p2 = solution.paths["a2"]
        max_t = max(len(p1), len(p2))
        for t in range(max_t):
            pos1 = p1[t] if t < len(p1) else p1[-1]
            pos2 = p2[t] if t < len(p2) else p2[-1]
            self.assertNotEqual(pos1, pos2, f"Vertex conflict detected at t={t}")

    def test_cbs_head_on_edge_swap_conflict(self) -> None:
        """Verify CBS resolves an edge-swap conflict in a corridor with a sidestep."""
        # Grid 3x2: row 0 is a corridor, row 1 has an alcove at (1, 1)
        # Agent 1 starts (0, 0), goal (2, 0)
        # Agent 2 starts (2, 0), goal (0, 0)
        # Walls at (0, 1) and (2, 1) -> only (1, 1) is open for sidestep
        walls = {Position(0, 1), Position(2, 1)}
        solver = CBSSolver(width=3, height=2, walls=walls)
        starts = {"a1": Position(0, 0), "a2": Position(2, 0)}
        goals = {"a1": Position(2, 0), "a2": Position(0, 0)}

        solution = solver.solve(starts, goals)
        self.assertTrue(solution.success)
        self.assertEqual(solution.status, "OPTIMAL")

        # Verify no edge swap conflict between a1 and a2
        p1 = solution.paths["a1"]
        p2 = solution.paths["a2"]
        max_t = max(len(p1), len(p2))
        for t in range(1, max_t):
            u1 = p1[t - 1] if (t - 1) < len(p1) else p1[-1]
            v1 = p1[t] if t < len(p1) else p1[-1]
            u2 = p2[t - 1] if (t - 1) < len(p2) else p2[-1]
            v2 = p2[t] if t < len(p2) else p2[-1]
            self.assertFalse(u1 == v2 and v1 == u2 and u1 != v1, f"Edge swap detected at t={t}")

    def test_cbs_unsolvable_narrow_corridor_without_alcove(self) -> None:
        """Verify CBS recognizes unsolvable deadlock when two agents face off in a 1-cell corridor with no detour."""
        # 3x1 corridor, no sidesteps possible
        solver = CBSSolver(width=3, height=1, max_time=15, max_expansions=50)
        starts = {"a1": Position(0, 0), "a2": Position(2, 0)}
        goals = {"a1": Position(2, 0), "a2": Position(0, 0)}

        solution = solver.solve(starts, goals)
        self.assertFalse(solution.success)
        self.assertIn(solution.status, ("UNSOLVABLE", "MAX_EXPANSIONS", "TIMEOUT"))

    def test_cbs_dynamic_obstacle_schedule_avoidance(self) -> None:
        """Verify CBS plans around dynamic obstacle positions scheduled in time."""
        solver = CBSSolver(width=5, height=3)
        starts = {"a1": Position(0, 1)}
        goals = {"a1": Position(4, 1)}
        # A moving hazard passes through (2, 1) at t=2
        dynamic_schedule = {2: {Position(2, 1)}}

        solution = solver.solve(starts, goals, dynamic_obstacles_schedule=dynamic_schedule)
        self.assertTrue(solution.success)
        path = solution.paths["a1"]
        # Agent must not be at (2, 1) at t=2
        if len(path) > 2:
            self.assertNotEqual(path[2], Position(2, 1))

    def test_cbs_policy_action_generation(self) -> None:
        """Verify CBSPolicy translates path sequence into Action enums."""
        path = [Position(0, 0), Position(1, 0), Position(1, 1), Position(1, 1)]
        policy = CBSPolicy(agent_id="a1", planned_path=path)

        obs1 = MultiAgentObservation(
            agent_id="a1",
            entity_position=Position(0, 0),
            cells=(),
            energy=10,
            target=Position(1, 1),
        )
        self.assertEqual(policy.choose_action(obs1), Action.RIGHT)

        obs2 = MultiAgentObservation(
            agent_id="a1",
            entity_position=Position(1, 0),
            cells=(),
            energy=9,
            target=Position(1, 1),
        )
        self.assertEqual(policy.choose_action(obs2), Action.DOWN)


if __name__ == "__main__":
    unittest.main()
