"""WorldSim Procedural Scenario Generator for Genuinely Held-Out Evaluations.

Generates structurally diverse multi-agent grid environments across:
- Bottleneck funnels (variable width W in {1, 2}, length, and constriction point)
- Multi-chamber rooms connected by constrictions
- Connected maze/obstacle fields with guaranteed reachability
- Multi-lane dynamic hazard crossings with randomized velocities and phases
- Multi-way junctions (T-junctions, Y-crossings)
- Energy-critical recharge chokepoints

Every generator is strictly deterministic for a given `env_seed`, producing
reproducible layouts with verified reachability.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from worldsim.dynamic_obstacles import DynamicObstacleConfig
from worldsim.fingerprint import compute_environment_fingerprint
from worldsim.models import Position, Resource
from worldsim.multi_agent import AgentConfig, MultiAgentWorldConfig


def bfs_reachable_cells(width: int, height: int, walls: set[Position], start: Position) -> set[Position]:
    """Find all reachable cells from start avoiding walls."""
    if start in walls:
        return set()
    queue = [start]
    visited = {start}
    while queue:
        curr = queue.pop(0)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = curr.x + dx, curr.y + dy
            if 0 <= nx < width and 0 <= ny < height:
                p = Position(nx, ny)
                if p not in walls and p not in visited:
                    visited.add(p)
                    queue.append(p)
    return visited


def generate_heldout_scenario(
    family_id: str,
    env_seed: int,
) -> tuple[MultiAgentWorldConfig, set[Position], str]:
    """Generate a genuinely held-out test scenario based on family_id and env_seed.
    
    Returns:
    - MultiAgentWorldConfig
    - Set of wall positions
    - Canonical environment fingerprint
    """
    rng = Random(env_seed)

    if family_id == "H01_FUNNEL_BOTTLENECK":
        # Asymmetric funnel: 10x6 grid. Wide left chamber funnels into narrow corridor of width 1 or 2
        width, height = 10, 6
        corridor_w = 1 if (rng.random() < 0.5) else 2
        funnel_x = 4 + rng.randint(0, 1)  # constriction column 4 or 5
        corridor_y_start = (height - corridor_w) // 2

        walls: set[Position] = set()
        # Build constriction wall
        for y in range(height):
            if not (corridor_y_start <= y < corridor_y_start + corridor_w):
                walls.add(Position(funnel_x, y))

        # 4 agents: 2 from left moving right, 2 from right moving left
        a1_y = 1 + rng.randint(0, 1)
        a2_y = 3 + rng.randint(0, 1)
        agents = [
            AgentConfig(id="agent_w1", initial_x=0, initial_y=a1_y, initial_energy=16, target=Position(9, a1_y)),
            AgentConfig(id="agent_w2", initial_x=0, initial_y=a2_y, initial_energy=16, target=Position(9, a2_y)),
            AgentConfig(id="agent_e1", initial_x=9, initial_y=a1_y, initial_energy=16, target=Position(0, a1_y)),
            AgentConfig(id="agent_e2", initial_x=9, initial_y=a2_y, initial_energy=16, target=Position(0, a2_y)),
        ]
        resources = [Resource(Position(funnel_x + 2, corridor_y_start), recharge_amount=8, consumable=False)]
        dyn_obstacles = []
        max_e = 16

    elif family_id == "H02_DOUBLE_CHAMBER_SWAP":
        # Two rooms connected by a single doorway with asymmetric starting energies
        width, height = 9, 7
        door_x = 4
        door_y = 2 + rng.randint(0, 2)  # Doorway at row 2, 3, or 4

        walls = {Position(door_x, y) for y in range(height) if y != door_y}

        e_budget_w = 8 + rng.randint(0, 3)
        e_budget_e = 8 + rng.randint(0, 3)

        agents = [
            AgentConfig(id="agent_w", initial_x=1, initial_y=door_y, initial_energy=e_budget_w, target=Position(7, door_y)),
            AgentConfig(id="agent_e", initial_x=7, initial_y=door_y, initial_energy=e_budget_e, target=Position(1, door_y)),
        ]
        # Recharge station located in west or east room
        res_x = 1 if rng.random() < 0.5 else 7
        res_y = 1 if door_y > 2 else 5
        resources = [Resource(Position(res_x, res_y), recharge_amount=6, consumable=False)]
        dyn_obstacles = []
        max_e = 12

    elif family_id == "H03_CONNECTED_MAZE":
        # 8x8 grid with randomized obstacle density p=0.18, validated for connectivity
        width, height = 8, 8
        max_attempts = 100
        for _ in range(max_attempts):
            candidate_walls: set[Position] = set()
            for x in range(width):
                for y in range(height):
                    # Keep corners and center clear
                    if (x in (0, 7) and y in (0, 7)) or (x in (3, 4) and y in (3, 4)):
                        continue
                    if rng.random() < 0.18:
                        candidate_walls.add(Position(x, y))

            # Verify connectivity between 4 corners
            corners = [Position(0, 0), Position(7, 0), Position(0, 7), Position(7, 7)]
            reachable = bfs_reachable_cells(width, height, candidate_walls, corners[0])
            if all(c in reachable for c in corners):
                walls = candidate_walls
                break
        else:
            walls = set()

        agents = [
            AgentConfig(id="agent_nw", initial_x=0, initial_y=0, initial_energy=18, target=Position(7, 7)),
            AgentConfig(id="agent_ne", initial_x=7, initial_y=0, initial_energy=18, target=Position(0, 7)),
            AgentConfig(id="agent_sw", initial_x=0, initial_y=7, initial_energy=18, target=Position(7, 0)),
            AgentConfig(id="agent_se", initial_x=7, initial_y=7, initial_energy=18, target=Position(0, 0)),
        ]
        resources = [Resource(Position(3, 3), recharge_amount=8, consumable=False)]
        dyn_obstacles = []
        max_e = 18

    elif family_id == "H04_DYNAMIC_HIGHWAY":
        # 4 agents crossing two dynamic obstacle patrol lanes with randomized velocities and initial phases
        width, height = 8, 6
        walls = set()
        v_speed1 = 1 if rng.random() < 0.5 else -1
        v_speed2 = -v_speed1
        start_y1 = rng.randint(0, height - 1)
        start_y2 = rng.randint(0, height - 1)

        lane1_x = 2
        lane2_x = 5

        dyn_obstacles = [
            DynamicObstacleConfig(
                id="hazard_lane_1",
                initial_x=lane1_x,
                initial_y=start_y1,
                velocity=(0, v_speed1),
                pattern="patrol",
                waypoints=(Position(lane1_x, 0), Position(lane1_x, height - 1)),
            ),
            DynamicObstacleConfig(
                id="hazard_lane_2",
                initial_x=lane2_x,
                initial_y=start_y2,
                velocity=(0, v_speed2),
                pattern="patrol",
                waypoints=(Position(lane2_x, height - 1), Position(lane2_x, 0)),
            ),
        ]

        row_a1 = 1 + rng.randint(0, 1)
        row_a2 = 3 + rng.randint(0, 1)
        agents = [
            AgentConfig(id="agent_west_1", initial_x=0, initial_y=row_a1, initial_energy=14, target=Position(7, row_a1)),
            AgentConfig(id="agent_west_2", initial_x=0, initial_y=row_a2, initial_energy=14, target=Position(7, row_a2)),
            AgentConfig(id="agent_east_1", initial_x=7, initial_y=row_a1, initial_energy=14, target=Position(0, row_a1)),
            AgentConfig(id="agent_east_2", initial_x=7, initial_y=row_a2, initial_energy=14, target=Position(0, row_a2)),
        ]
        resources = []
        max_e = 14

    elif family_id == "H05_THREE_WAY_JUNCTION":
        # T-junction: Agents approach a 1-cell intersection from North, West, and East
        width, height = 7, 7
        cx, cy = 3, 3  # Junction center

        walls = set()
        # Create T-corridor: horizontal row cy, vertical column cx for y >= cy
        for x in range(width):
            for y in range(height):
                is_corridor = (y == cy) or (x == cx and y <= cy)
                if not is_corridor:
                    walls.add(Position(x, y))

        start_w = rng.randint(0, 1)
        start_e = 6 - rng.randint(0, 1)
        start_n = rng.randint(0, 1)
        agents = [
            AgentConfig(id="agent_west", initial_x=start_w, initial_y=cy, initial_energy=10, target=Position(6, cy)),
            AgentConfig(id="agent_east", initial_x=start_e, initial_y=cy, initial_energy=10, target=Position(cx, 0)),
            AgentConfig(id="agent_north", initial_x=cx, initial_y=start_n, initial_energy=10, target=Position(0, cy)),
        ]
        resources = []
        dyn_obstacles = []
        max_e = 10

    else:  # H06_CRITICAL_RECHARGE_CHOKE
        # Single recharge pad inside a dead-end alcove
        width, height = 8, 4
        alcove_x = 3 + rng.randint(0, 1)  # Symmetrical central alcove column (3 or 4)
        alcove_y = 0

        walls = set()
        # Wall separating top row except at alcove_x
        for x in range(width):
            if x != alcove_x:
                walls.add(Position(x, 1))

        # Start positions with variable offsets
        x_l = rng.randint(0, 1)
        x_r = 7 - rng.randint(0, 1)
        d_direct_l = abs(x_l - 7)
        d_direct_r = abs(x_r - 0)
        d_alcove_l = abs(x_l - alcove_x) + 2
        d_alcove_r = abs(x_r - alcove_x) + 2

        # Initial energy guarantees reachability to alcove but requires recharge for goal
        e_l = min(d_direct_l - 1, d_alcove_l + rng.randint(0, 1))
        e_r = min(d_direct_r - 1, d_alcove_r + rng.randint(0, 1))

        agents = [
            AgentConfig(id="agent_left", initial_x=x_l, initial_y=2, initial_energy=e_l, target=Position(7, 2)),
            AgentConfig(id="agent_right", initial_x=x_r, initial_y=2, initial_energy=e_r, target=Position(0, 2)),
        ]
        resources = [Resource(Position(alcove_x, alcove_y), recharge_amount=8, consumable=False)]
        dyn_obstacles = []
        max_e = 10

    cfg = MultiAgentWorldConfig(
        width=width,
        height=height,
        agents=tuple(agents),
        resources=tuple(resources),
        dynamic_obstacles=tuple(dyn_obstacles),
        max_energy=max_e,
        movement_energy_cost=1,
        sensory_radius=2.0,
        communication_radius=2.0,
        enable_coordination=True,
    )

    fingerprint = compute_environment_fingerprint(cfg, walls=walls)
    return cfg, walls, fingerprint
