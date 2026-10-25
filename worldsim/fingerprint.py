"""WorldSim Environment Fingerprinting.

Produces deterministic, canonical cryptographic fingerprints (SHA-256) of
simulation configurations to verify whether random seeds genuinely alter
environments or reproduce identical instances.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from worldsim.models import Position, Resource
from worldsim.multi_agent import AgentConfig, MultiAgentWorldConfig


def compute_environment_fingerprint(
    config: MultiAgentWorldConfig | dict[str, Any],
    walls: Any = None,
) -> str:
    """Compute an immutable 16-character hex fingerprint of the environment layout.

    Captures:
    - Grid dimensions (width, height)
    - Static wall positions (sorted)
    - Dynamic obstacles (sorted by id, initial position, velocities, waypoints)
    - Resources (sorted by position, recharge amount, consumable flag)
    - Agents (sorted by id, start coordinates, target coordinates, initial energy)
    - Max energy & movement cost
    """
    raw_walls = walls
    if raw_walls is None and isinstance(config, dict):
        raw_walls = config.get("walls")

    walls_list: list[tuple[int, int]] = []
    if raw_walls:
        for p in raw_walls:
            if hasattr(p, "x") and hasattr(p, "y"):
                walls_list.append((p.x, p.y))
            elif isinstance(p, (tuple, list)) and len(p) >= 2:
                walls_list.append((p[0], p[1]))
    walls_list.sort()

    if isinstance(config, MultiAgentWorldConfig):
        cfg_dict: dict[str, Any] = {
            "width": config.width,
            "height": config.height,
            "max_energy": config.max_energy,
            "movement_energy_cost": config.movement_energy_cost,
            "walls": walls_list,
            "agents": sorted(
                [
                    {
                        "id": a.id,
                        "start": (a.initial_x, a.initial_y),
                        "target": (a.target.x, a.target.y) if a.target else None,
                        "energy": a.initial_energy,
                    }
                    for a in config.agents
                ],
                key=lambda x: str(x["id"]),
            ),
            "resources": sorted(
                [
                    {
                        "pos": (r.position.x, r.position.y),
                        "amount": r.recharge_amount,
                        "consumable": r.consumable,
                    }
                    for r in config.resources
                ],
                key=lambda x: (x["pos"][0], x["pos"][1]),
            ),
            "dynamic_obstacles": sorted(
                [
                    {
                        "id": o.id,
                        "pos": (o.initial_x, o.initial_y),
                        "velocity": getattr(o, "velocity", (0, 0)),
                        "pattern": o.pattern,
                        "waypoints": [(p.x, p.y) for p in o.waypoints] if o.waypoints else [],
                    }
                    for o in config.dynamic_obstacles
                ],
                key=lambda x: str(x["id"]),
            ),
        }
    else:
        # dict representation
        cfg_dict = {
            "width": config.get("width"),
            "height": config.get("height"),
            "max_energy": config.get("max_energy"),
            "walls": walls_list,
            "agents": sorted(
                [
                    {
                        "id": a.get("id"),
                        "start": (a.get("initial_x"), a.get("initial_y")),
                        "target": (a.get("target_x"), a.get("target_y")),
                        "energy": a.get("initial_energy"),
                    }
                    for a in config.get("agents", [])
                ],
                key=lambda x: str(x["id"]),
            ),
            "resources": sorted(
                [
                    {
                        "pos": (r.get("x"), r.get("y")),
                        "amount": r.get("recharge_amount"),
                        "consumable": r.get("consumable"),
                    }
                    for r in config.get("resources", [])
                ],
                key=lambda x: (x["pos"][0], x["pos"][1]),
            ),
            "dynamic_obstacles": sorted(
                [
                    {
                        "id": o.get("id"),
                        "pos": (o.get("initial_x"), o.get("initial_y")),
                        "velocity": o.get("velocity", (o.get("vx", 0), o.get("vy", 0))),
                    }
                    for o in config.get("dynamic_obstacles", [])
                ],
                key=lambda x: str(x["id"]),
            ),
        }

    canonical_json = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]
