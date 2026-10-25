"""WorldSim Scenario Registry with Strict Train/Val/Test Split Isolation.

Enforces sound experimental separation:
- TRAIN: Environments used to optimize feature weights and policy representations.
- VAL: Validation environments used for hyperparameter selection and remediation tuning.
- TEST: Genuinely held-out test environments reserved strictly for final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

from worldsim.fingerprint import compute_environment_fingerprint
from worldsim.models import Position, Resource
from worldsim.multi_agent import AgentConfig, MultiAgentWorldConfig
from worldsim.scenario_generator import generate_heldout_scenario


class Split(str, Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "held_out_test"


@dataclass(frozen=True)
class ScenarioDescriptor:
    scenario_id: str
    scenario_name: str
    split: Split
    family: str
    description: str
    is_procedural: bool = False
    default_env_seed: int = 42


# Master Scenario Registry
_REGISTRY: dict[str, ScenarioDescriptor] = {
    # --------------------------------------------------------------------------
    # 1. TRAINING SPLIT (Used for RL representation learning)
    # --------------------------------------------------------------------------
    "TRAIN_ENV1": ScenarioDescriptor(
        scenario_id="TRAIN_ENV1",
        scenario_name="Training Task 1: Linear Navigation",
        split=Split.TRAIN,
        family="Single-Task RL",
        description="4x4 grid with resource at (2,0) and target at (3,2).",
    ),
    "TRAIN_ENV2": ScenarioDescriptor(
        scenario_id="TRAIN_ENV2",
        scenario_name="Training Task 2: Perpendicular Resource",
        split=Split.TRAIN,
        family="Single-Task RL",
        description="4x4 grid with resource at (0,2) and target at (3,2).",
    ),
    "TRAIN_ENV3": ScenarioDescriptor(
        scenario_id="TRAIN_ENV3",
        scenario_name="Training Task 3: Inverted Diagonal",
        split=Split.TRAIN,
        family="Single-Task RL",
        description="4x4 grid breaking collinearity between resource and goal.",
    ),
    "TRAIN_D04": ScenarioDescriptor(
        scenario_id="TRAIN_D04",
        scenario_name="Training Task 4: Dynamic Resource Approach",
        split=Split.TRAIN,
        family="Dynamic RL",
        description="Dynamic obstacle training environment from Phase 14.4.",
    ),

    # --------------------------------------------------------------------------
    # 2. VALIDATION & TUNING SPLIT (Used for remediation and parameter tuning)
    # --------------------------------------------------------------------------
    "VAL_D01": ScenarioDescriptor(
        scenario_id="VAL_D01",
        scenario_name="Validation D01: Dual Patrol Crossing",
        split=Split.VAL,
        family="Dynamic Patrol",
        description="2 agents crossing perpendicular patrol lanes (Phase 14.3 development).",
    ),
    "VAL_D02": ScenarioDescriptor(
        scenario_id="VAL_D02",
        scenario_name="Validation D02: Diamond Intersection",
        split=Split.VAL,
        family="Dynamic Patrol",
        description="4 agents in dynamic diamond layout (Phase 14.3 development).",
    ),
    "VAL_D03": ScenarioDescriptor(
        scenario_id="VAL_D03",
        scenario_name="Validation D03: Dynamic Corridor Mesh",
        split=Split.VAL,
        family="Dynamic Bottleneck",
        description="4 agents with 4 moving hazards (Phase 14.4 tuning).",
    ),
    "VAL_U06": ScenarioDescriptor(
        scenario_id="VAL_U06",
        scenario_name="Validation U06: 8-Agent Asymmetric Bottleneck",
        split=Split.VAL,
        family="Bottleneck Mesh",
        description="8 agents in asymmetric quadrants (Phase 14.2 remediation).",
    ),
    "VAL_U14": ScenarioDescriptor(
        scenario_id="VAL_U14",
        scenario_name="Validation U14: Corridor Swap Reusable",
        split=Split.VAL,
        family="Narrow Corridor",
        description="2 agents swapping across 1-cell bottleneck (Phase 14.4.2 audit).",
    ),
    "VAL_U01": ScenarioDescriptor(
        scenario_id="VAL_U01",
        scenario_name="Validation U01: Diagonal Crossing",
        split=Split.VAL,
        family="Generalization Audit",
        description="Unseen audit scenario from Phase 13.5.",
    ),

    # --------------------------------------------------------------------------
    # 3. GENUINELY HELD-OUT TEST SPLIT (Untouched for final evaluation)
    # --------------------------------------------------------------------------
    "TEST_H01": ScenarioDescriptor(
        scenario_id="TEST_H01",
        scenario_name="Held-Out H01: Asymmetric Funnel Bottleneck",
        split=Split.TEST,
        family="Funnel Topologies",
        description="Procedural funnel constriction with variable corridor widths W in {1, 2}.",
        is_procedural=True,
    ),
    "TEST_H02": ScenarioDescriptor(
        scenario_id="TEST_H02",
        scenario_name="Held-Out H02: Double-Chamber Doorway Swap",
        split=Split.TEST,
        family="Multi-Chamber",
        description="Two rooms connected by a single doorway with asymmetric starting energies.",
        is_procedural=True,
    ),
    "TEST_H03": ScenarioDescriptor(
        scenario_id="TEST_H03",
        scenario_name="Held-Out H03: Procedural Connected Maze",
        split=Split.TEST,
        family="Maze Fields",
        description="8x8 obstacle field with guaranteed 4-corner reachability.",
        is_procedural=True,
    ),
    "TEST_H04": ScenarioDescriptor(
        scenario_id="TEST_H04",
        scenario_name="Held-Out H04: Multi-Lane Dynamic Hazard Highway",
        split=Split.TEST,
        family="Dynamic Hazards",
        description="4 agents crossing two randomized counter-flowing hazard lanes.",
        is_procedural=True,
    ),
    "TEST_H05": ScenarioDescriptor(
        scenario_id="TEST_H05",
        scenario_name="Held-Out H05: Three-Way T-Junction Convergence",
        split=Split.TEST,
        family="Junctions",
        description="3 agents meeting at a 1-cell T-junction from orthogonal directions.",
        is_procedural=True,
    ),
    "TEST_H06": ScenarioDescriptor(
        scenario_id="TEST_H06",
        scenario_name="Held-Out H06: Critical Recharge Alcove Chokepoint",
        split=Split.TEST,
        family="Energy Chokepoints",
        description="Single recharge station placed inside a dead-end alcove under tight energy budgets.",
        is_procedural=True,
    ),
}


def get_scenario_descriptor(scenario_id: str) -> ScenarioDescriptor:
    if scenario_id not in _REGISTRY:
        raise KeyError(f"Unknown scenario ID: {scenario_id}")
    return _REGISTRY[scenario_id]


def list_scenarios(split: Split | None = None) -> list[ScenarioDescriptor]:
    if split is None:
        return list(_REGISTRY.values())
    return [s for s in _REGISTRY.values() if s.split == split]


def assert_not_held_out(scenario_id: str, context: str = "Training/Tuning") -> None:
    """Safety guard ensuring held-out scenarios cannot accidentally be used in training."""
    desc = get_scenario_descriptor(scenario_id)
    if desc.split == Split.TEST:
        raise RuntimeError(
            f"DATA CONTAMINATION DETECTED! Scenario '{scenario_id}' belongs to Split.TEST "
            f"and cannot be accessed during {context}."
        )


def assert_split(
    scenario_id: str,
    allowed_splits: Sequence[Split] | Split,
    context: str = "Pipeline Execution",
) -> None:
    """Enforce that a scenario belongs strictly to the allowed split(s)."""
    desc = get_scenario_descriptor(scenario_id)
    if isinstance(allowed_splits, Split):
        allowed = {allowed_splits}
    else:
        allowed = set(allowed_splits)

    if desc.split not in allowed:
        raise RuntimeError(
            f"SPLIT ISOLATION VIOLATION! Scenario '{scenario_id}' belongs to {desc.split.name}, "
            f"which is not permitted during {context} (allowed: {[s.name for s in allowed]})."
        )


def build_scenario_instance(
    scenario_id: str,
    env_seed: int = 42,
    required_split: Split | None = None,
) -> tuple[MultiAgentWorldConfig, set[Position], str]:
    """Instantiate a scenario with an explicit environment seed, returning (config, walls, fingerprint)."""
    desc = get_scenario_descriptor(scenario_id)
    if required_split is not None and desc.split != required_split:
        raise RuntimeError(
            f"SPLIT ISOLATION VIOLATION! Requested split {required_split.name} but scenario '{scenario_id}' "
            f"is in {desc.split.name}."
        )

    if desc.is_procedural:
        family_map = {
            "TEST_H01": "H01_FUNNEL_BOTTLENECK",
            "TEST_H02": "H02_DOUBLE_CHAMBER_SWAP",
            "TEST_H03": "H03_CONNECTED_MAZE",
            "TEST_H04": "H04_DYNAMIC_HIGHWAY",
            "TEST_H05": "H05_THREE_WAY_JUNCTION",
            "TEST_H06": "H06_CRITICAL_RECHARGE_CHOKE",
        }
        family = family_map[scenario_id]
        return generate_heldout_scenario(family, env_seed)
    else:
        # Static validation/training scenarios
        from experiments.phase14_3_dynamic_obstacle_eval import build_controlled_dynamic_scenarios
        from experiments.phase13_5_unseen_audit import build_all_unseen_scenarios

        dyn_dict = {s["id"]: s for s in build_controlled_dynamic_scenarios()}
        unseen_dict = {spec.scenario_id: (spec, prep) for spec, prep in build_all_unseen_scenarios()}

        clean_id = scenario_id.replace("VAL_", "").replace("TRAIN_", "")
        if clean_id in dyn_dict:
            s = dyn_dict[clean_id]
            cfg = MultiAgentWorldConfig(
                width=s["width"],
                height=s["height"],
                agents=tuple(s["agents"]),
                resources=tuple(s.get("resources", ())),
                dynamic_obstacles=tuple(s.get("dynamic_obstacles", ())),
                max_energy=s.get("max_energy", 12),
                enable_coordination=True,
            )
            fp = compute_environment_fingerprint(cfg, walls=set())
            return cfg, set(), fp
        elif clean_id in unseen_dict:
            spec, prep = unseen_dict[clean_id]
            agents = [
                AgentConfig(
                    id=a["id"],
                    initial_x=a["initial_x"],
                    initial_y=a["initial_y"],
                    initial_energy=a["initial_energy"],
                    target=Position(a["target_x"], a["target_y"]) if a.get("target_x") is not None else None,
                )
                for a in spec.active_agents
            ]
            res = [Resource(Position(r["x"], r["y"]), r["recharge_amount"], r["consumable"]) for r in spec.resources]
            walls = {Position(p["x"], p["y"]) for p in spec.pre_positioned if p.get("status") == "exhausted"}
            cfg = MultiAgentWorldConfig(
                width=spec.width,
                height=spec.height,
                agents=tuple(agents),
                resources=tuple(res),
                max_energy=spec.max_energy,
                enable_coordination=True,
            )
            fp = compute_environment_fingerprint(cfg, walls=walls)
            return cfg, walls, fp
        elif clean_id in ("ENV1", "ENV2", "ENV3"):
            res_coords = {"ENV1": (2, 0), "ENV2": (0, 2), "ENV3": (2, 2)}[clean_id]
            cfg = MultiAgentWorldConfig(
                width=4,
                height=4,
                agents=(AgentConfig(id="agent_0", initial_x=0, initial_y=0, initial_energy=4, target=Position(3, 2)),),
                resources=(Resource(Position(res_coords[0], res_coords[1]), recharge_amount=4, consumable=True),),
                max_energy=6,
                enable_coordination=False,
            )
            fp = compute_environment_fingerprint(cfg, walls=set())
            return cfg, set(), fp
        else:
            raise KeyError(f"No factory implementation for static scenario: {scenario_id}")
