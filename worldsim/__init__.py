"""WorldSim: artificial-life simulation through Phase 13.1 (multi-agent coordination & congestion control)."""

from .actions import Action
from .benchmark import run_benchmark_audit
from .config import WorldConfig
from .coordination import (
    DeadlockDetector,
    DeadlockEvent,
    FairnessMonitor,
    MultiAgentCoordinator,
    ResourceArbitrator,
    jains_fairness_index,
)
from .engine import Simulation
from .experience import Experience
from .experiment import ExperimentResult, run_experiment, run_policy_experiment
from .learning import EpisodeResult, EvaluationResult, LearningConfig, LearningRunner, QLearner
from .linear_learning import FeatureExtractor, LinearLearningConfig, LinearLearningRunner, LinearQLearner
from .memory import ExperienceMemory, MemoryEntry
from .dynamic_obstacles import (
    DynamicObstacleConfig,
    DynamicObstacleState,
    predict_next_obstacle_state,
    predict_obstacle_trajectory,
)
from .models import Entity, Observation, Position, Resource, StepResult, WorldState
from .multi_agent import (
    AgentConfig,
    AgentState,
    AgentStepResult,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentStepResult,
    MultiAgentWorldConfig,
    MultiAgentWorldState,
)
from .multi_agent_experiment import (
    MultiAgentExperimentMetrics,
    run_all_multi_agent_experiments,
    run_multi_agent_episode,
)
from .planning import (
    EnergyPlanResult,
    EnergyPlanningProblem,
    PlanExecutionResult,
    PlanResult,
    PlanningProblem,
    execute_energy_plan,
    execute_plan,
    plan,
    plan_energy,
)
from .policy import EnergyAwarePolicy, EnergyIgnorantPolicy, EnergyPlannerPolicy, RandomPolicy
from .robust_policy import (
    AgentMemory,
    CycleDetectionResult,
    CycleDetector,
    ExplorationController,
    RechargeAssessment,
    RechargeEvaluator,
    ResourceUtility,
    ResourceUtilityEvaluator,
    RobustResourcePolicy,
    RouteComparison,
)
from .viewer import MultiAgentWorldViewer, WorldViewer

__all__ = [
    "Action",
    "AgentConfig",
    "AgentMemory",
    "AgentState",
    "AgentStepResult",
    "CycleDetectionResult",
    "CycleDetector",
    "DeadlockDetector",
    "DeadlockEvent",
    "DynamicObstacleConfig",
    "DynamicObstacleState",
    "EnergyAwarePolicy",
    "EnergyIgnorantPolicy",
    "EnergyPlanResult",
    "EnergyPlannerPolicy",
    "EnergyPlanningProblem",
    "Entity",
    "EpisodeResult",
    "EvaluationResult",
    "Experience",
    "ExperienceMemory",
    "ExperimentResult",
    "ExplorationController",
    "FairnessMonitor",
    "FeatureExtractor",
    "LearningConfig",
    "LearningRunner",
    "LinearLearningConfig",
    "LinearLearningRunner",
    "LinearQLearner",
    "MemoryEntry",
    "MultiAgentCoordinator",
    "MultiAgentExperimentMetrics",
    "MultiAgentObservation",
    "MultiAgentSimulation",
    "MultiAgentStepResult",
    "MultiAgentWorldConfig",
    "MultiAgentWorldState",
    "MultiAgentWorldViewer",
    "Observation",
    "PlanExecutionResult",
    "PlanResult",
    "PlanningProblem",
    "Position",
    "QLearner",
    "RandomPolicy",
    "RechargeAssessment",
    "RechargeEvaluator",
    "Resource",
    "ResourceArbitrator",
    "ResourceUtility",
    "ResourceUtilityEvaluator",
    "RobustResourcePolicy",
    "RouteComparison",
    "Simulation",
    "StepResult",
    "WorldConfig",
    "WorldState",
    "WorldViewer",
    "execute_energy_plan",
    "execute_plan",
    "jains_fairness_index",
    "plan",
    "plan_energy",
    "run_all_multi_agent_experiments",
    "run_benchmark_audit",
    "run_experiment",
    "run_multi_agent_episode",
    "run_policy_experiment",
]



