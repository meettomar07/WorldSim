"""Script to run the full WorldSim Phase 13.2 benchmark suite and output benchmark_results_phase13_2.json."""

from worldsim.multi_agent_experiment_phase13_2 import run_all_phase13_2_experiments

if __name__ == "__main__":
    run_all_phase13_2_experiments("benchmark_results_phase13_2.json")
