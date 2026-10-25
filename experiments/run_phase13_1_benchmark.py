"""Runner script for WorldSim Phase 13.1 Benchmark Suite.

Executes all 10 Phase 13.1 controlled experiments and ablations, reporting
quantitative metrics and exporting results to benchmark_results_phase13_1.json.
"""

from worldsim.multi_agent_experiment_phase13_1 import run_all_phase13_1_experiments

if __name__ == "__main__":
    run_all_phase13_1_experiments("benchmark_results_phase13_1.json")
