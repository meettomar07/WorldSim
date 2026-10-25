"""Runner script for WorldSim Phase 13.4 Independent Validation & Robustness Audit."""

import sys
from worldsim.multi_agent_experiment_phase13_4 import run_all_phase13_4_experiments

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results_phase13_4.json"
    results = run_all_phase13_4_experiments(out_file)
    print(f"\nPhase 13.4 benchmark completed successfully with {len(results['scenarios'])} scenarios.")
