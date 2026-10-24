"""Root entry point for executing WorldSim Phase 13 Multi-Agent benchmark."""

import sys
from worldsim.multi_agent_experiment import run_all_multi_agent_experiments


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results_phase13.json"
    run_all_multi_agent_experiments(output_path=output_path)


if __name__ == "__main__":
    main()
