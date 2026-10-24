"""Root entry point for executing WorldSim Phase 12.2 benchmark."""

import sys
from worldsim.benchmark_phase12_2 import run_phase12_2_benchmark


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results_phase12_2.json"
    run_phase12_2_benchmark(output_path=output_path)


if __name__ == "__main__":
    main()
