"""Run script for WorldSim Phase 13.3 Multi-Agent Benchmark.

Executes all 9 benchmark scenarios across fixed, recorded random seeds,
measures overall completion and per-agent success separately, evaluates
resource fairness and starvation prevention, and exports results to
benchmark_results_phase13_3.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worldsim.multi_agent_experiment_phase13_3 import run_all_phase13_3_experiments


def main() -> None:
    output_file = PROJECT_ROOT / "benchmark_results_phase13_3.json"
    results = run_all_phase13_3_experiments(output_path=str(output_file))
    print("\nPhase 13.3 Benchmark Execution Complete.")
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()
