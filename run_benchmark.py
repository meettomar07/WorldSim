"""WorldSim Benchmark Runner."""

from pathlib import Path
from worldsim.benchmark import run_benchmark_audit

if __name__ == "__main__":
    out_file = Path("benchmark_results.json")
    run_benchmark_audit(out_file)
