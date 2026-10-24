#!/usr/bin/env python3
"""WorldSim Phase 15.0 — CLI Launcher for the Real-Time 2D World Viewer.

Usage:
    python run_viewer.py [--port 8000] [--scenario D01] [--policy hybrid_comm] [--seed 42] [--no-browser]

Options:
    --host          Host interface to bind (default: 127.0.0.1)
    --port          HTTP port to bind (default: 8000)
    --scenario      Initial scenario ID:
                      * Dynamic: D01, D02, D03, D04
                      * Generalization: U01..U16 (e.g. U06, U14)
                      * Baseline: B01, B02
    --policy        Coordination policy mode:
                      * hybrid_comm (default)
                      * remediated_comm
                      * learned_comm
                      * learned_no_comm
                      * heuristic
                      * centralized
    --seed          Integer RNG seed (default: 42)
    --no-browser    Do not open the default web browser automatically
"""

import argparse
import sys
import time

from worldsim.web_viewer import launch_web_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WorldSim Live 2D World Viewer (Phase 15.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--scenario",
        default="D01",
        help="Initial scenario ID: D01..D04, U01..U16, B01..B02 (default: D01)",
    )
    parser.add_argument(
        "--policy",
        default="hybrid_comm",
        choices=["hybrid_comm", "remediated_comm", "learned_comm", "learned_no_comm", "heuristic", "centralized"],
        help="Coordination policy mode (default: hybrid_comm)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the web browser automatically")

    args = parser.parse_args()

    print("=" * 70)
    print(" WorldSim Phase 15.0 — Real-Time 2D World Viewer")
    print("=" * 70)

    try:
        server = launch_web_viewer(
            host=args.host,
            port=args.port,
            scenario=args.scenario,
            policy=args.policy,
            seed=args.seed,
            open_browser=not args.no_browser,
        )
    except OSError as e:
        print(f"Error starting server on port {args.port}: {e}")
        print("Try specifying a different port using --port <port_number>.")
        sys.exit(1)

    print("\nViewer server running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping WorldSim Viewer server...")
        server.stop()
        print("Viewer stopped cleanly.")


if __name__ == "__main__":
    main()
