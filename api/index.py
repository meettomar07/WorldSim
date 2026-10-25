"""Vercel Serverless Function entry point for WorldSim Live 2D Viewer."""

from __future__ import annotations

import http.server
from pathlib import Path
import sys

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from worldsim.web_viewer import SimulationHTTPRequestHandler, SimulationSession

# Shared in-memory simulation session
_SHARED_SESSION = SimulationSession(default_scenario="D01", default_policy="hybrid_comm", default_seed=42)


class handler(SimulationHTTPRequestHandler):
    """Vercel Serverless Python request handler."""

    def __init__(self, *args, **kwargs):
        self.session = _SHARED_SESSION
        super().__init__(*args, **kwargs)
