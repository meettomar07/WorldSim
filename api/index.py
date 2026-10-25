"""Vercel Serverless Function entry point for WorldSim Live 2D Viewer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import sys
import traceback
import urllib.parse

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

_SHARED_SESSION = None


def get_session():
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        from worldsim.web_viewer import SimulationSession
        _SHARED_SESSION = SimulationSession(default_scenario="D01", default_policy="heuristic", default_seed=42)
    return _SHARED_SESSION


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Python request handler."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def _get_target_path(self) -> str:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if "endpoint" in qs:
            return "/api/" + qs["endpoint"][0]
        matched = self.headers.get("x-matched-path") or self.headers.get("x-vercel-matched-path")
        if matched and matched.startswith("/api/"):
            return matched.split("?")[0]
        return parsed.path

    def do_GET(self) -> None:
        try:
            path = self._get_target_path()

            # Lightweight probe endpoint
            if path == "/api/test":
                data = json.dumps({
                    "status": "ok",
                    "python": sys.version,
                    "cwd": os.getcwd(),
                    "root_dir": str(root_dir),
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
                return

            if path in {"/", "/index.html"}:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._send_cors()
                self.end_headers()
                html_path = root_dir / "worldsim" / "static" / "index.html"
                if not html_path.exists():
                    html_path = root_dir / "public" / "index.html"
                self.wfile.write(html_path.read_bytes())
                return

            session = get_session()

            if path == "/api/state":
                data = json.dumps(session.get_snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/scenarios":
                data = json.dumps(session.get_available_scenarios()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/policies":
                data = json.dumps(session.get_available_policies()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/health":
                data = json.dumps({
                    "status": "ok",
                    "tick": session.step_count,
                    "scenario": session.scenario_id,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/favicon.ico":
                self.send_response(204)
                self._send_cors()
                self.end_headers()
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "not_found", "path": path, "raw": self.path}).encode("utf-8"))
        except Exception as e:
            err_data = json.dumps({
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(err_data)

    def do_POST(self) -> None:
        try:
            path = self._get_target_path()
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                payload = {}

            session = get_session()

            if path == "/api/step":
                data = json.dumps(session.step()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/reset":
                data = json.dumps(session.reset()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/select":
                scen_id = payload.get("scenario", "D01")
                pol = payload.get("policy", "heuristic")
                seed = int(payload.get("seed", 42))
                data = json.dumps(session.load_scenario(scen_id, seed, pol)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            err_data = json.dumps({
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(err_data)
