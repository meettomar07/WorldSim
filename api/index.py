"""Vercel Serverless Function entry point for WorldSim Live 2D Viewer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sys
import traceback
import urllib.parse

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from worldsim.web_viewer import SimulationSession

# Shared in-memory simulation session
_SHARED_SESSION = SimulationSession(default_scenario="D01", default_policy="hybrid_comm", default_seed=42)


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Python request handler."""

    session = _SHARED_SESSION

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
            if path in {"/", "/index.html"}:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._send_cors()
                self.end_headers()
                html_path = root_dir / "worldsim" / "static" / "index.html"
                self.wfile.write(html_path.read_bytes())
            elif path == "/api/state":
                data = json.dumps(self.session.get_snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/scenarios":
                data = json.dumps(self.session.get_available_scenarios()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/policies":
                data = json.dumps(self.session.get_available_policies()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/health":
                data = json.dumps({
                    "status": "ok",
                    "tick": self.session.step_count,
                    "scenario": self.session.scenario_id,
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
            err_data = json.dumps({"error": str(e), "traceback": traceback.format_exc()}).encode("utf-8")
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

            if path == "/api/step":
                data = json.dumps(self.session.step()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/reset":
                data = json.dumps(self.session.reset()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/select":
                scen_id = payload.get("scenario", "D01")
                pol = payload.get("policy", "hybrid_comm")
                seed = int(payload.get("seed", 42))
                data = json.dumps(self.session.load_scenario(scen_id, seed, pol)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            err_data = json.dumps({"error": str(e), "traceback": traceback.format_exc()}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(err_data)
