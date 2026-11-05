"""Vercel Serverless Function entry point for WorldSim Live 2D Viewer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any
import urllib.parse

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

_SHARED_SESSION = None
_TMP_CONFIG_FILE = Path("/tmp/worldsim_session_config.json")


def _save_session_config(scenario_id: str, policy_mode: str, seed: int) -> None:
    try:
        _TMP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TMP_CONFIG_FILE.write_text(
            json.dumps({"scenario": scenario_id, "policy": policy_mode, "seed": seed}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_saved_session_config() -> dict[str, Any]:
    try:
        if _TMP_CONFIG_FILE.exists():
            return json.loads(_TMP_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def get_session():
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        from worldsim.web_viewer import SimulationSession
        saved = _load_saved_session_config()
        default_scenario = saved.get("scenario", "D01")
        default_policy = saved.get("policy", "hybrid_comm")
        default_seed = int(saved.get("seed", 42))
        _SHARED_SESSION = SimulationSession(
            default_scenario=default_scenario,
            default_policy=default_policy,
            default_seed=default_seed,
        )
    return _SHARED_SESSION


def _sync_session_if_needed(session: Any, req_scen: str | None, req_pol: str | None, req_seed: int | None) -> None:
    """Ensure session matches requested scenario/policy/seed (guards against serverless container mismatch)."""
    target_scen = req_scen or session.scenario_id
    target_pol = req_pol or session.policy_mode
    target_seed = req_seed if req_seed is not None else session.seed

    needs_reload = False
    if req_scen is not None and req_scen != session.scenario_id:
        needs_reload = True
    if req_pol is not None and req_pol != session.policy_mode:
        needs_reload = True
    if req_seed is not None and req_seed != session.seed:
        needs_reload = True

    if needs_reload:
        session.load_scenario(target_scen, target_seed, target_pol)
        _save_session_config(target_scen, target_pol, target_seed)


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Python request handler."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, x-scenario-id, x-policy-mode, x-seed")

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

            # Check query params / headers for session synchronization
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            req_scen = self.headers.get("x-scenario-id") or qs.get("scenario", [None])[0] or qs.get("scenario_id", [None])[0]
            req_pol = self.headers.get("x-policy-mode") or qs.get("policy", [None])[0] or qs.get("policy_mode", [None])[0]
            raw_seed = self.headers.get("x-seed") or qs.get("seed", [None])[0]
            try:
                req_seed = int(raw_seed) if raw_seed is not None else None
            except (ValueError, TypeError):
                req_seed = None

            if path == "/api/state":
                _sync_session_if_needed(session, req_scen, req_pol, req_seed)
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

            # Robust payload / header extraction supporting all naming conventions
            req_scen = (
                payload.get("scenario_id")
                or payload.get("scenario")
                or self.headers.get("x-scenario-id")
            )
            req_pol = (
                payload.get("policy_mode")
                or payload.get("policy")
                or self.headers.get("x-policy-mode")
            )
            raw_seed = payload.get("seed") or self.headers.get("x-seed")
            try:
                req_seed = int(raw_seed) if raw_seed is not None else None
            except (ValueError, TypeError):
                req_seed = None

            if path == "/api/step":
                _sync_session_if_needed(session, req_scen, req_pol, req_seed)
                data = json.dumps(session.step()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/reset":
                _sync_session_if_needed(session, req_scen, req_pol, req_seed)
                data = json.dumps(session.reset()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/select":
                scen_id = req_scen or "D01"
                pol = req_pol or "hybrid_comm"
                seed = req_seed if req_seed is not None else 42
                snapshot = session.load_scenario(scen_id, seed, pol)
                _save_session_config(scen_id, pol, seed)
                data = json.dumps(snapshot).encode("utf-8")
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
