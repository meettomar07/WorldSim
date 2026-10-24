"""Tests for WorldSim Phase 15.0 Live 2D Web Viewer & Simulation Interface."""

import json
import urllib.request
import pytest

from worldsim.web_viewer import (
    LiveViewerServer,
    SimulationSession,
)


def test_session_init_and_default_scenario():
    """Verify default initialization loads D01 with correct entities and dimensions."""
    session = SimulationSession(default_scenario="D01", default_seed=42)
    assert session.scenario_id == "D01"
    assert session.width == 8
    assert session.height == 5
    assert session.step_count == 0
    assert len(session.active_agent_ids) == 2
    assert "agent_w" in session.active_agent_ids
    assert "agent_e" in session.active_agent_ids
    assert session.sim is not None
    assert len(session.sim.state.dynamic_obstacles) == 2


def test_session_snapshot_schema():
    """Verify complete snapshot schema for real-time visualization."""
    session = SimulationSession(default_scenario="D01", default_seed=42)
    snap = session.get_snapshot()

    # Top-level keys
    assert "scenario" in snap
    assert "tick" in snap
    assert "overall_completed" in snap
    assert "walls" in snap
    assert "resources" in snap
    assert "dynamic_obstacles" in snap
    assert "agents" in snap
    assert "metrics" in snap
    assert "recent_events" in snap

    # Scenario metadata
    assert snap["scenario"]["id"] == "D01"
    assert snap["scenario"]["width"] == 8
    assert snap["scenario"]["height"] == 5
    assert snap["tick"] == 0

    # Agent record schema
    assert "agent_w" in snap["agents"]
    ag = snap["agents"]["agent_w"]
    assert ag["id"] == "agent_w"
    assert ag["x"] == 0
    assert ag["y"] == 2
    assert "energy" in ag
    assert "max_energy" in ag
    assert "decision_attribution" in ag
    assert "observation_cells" in ag
    assert "trail" in ag
    assert isinstance(ag["observation_cells"], list)
    assert len(ag["observation_cells"]) > 0

    # Dynamic obstacles schema
    assert len(snap["dynamic_obstacles"]) == 2
    obs = snap["dynamic_obstacles"][0]
    assert "id" in obs
    assert "x" in obs
    assert "y" in obs
    assert "vx" in obs
    assert "vy" in obs
    assert "pattern" in obs

    # Telemetry metrics schema
    metrics = snap["metrics"]
    assert metrics["active_agents"] == 2
    assert metrics["reached_agents"] == 0
    assert metrics["exhausted_agents"] == 0
    assert metrics["total_agents"] == 2
    assert metrics["completion_rate_pct"] == 0.0
    assert metrics["physical_collisions"] == 0
    assert "attempted_conflicts" in metrics
    assert "vertex_collisions" in metrics
    assert "swap_collisions" in metrics
    assert "occupied_collisions" in metrics
    assert "dynamic_obstacle_collisions" in metrics


def test_strict_physical_collision_invariant():
    """Verify strict physical collision invariant (0 illegal overlaps) and conflict accounting."""
    session = SimulationSession(default_scenario="D01", default_seed=42)
    # Step simulation
    for _ in range(5):
        snap = session.step()
        # Physical collision invariant: engine guarantees zero overlap
        assert snap["metrics"]["physical_collisions"] == 0
        # No two agents or dynamic obstacles ever occupy the exact same cell
        agent_positions = [
            (a["x"], a["y"]) for a in snap["agents"].values() if a["active"]
        ]
        assert len(agent_positions) == len(set(agent_positions))


def test_session_step_advancement():
    """Verify one-step advancement through visualization interface."""
    session = SimulationSession(default_scenario="D01", default_seed=42)
    assert session.step_count == 0

    snap1 = session.step()
    assert session.step_count == 1
    assert snap1["tick"] == 1
    assert len(snap1["agents"]["agent_w"]["trail"]) == 2

    snap2 = session.step()
    assert session.step_count == 2
    assert snap2["tick"] == 2
    assert len(snap2["agents"]["agent_w"]["trail"]) == 3


def test_session_reset():
    """Verify simulation reset restores initial state, energy, and tick 0."""
    session = SimulationSession(default_scenario="D01", default_seed=42)
    for _ in range(4):
        session.step()
    assert session.step_count == 4

    snap_reset = session.reset()
    assert session.step_count == 0
    assert snap_reset["tick"] == 0
    assert snap_reset["agents"]["agent_w"]["x"] == 0
    assert snap_reset["agents"]["agent_w"]["y"] == 2
    assert snap_reset["agents"]["agent_w"]["energy"] == 10
    assert len(snap_reset["agents"]["agent_w"]["trail"]) == 1


def test_session_scenario_selection():
    """Verify dynamic loading of D04, U06, and U14 scenarios."""
    session = SimulationSession(default_scenario="D01", default_seed=42)

    # 1. Test D04 Resource Approach
    snap_d04 = session.load_scenario("D04", seed=42)
    assert snap_d04["scenario"]["id"] == "D04"
    assert snap_d04["scenario"]["width"] == 7
    assert snap_d04["scenario"]["height"] == 7
    assert len(snap_d04["resources"]) == 1
    assert snap_d04["resources"][0]["x"] == 3
    assert snap_d04["resources"][0]["y"] == 3
    assert snap_d04["resources"][0]["recharge_amount"] == 6

    # 2. Test U14 Multi-Corner Blind Junction
    snap_u14 = session.load_scenario("U14", seed=42)
    assert snap_u14["scenario"]["id"] == "U14"
    assert snap_u14["scenario"]["width"] == 7
    assert snap_u14["scenario"]["height"] == 7
    # Corner wall pillars at (2,2), (4,2), (2,4), (4,4)
    wall_coords = {(w["x"], w["y"]) for w in snap_u14["walls"]}
    assert (2, 2) in wall_coords
    assert (4, 2) in wall_coords
    assert (2, 4) in wall_coords
    assert (4, 4) in wall_coords

    # 3. Test U06 8-Agent Staggered Quadrant Mesh
    snap_u06 = session.load_scenario("U06", seed=42)
    assert snap_u06["scenario"]["id"] == "U06"
    assert snap_u06["metrics"]["total_agents"] == 8
    assert len(snap_u06["resources"]) == 4


def test_session_policy_modes():
    """Verify execution across all supported policy modes."""
    policies_to_test = [
        "hybrid_comm",
        "remediated_comm",
        "learned_comm",
        "learned_no_comm",
        "heuristic",
        "centralized",
    ]
    for pol_mode in policies_to_test:
        session = SimulationSession(default_scenario="D01", default_policy=pol_mode, default_seed=42)
        assert session.policy_mode == pol_mode
        snap = session.step()
        assert snap["tick"] == 1
        for ag in snap["agents"].values():
            assert ag["action"] in {"up", "down", "left", "right", "recharge"}


def test_http_server_endpoints():
    """Verify HTTP API endpoints (GET /api/state, POST /api/step, etc.)."""
    session = SimulationSession(default_scenario="D01", default_seed=42)
    server = LiveViewerServer(host="127.0.0.1", port=0, session=session)
    server.start()

    base_url = f"http://127.0.0.1:{server.port}"

    try:
        # GET /api/health
        with urllib.request.urlopen(f"{base_url}/api/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert data["scenario"] == "D01"

        # GET /api/scenarios
        with urllib.request.urlopen(f"{base_url}/api/scenarios") as resp:
            assert resp.status == 200
            scenarios = json.loads(resp.read().decode())
            assert isinstance(scenarios, list)
            scen_ids = [s["id"] for s in scenarios]
            assert "D01" in scen_ids
            assert "D04" in scen_ids
            assert "U14" in scen_ids

        # GET /api/policies
        with urllib.request.urlopen(f"{base_url}/api/policies") as resp:
            assert resp.status == 200
            policies = json.loads(resp.read().decode())
            assert any(p["id"] == "hybrid_comm" for p in policies)

        # GET /api/state
        with urllib.request.urlopen(f"{base_url}/api/state") as resp:
            assert resp.status == 200
            snap = json.loads(resp.read().decode())
            assert snap["scenario"]["id"] == "D01"
            assert snap["tick"] == 0

        # POST /api/step
        req_step = urllib.request.Request(f"{base_url}/api/step", data=b"{}", method="POST")
        with urllib.request.urlopen(req_step) as resp:
            assert resp.status == 200
            snap_step = json.loads(resp.read().decode())
            assert snap_step["tick"] == 1

        # POST /api/reset
        req_reset = urllib.request.Request(f"{base_url}/api/reset", data=b"{}", method="POST")
        with urllib.request.urlopen(req_reset) as resp:
            assert resp.status == 200
            snap_reset = json.loads(resp.read().decode())
            assert snap_reset["tick"] == 0

        # POST /api/select
        select_payload = json.dumps({"scenario_id": "D04", "seed": 42, "policy_mode": "hybrid_comm"}).encode()
        req_select = urllib.request.Request(
            f"{base_url}/api/select",
            data=select_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_select) as resp:
            assert resp.status == 200
            snap_select = json.loads(resp.read().decode())
            assert snap_select["scenario"]["id"] == "D04"
            assert snap_select["tick"] == 0

        # GET / (Index HTML)
        with urllib.request.urlopen(f"{base_url}/") as resp:
            assert resp.status == 200
            html_text = resp.read().decode("utf-8")
            assert "WorldSim" in html_text
            assert "<canvas" in html_text

    finally:
        server.stop()


def test_authoritative_determinism_parity():
    """Verify that running through SimulationSession produces exact bit-for-bit identical trajectory."""
    # Run 10 ticks in session 1
    session1 = SimulationSession(default_scenario="D01", default_policy="hybrid_comm", default_seed=42)
    traj1 = []
    for _ in range(10):
        snap = session1.step()
        traj1.append({aid: (ag["x"], ag["y"], ag["energy"]) for aid, ag in snap["agents"].items()})

    # Run 10 ticks in session 2 with identical seed
    session2 = SimulationSession(default_scenario="D01", default_policy="hybrid_comm", default_seed=42)
    traj2 = []
    for _ in range(10):
        snap = session2.step()
        traj2.append({aid: (ag["x"], ag["y"], ag["energy"]) for aid, ag in snap["agents"].items()})

    assert traj1 == traj2


def test_u14_full_episode_completion():
    """Verify full multi-agent episode completion and deadlock recovery on U14."""
    session = SimulationSession(default_scenario="U14", default_policy="hybrid_comm", default_seed=42)
    assert session.scenario_id == "U14"
    assert len(session.wall_positions) == 4
    for _ in range(30):
        snap = session.step()
        if snap["overall_completed"]:
            break
    assert session.overall_completed
    assert snap["metrics"]["reached_agents"] == 4
    assert snap["metrics"]["physical_collisions"] == 0

