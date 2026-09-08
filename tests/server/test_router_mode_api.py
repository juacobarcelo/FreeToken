"""State-machine tests for POST /v1/router/mode (api_server), the runtime switch between the
stock MoE offload path and the project router.

The switch shares the cache-rebuild maintenance gate and future map, so the same three edge
paths must hold: a dispatch error reopens the gate, an HTTP timeout keeps it closed until the
scheduler's reply resolves it, and only a genuine "failed" latches. On top of that, every reply
carries the routing path actually serving, which /v1/stats and GET /v1/router/mode report.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from freetoken.message import RouterModeMsg
from freetoken.server import api_server
from freetoken.server.api_server import FrontendManager, RouterModeRequest, dispatch_router_mode
from freetoken.server.stats import router_state


class _FakeState:
    def __init__(self, send_impl, *, maintenance_state="serving", fatal_error=None, config=None):
        self.rebuild_futures: dict = {}
        self.maintenance_state = maintenance_state
        self.fatal_error = fatal_error
        self.last_rebuild = None
        self.last_router_mode = None
        self.config = config
        self._loop = None
        self._send_impl = send_impl

    async def send_one(self, msg):
        await self._send_impl(msg)


def _reply(request_id, status, *, mode="off", epoch=0, error=None):
    return SimpleNamespace(request_id=request_id, status=status, mode=mode, epoch=epoch, error=error)


def test_dispatch_sends_a_router_mode_message_and_awaits_the_reply():
    sent = []

    async def capture(msg):
        sent.append(msg)

    state = _FakeState(capture)

    async def _run():
        task = asyncio.create_task(dispatch_router_mode(state, mode="active", timeout=5.0))
        await asyncio.sleep(0)
        assert state.maintenance_state == "rebuilding"  # generation is gated meanwhile
        (request_id,) = state.rebuild_futures
        FrontendManager._resolve_router_mode(state, _reply(request_id, "ok", mode="active", epoch=1))
        return await task

    result = asyncio.run(_run())
    assert isinstance(sent[0], RouterModeMsg)
    assert (sent[0].mode, sent[0].when) == ("active", "if_idle")
    assert (result["status"], result["mode"], result["epoch"]) == ("ok", "active", 1)
    assert state.maintenance_state == "serving"
    assert state.rebuild_futures == {}
    assert router_state(state) == {"mode": "active", "epoch": 1, "configured": False}


def test_dispatch_exception_returns_to_serving():
    async def boom(_msg):
        raise RuntimeError("zmq push failed")

    state = _FakeState(boom)
    result = asyncio.run(dispatch_router_mode(state, mode="off"))
    assert result["status"] == "failed"
    assert "zmq push failed" in result["error"]
    assert state.maintenance_state == "serving"
    assert state.rebuild_futures == {}


def test_timeout_keeps_gate_closed_until_the_late_reply():
    async def ok(_msg):
        return None

    state = _FakeState(ok)
    result = asyncio.run(dispatch_router_mode(state, mode="active", timeout=0.01))
    assert result["status"] == "timeout"
    assert state.maintenance_state == "rebuilding"
    assert state.rebuild_futures == {}

    FrontendManager._resolve_router_mode(state, _reply(result["request_id"], "ok", mode="active", epoch=3))
    assert state.maintenance_state == "serving"
    assert state.last_router_mode["epoch"] == 3


def test_only_failed_latches_and_every_reply_reports_the_serving_path():
    for status in ("ok", "busy", "rejected", "unsupported"):
        state = _FakeState(None, maintenance_state="rebuilding")
        FrontendManager._resolve_router_mode(state, _reply("r1", status, mode="planning-only", epoch=2))
        assert state.maintenance_state == "serving", status
        assert router_state(state)["mode"] == "planning-only", status

    state = _FakeState(None, maintenance_state="rebuilding")
    FrontendManager._resolve_router_mode(state, _reply("r1", "failed", error="reset kernel failed"))
    assert state.maintenance_state == "failed"
    assert state.last_router_mode["error"] == "reset kernel failed"


def test_fatal_latch_outranks_a_late_reply():
    async def _run():
        state = _FakeState(None, maintenance_state="failed", fatal_error="scheduler exited")
        fut = asyncio.get_running_loop().create_future()
        state.rebuild_futures["r1"] = fut
        FrontendManager._resolve_router_mode(state, _reply("r1", "ok", mode="active", epoch=1))
        assert state.maintenance_state == "failed"
        assert fut.done() and fut.result()["mode"] == "active"
        assert state.rebuild_futures == {}

    asyncio.run(_run())


def test_watchdog_failure_wakes_a_router_mode_waiter_too():
    # The waiter lives in rebuild_futures precisely so the crash path needs no second map.
    async def ok(_msg):
        return None

    async def _run():
        state = _FakeState(ok)
        state._loop = asyncio.get_running_loop()
        task = asyncio.create_task(dispatch_router_mode(state, mode="active", timeout=30.0))
        await asyncio.sleep(0)
        assert state.rebuild_futures
        FrontendManager.fail_pending_rebuilds(state, "scheduler exited")
        result = await asyncio.wait_for(task, timeout=5.0)
        assert result == {"status": "failed", "error": "scheduler exited"}
        assert state.rebuild_futures == {}

    asyncio.run(_run())


def test_request_model_rejects_unknown_modes_and_when_values():
    assert RouterModeRequest(mode="off").when == "if_idle"
    assert RouterModeRequest(mode="planning-only").timeout == 120.0
    with pytest.raises(ValidationError):
        RouterModeRequest(mode="drain")
    with pytest.raises(ValidationError):
        RouterModeRequest(mode="active", when="drain")


def test_router_state_falls_back_to_the_startup_configuration():
    unconfigured = SimpleNamespace(config=SimpleNamespace(moe_router_config=None, moe_router_mode="active"))
    assert router_state(unconfigured) == {"mode": "off", "epoch": 0, "configured": False}
    configured = SimpleNamespace(
        config=SimpleNamespace(moe_router_config="router.yaml", moe_router_mode="off"),
        last_router_mode=None,
    )
    assert router_state(configured) == {"mode": "off", "epoch": 0, "configured": True}
    configured.last_router_mode = {"mode": "active", "epoch": 4, "status": "ok"}
    assert router_state(configured) == {"mode": "active", "epoch": 4, "configured": True}


def _with_global_state(state):
    class _Ctx:
        def __enter__(self):
            self.prev = api_server._GLOBAL_STATE
            api_server._GLOBAL_STATE = state
            return TestClient(api_server.app)

        def __exit__(self, *exc):
            api_server._GLOBAL_STATE = self.prev

    return _Ctx()


def test_endpoint_is_gated_while_loading_rebuilding_or_failed():
    for mstate, code in (("loading", 503), ("failed", 503), ("rebuilding", 409), ("stopping", 409)):
        state = SimpleNamespace(maintenance_state=mstate, rebuild_futures={}, last_router_mode=None)
        with _with_global_state(state) as client:
            r = client.post("/v1/router/mode", json={"mode": "active"})
        assert r.status_code == code, mstate
        assert state.maintenance_state == mstate  # a refused request changes nothing


def test_endpoint_validates_the_mode_before_reaching_the_scheduler():
    sent = []

    async def send_one(msg):
        sent.append(msg)

    state = SimpleNamespace(
        maintenance_state="serving", rebuild_futures={}, last_router_mode=None, send_one=send_one
    )
    with _with_global_state(state) as client:
        r = client.post("/v1/router/mode", json={"mode": "drain"})
    assert r.status_code == 422
    assert sent == []
    assert state.maintenance_state == "serving"


def test_endpoint_timeout_keeps_gate_closed_and_reports_504():
    sent = []

    async def send_one(msg):
        sent.append(msg)

    state = SimpleNamespace(
        maintenance_state="serving", rebuild_futures={}, last_router_mode=None, send_one=send_one
    )
    with _with_global_state(state) as client:
        r = client.post("/v1/router/mode", json={"mode": "active", "timeout": 0.05})
    assert r.status_code == 504
    assert state.maintenance_state == "rebuilding"
    assert len(sent) == 1 and sent[0].mode == "active"


def test_get_endpoint_and_stats_expose_the_same_router_state():
    state = SimpleNamespace(
        maintenance_state="serving",
        rebuild_futures={},
        last_router_mode={"mode": "planning-only", "epoch": 2, "status": "ok"},
        config=SimpleNamespace(moe_router_config="router.yaml", moe_router_mode="off"),
    )
    with _with_global_state(state) as client:
        r = client.get("/v1/router/mode")
    assert r.status_code == 200
    assert r.json() == {"mode": "planning-only", "epoch": 2, "configured": True}
