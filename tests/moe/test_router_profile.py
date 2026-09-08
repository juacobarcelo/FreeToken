"""The diagnostic profiling proxy around the router adapter (no GPU: fake events and clock)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.moe.router_profile import ProfilingRouter, RouterProfile, router_profile_snapshot


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.steps: list[float] = []

    def __call__(self):
        value = self.now
        if self.steps:
            self.now += self.steps.pop(0)
        return value


class _Events:
    """Fake stream/event machinery: each recorded event carries a fixed GPU timeline."""

    def __init__(self, gpu_seconds_per_call):
        self.gpu_seconds_per_call = gpu_seconds_per_call
        self.synchronized = 0
        self._tick = 0.0

    def stream(self):
        return "stream"

    def synchronize(self, stream):
        assert stream == "stream"
        self.synchronized += 1

    def record(self, stream):
        event = self._tick
        self._tick += self.gpu_seconds_per_call / 2
        return event

    def elapsed_s(self, start, end):
        return end - start


class _Adapter:
    def __init__(self):
        self.calls = []
        self.marker = "real-adapter"

    def forward(self, **kwargs):
        self.calls.append(("forward", kwargs["layer"].layer_id, kwargs["is_prefill"]))
        return "out"

    def plan_only(self, **kwargs):
        self.calls.append(("plan", kwargs["layer"].layer_id))


def test_proxy_delegates_and_separates_wait_host_and_gpu_time():
    adapter = _Adapter()
    clock = _Clock()
    # Per timed call the clock is read three times: before sync (+wait), after sync (+host), after.
    clock.steps = [0.5, 0.2, 0.0, 0.5, 0.3, 0.0, 0.1, 0.4, 0.0]
    events = _Events(gpu_seconds_per_call=2.0)
    proxy = ProfilingRouter(adapter, num_layers=2, events=events, clock=clock)
    layer0, layer1 = SimpleNamespace(layer_id=0), SimpleNamespace(layer_id=1)

    assert proxy.forward(layer=layer0, is_prefill=False, hidden_states=None, topk_weights=None, topk_ids=None) == "out"
    assert proxy.forward(layer=layer1, is_prefill=False, hidden_states=None, topk_weights=None, topk_ids=None) == "out"
    proxy.plan_only(layer=layer0, hidden_states=None, topk_weights=None, topk_ids=None)
    assert adapter.calls == [("forward", 0, False), ("forward", 1, False), ("plan", 0)]
    assert events.synchronized == 3  # one sync per timed call, before the adapter runs
    assert proxy.marker == "real-adapter"  # other attributes fall through to the adapter

    snapshot = proxy.snapshot()
    assert snapshot["calls"] == {"prefill": [0, 0], "decode": [1, 1], "plan": [1, 0]}
    assert snapshot["wait_s"]["decode"] == pytest.approx([0.5, 0.5])
    assert snapshot["host_s"]["decode"] == pytest.approx([0.2, 0.3])
    assert snapshot["host_s"]["plan"] == pytest.approx([0.4, 0.0])
    assert snapshot["gpu_s"]["decode"] == pytest.approx([1.0, 1.0])
    assert snapshot["totals"]["decode"] == pytest.approx({"calls": 2, "wait_s": 1.0, "host_s": 0.5, "gpu_s": 2.0})
    assert snapshot["decode_host_over_gpu"] == pytest.approx(0.25)
    # snapshot() resets: a second read is empty and the ratio undefined.
    empty = proxy.snapshot()
    assert empty["totals"]["decode"]["calls"] == 0 and empty["decode_host_over_gpu"] is None


def test_prefill_calls_are_kept_apart_from_decode():
    adapter, clock, events = _Adapter(), _Clock(), _Events(gpu_seconds_per_call=1.0)
    clock.steps = [0.0, 0.1, 0.0]
    proxy = ProfilingRouter(adapter, num_layers=1, events=events, clock=clock)
    proxy.forward(layer=SimpleNamespace(layer_id=0), is_prefill=True, hidden_states=None, topk_weights=None, topk_ids=None)
    snapshot = proxy.snapshot()
    assert snapshot["totals"]["prefill"]["calls"] == 1 and snapshot["totals"]["decode"]["calls"] == 0
    assert snapshot["decode_host_over_gpu"] is None


def test_exception_in_adapter_still_records_and_propagates():
    class _Broken(_Adapter):
        def forward(self, **kwargs):
            raise RuntimeError("kernel failed")

    proxy = ProfilingRouter(_Broken(), num_layers=1, events=_Events(1.0), clock=_Clock())
    try:
        proxy.forward(layer=SimpleNamespace(layer_id=0), is_prefill=False, hidden_states=None, topk_weights=None, topk_ids=None)
    except RuntimeError as error:
        assert "kernel failed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("adapter error was swallowed")
    assert proxy.snapshot()["totals"]["decode"]["calls"] == 1


def test_snapshot_helper_only_reports_profiling_adapters():
    assert router_profile_snapshot(None) is None
    assert router_profile_snapshot(_Adapter()) is None
    proxy = ProfilingRouter(_Adapter(), num_layers=1, events=_Events(1.0), clock=_Clock())
    assert isinstance(proxy.profile, RouterProfile)
    assert router_profile_snapshot(proxy)["schema_version"] == "1.0"


def test_scheduler_reply_drains_the_profile_only_on_ok():
    """``Scheduler._reply_router_mode`` is exercised unbound (the scheduler module needs the
    CUDA kernel packages): a busy reply while a block drains must leave the counters intact;
    the applied switch's ``ok`` reply carries them and resets, so one profile spans exactly
    the blocks between two applied switches."""
    import ast
    from pathlib import Path

    from freetoken.message import RouterModeResultMsg

    source = Path(__file__).resolve().parents[2] / "python/freetoken/scheduler/scheduler.py"
    node = next(
        n for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_reply_router_mode"
    )
    namespace: dict = {"RouterModeResultMsg": RouterModeResultMsg}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    reply = namespace["_reply_router_mode"]

    router = ProfilingRouter(_Adapter(), num_layers=2, events=_Events(0.5), clock=_Clock())
    router.forward(layer=SimpleNamespace(layer_id=1), is_prefill=False)
    cache = SimpleNamespace(causal_router=router, router_mode="active")
    sent: list = []
    scheduler = SimpleNamespace(
        engine=SimpleNamespace(moe_offload_cache=cache, router_mode_epoch=4, _router_adapter=router),
        send_result=lambda messages: sent.extend(messages),
    )

    reply(scheduler, "req-1", "busy")
    assert sent[-1].status == "busy" and sent[-1].profile is None
    assert sent[-1].mode == "active" and sent[-1].epoch == 4
    assert router.profile.calls["decode"] == [0, 1]  # untouched by the refused request

    reply(scheduler, "req-2", "ok")
    assert sent[-1].status == "ok"
    assert sent[-1].profile["totals"]["decode"]["calls"] == 1
    assert router.profile.calls["decode"] == [0, 0]  # drained exactly once, by the applied switch

    reply(scheduler, "req-3", "ok")
    assert sent[-1].profile["totals"]["decode"]["calls"] == 0
