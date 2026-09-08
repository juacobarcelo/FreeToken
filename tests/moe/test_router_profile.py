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
