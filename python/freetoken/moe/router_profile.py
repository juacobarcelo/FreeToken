"""Opt-in host/GPU timing of the project router, wrapped around the adapter from outside.

The adapter's own files are bound by the expert-equivalence proof, so nothing in them may
change for instrumentation. This proxy exposes the two methods the MoE layer calls
(``plan_only`` and ``forward``) and, per layer and phase, accumulates:

* ``wait_s``  -- time spent synchronising the compute stream before the call, i.e. the GPU work
  of earlier layers the host would otherwise block on inside the adapter's ``.cpu()`` calls;
* ``host_s``  -- wall time of the adapter call itself once the stream is idle, i.e. host-side
  planning plus kernel enqueue with no GPU wait hidden inside;
* ``gpu_s``   -- elapsed time between CUDA events recorded on the compute stream around the
  call, i.e. the GPU time of the layer's expert kernels and copies the adapter issued.

The extra per-layer synchronisation is why this is a diagnostic-only mode
(``--moe-router-profile``): timed blocks never run with it. ``snapshot()`` is taken by the
scheduler when it answers a router-mode switch, so a client collects a block's profile with
the same call that ends the block.
"""

from __future__ import annotations

import time
from typing import Any, Callable

PHASES = ("prefill", "decode", "plan")


class _CudaEvents:
    """Real CUDA events; created lazily so the module imports without a GPU."""

    def __init__(self, stream_factory: Callable[[], Any]) -> None:
        import torch

        self._torch = torch
        self._stream_factory = stream_factory

    def stream(self) -> Any:
        return self._stream_factory()

    def synchronize(self, stream: Any) -> None:
        stream.synchronize()

    def record(self, stream: Any) -> Any:
        event = self._torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def elapsed_s(self, start: Any, end: Any) -> float:
        end.synchronize()
        return start.elapsed_time(end) / 1000.0


class RouterProfile:
    """Per-layer accumulators for one phase; ``snapshot`` sums the pending GPU event pairs."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.reset()

    def reset(self) -> None:
        self.calls = {phase: [0] * self.num_layers for phase in PHASES}
        self.wait_s = {phase: [0.0] * self.num_layers for phase in PHASES}
        self.host_s = {phase: [0.0] * self.num_layers for phase in PHASES}
        self.gpu_s = {phase: [0.0] * self.num_layers for phase in PHASES}
        self.pending: list[tuple[str, int, Any, Any]] = []

    def snapshot(self, events: Any, *, reset: bool = True) -> dict[str, Any]:
        for phase, layer_id, start, end in self.pending:
            self.gpu_s[phase][layer_id] += events.elapsed_s(start, end)
        self.pending = []
        decode_host = sum(self.host_s["decode"])
        decode_gpu = sum(self.gpu_s["decode"])
        result = {
            "schema_version": "1.0",
            "num_layers": self.num_layers,
            "calls": {phase: list(values) for phase, values in self.calls.items()},
            "wait_s": {phase: list(values) for phase, values in self.wait_s.items()},
            "host_s": {phase: list(values) for phase, values in self.host_s.items()},
            "gpu_s": {phase: list(values) for phase, values in self.gpu_s.items()},
            "totals": {phase: {"calls": sum(self.calls[phase]), "wait_s": sum(self.wait_s[phase]),
                               "host_s": sum(self.host_s[phase]), "gpu_s": sum(self.gpu_s[phase])}
                       for phase in PHASES},
            # The pre-registered decision quantity: host planning time per unit of GPU layer time.
            "decode_host_over_gpu": (decode_host / decode_gpu) if decode_gpu > 0 else None,
        }
        if reset:
            self.reset()
        return result


class ProfilingRouter:
    """Drop-in stand-in for the adapter: same two entry points, timed around the real call."""

    def __init__(self, adapter: Any, *, num_layers: int, events: Any | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        if events is None:
            import torch

            events = _CudaEvents(torch.cuda.current_stream)
        self.adapter = adapter
        self.profile = RouterProfile(num_layers)
        self._events = events
        self._clock = clock

    def _timed(self, phase: str, layer_id: int, call: Callable[[], Any]) -> Any:
        stream = self._events.stream()
        waited = self._clock()
        self._events.synchronize(stream)
        started = self._clock()
        start_event = self._events.record(stream)
        try:
            return call()
        finally:
            end_event = self._events.record(stream)
            finished = self._clock()
            profile = self.profile
            profile.calls[phase][layer_id] += 1
            profile.wait_s[phase][layer_id] += started - waited
            profile.host_s[phase][layer_id] += finished - started
            profile.pending.append((phase, layer_id, start_event, end_event))

    def plan_only(self, **kwargs: Any) -> None:
        return self._timed("plan", kwargs["layer"].layer_id, lambda: self.adapter.plan_only(**kwargs))

    def forward(self, **kwargs: Any) -> Any:
        phase = "prefill" if kwargs.get("is_prefill") else "decode"
        return self._timed(phase, kwargs["layer"].layer_id, lambda: self.adapter.forward(**kwargs))

    def snapshot(self, *, reset: bool = True) -> dict[str, Any]:
        return self.profile.snapshot(self._events, reset=reset)

    def __getattr__(self, name: str) -> Any:  # anything else the layer or tests reach for
        return getattr(self.adapter, name)


def router_profile_snapshot(adapter: Any) -> dict[str, Any] | None:
    """The profile carried by a router-mode reply when profiling is on, else None."""
    snapshot = getattr(adapter, "snapshot", None)
    if adapter is None or not isinstance(getattr(adapter, "profile", None), RouterProfile) or snapshot is None:
        return None
    return snapshot(reset=True)
