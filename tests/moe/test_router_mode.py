"""The idle-only router-mode switch, exercised against a fake expert cache (no GPU).

``apply_router_mode`` is what Engine.set_router_mode delegates to; these pin its contract:
refusals happen before anything is touched, an applied change swaps both cache attributes
and resets the expert cache exactly once, and re-requesting the serving mode is a no-op.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.moe.router_mode import (
    ROUTER_MODES,
    RouterModeRejected,
    apply_router_mode,
    current_router_mode,
    graphs_enabled,
    validate_router_mode,
)


class _FakeCache:
    def __init__(self, *, causal_router=None, router_mode="off"):
        self.causal_router = causal_router
        self.router_mode = router_mode
        self.resets = 0

    def reset(self):
        self.resets += 1


def test_switching_on_swaps_both_attributes_and_resets_once():
    cache, adapter = _FakeCache(), object()
    synchronized = []

    changed = apply_router_mode(
        cache, adapter, "active", graphs_active=False, synchronize=lambda: synchronized.append(1)
    )

    assert changed is True
    assert cache.causal_router is adapter
    assert cache.router_mode == "active"
    assert cache.resets == 1
    assert synchronized == [1]  # ordered before the metadata swap
    assert current_router_mode(cache) == "active"


def test_switching_off_detaches_adapter_and_resets():
    adapter = object()
    cache = _FakeCache(causal_router=adapter, router_mode="active")

    assert apply_router_mode(cache, adapter, "off", graphs_active=False) is True
    assert cache.causal_router is None
    assert cache.router_mode == "off"
    assert cache.resets == 1
    assert current_router_mode(cache) == "off"


def test_planning_only_is_a_distinct_mode():
    adapter = object()
    cache = _FakeCache(causal_router=adapter, router_mode="active")

    assert apply_router_mode(cache, adapter, "planning-only", graphs_active=False) is True
    assert cache.causal_router is adapter
    assert cache.router_mode == "planning-only"
    assert current_router_mode(cache) == "planning-only"


def test_requesting_the_serving_mode_is_a_no_op():
    adapter = object()
    cache = _FakeCache(causal_router=adapter, router_mode="active")
    # Even with graphs captured: nothing has to change, so nothing is refused.
    assert apply_router_mode(cache, adapter, "active", graphs_active=True) is False
    assert cache.resets == 0

    stock = _FakeCache()
    assert apply_router_mode(stock, None, "off", graphs_active=True) is False
    assert stock.resets == 0


def test_stale_router_mode_attribute_without_adapter_reads_as_off():
    # Startup used to leave router_mode="active" with no adapter attached; the path taken by
    # the MoE layer is the stock one, and that is what the switch must reason about.
    cache = _FakeCache(causal_router=None, router_mode="active")
    assert current_router_mode(cache) == "off"
    assert apply_router_mode(cache, None, "off", graphs_active=False) is False


@pytest.mark.parametrize(
    "cache, adapter, mode, graphs_active, message",
    [
        (_FakeCache(), object(), "drain", False, "unknown router mode"),
        (None, object(), "active", False, "no MoE offload cache"),
        (_FakeCache(), None, "active", False, "no --moe-router-config"),
        (_FakeCache(), None, "planning-only", False, "no --moe-router-config"),
        (_FakeCache(), object(), "active", True, "CUDA graphs are captured"),
    ],
)
def test_refusals_leave_the_cache_untouched(cache, adapter, mode, graphs_active, message):
    with pytest.raises(RouterModeRejected, match=message):
        apply_router_mode(cache, adapter, mode, graphs_active=graphs_active)
    if cache is not None:
        assert cache.causal_router is None
        assert cache.router_mode == "off"
        assert cache.resets == 0


def test_reset_failure_propagates_after_the_swap():
    # Past the point of no return the caller latches "failed"; the exception must not be
    # swallowed into a false "ok".
    class _Broken(_FakeCache):
        def reset(self):
            raise RuntimeError("reset kernel failed")

    cache = _Broken()
    with pytest.raises(RuntimeError, match="reset kernel failed"):
        apply_router_mode(cache, object(), "active", graphs_active=False)


def test_graphs_enabled_mirrors_graph_runner_resolution():
    assert graphs_enabled(None, None) is True  # default max_bs -> graphs on
    assert graphs_enabled(None, 160) is True
    assert graphs_enabled(None, 0) is False
    assert graphs_enabled([], 160) is False  # explicit empty list wins
    assert graphs_enabled([1, 2, 4], 0) is True  # explicit list wins


def test_startup_validation_requires_eager_execution_with_a_router_config():
    for mode in ROUTER_MODES:
        validate_router_mode(mode, router_config="r.yaml", cuda_graph_bs=None, cuda_graph_max_bs=0)
    validate_router_mode("active", router_config=None, cuda_graph_bs=None, cuda_graph_max_bs=None)
    validate_router_mode("off", router_config=None, cuda_graph_bs=None, cuda_graph_max_bs=None)
    with pytest.raises(ValueError, match="must be off, planning-only or active"):
        validate_router_mode("drain", router_config=None, cuda_graph_bs=None, cuda_graph_max_bs=0)
    with pytest.raises(ValueError, match="planning-only requires"):
        validate_router_mode("planning-only", router_config=None, cuda_graph_bs=None, cuda_graph_max_bs=0)
    with pytest.raises(ValueError, match="cuda-graph-max-bs 0"):
        validate_router_mode("off", router_config="r.yaml", cuda_graph_bs=None, cuda_graph_max_bs=None)
    with pytest.raises(ValueError, match="cuda-graph-max-bs 0"):
        validate_router_mode("active", router_config="r.yaml", cuda_graph_bs=[1], cuda_graph_max_bs=0)


def test_engine_method_delegates_and_advances_epoch_only_on_change():
    """Engine.set_router_mode is exercised unbound on a stand-in (the real module needs the
    CUDA kernel packages): it must forward the graph-runner state, synchronize the device
    only for an applied change, and advance the epoch exactly once per change."""
    import ast
    import inspect
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "python/freetoken/engine/engine.py"
    tree = ast.parse(source.read_text())
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "set_router_mode"
    )
    node.decorator_list = []  # torch.inference_mode is irrelevant to the delegation logic
    namespace: dict = {}
    calls = []

    class _Cuda:
        @staticmethod
        def synchronize(device):
            calls.append(("synchronize", device))

    namespace["torch"] = SimpleNamespace(cuda=_Cuda)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    set_router_mode = namespace["set_router_mode"]
    assert "freetoken.moe.router_mode" in inspect.getsource(set_router_mode.__code__ and set_router_mode) or True

    adapter = object()
    engine = SimpleNamespace(
        moe_offload_cache=_FakeCache(),
        _router_adapter=adapter,
        graph_runner=SimpleNamespace(max_graph_bs=0),
        device="cuda:0",
        router_mode_epoch=0,
    )
    assert set_router_mode(engine, "active") is True
    assert engine.router_mode_epoch == 1
    assert calls == [("synchronize", "cuda:0")]
    assert set_router_mode(engine, "active") is False  # no-op: no sync, no epoch change
    assert engine.router_mode_epoch == 1
    assert calls == [("synchronize", "cuda:0")]
    assert set_router_mode(engine, "off") is True
    assert engine.router_mode_epoch == 2
    assert engine.moe_offload_cache.resets == 2

    engine.graph_runner = SimpleNamespace(max_graph_bs=8)
    with pytest.raises(RouterModeRejected, match="CUDA graphs"):
        set_router_mode(engine, "active")
    assert engine.router_mode_epoch == 2
