"""Execute the real preparation and dispatch methods with CPU protocol doubles.

These tests need no Torch, CUDA, FreeToken dependency install, or checkpoint.
They validate the observation boundary, not GPU arithmetic or TP2 behavior.
"""

import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, data, *, dtype="int32"):
        self.data = deepcopy(data)
        self.dtype, self.device = dtype, "cpu"
        self.shape = self._shape(data)
        self.ndim = len(self.shape)
        self.reads = 0

    @staticmethod
    def _shape(data):
        return (len(data), *Tensor._shape(data[0])) if isinstance(data, list) and data else ()

    def detach(self):
        return self

    def cpu(self):
        self.reads += 1
        return self

    def tolist(self):
        return deepcopy(self.data)

    def item(self):
        self.reads += 1
        return self.data


@pytest.fixture
def methods(monkeypatch):
    module = ModuleType("_isp_router_boundary_under_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(torch=SimpleNamespace(int32="int32"), diagnostic=SimpleNamespace(observer=None))
    source = ast.parse((ROOT / "python/freetoken/moe/causal_router.py").read_text())
    source.body = [node for node in source.body if not (
        isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "freetoken.moe")]
    exec(compile(source, "causal_router.py", "exec"), module.__dict__)
    source = ast.parse((ROOT / "python/freetoken/layers/moe.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "OffloadMoELayer")
    routed = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "routed_forward")
    # The sole local import gets the same explicitly inactive observation double.
    routed.body = [n for n in routed.body if not isinstance(n, ast.ImportFrom)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, routed], type_ignores=[]))
    exec(compile(tree, "moe.py:routed_forward", "exec"), module.__dict__)
    return module


def adapter_fixture(methods):
    from inference_system_planner.moe_router.live import plan_alternative_a_layer
    from inference_system_planner.moe_router import ExpertKey

    adapter = object.__new__(methods.AlternativeAAdapter)
    adapter.cache = SimpleNamespace(
        num_layers=2, num_experts=4, device="cpu",
        id_of_slot=Tensor([0, -1, -1, -1]), usage=Tensor([7, 0, 0, 0]), step=Tensor(7))
    adapter._plan_layer = plan_alternative_a_layer
    adapter._ExpertKey = ExpertKey
    adapter.config = SimpleNamespace(profile=lambda key: SimpleNamespace(transfer_time_us=key.expert_id + 1))
    inputs = dict(layer=SimpleNamespace(layer_id=0, top_k=2),
                  hidden_states=Tensor([[1, 2], [3, 4]], dtype="float"),
                  topk_weights=Tensor([[0.75, 0.25], [0.5, 0.5]], dtype="float"),
                  topk_ids=Tensor([[0, 2], [1, 2]]))
    return adapter, inputs


def snapshot(adapter, inputs):
    return {name: deepcopy(value.data) for name, value in {
        **{k: v for k, v in inputs.items() if isinstance(v, Tensor)},
        **{k: v for k, v in vars(adapter.cache).items() if isinstance(v, Tensor)},
    }.items()}


def test_planning_only_executes_actual_planner_and_preserves_all_source_operands(methods):
    adapter, inputs = adapter_fixture(methods)
    before = snapshot(adapter, inputs)
    original = adapter._plan_layer
    plans = []

    def actual(**kwargs):
        plan = original(**kwargs)
        plans.append(plan)
        return plan
    adapter._plan_layer = actual
    adapter.forward = lambda **kwargs: pytest.fail("B must never apply/undo an active forward")
    assert adapter.plan_only(**inputs) is None
    assert [g.expert_id for g in plans[0].groups] == [0, 2, 1]
    assert snapshot(adapter, inputs) == before
    assert inputs["topk_ids"].reads == 1
    assert adapter.cache.id_of_slot.reads == adapter.cache.usage.reads == adapter.cache.step.reads == 1


def test_faulty_planner_can_only_mutate_private_host_copies(methods):
    adapter, inputs = adapter_fixture(methods)
    before = snapshot(adapter, inputs)

    def faulty(**kwargs):
        kwargs["topk_rows"][0][0] = 3
        kwargs["resident_expert_ids"].clear()
        raise RuntimeError("injected planning fault")
    adapter._plan_layer = faulty
    with pytest.raises(RuntimeError, match="injected"):
        adapter.plan_only(**inputs)
    assert snapshot(adapter, inputs) == before


@pytest.mark.parametrize("prefill", [False, True])
def test_B_preserves_stock_operation_and_argument_identity_before_ID_rewrite(methods, prefill):
    adapter, inputs = adapter_fixture(methods)
    calls, seen = [], []
    actual_plan = adapter.plan_only

    def plan(**kwargs):
        seen.append(deepcopy(kwargs["topk_ids"].data))
        actual_plan(**kwargs)
    adapter.plan_only = plan
    adapter.forward = lambda **kwargs: pytest.fail("unexpected active path")
    methods.get_global_ctx = lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=prefill))
    result = object()

    def stock(hidden, weights, ids):
        assert hidden is inputs["hidden_states"] and weights is inputs["topk_weights"] and ids is inputs["topk_ids"]
        calls.append("prefill" if prefill else "decode")
        if not prefill:
            ids.data[0][0] = 99  # stock's intentional expert-to-cache-slot rewrite
        return result
    layer = inputs["layer"]
    layer._instrumentation = lambda: None
    layer._prefill_routed = layer._decode_routed = stock
    layer._maybe_all_reduce = lambda output: (calls.append("all_reduce"), output)[1]
    layer.offload_cache = SimpleNamespace(causal_router=adapter, router_mode="planning-only")
    assert methods.routed_forward(layer, inputs["hidden_states"], inputs["topk_weights"], inputs["topk_ids"]) is result
    assert seen == [[[0, 2], [1, 2]]]
    assert calls == ["prefill" if prefill else "decode", "all_reduce"]


def test_A_does_not_prepare_metadata_or_create_a_plan(methods):
    _, inputs = adapter_fixture(methods)
    layer = inputs["layer"]
    layer._instrumentation = lambda: None
    layer.offload_cache = SimpleNamespace(causal_router=None, router_mode="active")
    layer._prefill_routed = lambda *args: args[0]
    layer._maybe_all_reduce = lambda output: output
    methods.get_global_ctx = lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=True))
    methods.routed_forward(layer, inputs["hidden_states"], inputs["topk_weights"], inputs["topk_ids"])
    assert all(value.reads == 0 for value in inputs.values() if isinstance(value, Tensor))


@pytest.mark.parametrize("prefill", [False, True])
def test_active_dispatch_passes_the_actual_serving_phase(methods, prefill):
    adapter, inputs = adapter_fixture(methods)
    calls = []
    adapter.forward = lambda **kwargs: (calls.append(kwargs), inputs["hidden_states"])[1]
    layer = inputs["layer"]
    layer._instrumentation = lambda: None
    layer.offload_cache = SimpleNamespace(causal_router=adapter, router_mode="active")
    layer._maybe_all_reduce = lambda output: output
    methods.get_global_ctx = lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=prefill))
    methods.routed_forward(layer, inputs["hidden_states"], inputs["topk_weights"], inputs["topk_ids"])
    assert len(calls) == 1 and calls[0]["is_prefill"] is prefill
    assert calls[0]["topk_ids"] is inputs["topk_ids"]


def test_prefill_configuration_uses_full_forward_geometry_before_expert_grouping():
    source = ast.parse((ROOT / "python/freetoken/moe/fused_mxfp4.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "mxfp4_prefill_config")
    calls = []
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 8}
    namespace = {"try_get_optimal_moe_config": lambda *args: (calls.append(args), config)[1]}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fused_mxfp4.py:config", "exec"), namespace)
    actual = namespace["mxfp4_prefill_config"](
        num_tokens=286, num_experts=128, hidden_size=2880, local_intermediate_size=1440, top_k=4)
    assert calls == [((128, 2880, 2880), (128, 2880, 1440), 4, 286)]
    assert actual == {**config, "BLOCK_SIZE_K": 64}
    assert config["BLOCK_SIZE_K"] == 16


@pytest.fixture
def decode_config_module(monkeypatch):
    """Use the real geometry selector without importing Torch or GPU packages."""
    module = ModuleType("freetoken.moe.fused_mxfp4")
    tree = ast.parse((ROOT / "python/freetoken/moe/fused_mxfp4.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in {"_decode_split_count", "mxfp4_decode_config"}]
    exec(compile(tree, "fused_mxfp4.py:decode-config", "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize("rows,expected", [(1, (45, 18)), (2, (23, 9))])
def test_decode_config_differs_from_regrouped_route_count(decode_config_module, rows, expected):
    config = decode_config_module.mxfp4_decode_config(
        num_tokens=rows, hidden_size=2880, local_intermediate_size=1440, top_k=4)
    group = decode_config_module.mxfp4_decode_config(
        num_tokens=rows, hidden_size=2880, local_intermediate_size=1440, top_k=1)
    assert (config["gate_up_num_splits"], config["down_num_splits"]) == expected
    assert group != config  # These shapes expose the old per-group recomputation bug.


@pytest.mark.parametrize("rows", [1, 2])
def test_active_decode_freezes_geometry_and_uses_stock_topk_reduction(
    methods, decode_config_module, rows,
):
    """Run the actual adapter forward across resident and newly loaded groups."""
    adapter = object.__new__(methods.AlternativeAAdapter)
    adapter.cache = SimpleNamespace(
        banks=[([], SimpleNamespace(shape=(4, 1440, 2880)))],
        usage=[0] * 4, step=SimpleNamespace(fill_=lambda value: None))
    groups = [SimpleNamespace(expert_id=i) for i in range(4)]
    prepared = SimpleNamespace(
        plan=SimpleNamespace(resident_groups=groups[:1], missing_groups=groups[1:]),
        id_of_slot=[0, -1, -1, -1], usage=[1, 0, 0, 0], step=1,
        slot_for_layer=[0, -1, -1, -1])
    adapter.prepare_layer = lambda **kwargs: prepared
    adapter._schedule_load = lambda **kwargs: (kwargs["expert_id"], None)
    calls = []
    adapter._compute_group = lambda **kwargs: calls.append(kwargs)
    reduction = []
    output = object()

    class Partials:
        def sum(self, *, dim):
            reduction.append(dim)
            return SimpleNamespace(to=lambda dtype: output)

    partials = Partials()
    methods.torch.empty = lambda *args, **kwargs: partials
    hidden = SimpleNamespace(shape=(rows, 2880), dtype="bfloat16", device="cpu")
    result = adapter.forward(
        layer=SimpleNamespace(layer_id=0, top_k=4), hidden_states=hidden,
        topk_weights=object(), topk_ids=SimpleNamespace(shape=(rows, 4)), is_prefill=False)
    assert result is output and reduction == [1]
    assert len(calls) == 4
    config = decode_config_module.mxfp4_decode_config(
        num_tokens=rows, hidden_size=2880, local_intermediate_size=1440, top_k=4)
    for call in calls:
        assert call["decode_config"] == config
        assert call["decode_config"] is calls[0]["decode_config"]
        assert call["prefill_config"] is None
        assert call["partials"] is partials
        assert call["is_prefill"] is False


@pytest.fixture
def reset_dispatch():
    """Load the actual scheduler admission methods without importing the GPU runtime."""
    source = ast.parse((ROOT / "python/freetoken/scheduler/scheduler.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    names = {"_process_one_msg", "_diagnostic_reset_allowed"}
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == 2
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[]))
    namespace = {name: type(name, (), {}) for name in
                 ("BatchBackendMsg", "ExitMsg", "UserMsg", "AbortBackendMsg", "CacheRebuildBackendMsg")}
    exec(compile(tree, "scheduler.py:reset-admission", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("case,expected", [
    ("diagnostic", None), ("ordinary", "unsupported"), ("armed", "unsupported"),
    ("different-pages", "unsupported"), ("moe", "unsupported"), ("mamba", "unsupported"),
    ("swa", "unsupported"), ("tp4", "unsupported"), ("busy", "busy"),
    ("drain", "unsupported"), ("unsupported-cache", "unsupported"),
])
def test_actual_TP_reset_admission_is_limited_to_one_idle_same_size_diagnostic(reset_dispatch, case, expected):
    methods = reset_dispatch
    diagnostic = None if case == "ordinary" else SimpleNamespace(armed=case == "armed")
    replies = []
    scheduler = SimpleNamespace(
        engine=SimpleNamespace(router_diagnostic=diagnostic, num_pages=8192),
        config=SimpleNamespace(tp_info=SimpleNamespace(size=4 if case == "tp4" else 2)),
        cache_manager=SimpleNamespace(supports_runtime_rebuild=case != "unsupported-cache"),
        prefill_manager=SimpleNamespace(runnable=case == "busy"),
        decode_manager=SimpleNamespace(runnable=False), _pending_rebuild=None,
        _reply_rebuild=lambda *args: replies.append(args),
    )
    scheduler._diagnostic_reset_allowed = lambda **kw: methods['_diagnostic_reset_allowed'](scheduler, **kw)
    message = methods['CacheRebuildBackendMsg']()
    message.request_id = 'reset'
    message.mode = 'drain' if case == 'drain' else 'if_idle'
    message.num_pages = 4096 if case == 'different-pages' else 8192
    message.moe_cache_size = 2395 if case == 'moe' else None
    message.num_mamba_slots = 1 if case == 'mamba' else None
    message.num_swa_pages = 8192 if case == 'swa' else None
    methods['_process_one_msg'](scheduler, message)
    if expected is None:
        assert scheduler._pending_rebuild is message
        assert replies == []
    else:
        assert scheduler._pending_rebuild is None
        assert replies[0][:2] == ('reset', expected)
