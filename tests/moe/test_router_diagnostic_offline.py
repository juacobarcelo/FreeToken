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
