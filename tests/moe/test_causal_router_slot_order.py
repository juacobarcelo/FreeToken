"""Run actual forward/load/victim methods with CPU cache and stream doubles.

This verifies slot choices, copy targets and event dependencies, not GPU math.
"""

import ast
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from inference_system_planner.moe_router import ordered_lru_slots, select_lru_slot
from inference_system_planner.moe_router.live import plan_alternative_a_layer


@pytest.fixture
def implementation(monkeypatch):
    module = ModuleType("_isp_slot_order_adapter_under_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.diagnostic = SimpleNamespace(observer=None)
    source_path = Path(__file__).resolve().parents[2] / "python/freetoken/moe/causal_router.py"
    source = ast.parse(source_path.read_text())
    source.body = [node for node in source.body if not (
        isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "freetoken.moe")]
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    kernel = ModuleType("freetoken.kernel")
    kernel.moe_sum_reduce_triton = lambda partials, output: None
    monkeypatch.setitem(sys.modules, "freetoken.kernel", kernel)
    return module


def legacy_victim(self, *, id_of_slot, usage, protected_until, slot_order):
    # The pre-optimization adapter algorithm, retained as an executable oracle.
    immediately_safe = [slot for slot in range(len(id_of_slot)) if slot not in protected_until]
    if immediately_safe:
        return select_lru_slot(
            id_of_slot=id_of_slot, usage_by_slot=usage,
            num_experts=self.cache.num_experts, candidate_slot_ids=immediately_safe,
        ), None
    if not protected_until:
        raise RuntimeError("Alternative A found no evictable expert-cache slot")
    slot = next(iter(protected_until))
    return slot, protected_until[slot]


@pytest.mark.parametrize("capacity", [1, 2, 7])
def test_all_protected_preserves_exact_first_event_identity(implementation, capacity):
    adapter = object.__new__(implementation.AlternativeAAdapter)
    adapter.cache = SimpleNamespace(num_experts=8)
    adapter._ordered_lru_slots = ordered_lru_slots
    protected = {slot: object() for slot in reversed(range(capacity))}
    slot, wait = adapter._select_victim(
        id_of_slot=list(range(capacity)), usage=list(range(capacity)),
        protected_until=protected, slot_order=implementation.LayerSlotOrder(),
    )
    assert slot == capacity - 1
    assert wait is protected[slot]


class Array:
    def __init__(self, data):
        self.data = deepcopy(data)
        self.shape = (3, 2)
        self.dtype, self.device = "float", "cpu"

    def __getitem__(self, index):
        return self.data[index[0]][index[1]] if isinstance(index, tuple) else self.data[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            self.data[index[0]][index[1]] = value
        else:
            self.data[index] = value

    def fill_(self, value):
        self.data = value


def run_forwards(module, initial_ids, initial_usage, layer_routes, *, legacy):
    adapter = object.__new__(module.AlternativeAAdapter)
    experts = 16
    mapping = [[-1] * experts for _ in range(4)]
    for slot, flat_id in enumerate(initial_ids):
        if flat_id >= 0:
            mapping[flat_id // experts][flat_id % experts] = slot
    adapter.cache = SimpleNamespace(
        num_experts=experts, device="cpu", id_of_slot=Array(initial_ids),
        usage=Array(initial_usage), step=Array(max(initial_usage, default=0)),
        slot_for_id=Array(mapping), banks=[],
    )
    trace, builds = [], []

    class BankSlot:
        def __init__(self, slot, value):
            self.slot, self.value = slot, value

        def copy_(self, source, *, non_blocking):
            assert non_blocking is True
            self.value = source
            trace.append(("copy", self.slot, source))

    bank = [BankSlot(slot, value) for slot, value in enumerate(initial_ids)]
    adapter.cache.banks = [([[layer * experts + expert for expert in range(experts)]
                            for layer in range(4)], bank)]

    class Stream:
        def wait_event(self, event):
            trace.append(("wait", event.identity))

    class Event:
        def record(self, stream):
            pass

    adapter.copy_stream = Stream()
    module.torch = SimpleNamespace(
        empty=lambda *args, **kwargs: None, empty_like=lambda value: value,
        cuda=SimpleNamespace(Event=Event, stream=lambda stream: nullcontext()),
    )

    def ordered(**kwargs):
        builds.append(tuple(kwargs["id_of_slot"]))
        return ordered_lru_slots(**kwargs)

    adapter._ordered_lru_slots = ordered
    if legacy:
        adapter._select_victim = legacy_victim.__get__(adapter)

    def compute(*, layer, group, slot_id, ready, **kwargs):
        expected_id = layer.layer_id * experts + group.expert_id
        assert adapter.cache.id_of_slot[slot_id] == expected_id
        assert adapter.cache.slot_for_id[layer.layer_id, group.expert_id] == slot_id
        assert bank[slot_id].value == expected_id
        identity = (len(trace), layer.layer_id, group.expert_id, slot_id)
        trace.append(("compute", identity, ready is not None))
        event = Event()
        event.identity = identity
        return event

    adapter._compute_group = compute
    expected_builds = 0
    for layer_id, rows in layer_routes:
        ids = list(adapter.cache.id_of_slot.data)
        usage = list(adapter.cache.usage.data)
        residents = [expert for expert, slot in enumerate(mapping[layer_id]) if slot >= 0]
        needed = {expert for row in rows for expert in row}
        plan = plan_alternative_a_layer(
            layer_id=layer_id, topk_rows=rows, resident_expert_ids=residents,
            transfer_time_us_by_expert={expert: 1 for expert in needed - set(residents)},
        )
        prepared = module.PreparedLayer(
            plan, ids, usage, adapter.cache.step.data, list(mapping[layer_id]))
        adapter.prepare_layer = lambda **kwargs: prepared
        expected_builds += bool(plan.missing_groups)
        adapter.forward(
            layer=SimpleNamespace(layer_id=layer_id, top_k=len(rows[0])),
            hidden_states=Array([]), topk_weights=Array([]), topk_ids=Array([]),
            is_prefill=False,
        )
        # The real load method updates the GPU mapping double in place.
        mapping = adapter.cache.slot_for_id.data
        trace.append(("state", list(adapter.cache.id_of_slot.data),
                      deepcopy(mapping), list(adapter.cache.usage.data), adapter.cache.step.data))
    if not legacy:
        assert len(builds) == expected_builds
    return trace


@pytest.mark.parametrize("capacity", [1, 2, 5, 17, 64])
@pytest.mark.parametrize("initially_empty", [False, True])
def test_full_forward_decisions_and_waits_match_legacy_across_layers(
    implementation, capacity, initially_empty,
):
    rng = random.Random(380010 + capacity)
    ids = [-1] * capacity if initially_empty else rng.sample(range(64), capacity)
    usage = [rng.randrange(4) for _ in ids]
    layers = [(layer, [rng.sample(range(16), 4) for _ in range(12)])
              for layer in [0, 1, 2, 3, 0, 0, 2]]
    expected = run_forwards(implementation, ids, usage, layers, legacy=True)
    assert run_forwards(implementation, ids, usage, layers, legacy=False) == expected


def test_resident_only_forward_never_constructs_an_eviction_order(implementation):
    layers = [(0, [[0, 1], [1, 0]]), (0, [[0, 1]])]
    assert run_forwards(implementation, [0, 1], [0, 0], layers, legacy=False) == run_forwards(
        implementation, [0, 1], [0, 0], layers, legacy=True)
