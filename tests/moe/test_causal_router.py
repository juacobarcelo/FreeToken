"""Boundaries for the optional InferenceSystemPlanner Alternative A adapter."""

from types import SimpleNamespace

import pytest
import torch


def _router_config(**overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    values = {
        "model_path": "/tmp/gpt-oss-120b",
        "tp_info": DistributedInfo(rank=0, size=2),
        "dtype": torch.bfloat16,
        "attention_backend": "triton",
        "moe_backend": "offload",
        "moe_cache_size": 256,
        "moe_router_config": "/tmp/alternative-a.yaml",
        "cuda_graph_bs": [1, 2, 4, 8, 16],
        "cuda_graph_max_bs": 16,
    }
    values.update(overrides)
    config = EngineConfig(**values)
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            model_type="gpt_oss",
            single_stream_only=False,
            dsv4_args=None,
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=36,
            num_moe_layers=36,
            num_experts=128,
            expert_quant="none",
            moe_weight_format="mxfp4",
            hidden_act="gpt_oss_swiglu",
        ),
    )
    return config


def test_adjust_config_enables_only_bounded_eager_tp2_path() -> None:
    from freetoken.engine.engine import _adjust_config

    config = _router_config()
    _adjust_config(config)

    assert config.moe_backend == "offload"
    assert config.moe_prefill_overlap is False
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tp_info": None}, "tensor parallel size 2"),
        ({"moe_backend": "fused"}, "requires --moe-backend offload"),
        ({"moe_cpu_layers": "1"}, "cannot be combined with --moe-cpu-layers"),
        (
            {
                "moe_instrumentation_dir": "/tmp/evidence",
                "moe_instrumentation_run_id": "router-test",
            },
            "cannot be combined with performance instrumentation",
        ),
    ],
)
def test_adjust_config_rejects_incompatible_router_modes(overrides, message) -> None:
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.engine import _adjust_config

    if "tp_info" in overrides and overrides["tp_info"] is None:
        overrides = {**overrides, "tp_info": DistributedInfo(rank=0, size=1)}
    config = _router_config(**overrides)

    with pytest.raises(ValueError, match=message):
        _adjust_config(config)


def test_adjust_config_validates_the_router_mode_after_forcing_the_eager_path() -> None:
    """The mode check runs on the adjusted configuration, not the requested one.

    A router config always lands on the eager path inside ``_adjust_config``; checking the
    graph settings before that adjustment rejected a configuration the function was about to
    make legal (it did, until this test). What the check must still catch is the mode itself.
    """
    from freetoken.engine.engine import _adjust_config

    # Graphs requested and a router config present: adjusted to eager, then accepted.
    for mode in ("off", "planning-only", "active"):
        config = _router_config(moe_router_mode=mode)
        _adjust_config(config)
        assert config.cuda_graph_max_bs == 0 and config.cuda_graph_bs == []

    with pytest.raises(ValueError, match="moe_router_mode must be"):
        _adjust_config(_router_config(moe_router_mode="sometimes"))

    # planning-only asks for plans that are never applied: without a configuration to plan
    # with it is meaningless and refused. "active" is the default value of the flag, so a
    # plain run without a router configuration keeps the stock path instead of failing.
    with pytest.raises(ValueError, match="planning-only requires"):
        _adjust_config(_router_config(moe_router_config=None, moe_router_mode="planning-only"))
    for mode in ("off", "active"):
        stock = _router_config(moe_router_config=None, moe_router_mode=mode)
        _adjust_config(stock)
        assert stock.cuda_graph_max_bs == 16 and stock.cuda_graph_bs == [1, 2, 4, 8, 16]


def test_adjust_config_rejects_router_for_non_gpt_oss() -> None:
    from freetoken.engine.engine import _adjust_config

    config = _router_config()
    object.__setattr__(config.model_config, "model_type", "other_moe")

    with pytest.raises(ValueError, match="supports GPT-OSS only"):
        _adjust_config(config)


def test_routed_forward_delegates_to_adapter_without_changing_arguments(monkeypatch) -> None:
    from freetoken.distributed import DistributedInfo
    from freetoken.layers.moe import OffloadMoELayer

    monkeypatch.setattr(
        "freetoken.layers.moe.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )
    layer = OffloadMoELayer(
        layer_id=3,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=16,
    )
    calls = []

    class Adapter:
        def forward(self, **kwargs):
            calls.append(kwargs)
            return kwargs["hidden_states"] + 1

    layer.offload_cache = SimpleNamespace(causal_router=Adapter(), router_mode="active")
    monkeypatch.setattr(
        "freetoken.layers.moe.get_global_ctx",
        lambda: SimpleNamespace(
            batch=SimpleNamespace(is_prefill=False),
            moe_instrumentation=None,
        ),
    )
    monkeypatch.setattr(layer, "_maybe_all_reduce", lambda value: value)
    hidden = torch.zeros((2, 8))
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)

    output = layer.routed_forward(hidden, weights, ids)

    assert torch.equal(output, hidden + 1)
    assert len(calls) == 1
    assert calls[0]["layer"] is layer
    assert calls[0]["hidden_states"] is hidden
    assert calls[0]["topk_weights"] is weights
    assert calls[0]["topk_ids"] is ids
    assert calls[0]["is_prefill"] is False


@pytest.mark.parametrize("rows,is_prefill", [(1, True), (17, True), (1, False), (2, False)])
def test_compute_group_preserves_phase_geometry_and_operand_placement(
    monkeypatch, rows, is_prefill,
) -> None:
    from freetoken.moe.causal_router import AlternativeAAdapter

    class FakeStream:
        def wait_event(self, event) -> None:
            raise AssertionError("no copy event expected")

    class FakeEvent:
        def __init__(self) -> None:
            self.recorded_on = None

        def record(self, stream) -> None:
            self.recorded_on = stream

    stream = FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)

    cache_size = 2395
    slot_id = 1200
    bank_caches = [
        torch.arange(cache_size * 2).reshape(cache_size, 2) + bank_id
        for bank_id in range(6)
    ]
    adapter = object.__new__(AlternativeAAdapter)
    adapter.cache = SimpleNamespace(
        device=torch.device("cpu"),
        banks=[([], bank_cache) for bank_cache in bank_caches],
    )
    calls = []

    def fake_kernel(hidden, weights, ids, *views, **kwargs):
        calls.append((hidden, weights, ids, views, kwargs))
        return torch.full_like(hidden, 7)

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("group size must not change the serving forward's phase")

    monkeypatch.setattr(
        "freetoken.moe.fused_mxfp4.run_mxfp4_prefill_experts_t",
        fake_kernel if is_prefill else unexpected_kernel,
    )
    monkeypatch.setattr(
        "freetoken.moe.fused_mxfp4.run_mxfp4_splitk_decode_experts",
        unexpected_kernel if is_prefill else fake_kernel,
    )

    occurrences = [
        SimpleNamespace(token_row=row, topk_column=row % 2) for row in reversed(range(rows))
    ]
    group = SimpleNamespace(occurrences=occurrences, token_row_count=rows)
    layer = SimpleNamespace(hidden_act_alpha=1.702, swiglu_limit=7.0)
    hidden = torch.arange(17 * 4, dtype=torch.float32).reshape(17, 4)
    weights = torch.arange(17 * 2, dtype=torch.float32).reshape(17, 2) / (17 * 2)
    partials = torch.zeros((17, 2, 4))
    config = ({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
              if is_prefill else {"gate_up_num_splits": 45, "down_num_splits": 18})

    done = adapter._compute_group(
        layer=layer,
        hidden_states=hidden,
        topk_weights=weights,
        group=group,
        slot_id=slot_id,
        ready=None,
        partials=partials,
        is_prefill=is_prefill,
        prefill_config=config if is_prefill else None,
        decode_config=None if is_prefill else config,
    )

    assert len(calls) == 1
    group_hidden, group_weights, ids, views, kwargs = calls[0]
    ordered_rows = [item.token_row for item in occurrences]
    ordered_columns = [item.topk_column for item in occurrences]
    assert torch.equal(group_hidden, hidden[ordered_rows])
    assert torch.equal(group_weights[:, 0], weights[ordered_rows, ordered_columns])
    assert torch.equal(ids, torch.zeros((rows, 1), dtype=torch.int32))
    assert len(views) == len(bank_caches)
    for view, bank_cache in zip(views, bank_caches, strict=True):
        assert view.shape == (1, 2)
        assert view.is_contiguous()
        assert view.data_ptr() == bank_cache[slot_id].data_ptr()
    assert kwargs["top_k"] == 1
    assert kwargs["kernel_config"] is config
    expected_partials = torch.zeros_like(partials)
    expected_partials[ordered_rows, ordered_columns] = 7
    assert torch.equal(partials, expected_partials)
    assert done.recorded_on is stream


def test_runtime_rejects_router_cache_resize_before_teardown() -> None:
    from freetoken.engine.engine import CacheRebuildRejected, Engine

    engine = object.__new__(Engine)
    engine.config = SimpleNamespace()
    engine.moe_instrumentation = None
    engine.moe_offload_cache = SimpleNamespace(causal_router=object())

    with pytest.raises(CacheRebuildRejected, match="causal router"):
        engine.rebuild_runtime_cache(moe_cache_size=512)
