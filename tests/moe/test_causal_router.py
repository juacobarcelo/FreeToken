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

    if overrides.get("tp_info") is None:
        overrides = {**overrides, "tp_info": DistributedInfo(rank=0, size=1)}
    config = _router_config(**overrides)

    with pytest.raises(ValueError, match=message):
        _adjust_config(config)


def test_adjust_config_rejects_router_for_non_gpt_oss() -> None:
    from freetoken.engine.engine import _adjust_config

    config = _router_config()
    object.__setattr__(config.model_config, "model_type", "other_moe")

    with pytest.raises(ValueError, match="supports GPT-OSS only"):
        _adjust_config(config)


def test_routed_forward_delegates_to_adapter_without_changing_arguments(monkeypatch) -> None:
    from freetoken.layers.moe import OffloadMoELayer

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

    layer.offload_cache = SimpleNamespace(causal_router=Adapter())
    monkeypatch.setattr(
        "freetoken.layers.moe.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False)),
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


def test_runtime_rejects_router_cache_resize_before_teardown() -> None:
    from freetoken.engine.engine import CacheRebuildRejected, Engine

    engine = object.__new__(Engine)
    engine.config = SimpleNamespace()
    engine.moe_instrumentation = None
    engine.moe_offload_cache = SimpleNamespace(causal_router=object())

    with pytest.raises(CacheRebuildRejected, match="causal router"):
        engine.rebuild_runtime_cache(moe_cache_size=512)
