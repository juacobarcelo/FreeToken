"""gpt-oss mxfp4 MoE tests: kernel numerics (vs dequant reference) + offload
movement (bit-identical vs running the same kernel directly on the banks).

Lean suite — five gates:
  1. fused routing == softmax-topk-renorm
  2. split-K decode kernel == dequant reference
  3. _t prefill kernel == dequant reference
  4. offload decode == direct split-K, *under real LRU eviction* (ledger transparency)
  5. offload prefill (overlap=True) runs + == direct _t  (begin_prefill crash guard)
"""

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tiny_config():
    """Minimal gpt-oss-like ModelConfig for the mxfp4 offload path."""
    from freetoken.models.config import ModelConfig, RotaryConfig
    rotary = RotaryConfig(
        head_dim=64,
        rotary_dim=64,
        max_position=2048,
        base=10000.0,
        scaling=None,
    )
    return ModelConfig(
        num_layers=2,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=128,
        vocab_size=256,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rotary_config=rotary,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        norm_topk_prob=True,
        model_type="gpt_oss",
        architectures=["GptOssForCausalLM"],
        moe_weight_format="mxfp4",
        hidden_act_alpha=1.702,
        swiglu_limit=7.0,
    )


@pytest.fixture
def tp1(monkeypatch):
    import freetoken.layers.moe as moe_mod
    import freetoken.models.gpt_oss.weight as gpt_weight
    from freetoken.distributed import DistributedInfo

    tp_info = DistributedInfo(rank=0, size=1)
    monkeypatch.setattr(moe_mod, "get_tp_info", lambda: tp_info)
    monkeypatch.setattr(gpt_weight, "get_tp_info", lambda: tp_info)
    return tp_info


def _make_offload_cache(config, device, *, cache_size=None, prefill_overlap=False):
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    banks = load_expert_banks(
        None,
        config,
        device=device,
        dtype=torch.bfloat16,
        dummy=True,
    )
    cache = OffloadMoeCache(
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        cache_size=cache_size if cache_size is not None else config.num_layers * config.num_experts,
        device=device,
        cache_policy="lru",
        prefill_overlap=prefill_overlap,
        quant_format=banks.quant_format,
    )
    cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    return cache


# ── kernel numerics ────────────────────────────────────────────────────────


@CUDA
@pytest.mark.parametrize("num_experts", [40, 128])  # 40 exercises the non-pow2 mask path
def test_gpt_oss_fused_routing_matches_softmax_topk_renorm(num_experts):
    from freetoken.kernel import gpt_oss_fused_routing

    device = torch.device("cuda")
    torch.manual_seed(0)
    tokens, top_k = 5, 4
    logits = torch.randn(tokens, num_experts, device=device, dtype=torch.bfloat16)

    weights, ids = gpt_oss_fused_routing(logits, top_k)
    assert weights.shape == (tokens, top_k)
    assert ids.shape == (tokens, top_k)
    assert ids.dtype == torch.int32
    assert weights.dtype == torch.float32

    # reference: softmax over all -> top-k -> renormalize (== softmax over top-k)
    probs = torch.softmax(logits.float(), dim=-1)
    ref_w, ref_id = torch.topk(probs, top_k, dim=-1)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)

    # compare as {expert_id: weight} maps per token (order-independent)
    for t in range(tokens):
        got = {int(ids[t, k]): float(weights[t, k]) for k in range(top_k)}
        want = {int(ref_id[t, k]): float(ref_w[t, k]) for k in range(top_k)}
        assert set(got) == set(want), f"token {t}: {set(got)} != {set(want)}"
        for e in want:
            assert abs(got[e] - want[e]) < 1e-3, f"token {t} expert {e}: {got[e]} vs {want[e]}"


def _mxfp4_dequant_reference(run, M, seed):
    """Run `run` (a split-K decode or _t prefill helper) on random mxfp4 experts and
    compare against a full dequant -> linear -> swiglu -> linear reference."""
    from freetoken.moe.fused_mxfp4 import (
        _transpose_mxfp4_for_decode,
        dequant_mxfp4_blocks,
        gpt_oss_swiglu,
    )

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    E, H, I = 8, 128, 64
    top_k = 2
    alpha, limit = 1.702, 7.0
    hb, ib = H // 32, I // 32

    gu_b = torch.randint(0, 256, (E, 2 * I, hb, 16), device=device, dtype=torch.uint8, generator=gen)
    gu_s = torch.full((E, 2 * I, hb), 124, device=device, dtype=torch.uint8)
    gu_bias = 0.05 * torch.randn(E, 2 * I, device=device, dtype=torch.bfloat16, generator=gen)
    dn_b = torch.randint(0, 256, (E, H, ib, 16), device=device, dtype=torch.uint8, generator=gen)
    dn_s = torch.full((E, H, ib), 124, device=device, dtype=torch.uint8)
    dn_bias = 0.05 * torch.randn(E, H, device=device, dtype=torch.bfloat16, generator=gen)

    hidden = 0.1 * torch.randn(M, H, device=device, dtype=torch.bfloat16, generator=gen)
    tid = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32, generator=gen)
    tw = torch.rand(M, top_k, device=device, dtype=torch.float32, generator=gen)

    gbt, gst = _transpose_mxfp4_for_decode(gu_b, gu_s)
    dbt, dst = _transpose_mxfp4_for_decode(dn_b, dn_s)
    out = run(
        hidden, tw, tid, gbt, gst, gu_bias, dbt, dst, dn_bias,
        top_k=top_k, hidden_act_alpha=alpha, swiglu_limit=limit,
    )

    expected = torch.zeros(M, H, device=device, dtype=torch.float32)
    for t in range(M):
        for r in range(top_k):
            e = int(tid[t, r])
            gw = dequant_mxfp4_blocks(gu_b[e], gu_s[e], out_dtype=torch.float32)
            gate_up = torch.nn.functional.linear(hidden[t].float(), gw, gu_bias[e].float())
            act = gpt_oss_swiglu(gate_up, alpha=alpha, limit=limit)
            dw = dequant_mxfp4_blocks(dn_b[e], dn_s[e], out_dtype=torch.float32)
            expected[t] += torch.nn.functional.linear(act, dw, dn_bias[e].float()) * float(tw[t, r])
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=6e-2, atol=6e-2)


@CUDA
@pytest.mark.parametrize("M", [1, 3])  # M=1 broadcast path; M>1 gather path
def test_run_mxfp4_splitk_decode_experts_matches_reference(M):
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts

    _mxfp4_dequant_reference(run_mxfp4_splitk_decode_experts, M, seed=21)


@CUDA
@pytest.mark.parametrize("M", [24])  # M > MXFP4_DECODE_MAX_TOKENS (16): prefill path
def test_run_mxfp4_prefill_experts_t_matches_reference(M):
    from freetoken.moe.fused_mxfp4 import run_mxfp4_prefill_experts_t

    _mxfp4_dequant_reference(run_mxfp4_prefill_experts_t, M, seed=31)


# ── offload movement (bit-identical vs the same kernel on the banks) ─────────


@CUDA
def test_offload_decode_bit_identical_under_eviction(tp1):
    """Offload decode matches direct split-K across a real evict+reload cycle."""
    from freetoken.models.gpt_oss.moe import GptOssMxfp4OffloadMoELayer
    from freetoken.moe.fused_mxfp4 import (
        run_mxfp4_splitk_decode_experts as _run_mxfp4_splitk_decode_experts,
    )

    dev = torch.device("cuda")
    c = _tiny_config()
    E, H, tk = c.num_experts, c.hidden_size, c.num_experts_per_tok

    cache = _make_offload_cache(c, dev, cache_size=E)  # 4 slots, 8 total experts -> eviction

    layer1 = GptOssMxfp4OffloadMoELayer(c, layer_id=1)
    layer1.offload_cache = cache
    layer0 = GptOssMxfp4OffloadMoELayer(c, layer_id=0)
    layer0.offload_cache = cache

    torch.manual_seed(42)
    hidden = 0.1 * torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    tw = torch.tensor([[0.6, 0.4]], dtype=torch.float32, device=dev)

    def ref(layer_id: int, expert_ids: list[int]) -> torch.Tensor:
        g = lambda k: cache.bank_sources[k][layer_id].to(dev)
        tid_ref = torch.tensor([expert_ids], dtype=torch.int32, device=dev)
        return _run_mxfp4_splitk_decode_experts(
            hidden, tw, tid_ref,
            g("gate_up_blocks"), g("gate_up_scales"), g("gate_up_bias"),
            g("down_blocks"), g("down_scales"), g("down_bias"),
            top_k=tk, hidden_act_alpha=c.hidden_act_alpha, swiglu_limit=c.swiglu_limit,
        )

    # A: cold-load layer-1 experts 0,1
    out_a = layer1._decode_routed(hidden, tw, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    assert torch.equal(out_a, ref(1, [0, 1])), "Call A (cold load) failed"

    # B: cold-load layer-1 experts 2,3  (cache now full: L1E0..L1E3)
    out_b = layer1._decode_routed(hidden, tw, torch.tensor([[2, 3]], dtype=torch.int32, device=dev))
    assert torch.equal(out_b, ref(1, [2, 3])), "Call B (fill cache) failed"

    # Evictor: layer0 [0,1] claims the two LRU-oldest slots (A's) -> L1E0,L1E1 evicted
    out_ev = layer0._decode_routed(hidden, tw, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    assert torch.equal(out_ev, ref(0, [0, 1])), "Evictor call failed"
    assert int(cache.slot_for_id[1, 0].item()) == -1, "Eviction did not happen (L1E0 still resident)"
    assert int(cache.slot_for_id[1, 1].item()) == -1, "Eviction did not happen (L1E1 still resident)"

    # C: L1E0,L1E1 missing -> reload (evicting L1E2,L1E3)
    out_c = layer1._decode_routed(hidden, tw, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    assert torch.equal(out_c, ref(1, [0, 1])), "Call C (reload after eviction) failed"


@CUDA
@pytest.mark.parametrize("M", [16])
def test_offload_prefill_overlap_matches_reference(M, tp1):
    """Prefill overlap must not crash and must match the direct _t kernel."""
    from freetoken.models.gpt_oss.moe import GptOssMxfp4OffloadMoELayer
    from freetoken.moe.fused_mxfp4 import (
        run_mxfp4_prefill_experts_t as _run_mxfp4_prefill_experts_t,
    )

    dev = torch.device("cuda")
    c = _tiny_config()
    E, H, tk = c.num_experts, c.hidden_size, c.num_experts_per_tok

    cache = _make_offload_cache(c, dev, prefill_overlap=True)
    # layer_id=0 is mandatory: _wait_prefill_overlap gates begin_prefill() on layer_id==0,
    # so only layer 0 exercises the previously-crashing code path.
    layer = GptOssMxfp4OffloadMoELayer(c, layer_id=0)
    layer.offload_cache = cache

    torch.manual_seed(2)
    hidden = 0.1 * torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    tw, tid = layer._topk(logits.contiguous())

    out_overlap = layer._prefill_routed(hidden, tw, tid.clone())  # must not raise

    g = lambda k: cache.bank_sources[k][0].to(dev)  # layer-0 source rows
    out_ref = _run_mxfp4_prefill_experts_t(
        hidden, tw, tid,
        g("gate_up_blocks"), g("gate_up_scales"), g("gate_up_bias"),
        g("down_blocks"), g("down_scales"), g("down_bias"),
        top_k=tk, hidden_act_alpha=c.hidden_act_alpha, swiglu_limit=c.swiglu_limit,
    )
    max_diff = (out_overlap - out_ref).abs().max().item()
    assert torch.equal(out_overlap, out_ref), f"overlap prefill differs; max_diff={max_diff}"


@CUDA
@pytest.mark.parametrize(
    ("num_tokens", "max_running_requests", "cache_size", "is_prefill", "rare_group"),
    [(1, 1, 4, False, False), (2, 2, 4, False, False),
     (3, 3, 4, False, False), (3, 3, 4, True, False),
     (32, 16, 1024, True, False), (32, 16, 1024, True, True)],
)
def test_causal_router_preserves_mxfp4_result(
    tmp_path, tp1, monkeypatch, num_tokens, max_running_requests, cache_size, is_prefill, rare_group
):
    """The optional adapter changes scheduling, not routed expert semantics."""

    pytest.importorskip("inference_system_planner.moe_router")
    from freetoken.models.gpt_oss.moe import GptOssMxfp4OffloadMoELayer
    from freetoken.moe.causal_router import AlternativeAAdapter
    from freetoken.moe.fused_mxfp4 import (
        run_mxfp4_prefill_experts_t,
        run_mxfp4_splitk_decode_experts,
    )

    dev = torch.device("cuda:0")
    config = _tiny_config()
    if not is_prefill and num_tokens in (1, 2):
        from dataclasses import replace

        # Small H/I cap all split counts at the same value and miss the old
        # regrouping bug. These shapes expose different baseline/group splits,
        # four-column summation and cache pressure with two logical users.
        config = replace(config, hidden_size=2048, intermediate_size=1024,
                         moe_intermediate_size=1024, num_experts=8, num_experts_per_tok=4)
    cache = _make_offload_cache(config, dev, cache_size=cache_size)
    layer = GptOssMxfp4OffloadMoELayer(config, layer_id=0)
    layer.offload_cache = cache
    per_rank_bytes = sum(
        bank[0].numel() * bank.element_size() for _, bank in cache.banks
    )
    aggregate_bytes = 2 * per_rank_bytes
    router_path = tmp_path / "alternative-a.yaml"
    router_path.write_text(
        f"""schema_version: "1.0"
configuration_id: freetoken-gpu-test
policy:
  policy_id: alternative-a-layer-synchronous
  batch_size: {max_running_requests}
  microbatch_size: {max_running_requests}
  synchronization: layer-synchronous
  waiting_rule: scheduled-resource-completion-only
  tie_breaks:
    resident: [ready-token-rows-desc, oldest-reveal-us, layer-id, expert-id]
    missing: [ready-token-rows-desc, transfer-time-us, oldest-reveal-us, layer-id, expert-id]
    eviction: [last-used-us, layer-id, expert-id]
resources:
  fast_memory_capacity_bytes: {aggregate_bytes * cache.cache_size}
  activation_reserve_bytes: 0
  pinned_kv_reserve_bytes: 0
  initial_resident_experts: []
  kv_cache_policy: pinned
  compute_slots: 1
  pcie_transfer_slots: 1
costs:
  default_profile:
    expert_size: {{value: {aggregate_bytes}, unit: byte, evidence_type: measured, provenance: gpu-test}}
    h2d_transfer_time: {{value: 1, unit: microsecond, evidence_type: measured, provenance: gpu-test}}
    compute_fixed_time: {{value: 1, unit: microsecond, evidence_type: measured, provenance: gpu-test}}
    compute_per_token_row_time: {{value: 1, unit: microsecond-per-token-row, evidence_type: measured, provenance: gpu-test}}
  overrides: []
"""
    )
    adapter = AlternativeAAdapter(
        config_path=str(router_path),
        cache=cache,
        batch_size=max_running_requests,
        tensor_parallel_size=2,
    )

    torch.manual_seed(71)
    hidden = 0.1 * torch.randn(
        num_tokens, config.hidden_size, device=dev, dtype=torch.bfloat16
    )
    if num_tokens in (1, 2):
        weights = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]][:num_tokens],
            dtype=torch.float32, device=dev)
        expert_ids = torch.tensor(
            [[0, 1, 2, 3], [2, 3, 4, 5]][:num_tokens], dtype=torch.int32, device=dev)
    elif num_tokens == 3:
        weights = torch.tensor(
            [[0.6, 0.4], [0.7, 0.3], [0.55, 0.45]],
            dtype=torch.float32,
            device=dev,
        )
        expert_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3]], dtype=torch.int32, device=dev
        )
    else:
        weights = torch.tensor(
            [[0.6, 0.4]], dtype=torch.float32, device=dev
        ).repeat(num_tokens, 1)
        expert_ids = torch.tensor(
            [[0, 1]], dtype=torch.int32, device=dev
        ).repeat(num_tokens, 1)
        if rare_group:
            expert_ids[-1, 1] = 2  # one routed row must still use the prefill arithmetic

    def source(name):
        return cache.bank_sources[name][0].to(dev)

    expected_kernel = (
        run_mxfp4_prefill_experts_t if is_prefill else run_mxfp4_splitk_decode_experts
    )
    from freetoken.moe import diagnostic

    class Capture:
        group = None
        reference_partials = None
        placed = None

        def __init__(self):
            self.observed = torch.empty(
                (num_tokens, config.num_experts_per_tok, config.hidden_size),
                dtype=hidden.dtype, device=dev)
            self.coverage = torch.zeros(expert_ids.shape, dtype=torch.int32, device=dev)

        def kernel_begin(self, kind, *args, **kwargs):
            assert kind == ("prefill" if is_prefill else "decode")

        def kernel_end(self, partials, output):
            if self.group is None:
                self.reference_partials = partials.clone()
            else:
                rows, columns = self.group
                # Verify the actual expert output before the one-column helper
                # reduction and the returned operand later scattered by C.
                assert torch.equal(partials[:, 0], output)
                self.observed[rows, columns] = partials[:, 0]
                self.coverage[rows, columns] += 1

        def group_begin(self, rows, columns, expert_id, slot_id):
            assert torch.all(expert_ids[rows, columns] == expert_id)
            self.group = rows, columns

        def group_end(self):
            self.group = None

        def placed_partials(self, partials):
            self.placed = partials.clone()

    capture = Capture()
    monkeypatch.setattr(diagnostic, "observer", capture)
    expected = expected_kernel(
        hidden,
        weights,
        expert_ids,
        source("gate_up_blocks"),
        source("gate_up_scales"),
        source("gate_up_bias"),
        source("down_blocks"),
        source("down_scales"),
        source("down_bias"),
        top_k=config.num_experts_per_tok,
        hidden_act_alpha=config.hidden_act_alpha,
        swiglu_limit=config.swiglu_limit,
    )
    expected_partials = capture.reference_partials.clone()
    monkeypatch.setattr(diagnostic, "observer", None)
    # Issue 36: the real planning boundary may read, but must not mutate operands,
    # cache metadata, scratch buffers or any expert parameter bank.
    state = [hidden, weights, expert_ids, cache.id_of_slot, cache.slot_for_id,
             cache.usage, cache.step, cache.evict_slots, cache.src_indices,
             cache.num_indices, cache.num_missing_full]
    snapshots = [tensor.clone() for tensor in state]
    parameter_snapshots = [bank.view(torch.uint8).clone() for _, bank in cache.banks]
    adapter.plan_only(layer=layer, hidden_states=hidden, topk_weights=weights, topk_ids=expert_ids)
    for tensor, snapshot in zip(state, snapshots, strict=True):
        assert torch.equal(tensor, snapshot)
    for (_, bank), snapshot in zip(cache.banks, parameter_snapshots, strict=True):
        assert torch.equal(bank.view(torch.uint8), snapshot)
    after_planning = expected_kernel(
        hidden, weights, expert_ids,
        source("gate_up_blocks"), source("gate_up_scales"), source("gate_up_bias"),
        source("down_blocks"), source("down_scales"), source("down_bias"),
        top_k=config.num_experts_per_tok, hidden_act_alpha=config.hidden_act_alpha,
        swiglu_limit=config.swiglu_limit,
    )
    assert torch.equal(after_planning, expected)
    monkeypatch.setattr(diagnostic, "observer", capture)
    actual = adapter.forward(
        layer=layer,
        hidden_states=hidden,
        topk_weights=weights,
        topk_ids=expert_ids,
        is_prefill=is_prefill,
    )
    torch.cuda.synchronize(dev)

    assert torch.all(capture.coverage == 1)
    assert torch.equal(capture.observed, expected_partials)
    assert torch.equal(capture.placed, expected_partials)
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()
    expected_residents = sorted(set(expert_ids.flatten().tolist()))
    actual_residents = sorted(
        slot for slot in cache.slot_for_id[0].tolist() if slot >= 0
    )
    assert actual_residents == list(range(min(len(expected_residents), cache_size)))
