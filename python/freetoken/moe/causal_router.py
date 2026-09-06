"""Bounded FreeToken adapter for InferenceSystemPlanner Alternative A.

The policy and its deterministic priorities live in InferenceSystemPlanner. This
module applies one current-layer plan with FreeToken's existing host expert banks,
GPU slot cache, CUDA streams, and GPT-OSS MXFP4 kernels. No future route is passed
across the adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from freetoken.moe import diagnostic

if TYPE_CHECKING:
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache


@dataclass
class PreparedLayer:
    """Materialized planning result and private host copies; no cache writes."""

    plan: object
    id_of_slot: list[int]
    usage: list[int]
    step: int
    slot_for_layer: list[int]


class AlternativeAAdapter:
    """Apply a resident-first, one-transfer-ahead plan to one visible layer."""

    def __init__(
        self,
        *,
        config_path: str,
        cache: "OffloadMoeCache",
        batch_size: int,
        tensor_parallel_size: int,
    ) -> None:
        try:
            from inference_system_planner.moe_router import (
                AlternativeAPolicy,
                ExpertKey,
                load_router_config,
                plan_alternative_a_layer,
                select_lru_slot,
            )
        except ImportError as error:
            raise RuntimeError(
                "--moe-router-config requires the InferenceSystemPlanner package "
                "from the issue-22 branch"
            ) from error

        config = load_router_config(Path(config_path))
        if config.policy_id != AlternativeAPolicy.policy_id:
            raise ValueError(
                "--moe-router-config must select alternative-a-layer-synchronous"
            )
        if config.batch_size != batch_size or config.microbatch_size != batch_size:
            raise ValueError(
                "router batch and microbatch sizes must equal FreeToken max_running_req"
            )
        if tensor_parallel_size != 2:
            raise ValueError("Alternative A live adapter requires tensor parallel size 2")
        if cache.quant_format != "mxfp4_triton":
            raise ValueError("Alternative A live adapter requires GPT-OSS MXFP4 banks")
        if config.initial_residents:
            raise ValueError(
                "Alternative A live adapter requires an empty declared initial cache"
            )
        configured_keys = set(config.profile_overrides) | {
            resident.key for resident in config.initial_residents
        }
        for key in configured_keys:
            if key.layer_id >= cache.num_layers or key.expert_id >= cache.num_experts:
                raise ValueError(
                    "router configuration refers to an expert outside the runtime model: "
                    f"layer={key.layer_id}, expert={key.expert_id}"
                )

        per_rank_expert_bytes = sum(
            int(bank_cache[0].numel() * bank_cache.element_size())
            for _, bank_cache in cache.banks
        )
        aggregate_expert_bytes = per_rank_expert_bytes * tensor_parallel_size
        for layer_id in range(cache.num_layers):
            for expert_id in range(cache.num_experts):
                profile = config.profile(ExpertKey(layer_id, expert_id))
                if profile.size_bytes != aggregate_expert_bytes:
                    raise ValueError(
                        "router expert size does not match the aggregate TP2 runtime object: "
                        f"configured={profile.size_bytes}, runtime={aggregate_expert_bytes}"
                    )
        runtime_capacity = aggregate_expert_bytes * cache.cache_size
        if config.expert_capacity_bytes != runtime_capacity:
            raise ValueError(
                "router expert capacity does not match the TP2 slot cache: "
                f"configured={config.expert_capacity_bytes}, runtime={runtime_capacity}"
            )

        self.config = config
        self.cache = cache
        self._ExpertKey = ExpertKey
        self._plan_layer = plan_alternative_a_layer
        self._select_lru_slot = select_lru_slot
        self.copy_stream = torch.cuda.Stream(device=cache.device)

    def _select_victim(
        self,
        *,
        id_of_slot: list[int],
        usage: list[int],
        protected_until: dict[int, torch.cuda.Event],
    ) -> tuple[int, torch.cuda.Event | None]:
        immediately_safe = [
            slot_id
            for slot_id in range(len(id_of_slot))
            if slot_id not in protected_until
        ]
        if immediately_safe:
            return (
                self._select_lru_slot(
                    id_of_slot=id_of_slot,
                    usage_by_slot=usage,
                    num_experts=self.cache.num_experts,
                    candidate_slot_ids=immediately_safe,
                ),
                None,
            )
        if not protected_until:
            raise RuntimeError("Alternative A found no evictable expert-cache slot")
        # Every event belongs to the same compute stream and the dict preserves
        # enqueue order. Wait for the first group to finish; choosing a later LRU
        # event would idle both transfer and compute unnecessarily.
        slot_id = next(iter(protected_until))
        return slot_id, protected_until[slot_id]

    def _schedule_load(
        self,
        *,
        layer_id: int,
        expert_id: int,
        id_of_slot: list[int],
        slot_for_layer: list[int],
        usage: list[int],
        protected_until: dict[int, torch.cuda.Event],
    ) -> tuple[int, torch.cuda.Event]:
        existing_slot = slot_for_layer[expert_id]
        if existing_slot >= 0:
            raise RuntimeError(
                f"Alternative A attempted a duplicate load for layer={layer_id}, "
                f"expert={expert_id}"
            )
        slot_id, safe_after = self._select_victim(
            id_of_slot=id_of_slot,
            usage=usage,
            protected_until=protected_until,
        )
        old_flat_id = id_of_slot[slot_id]
        new_flat_id = layer_id * self.cache.num_experts + expert_id
        ready = torch.cuda.Event()
        with torch.cuda.stream(self.copy_stream):
            if safe_after is not None:
                self.copy_stream.wait_event(safe_after)
            if old_flat_id >= 0:
                old_layer = old_flat_id // self.cache.num_experts
                old_expert = old_flat_id % self.cache.num_experts
                self.cache.slot_for_id[old_layer, old_expert] = -1
                if old_layer == layer_id:
                    slot_for_layer[old_expert] = -1
            self.cache.id_of_slot[slot_id] = new_flat_id
            self.cache.slot_for_id[layer_id, expert_id] = slot_id
            for sources, bank_cache in self.cache.banks:
                bank_cache[slot_id].copy_(
                    sources[layer_id][expert_id], non_blocking=True
                )
            ready.record(self.copy_stream)
        id_of_slot[slot_id] = new_flat_id
        slot_for_layer[expert_id] = slot_id
        protected_until.pop(slot_id, None)
        return slot_id, ready

    @staticmethod
    def _group_indices(group, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.tensor(
            [item.token_row for item in group.occurrences],
            dtype=torch.int64,
            device=device,
        )
        columns = torch.tensor(
            [item.topk_column for item in group.occurrences],
            dtype=torch.int64,
            device=device,
        )
        return rows, columns

    def _compute_group(
        self,
        *,
        layer: "OffloadMoELayer",
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        group,
        slot_id: int,
        ready: torch.cuda.Event | None,
        partials: torch.Tensor,
        is_prefill: bool,
        prefill_config: dict[str, int] | None,
    ) -> torch.cuda.Event:
        from freetoken.moe.fused_mxfp4 import (
            run_mxfp4_prefill_experts_t,
            run_mxfp4_splitk_decode_experts,
        )

        compute_stream = torch.cuda.current_stream(self.cache.device)
        if ready is not None:
            compute_stream.wait_event(ready)
        rows, columns = self._group_indices(group, device=self.cache.device)
        group_hidden = hidden_states.index_select(0, rows).contiguous()
        group_weights = topk_weights[rows, columns].reshape(-1, 1).contiguous()
        group_slots = torch.full(
            (group.token_row_count, 1),
            0,
            dtype=torch.int32,
            device=self.cache.device,
        )
        # A plan group contains exactly one expert. Keep the kernel's expert
        # axis at one instead of exposing the complete slot cache: the grouped
        # prefill aligner has a bounded expert axis, while a valid offload cache
        # may contain thousands of slots.
        views = tuple(
            bank_cache.narrow(0, slot_id, 1)
            for _, bank_cache in self.cache.banks
        )
        gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = views
        # A small expert group is still part of the original prompt forward.
        # Keep its kernel family and arithmetic geometry independent of grouping.
        if is_prefill and prefill_config is None:
            raise ValueError("prefill groups require the original forward's kernel configuration")
        run = run_mxfp4_prefill_experts_t if is_prefill else run_mxfp4_splitk_decode_experts
        kernel_options = {"kernel_config": prefill_config} if is_prefill else {}
        observer = diagnostic.observer
        if observer is not None:
            observer.group_begin(rows, columns, group.expert_id, slot_id)
        group_output = run(
            group_hidden,
            group_weights,
            group_slots,
            gu_blocks,
            gu_scales,
            gu_bias,
            dn_blocks,
            dn_scales,
            dn_bias,
            top_k=1,
            hidden_act_alpha=layer.hidden_act_alpha,
            swiglu_limit=layer.swiglu_limit,
            **kernel_options,
        )
        # Preserve the model router's original top-k column order. The final
        # reduction therefore uses the same boundary as the unmodified kernel.
        partials[rows, columns] = group_output
        if observer is not None:
            observer.group_end()
        done = torch.cuda.Event()
        done.record(compute_stream)
        return done

    def prepare_layer(
        self,
        *,
        layer: "OffloadMoELayer",
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> PreparedLayer:
        """Read current metadata and run the real planner without mutating execution state."""

        if layer.layer_id < 0 or layer.layer_id >= self.cache.num_layers:
            raise RuntimeError(f"invalid MoE layer id {layer.layer_id}")
        if topk_ids.ndim != 2 or topk_ids.shape[1] != layer.top_k:
            raise RuntimeError(
                "Alternative A requires a two-dimensional route matrix with "
                f"top_k={layer.top_k}"
            )
        if topk_ids.dtype != torch.int32:
            raise RuntimeError("Alternative A requires int32 expert ids")
        if topk_weights.shape != topk_ids.shape:
            raise RuntimeError("Alternative A route ids and weights must have equal shapes")
        if hidden_states.ndim != 2 or hidden_states.shape[0] != topk_ids.shape[0]:
            raise RuntimeError(
                "Alternative A hidden states and routes must have equal token rows"
            )
        if not (
            hidden_states.device == topk_weights.device == topk_ids.device == self.cache.device
        ):
            raise RuntimeError("Alternative A tensors and expert cache must share one device")
        route_rows = topk_ids.detach().cpu().tolist()
        if any(
            expert_id < 0 or expert_id >= self.cache.num_experts
            for row in route_rows
            for expert_id in row
        ):
            raise RuntimeError("Alternative A received an out-of-range expert id")
        id_of_slot = self.cache.id_of_slot.detach().cpu().tolist()
        usage = self.cache.usage.detach().cpu().tolist()
        occupied_ids = [flat_id for flat_id in id_of_slot if flat_id >= 0]
        if any(
            flat_id >= self.cache.num_layers * self.cache.num_experts
            for flat_id in occupied_ids
        ) or len(occupied_ids) != len(set(occupied_ids)):
            raise RuntimeError("Alternative A received an invalid expert-cache map")
        step = int(self.cache.step.item())
        slot_for_layer = [-1] * self.cache.num_experts
        for slot_id, flat_expert_id in enumerate(id_of_slot):
            if flat_expert_id < 0:
                continue
            resident_layer = flat_expert_id // self.cache.num_experts
            if resident_layer == layer.layer_id:
                slot_for_layer[flat_expert_id % self.cache.num_experts] = slot_id
        resident_ids = [
            expert_id
            for expert_id, slot_id in enumerate(slot_for_layer)
            if slot_id >= 0
        ]
        missing_ids = {
            expert_id
            for row in route_rows
            for expert_id in row
            if expert_id not in resident_ids
        }
        transfer_times = {
            expert_id: self.config.profile(
                self._ExpertKey(layer.layer_id, expert_id)
            ).transfer_time_us
            for expert_id in missing_ids
        }
        plan = self._plan_layer(
            layer_id=layer.layer_id,
            topk_rows=route_rows,
            resident_expert_ids=resident_ids,
            transfer_time_us_by_expert=transfer_times,
        )

        return PreparedLayer(plan, id_of_slot, usage, step, slot_for_layer)

    def plan_only(self, **kwargs) -> None:
        """Perform the same preparation as active execution, then discard its plan."""
        self.prepare_layer(**kwargs)

    def forward(
        self,
        *,
        layer: "OffloadMoELayer",
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        """Apply the current Alternative A policy; learned expert selection is unchanged."""
        prepared = self.prepare_layer(
            layer=layer, hidden_states=hidden_states,
            topk_weights=topk_weights, topk_ids=topk_ids,
        )
        plan = prepared.plan
        id_of_slot, usage, step = prepared.id_of_slot, prepared.usage, prepared.step
        slot_for_layer = prepared.slot_for_layer
        prefill_config = None
        if is_prefill:
            from freetoken.moe.fused_mxfp4 import mxfp4_prefill_config

            prefill_config = mxfp4_prefill_config(
                num_tokens=hidden_states.shape[0], num_experts=self.cache.num_experts,
                hidden_size=hidden_states.shape[1],
                local_intermediate_size=self.cache.banks[0][1].shape[2] // 2,
                top_k=layer.top_k,
            )

        partials = torch.empty(
            (hidden_states.shape[0], topk_ids.shape[1], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        protected_until: dict[int, torch.cuda.Event] = {}
        for group in plan.resident_groups:
            slot_id = slot_for_layer[group.expert_id]
            if slot_id < 0:
                raise RuntimeError("resident Alternative A group lost its cache slot")
            done = self._compute_group(
                layer=layer,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                group=group,
                slot_id=slot_id,
                ready=None,
                partials=partials,
                is_prefill=is_prefill,
                prefill_config=prefill_config,
            )
            protected_until[slot_id] = done
            step += 1
            usage[slot_id] = step
            self.cache.usage[slot_id] = step

        for group in plan.missing_groups:
            slot_id, ready = self._schedule_load(
                layer_id=layer.layer_id,
                expert_id=group.expert_id,
                id_of_slot=id_of_slot,
                slot_for_layer=slot_for_layer,
                usage=usage,
                protected_until=protected_until,
            )
            done = self._compute_group(
                layer=layer,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                group=group,
                slot_id=slot_id,
                ready=ready,
                partials=partials,
                is_prefill=is_prefill,
                prefill_config=prefill_config,
            )
            protected_until[slot_id] = done
            step += 1
            usage[slot_id] = step
            self.cache.usage[slot_id] = step

        self.cache.step.fill_(step)
        from freetoken.kernel import moe_sum_reduce_triton

        output = torch.empty_like(hidden_states)
        moe_sum_reduce_triton(partials, output)
        if diagnostic.observer is not None:
            diagnostic.observer.placed_partials(partials)
        return output


__all__ = ["AlternativeAAdapter"]
