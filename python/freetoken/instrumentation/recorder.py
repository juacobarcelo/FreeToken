"""Live CUDA-backed recorder for the versioned MoE evidence contract.

All policy decisions remain in the existing FreeToken kernels.  The recorder only
copies their inputs and outputs into dedicated buffers, records CUDA events, and
materializes those observations after the forward has completed.  The explicit
forward-boundary synchronization is part of the measured instrumentation overhead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .records import MoeEventWriter

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.engine.config import EngineConfig
    from freetoken.moe.offload_cache import OffloadMoeCache


_NO_D2H_REASON = (
    "FreeToken keeps the authoritative expert banks in host memory; cache eviction "
    "discards a device copy and performs no device-to-host expert transfer"
)
_NO_CACHE_REASON = "the fused backend has no expert residency cache"
_PREFILL_RESIDENCY_REASON = (
    "prefill streams complete expert layers, so requested-route cache hits and misses "
    "are not a scheduling input"
)


def _not_applicable_transfer(reason: str) -> dict[str, Any]:
    return {
        "availability": "not_applicable",
        "operation_count": None,
        "object_count": None,
        "bytes": None,
        "duration_ns": None,
        "duration_clock": None,
        "reason": reason,
    }


class MoeInstrumentationRecorder:
    """Capture one complete MoE observation per model forward."""

    def __init__(self, config: "EngineConfig", device: torch.device) -> None:
        mode = config.moe_backend
        if mode not in {"fused", "offload"}:
            raise ValueError(
                "MoE instrumentation currently supports the fused and offload backends; "
                f"resolved backend was {mode!r}"
            )
        if config.tp_info.size != 1:
            raise ValueError("MoE instrumentation currently supports tensor-parallel size 1")
        output_dir = getattr(config, "moe_instrumentation_dir", None)
        run_id = getattr(config, "moe_instrumentation_run_id", None)
        if not output_dir or not run_id:
            raise ValueError(
                "moe_instrumentation_dir and moe_instrumentation_run_id are both required"
            )

        model = config.model_config
        self.device = device
        self.execution_mode = mode
        self.num_layers = model.num_moe_layers
        self.num_experts = model.num_experts
        self.top_k = model.num_experts_per_tok
        self.max_rows = max(config.max_forward_len, config.max_running_req)
        self._requested = torch.full(
            (self.num_layers, self.max_rows, self.top_k),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._compute_started = [
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(self.num_layers)
        ]
        self._compute_ended = [
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(self.num_layers)
        ]
        self._transfer_started = [
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(self.num_layers)
        ]
        self._transfer_ended = [
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(self.num_layers)
        ]

        self.cache: OffloadMoeCache | None = None
        self._pre_ids: torch.Tensor | None = None
        self._post_ids: torch.Tensor | None = None
        self._load_experts: torch.Tensor | None = None
        self._load_slots: torch.Tensor | None = None
        self._load_count: torch.Tensor | None = None
        self._all_experts = torch.arange(
            self.num_experts, dtype=torch.int32, device=self.device
        )
        self._expert_object_bytes: int | None = None
        self._decode_transfer_operations = 0
        self._prefill_transfer_operations = 0
        self._forward_started_ns: int | None = None
        self._decode_step = 0

        # The writer needs cache geometry, so fused can open immediately while offload
        # calls attach_cache after its banks have been allocated.
        self._writer: MoeEventWriter | None = None
        self._writer_args = {
            "output_dir": Path(output_dir),
            "run_id": run_id,
            "execution_mode": mode,
            "model": {
                "id": getattr(config, "served_model_name", None) or config.model_path,
                "num_moe_layers": self.num_layers,
                "num_experts": self.num_experts,
                "experts_per_token": self.top_k,
            },
        }
        if mode == "fused":
            self._writer = MoeEventWriter(
                **self._writer_args,
                expert_cache={
                    "availability": "not_applicable",
                    "capacity_objects": None,
                    "policy": None,
                    "expert_object_bytes": None,
                    "reason": _NO_CACHE_REASON,
                },
            )

    def attach_cache(self, cache: "OffloadMoeCache") -> None:
        if self.execution_mode != "offload":
            raise AssertionError("only offload instrumentation can attach a cache")
        if cache.prefill_hit_d2d:
            raise ValueError(
                "MoE instrumentation does not yet support --moe-prefill-hit-d2d; "
                "the selected issue-13 modes use complete-layer prefill transfers"
            )
        self.cache = cache
        self._pre_ids = torch.full(
            (self.num_layers, cache.cache_size), -1, dtype=torch.int32, device=self.device
        )
        self._post_ids = torch.full_like(self._pre_ids, -1)
        self._load_experts = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int32, device=self.device
        )
        self._load_slots = torch.full_like(self._load_experts, -1)
        self._load_count = torch.zeros(
            (self.num_layers,), dtype=torch.int64, device=self.device
        )
        self._expert_object_bytes = sum(
            int(bank[1][0].numel() * bank[1].element_size()) for bank in cache.banks
        )
        self._decode_transfer_operations = 1 if cache._copy_fused_ok else len(cache.banks)
        self._prefill_transfer_operations = (
            len(cache.banks) if cache.prefill_overlap else self._decode_transfer_operations
        )
        self._writer = MoeEventWriter(
            **self._writer_args,
            expert_cache={
                "availability": "measured",
                "capacity_objects": cache.cache_size,
                "policy": cache.cache_policy,
                "expert_object_bytes": self._expert_object_bytes,
                "reason": None,
            },
        )

    def record_routes(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        ids = expert_ids.reshape(-1, self.top_k)
        if ids.shape[0] > self.max_rows:
            raise RuntimeError(
                f"MoE instrumentation route rows {ids.shape[0]} exceed {self.max_rows}"
            )
        self._requested[layer_id, : ids.shape[0]].copy_(ids)

    def record_cache_before(self, layer_id: int) -> None:
        assert self.cache is not None and self._pre_ids is not None
        self._pre_ids[layer_id].copy_(self.cache.id_of_slot)

    def record_cache_after(self, layer_id: int) -> None:
        assert self.cache is not None
        assert self._post_ids is not None
        assert self._load_experts is not None
        assert self._load_slots is not None
        assert self._load_count is not None
        self._post_ids[layer_id].copy_(self.cache.id_of_slot)
        self._load_experts[layer_id].copy_(self.cache.src_indices)
        self._load_slots[layer_id].copy_(self.cache.evict_slots)
        self._load_count[layer_id].copy_(self.cache.num_indices[0])

    def record_prefill_cache_after(self, layer_id: int, buffer_id: int) -> None:
        """Record a complete-layer transfer into one transient prefill buffer."""

        assert self.cache is not None
        assert self._post_ids is not None
        assert self._load_experts is not None
        assert self._load_slots is not None
        assert self._load_count is not None
        slot_start = buffer_id * self.num_experts
        self._post_ids[layer_id].copy_(self.cache.id_of_slot)
        self._load_experts[layer_id].copy_(self._all_experts)
        self._load_slots[layer_id].copy_(self._all_experts + slot_start)
        self._load_count[layer_id].fill_(self.num_experts)

    def begin_transfer(self, layer_id: int) -> None:
        self._transfer_started[layer_id].record(torch.cuda.current_stream(self.device))

    def end_transfer(self, layer_id: int) -> None:
        self._transfer_ended[layer_id].record(torch.cuda.current_stream(self.device))

    def begin_compute(self, layer_id: int) -> None:
        self._compute_started[layer_id].record(torch.cuda.current_stream(self.device))

    def end_compute(self, layer_id: int) -> None:
        self._compute_ended[layer_id].record(torch.cuda.current_stream(self.device))

    def begin_forward(self) -> None:
        if self._writer is None:
            raise AssertionError("offload instrumentation cache was not attached")
        self._forward_started_ns = time.monotonic_ns()

    @staticmethod
    def _elapsed_ns(start: torch.cuda.Event, end: torch.cuda.Event) -> int:
        return max(0, round(start.elapsed_time(end) * 1_000_000))

    def finish_forward(self, batch: "Batch") -> None:
        if self._forward_started_ns is None or self._writer is None:
            raise AssertionError("begin_forward must precede finish_forward")
        # This one explicit synchronization protects the shared capture buffers from the
        # next overlapped forward and is intentionally included in the overhead comparison.
        torch.cuda.synchronize(self.device)
        ended_ns = time.monotonic_ns()
        # CUDA graphs execute padded dummy rows too. Export them because they can
        # affect cache loads; the first request_count rows are the active rows.
        rows = batch.padded_size if batch.is_decode else int(batch.input_ids.numel())
        if rows > self.max_rows:
            raise RuntimeError(f"MoE instrumentation rows {rows} exceed {self.max_rows}")

        requested = self._requested[:, :rows].cpu().tolist()
        pre_ids = self._pre_ids.cpu().tolist() if self._pre_ids is not None else None
        post_ids = self._post_ids.cpu().tolist() if self._post_ids is not None else None
        load_experts = (
            self._load_experts.cpu().tolist() if self._load_experts is not None else None
        )
        load_slots = self._load_slots.cpu().tolist() if self._load_slots is not None else None
        load_counts = self._load_count.cpu().tolist() if self._load_count is not None else None

        layer_records: list[dict[str, Any]] = []
        for layer_id in range(self.num_layers):
            routes = requested[layer_id]
            unique_requested = sorted({expert for row in routes for expert in row})
            loads: list[dict[str, Any]] = []
            evictions: list[dict[str, Any]] = []
            transitions: list[dict[str, Any]] = []
            if self.execution_mode == "offload":
                assert pre_ids is not None and post_ids is not None
                assert load_experts is not None and load_slots is not None
                assert load_counts is not None and self._expert_object_bytes is not None
                count = int(load_counts[layer_id])
                if count > self.num_experts:
                    raise RuntimeError(
                        f"MoE instrumentation load count {count} exceeds "
                        f"num_experts={self.num_experts}"
                    )
                destination = (
                    "device_transient"
                    if batch.is_prefill and self.cache is not None and self.cache.prefill_overlap
                    else "device_cache"
                )
                for slot_id, old_flat_id in enumerate(pre_ids[layer_id]):
                    if old_flat_id < 0 or int(post_ids[layer_id][slot_id]) == old_flat_id:
                        continue
                    eviction = {
                        "layer_id": old_flat_id // self.num_experts,
                        "expert_id": old_flat_id % self.num_experts,
                        "slot_id": slot_id,
                    }
                    evictions.append(eviction)
                    transitions.append(
                        {
                            **eviction,
                            "from": "device_cache",
                            "to": "host_source",
                            "cause": "eviction",
                        }
                    )
                for index in range(count):
                    expert_id = int(load_experts[layer_id][index])
                    slot_id = int(load_slots[layer_id][index])
                    load = {
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "slot_id": slot_id,
                        "bytes": self._expert_object_bytes,
                        "destination": destination,
                    }
                    loads.append(load)
                    new_flat_id = layer_id * self.num_experts + expert_id
                    if destination == "device_cache":
                        if not 0 <= slot_id < len(pre_ids[layer_id]):
                            raise RuntimeError(
                                f"MoE instrumentation cache slot {slot_id} is out of range"
                            )
                        if int(post_ids[layer_id][slot_id]) != new_flat_id:
                            raise RuntimeError(
                                "MoE instrumentation cache snapshot does not reconcile "
                                f"for layer={layer_id}, expert={expert_id}, slot={slot_id}"
                            )
                    transitions.append(
                        {
                            "layer_id": layer_id,
                            "expert_id": expert_id,
                            "slot_id": slot_id,
                            "from": "host_source",
                            "to": destination,
                            "cause": "load",
                        }
                    )

                if batch.is_decode:
                    resident = set(int(item) for item in pre_ids[layer_id] if item >= 0)
                    hits = sum(
                        layer_id * self.num_experts + expert_id in resident
                        for expert_id in unique_requested
                    )
                    residency = {
                        "availability": "measured",
                        "hit_count": hits,
                        "miss_count": len(unique_requested) - hits,
                        "unit": "unique_expert_objects",
                        "reason": None,
                    }
                else:
                    residency = {
                        "availability": "not_applicable",
                        "hit_count": None,
                        "miss_count": None,
                        "unit": "unique_expert_objects",
                        "reason": _PREFILL_RESIDENCY_REASON,
                    }
                duration_ns = (
                    self._elapsed_ns(
                        self._transfer_started[layer_id], self._transfer_ended[layer_id]
                    )
                    if count
                    else 0
                )
                operations = (
                    self._prefill_transfer_operations
                    if batch.is_prefill
                    else self._decode_transfer_operations
                )
                h2d = {
                    "availability": "measured",
                    "operation_count": operations if count else 0,
                    "object_count": count,
                    "bytes": count * self._expert_object_bytes,
                    "duration_ns": duration_ns,
                    "duration_clock": "cuda_event",
                    "reason": None,
                }
            else:
                residency = {
                    "availability": "not_applicable",
                    "hit_count": None,
                    "miss_count": None,
                    "unit": "unique_expert_objects",
                    "reason": _NO_CACHE_REASON,
                }
                h2d = _not_applicable_transfer(_NO_CACHE_REASON)

            layer_records.append(
                {
                    "layer_id": layer_id,
                    "requested_expert_ids": routes,
                    "requested_shape": [rows, self.top_k],
                    "residency": residency,
                    "loads": loads,
                    "evictions": evictions,
                    "residency_transitions": transitions,
                    "transfers": {
                        "h2d": h2d,
                        "d2h": _not_applicable_transfer(_NO_D2H_REASON),
                    },
                    "compute": {
                        "availability": "measured",
                        "duration_ns": self._elapsed_ns(
                            self._compute_started[layer_id], self._compute_ended[layer_id]
                        ),
                        "duration_clock": "cuda_event",
                        "boundary": "routed_expert_kernel",
                        "reason": None,
                    },
                }
            )

        decode_step = self._decode_step if batch.is_decode else None
        self._writer.write_forward(
            {
                "phase": batch.phase,
                "decode_step": decode_step,
                "started_monotonic_ns": self._forward_started_ns,
                "ended_monotonic_ns": ended_ns,
                "batch": {
                    "request_count": batch.size,
                    "padded_request_count": batch.padded_size,
                    "token_row_count": rows,
                    "active_token_row_count": batch.size if batch.is_decode else rows,
                },
                "layers": layer_records,
            }
        )
        if batch.is_decode:
            self._decode_step += 1
        self._forward_started_ns = None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
