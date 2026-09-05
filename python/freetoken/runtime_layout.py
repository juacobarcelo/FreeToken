"""Allocation-free byte accounting for runtime expert-bank layouts."""

from __future__ import annotations


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def gpt_oss_mxfp4_expert_bytes_per_rank(
    *,
    hidden_size: int,
    intermediate_size: int,
    tensor_parallel_size: int,
    scalar_bytes: int,
) -> int:
    """Return one expert's bytes in one rank's MXFP4 Triton bank layout.

    This mirrors the six runtime banks built by
    ``freetoken.models.gpt_oss.weight._empty_mxfp4_triton_banks`` without
    importing Torch or allocating model tensors.
    """

    hidden = _positive_integer(hidden_size, "hidden_size")
    intermediate = _positive_integer(intermediate_size, "intermediate_size")
    parallel = _positive_integer(tensor_parallel_size, "tensor_parallel_size")
    scalar = _positive_integer(scalar_bytes, "scalar_bytes")
    if hidden % 32 != 0:
        raise ValueError("hidden_size must be divisible by 32")
    if intermediate % 32 != 0:
        raise ValueError("intermediate_size must be divisible by 32")

    intermediate_blocks = intermediate // 32
    blocks_per_rank = (intermediate_blocks + parallel - 1) // parallel
    local_intermediate = blocks_per_rank * 32

    gate_up_blocks = (hidden // 2) * (2 * local_intermediate)
    gate_up_scales = (hidden // 32) * (2 * local_intermediate)
    gate_up_bias = (2 * local_intermediate) * scalar
    down_blocks = (local_intermediate // 2) * hidden
    down_scales = (local_intermediate // 32) * hidden
    down_bias = hidden * scalar
    return sum(
        (
            gate_up_blocks,
            gate_up_scales,
            gate_up_bias,
            down_blocks,
            down_scales,
            down_bias,
        )
    )


__all__ = ["gpt_oss_mxfp4_expert_bytes_per_rank"]
