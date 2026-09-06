import pytest

from freetoken.runtime_layout import gpt_oss_mxfp4_expert_bytes_per_rank


def test_gpt_oss_120b_tp2_runtime_expert_bytes() -> None:
    per_rank = gpt_oss_mxfp4_expert_bytes_per_rank(
        hidden_size=2880,
        intermediate_size=2880,
        tensor_parallel_size=2,
        scalar_bytes=2,
    )

    assert per_rank == 6_621_120
    assert per_rank * 2 == 13_242_240
    assert per_rank * 2 - 13_236_480 == 5_760


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_size", 2879, "hidden_size must be divisible by 32"),
        ("intermediate_size", 2879, "intermediate_size must be divisible by 32"),
        ("tensor_parallel_size", 0, "tensor_parallel_size must be a positive integer"),
        ("scalar_bytes", True, "scalar_bytes must be a positive integer"),
    ],
)
def test_runtime_expert_bytes_reject_invalid_geometry(
    field: str,
    value: int,
    message: str,
) -> None:
    values = {
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "tensor_parallel_size": 2,
        "scalar_bytes": 2,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        gpt_oss_mxfp4_expert_bytes_per_rank(**values)
