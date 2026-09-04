from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from freetoken.instrumentation.records import (
    AGGREGATE_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    InstrumentationError,
    MoeEventWriter,
    aggregate_records,
    instrumentation_enabled,
    load_event_file,
    validate_forward_record,
    validate_run_id,
)


def _not_applicable(reason: str) -> dict:
    return {
        "availability": "not_applicable",
        "operation_count": None,
        "object_count": None,
        "bytes": None,
        "duration_ns": None,
        "duration_clock": None,
        "reason": reason,
    }


def _run(*, mode: str = "offload") -> dict:
    cache = (
        {
            "availability": "measured",
            "capacity_objects": 2,
            "policy": "lru",
            "expert_object_bytes": 10,
            "initial_resident_objects": [
                {"layer_id": 0, "expert_id": 3, "slot_id": 0},
                {"layer_id": 0, "expert_id": 1, "slot_id": 1},
            ],
            "initial_state_boundary": "before_first_observed_forward",
            "reason": None,
        }
        if mode == "offload"
        else {
            "availability": "not_applicable",
            "capacity_objects": None,
            "policy": None,
            "expert_object_bytes": None,
            "initial_resident_objects": None,
            "initial_state_boundary": None,
            "reason": "fused experts are resident",
        }
    )
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "record_type": "run",
        "sequence": 0,
        "run_id": "tiny-offload-01",
        "timestamp_monotonic_ns": 10,
        "created_at_utc": "2026-09-01T00:00:00.000Z",
        "runtime": "freetoken",
        "execution_mode": mode,
        "sequence_id_namespace": "freetoken-request-uid",
        "tensor_parallel": {
            "rank": 0,
            "world_size": 1,
            "route_replication": "replicated",
        },
        "model": {
            "id": "tiny-gpt-oss",
            "num_moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
        },
        "expert_cache": cache,
        "clock": {
            "timestamp_source": "time.monotonic_ns",
            "timestamp_unit": "nanosecond",
            "duration_source": "cuda_event",
            "duration_unit": "nanosecond",
        },
    }


def _offload_forward() -> dict:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "record_type": "forward",
        "sequence": 1,
        "run_id": "tiny-offload-01",
        "execution_mode": "offload",
        "phase": "decode",
        "decode_step": 0,
        "started_monotonic_ns": 20,
        "ended_monotonic_ns": 100,
        "batch": {
            "request_count": 1,
            "padded_request_count": 2,
            "token_row_count": 2,
            "active_token_row_count": 1,
            "request_sequence_ids": [7],
            "token_row_sequence_ids": [7, None],
            "token_positions": [12, None],
        },
        "layers": [
            {
                "layer_id": 0,
                "requested_expert_ids": [[1, 2], [0, 1]],
                "requested_shape": [2, 2],
                "residency": {
                    "availability": "measured",
                    "hit_count": 1,
                    "miss_count": 2,
                    "unit": "unique_expert_objects",
                    "reason": None,
                },
                "loads": [
                    {
                        "layer_id": 0,
                        "expert_id": 0,
                        "slot_id": 0,
                        "bytes": 10,
                        "destination": "device_cache",
                    },
                    {
                        "layer_id": 0,
                        "expert_id": 2,
                        "slot_id": 1,
                        "bytes": 10,
                        "destination": "device_cache",
                    },
                ],
                "evictions": [
                    {"layer_id": 0, "expert_id": 3, "slot_id": 0},
                    {"layer_id": 0, "expert_id": 1, "slot_id": 1},
                ],
                "residency_transitions": [
                    {
                        "layer_id": 0,
                        "expert_id": 3,
                        "slot_id": 0,
                        "from": "device_cache",
                        "to": "host_source",
                        "cause": "eviction",
                    },
                    {
                        "layer_id": 0,
                        "expert_id": 0,
                        "slot_id": 0,
                        "from": "host_source",
                        "to": "device_cache",
                        "cause": "load",
                    },
                    {
                        "layer_id": 0,
                        "expert_id": 1,
                        "slot_id": 1,
                        "from": "device_cache",
                        "to": "host_source",
                        "cause": "eviction",
                    },
                    {
                        "layer_id": 0,
                        "expert_id": 2,
                        "slot_id": 1,
                        "from": "host_source",
                        "to": "device_cache",
                        "cause": "load",
                    },
                ],
                "transfers": {
                    "h2d": {
                        "availability": "measured",
                        "operation_count": 1,
                        "object_count": 2,
                        "bytes": 20,
                        "duration_ns": 30,
                        "duration_clock": "cuda_event",
                        "reason": None,
                    },
                    "d2h": _not_applicable("eviction discards a device copy"),
                },
                "compute": {
                    "availability": "measured",
                    "duration_ns": 40,
                    "duration_clock": "cuda_event",
                    "boundary": "routed_expert_kernel",
                    "reason": None,
                },
            }
        ],
    }


def _run_end() -> dict:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "record_type": "run_end",
        "sequence": 2,
        "run_id": "tiny-offload-01",
        "timestamp_monotonic_ns": 120,
        "ended_at_utc": "2026-09-01T00:00:01.000Z",
        "status": "complete",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_instrumentation_is_disabled_without_opt_in(tmp_path: Path) -> None:
    assert instrumentation_enabled(None, None) is False
    assert list(tmp_path.iterdir()) == []


def test_instrumentation_requires_paired_opt_in_settings() -> None:
    with pytest.raises(InstrumentationError, match="must be supplied together"):
        instrumentation_enabled("evidence", None)


def test_valid_trace_reconciles_and_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "trace.events.jsonl"
    _write_jsonl(path, [_run(), _offload_forward(), _run_end()])

    records = load_event_file(path)
    aggregate = aggregate_records(records)

    assert aggregate["schema_version"] == AGGREGATE_SCHEMA_VERSION
    assert aggregate["complete"] is True
    assert aggregate["forward_count"] == 1
    totals = aggregate["totals"]
    assert totals["requested_expert_occurrences"]["value"] == 4
    assert totals["unique_requested_expert_objects"]["value"] == 3
    assert totals["residency_hits"]["value"] == 1
    assert totals["residency_misses"]["value"] == 2
    assert totals["expert_loads"]["value"] == 2
    assert totals["expert_evictions"]["value"] == 2
    assert totals["h2d_transfer_bytes"]["value"] == 20
    assert totals["d2h_transfer_bytes"]["availability"] == "not_applicable"


def test_rejects_non_monotonic_trace(tmp_path: Path) -> None:
    forward = _offload_forward()
    forward["started_monotonic_ns"] = 9
    path = tmp_path / "bad.events.jsonl"
    _write_jsonl(path, [_run(), forward])

    with pytest.raises(InstrumentationError, match="not monotonic"):
        load_event_file(path)


def test_rejects_cache_movement_that_cannot_replay(tmp_path: Path) -> None:
    run = _run()
    run["expert_cache"]["initial_resident_objects"][0]["expert_id"] = 0
    path = tmp_path / "bad-cache.events.jsonl"
    _write_jsonl(path, [run, _offload_forward()])

    with pytest.raises(InstrumentationError, match="cache replay"):
        load_event_file(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda event: event["layers"][0]["loads"][0].update(bytes=9), "cache geometry"),
        (
            lambda event: event["layers"][0]["residency_transitions"][1].update(
                expert_id=1
            ),
            "loads do not reconcile",
        ),
        (
            lambda event: event["layers"][0]["transfers"]["h2d"].update(bytes=19),
            "must equal load bytes",
        ),
        (
            lambda event: event["batch"]["token_row_sequence_ids"].__setitem__(1, 7),
            "identity and position must both be null",
        ),
    ],
)
def test_rejects_inconsistent_movement(mutation, message: str) -> None:
    event = _offload_forward()
    mutation(event)

    with pytest.raises(InstrumentationError, match=message):
        validate_forward_record(event, _run())


def test_fused_unavailable_values_remain_explicit() -> None:
    run = _run(mode="fused")
    run["run_id"] = "tiny-fused-01"
    forward = _offload_forward()
    forward.update(run_id="tiny-fused-01", execution_mode="fused")
    layer = forward["layers"][0]
    layer["residency"] = {
        "availability": "not_applicable",
        "hit_count": None,
        "miss_count": None,
        "unit": "unique_expert_objects",
        "reason": "fused experts are resident",
    }
    layer["loads"] = []
    layer["evictions"] = []
    layer["residency_transitions"] = []
    layer["transfers"]["h2d"] = _not_applicable("fused experts are resident")

    validate_forward_record(forward, run)
    aggregate = aggregate_records([run, forward])

    assert aggregate["totals"]["residency_hits"]["availability"] == "not_applicable"
    assert aggregate["totals"]["h2d_transfer_bytes"]["value"] is None


def test_writer_refuses_overwrite_and_binds_aggregate(tmp_path: Path) -> None:
    writer = MoeEventWriter(
        tmp_path,
        "writer-01",
        execution_mode="fused",
        model={
            "id": "tiny-gpt-oss",
            "num_moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
        },
        expert_cache={
            "availability": "not_applicable",
            "capacity_objects": None,
            "policy": None,
            "expert_object_bytes": None,
            "initial_resident_objects": None,
            "initial_state_boundary": None,
            "reason": "fused experts are resident",
        },
    )
    now = time.monotonic_ns()
    forward = _offload_forward()
    forward.pop("schema_version")
    forward.pop("record_type")
    forward.pop("sequence")
    forward.pop("run_id")
    forward.pop("execution_mode")
    forward.update(started_monotonic_ns=now, ended_monotonic_ns=now + 1)
    layer = forward["layers"][0]
    layer["residency"] = {
        "availability": "not_applicable",
        "hit_count": None,
        "miss_count": None,
        "unit": "unique_expert_objects",
        "reason": "fused experts are resident",
    }
    layer["loads"] = []
    layer["evictions"] = []
    layer["residency_transitions"] = []
    layer["transfers"]["h2d"] = _not_applicable("fused experts are resident")
    writer.write_forward(forward)
    aggregate = writer.close()

    assert aggregate is not None
    assert aggregate["complete"] is True
    assert aggregate["tensor_parallel"] == {
        "rank": 0,
        "world_size": 1,
        "route_replication": "replicated",
    }
    assert len(aggregate["source"]["events_sha256"]) == 64
    with pytest.raises(InstrumentationError, match="overwrite"):
        MoeEventWriter(
            tmp_path,
            "writer-01",
            execution_mode="fused",
            model={
                "id": "tiny-gpt-oss",
                "num_moe_layers": 1,
                "num_experts": 4,
                "experts_per_token": 2,
            },
            expert_cache={
                "availability": "not_applicable",
                "capacity_objects": None,
                "policy": None,
                "expert_object_bytes": None,
                "initial_resident_objects": None,
                "initial_state_boundary": None,
                "reason": "fused experts are resident",
            },
        )


def test_tp2_writers_use_unique_rank_paths_and_metadata(tmp_path: Path) -> None:
    common = {
        "execution_mode": "fused",
        "model": {
            "id": "tiny-gpt-oss",
            "num_moe_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
        },
        "expert_cache": {
            "availability": "not_applicable",
            "capacity_objects": None,
            "policy": None,
            "expert_object_bytes": None,
            "initial_resident_objects": None,
            "initial_state_boundary": None,
            "reason": "fused experts are resident",
        },
    }
    rank_zero = MoeEventWriter(
        tmp_path,
        "writer-tp2",
        **common,
        tensor_parallel={
            "rank": 0,
            "world_size": 2,
            "route_replication": "replicated",
        },
    )
    rank_one = MoeEventWriter(
        tmp_path,
        "writer-tp2",
        **common,
        tensor_parallel={
            "rank": 1,
            "world_size": 2,
            "route_replication": "replicated",
        },
    )

    assert rank_zero.events_path.name == "writer-tp2.tp-rank-00-of-02.events.jsonl"
    assert rank_one.events_path.name == "writer-tp2.tp-rank-01-of-02.events.jsonl"
    assert rank_zero.events_path != rank_one.events_path
    assert json.loads(rank_zero.events_path.read_text().splitlines()[0])[
        "tensor_parallel"
    ]["rank"] == 0
    assert json.loads(rank_one.events_path.read_text().splitlines()[0])[
        "tensor_parallel"
    ]["rank"] == 1
    rank_zero.close()
    rank_one.close()


def test_schema_10_trace_remains_readable(tmp_path: Path) -> None:
    records = [_run(), _offload_forward(), _run_end()]
    records[0].pop("tensor_parallel")
    for record in records:
        record["schema_version"] = "1.0"
    path = tmp_path / "legacy.events.jsonl"
    _write_jsonl(path, records)

    loaded = load_event_file(path)
    aggregate = aggregate_records(loaded)

    assert loaded[0]["schema_version"] == "1.0"
    assert aggregate["event_schema_version"] == "1.0"
    assert aggregate["tensor_parallel"]["world_size"] == 1


def test_schema_files_match_code_versions() -> None:
    schema_dir = (
        Path(__file__).parents[2]
        / "python"
        / "freetoken"
        / "instrumentation"
        / "schemas"
    )
    events = json.loads((schema_dir / "moe-events-v1.schema.json").read_text())
    aggregate = json.loads((schema_dir / "moe-aggregate-v1.schema.json").read_text())

    assert events["$defs"]["run"]["properties"]["schema_version"]["const"] == EVENT_SCHEMA_VERSION
    assert aggregate["properties"]["schema_version"]["const"] == AGGREGATE_SCHEMA_VERSION
    assert EVENT_SCHEMA_VERSION in aggregate["properties"]["event_schema_version"]["enum"]


@pytest.mark.parametrize("run_id", ["../escape", "has spaces", "", "x" * 129])
def test_run_id_is_filename_safe(run_id: str) -> None:
    with pytest.raises(InstrumentationError):
        validate_run_id(run_id)
