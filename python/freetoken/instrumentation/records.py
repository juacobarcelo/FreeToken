"""Versioned FreeToken MoE event records and deterministic aggregation."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

EVENT_SCHEMA_VERSION = "1.0"
AGGREGATE_SCHEMA_VERSION = "1.0"
AVAILABILITIES = frozenset({"measured", "unavailable", "unsupported", "not_applicable"})
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class InstrumentationError(ValueError):
    """A trace record violates the versioned evidence contract."""


def validate_run_id(run_id: str) -> None:
    """Validate the stable filename-safe run identifier."""

    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise InstrumentationError("run_id contains unsupported characters")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InstrumentationError(f"{path}: must be an object")
    return value


def _require_list(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise InstrumentationError(f"{path}: must be an array")
    return value


def _require_int(value: object, path: str, *, minimum: int = 0) -> int:
    if not _is_integer(value) or value < minimum:
        raise InstrumentationError(f"{path}: must be an integer >= {minimum}")
    return value


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InstrumentationError(f"{path}: must be a non-empty string")
    return value


def _validate_availability(value: object, path: str) -> Mapping[str, Any]:
    item = _require_mapping(value, path)
    availability = item.get("availability")
    if availability not in AVAILABILITIES:
        raise InstrumentationError(
            f"{path}.availability: must be one of {sorted(AVAILABILITIES)}"
        )
    reason = item.get("reason")
    if availability == "measured":
        if reason is not None:
            raise InstrumentationError(f"{path}.reason: measured values require null")
    elif not isinstance(reason, str) or not reason:
        raise InstrumentationError(f"{path}.reason: unavailable values require a cause")
    return item


def _validate_metric(value: object, path: str) -> Mapping[str, Any]:
    item = _validate_availability(value, path)
    unit = _require_string(item.get("unit"), f"{path}.unit")
    metric_value = item.get("value")
    if item["availability"] == "measured":
        _require_int(metric_value, f"{path}.value")
    elif metric_value is not None:
        raise InstrumentationError(f"{path}.value: must be null when not measured")
    if not unit:
        raise AssertionError("unit validation returned an empty value")
    return item


def _validate_transfer(value: object, path: str) -> Mapping[str, Any]:
    item = _validate_availability(value, path)
    fields = ("operation_count", "object_count", "bytes", "duration_ns")
    if item["availability"] == "measured":
        for field in fields:
            _require_int(item.get(field), f"{path}.{field}")
        if item.get("duration_clock") != "cuda_event":
            raise InstrumentationError(f"{path}.duration_clock: must be cuda_event")
    else:
        for field in fields:
            if item.get(field) is not None:
                raise InstrumentationError(f"{path}.{field}: must be null when not measured")
        if item.get("duration_clock") is not None:
            raise InstrumentationError(
                f"{path}.duration_clock: must be null when not measured"
            )
    return item


def _validate_compute(value: object, path: str) -> Mapping[str, Any]:
    item = _validate_availability(value, path)
    if item["availability"] == "measured":
        _require_int(item.get("duration_ns"), f"{path}.duration_ns")
        if item.get("duration_clock") != "cuda_event":
            raise InstrumentationError(f"{path}.duration_clock: must be cuda_event")
        _require_string(item.get("boundary"), f"{path}.boundary")
    else:
        for field in ("duration_ns", "duration_clock", "boundary"):
            if item.get(field) is not None:
                raise InstrumentationError(f"{path}.{field}: must be null when not measured")
    return item


def validate_run_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise InstrumentationError("run.schema_version: unsupported event schema")
    if record.get("record_type") != "run":
        raise InstrumentationError("run.record_type: must be run")
    if record.get("sequence") != 0:
        raise InstrumentationError("run.sequence: must be 0")
    run_id = _require_string(record.get("run_id"), "run.run_id")
    if not _RUN_ID.fullmatch(run_id):
        raise InstrumentationError("run.run_id: contains unsupported characters")
    _require_int(record.get("timestamp_monotonic_ns"), "run.timestamp_monotonic_ns")
    _require_string(record.get("created_at_utc"), "run.created_at_utc")
    if record.get("runtime") != "freetoken":
        raise InstrumentationError("run.runtime: must be freetoken")
    mode = record.get("execution_mode")
    if mode not in {"fused", "offload"}:
        raise InstrumentationError("run.execution_mode: must be fused or offload")
    if record.get("sequence_id_namespace") != "freetoken-request-uid":
        raise InstrumentationError("run.sequence_id_namespace: unsupported namespace")
    model = _require_mapping(record.get("model"), "run.model")
    for field in ("num_moe_layers", "num_experts", "experts_per_token"):
        _require_int(model.get(field), f"run.model.{field}", minimum=1)
    if model["experts_per_token"] > model["num_experts"]:
        raise InstrumentationError("run.model: experts_per_token exceeds num_experts")
    _require_string(model.get("id"), "run.model.id")
    cache = _validate_availability(record.get("expert_cache"), "run.expert_cache")
    if cache["availability"] == "measured":
        _require_int(cache.get("capacity_objects"), "run.expert_cache.capacity_objects", minimum=1)
        _require_string(cache.get("policy"), "run.expert_cache.policy")
        _require_int(
            cache.get("expert_object_bytes"),
            "run.expert_cache.expert_object_bytes",
            minimum=1,
        )
        initial = _require_list(
            cache.get("initial_resident_objects"),
            "run.expert_cache.initial_resident_objects",
        )
        if initial:
            raise InstrumentationError(
                "run.expert_cache.initial_resident_objects: FreeToken starts with an empty cache"
            )
        if cache.get("initial_state_boundary") != "before_first_observed_forward":
            raise InstrumentationError(
                "run.expert_cache.initial_state_boundary: unsupported boundary"
            )
    else:
        for field in (
            "capacity_objects",
            "policy",
            "expert_object_bytes",
            "initial_resident_objects",
            "initial_state_boundary",
        ):
            if cache.get(field) is not None:
                raise InstrumentationError(
                    f"run.expert_cache.{field}: must be null when cache is not applicable"
                )
    if mode == "offload" and cache["availability"] != "measured":
        raise InstrumentationError("run.expert_cache: offload requires measured cache geometry")
    if mode == "fused" and cache["availability"] != "not_applicable":
        raise InstrumentationError("run.expert_cache: fused requires not_applicable")
    clock = _require_mapping(record.get("clock"), "run.clock")
    if clock != {
        "duration_source": "cuda_event",
        "duration_unit": "nanosecond",
        "timestamp_source": "time.monotonic_ns",
        "timestamp_unit": "nanosecond",
    }:
        raise InstrumentationError("run.clock: unsupported clock declaration")


def _validate_layer(
    value: object,
    path: str,
    *,
    expected_layer: int,
    num_layers: int,
    num_experts: int,
    experts_per_token: int,
    execution_mode: str,
    phase: str,
    expert_object_bytes: int | None,
    cache_capacity: int | None,
) -> None:
    layer = _require_mapping(value, path)
    if layer.get("layer_id") != expected_layer:
        raise InstrumentationError(f"{path}.layer_id: layers must be ordered and contiguous")
    requested = _require_list(layer.get("requested_expert_ids"), f"{path}.requested_expert_ids")
    shape = _require_list(layer.get("requested_shape"), f"{path}.requested_shape")
    if len(shape) != 2:
        raise InstrumentationError(f"{path}.requested_shape: must have two dimensions")
    rows = _require_int(shape[0], f"{path}.requested_shape[0]")
    top_k = _require_int(shape[1], f"{path}.requested_shape[1]", minimum=1)
    if top_k != experts_per_token or len(requested) != rows:
        raise InstrumentationError(f"{path}.requested_shape: does not match routing rows")
    for row_index, row_value in enumerate(requested):
        row = _require_list(row_value, f"{path}.requested_expert_ids[{row_index}]")
        if len(row) != top_k:
            raise InstrumentationError(
                f"{path}.requested_expert_ids[{row_index}]: wrong top-k width"
            )
        if len(set(row)) != len(row):
            raise InstrumentationError(
                f"{path}.requested_expert_ids[{row_index}]: duplicate expert id"
            )
        for column, expert_id in enumerate(row):
            parsed = _require_int(
                expert_id,
                f"{path}.requested_expert_ids[{row_index}][{column}]",
            )
            if parsed >= num_experts:
                raise InstrumentationError(
                    f"{path}.requested_expert_ids[{row_index}][{column}]: out of range"
                )

    residency = _validate_availability(layer.get("residency"), f"{path}.residency")
    if residency.get("unit") != "unique_expert_objects":
        raise InstrumentationError(f"{path}.residency.unit: unsupported unit")
    unique_requested = len({expert for row in requested for expert in row})
    if residency["availability"] == "measured":
        hits = _require_int(residency.get("hit_count"), f"{path}.residency.hit_count")
        misses = _require_int(residency.get("miss_count"), f"{path}.residency.miss_count")
        if hits + misses != unique_requested:
            raise InstrumentationError(
                f"{path}.residency: hit_count + miss_count must equal unique requests"
            )
    else:
        for field in ("hit_count", "miss_count"):
            if residency.get(field) is not None:
                raise InstrumentationError(f"{path}.residency.{field}: must be null")

    loads = _require_list(layer.get("loads"), f"{path}.loads")
    for index, raw in enumerate(loads):
        load = _require_mapping(raw, f"{path}.loads[{index}]")
        if load.get("layer_id") != expected_layer:
            raise InstrumentationError(f"{path}.loads[{index}].layer_id: wrong layer")
        expert_id = _require_int(load.get("expert_id"), f"{path}.loads[{index}].expert_id")
        if expert_id >= num_experts:
            raise InstrumentationError(f"{path}.loads[{index}].expert_id: out of range")
        slot_id = _require_int(load.get("slot_id"), f"{path}.loads[{index}].slot_id")
        _require_int(load.get("bytes"), f"{path}.loads[{index}].bytes", minimum=1)
        if load.get("destination") not in {"device_cache", "device_transient"}:
            raise InstrumentationError(f"{path}.loads[{index}].destination: unsupported")
        if phase == "decode" and load.get("destination") != "device_cache":
            raise InstrumentationError(
                f"{path}.loads[{index}].destination: decode requires device_cache"
            )
        if expert_object_bytes is None or load.get("bytes") != expert_object_bytes:
            raise InstrumentationError(
                f"{path}.loads[{index}].bytes: does not match run cache geometry"
            )
        if cache_capacity is None or slot_id >= cache_capacity:
            raise InstrumentationError(f"{path}.loads[{index}].slot_id: out of range")
    load_keys = [
        (load["layer_id"], load["expert_id"], load["slot_id"], load["destination"])
        for load in loads
    ]
    if len(load_keys) != len(set(load_keys)):
        raise InstrumentationError(f"{path}.loads: duplicate load")
    loaded_experts = [load["expert_id"] for load in loads]
    if len(loaded_experts) != len(set(loaded_experts)):
        raise InstrumentationError(f"{path}.loads: expert loaded more than once")

    evictions = _require_list(layer.get("evictions"), f"{path}.evictions")
    for index, raw in enumerate(evictions):
        eviction = _require_mapping(raw, f"{path}.evictions[{index}]")
        evicted_layer = _require_int(
            eviction.get("layer_id"), f"{path}.evictions[{index}].layer_id"
        )
        if evicted_layer >= num_layers:
            raise InstrumentationError(f"{path}.evictions[{index}].layer_id: out of range")
        expert_id = _require_int(
            eviction.get("expert_id"), f"{path}.evictions[{index}].expert_id"
        )
        if expert_id >= num_experts:
            raise InstrumentationError(f"{path}.evictions[{index}].expert_id: out of range")
        eviction_slot = _require_int(
            eviction.get("slot_id"), f"{path}.evictions[{index}].slot_id"
        )
        if cache_capacity is None or eviction_slot >= cache_capacity:
            raise InstrumentationError(f"{path}.evictions[{index}].slot_id: out of range")

    transitions = _require_list(
        layer.get("residency_transitions"), f"{path}.residency_transitions"
    )
    if len(transitions) != len(loads) + len(evictions):
        raise InstrumentationError(
            f"{path}.residency_transitions: must reconcile with loads and evictions"
        )
    load_transition_keys: list[tuple[int, int, int, str]] = []
    eviction_transition_keys: list[tuple[int, int, int]] = []
    for index, raw in enumerate(transitions):
        transition = _require_mapping(raw, f"{path}.residency_transitions[{index}]")
        if transition.get("cause") not in {"load", "eviction"}:
            raise InstrumentationError(
                f"{path}.residency_transitions[{index}].cause: unsupported"
            )
        source = _require_string(
            transition.get("from"), f"{path}.residency_transitions[{index}].from"
        )
        destination = _require_string(
            transition.get("to"), f"{path}.residency_transitions[{index}].to"
        )
        transition_layer = _require_int(
            transition.get("layer_id"), f"{path}.residency_transitions[{index}].layer_id"
        )
        transition_expert = _require_int(
            transition.get("expert_id"), f"{path}.residency_transitions[{index}].expert_id"
        )
        transition_slot = _require_int(
            transition.get("slot_id"), f"{path}.residency_transitions[{index}].slot_id"
        )
        if transition_layer >= num_layers or transition_expert >= num_experts:
            raise InstrumentationError(
                f"{path}.residency_transitions[{index}]: object id out of range"
            )
        if transition["cause"] == "load":
            if source != "host_source" or destination not in {
                "device_cache",
                "device_transient",
            }:
                raise InstrumentationError(
                    f"{path}.residency_transitions[{index}]: invalid load transition"
                )
            load_transition_keys.append(
                (transition_layer, transition_expert, transition_slot, destination)
            )
        else:
            if source != "device_cache" or destination != "host_source":
                raise InstrumentationError(
                    f"{path}.residency_transitions[{index}]: invalid eviction transition"
                )
            eviction_transition_keys.append(
                (transition_layer, transition_expert, transition_slot)
            )
    eviction_keys = [
        (eviction["layer_id"], eviction["expert_id"], eviction["slot_id"])
        for eviction in evictions
    ]
    if len(eviction_keys) != len(set(eviction_keys)):
        raise InstrumentationError(f"{path}.evictions: duplicate eviction")
    if sorted(load_transition_keys) != sorted(load_keys):
        raise InstrumentationError(f"{path}.residency_transitions: loads do not reconcile")
    if sorted(eviction_transition_keys) != sorted(eviction_keys):
        raise InstrumentationError(f"{path}.residency_transitions: evictions do not reconcile")

    transfers = _require_mapping(layer.get("transfers"), f"{path}.transfers")
    h2d = _validate_transfer(transfers.get("h2d"), f"{path}.transfers.h2d")
    _validate_transfer(transfers.get("d2h"), f"{path}.transfers.d2h")
    d2h = transfers["d2h"]
    if d2h["availability"] != "not_applicable":
        raise InstrumentationError(f"{path}.transfers.d2h: schema 1.0 requires not_applicable")
    if h2d["availability"] == "measured":
        if h2d["object_count"] != len(loads):
            raise InstrumentationError(f"{path}.transfers.h2d.object_count: must equal loads")
        if h2d["bytes"] != sum(load["bytes"] for load in loads):
            raise InstrumentationError(f"{path}.transfers.h2d.bytes: must equal load bytes")
    elif loads:
        raise InstrumentationError(f"{path}.loads: cannot exist without measured H2D transfer")
    if execution_mode == "fused":
        if loads or evictions or transitions:
            raise InstrumentationError(f"{path}: fused mode cannot report cache movement")
        if residency["availability"] != "not_applicable":
            raise InstrumentationError(f"{path}.residency: fused mode requires not_applicable")
        if h2d["availability"] != "not_applicable":
            raise InstrumentationError(f"{path}.transfers.h2d: fused mode requires not_applicable")
    elif phase == "decode":
        if residency["availability"] != "measured":
            raise InstrumentationError(f"{path}.residency: offload decode requires measured")
        if residency["miss_count"] != len(loads):
            raise InstrumentationError(f"{path}.loads: count must equal decode misses")
        if not set(loaded_experts).issubset(
            {expert for row in requested for expert in row}
        ):
            raise InstrumentationError(f"{path}.loads: decode loaded an unrequested expert")
    elif residency["availability"] != "not_applicable":
        raise InstrumentationError(f"{path}.residency: offload prefill requires not_applicable")
    elif set(loaded_experts) != set(range(num_experts)):
        raise InstrumentationError(f"{path}.loads: prefill must load the complete layer")
    _validate_compute(layer.get("compute"), f"{path}.compute")


def validate_forward_record(record: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    if record.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise InstrumentationError("forward.schema_version: unsupported event schema")
    if record.get("record_type") != "forward":
        raise InstrumentationError("forward.record_type: must be forward")
    if record.get("run_id") != run.get("run_id"):
        raise InstrumentationError("forward.run_id: must match run record")
    _require_int(record.get("sequence"), "forward.sequence", minimum=1)
    started = _require_int(record.get("started_monotonic_ns"), "forward.started_monotonic_ns")
    ended = _require_int(record.get("ended_monotonic_ns"), "forward.ended_monotonic_ns")
    if ended < started:
        raise InstrumentationError("forward timestamps: end precedes start")
    phase = record.get("phase")
    if phase not in {"prefill", "decode"}:
        raise InstrumentationError("forward.phase: must be prefill or decode")
    decode_step = record.get("decode_step")
    if phase == "decode":
        _require_int(decode_step, "forward.decode_step")
    elif decode_step is not None:
        raise InstrumentationError("forward.decode_step: prefill requires null")
    if record.get("execution_mode") != run.get("execution_mode"):
        raise InstrumentationError("forward.execution_mode: must match run record")
    batch = _require_mapping(record.get("batch"), "forward.batch")
    request_count = _require_int(
        batch.get("request_count"), "forward.batch.request_count", minimum=1
    )
    padded_count = _require_int(
        batch.get("padded_request_count"), "forward.batch.padded_request_count", minimum=1
    )
    token_rows = _require_int(
        batch.get("token_row_count"), "forward.batch.token_row_count", minimum=1
    )
    active_rows = _require_int(
        batch.get("active_token_row_count"),
        "forward.batch.active_token_row_count",
        minimum=1,
    )
    request_sequence_ids = _require_list(
        batch.get("request_sequence_ids"), "forward.batch.request_sequence_ids"
    )
    token_sequence_ids = _require_list(
        batch.get("token_row_sequence_ids"), "forward.batch.token_row_sequence_ids"
    )
    token_positions = _require_list(
        batch.get("token_positions"), "forward.batch.token_positions"
    )
    if padded_count < request_count:
        raise InstrumentationError("forward.batch: padded count is smaller than request count")
    if active_rows > token_rows:
        raise InstrumentationError("forward.batch: active rows exceed executed token rows")
    if len(request_sequence_ids) != request_count:
        raise InstrumentationError(
            "forward.batch.request_sequence_ids: count must equal request_count"
        )
    for index, sequence_id in enumerate(request_sequence_ids):
        _require_int(sequence_id, f"forward.batch.request_sequence_ids[{index}]")
    if len(set(request_sequence_ids)) != len(request_sequence_ids):
        raise InstrumentationError("forward.batch.request_sequence_ids: values must be unique")
    if len(token_sequence_ids) != token_rows or len(token_positions) != token_rows:
        raise InstrumentationError(
            "forward.batch: token identity arrays must equal token_row_count"
        )
    for index, (sequence_id, position) in enumerate(
        zip(token_sequence_ids, token_positions, strict=True)
    ):
        if sequence_id is None or position is None:
            if sequence_id is not None or position is not None:
                raise InstrumentationError(
                    f"forward.batch.token_row_sequence_ids[{index}]: identity and position "
                    "must both be null"
                )
            continue
        _require_int(sequence_id, f"forward.batch.token_row_sequence_ids[{index}]")
        _require_int(position, f"forward.batch.token_positions[{index}]")
    if phase == "decode":
        if token_rows != padded_count:
            raise InstrumentationError(
                "forward.batch.token_row_count: decode rows must equal padded requests"
            )
        if active_rows != request_count:
            raise InstrumentationError(
                "forward.batch.active_token_row_count: decode rows must equal requests"
            )
        if list(token_sequence_ids[:active_rows]) != list(request_sequence_ids):
            raise InstrumentationError(
                "forward.batch.token_row_sequence_ids: active decode rows must match requests"
            )
        if any(value is not None for value in token_sequence_ids[active_rows:]):
            raise InstrumentationError(
                "forward.batch.token_row_sequence_ids: padded decode rows must be null"
            )
    else:
        if active_rows != token_rows or padded_count != request_count:
            raise InstrumentationError("forward.batch: prefill rows cannot be padding")
        expected_request_order = list(dict.fromkeys(token_sequence_ids))
        if expected_request_order != list(request_sequence_ids):
            raise InstrumentationError(
                "forward.batch.token_row_sequence_ids: prefill rows must follow request order"
            )
        prior_by_sequence: dict[int, int] = {}
        for sequence_id, position in zip(token_sequence_ids, token_positions, strict=True):
            assert isinstance(sequence_id, int) and isinstance(position, int)
            prior = prior_by_sequence.get(sequence_id)
            if prior is not None and position != prior + 1:
                raise InstrumentationError(
                    "forward.batch.token_positions: prefill positions must be contiguous"
                )
            prior_by_sequence[sequence_id] = position
    model = run["model"]
    layers = _require_list(record.get("layers"), "forward.layers")
    if len(layers) != model["num_moe_layers"]:
        raise InstrumentationError("forward.layers: wrong layer count")
    for layer_id, layer in enumerate(layers):
        _validate_layer(
            layer,
            f"forward.layers[{layer_id}]",
            expected_layer=layer_id,
            num_layers=model["num_moe_layers"],
            num_experts=model["num_experts"],
            experts_per_token=model["experts_per_token"],
            execution_mode=run["execution_mode"],
            phase=phase,
            expert_object_bytes=run["expert_cache"]["expert_object_bytes"],
            cache_capacity=run["expert_cache"]["capacity_objects"],
        )
        if layer["requested_shape"][0] != token_rows:
            raise InstrumentationError(
                f"forward.layers[{layer_id}].requested_shape: row count must match batch"
            )


def validate_run_end_record(record: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    if record.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise InstrumentationError("run_end.schema_version: unsupported event schema")
    if record.get("record_type") != "run_end":
        raise InstrumentationError("run_end.record_type: must be run_end")
    if record.get("run_id") != run.get("run_id"):
        raise InstrumentationError("run_end.run_id: must match run record")
    _require_int(record.get("sequence"), "run_end.sequence", minimum=1)
    _require_int(record.get("timestamp_monotonic_ns"), "run_end.timestamp_monotonic_ns")
    _require_string(record.get("ended_at_utc"), "run_end.ended_at_utc")
    if record.get("status") != "complete":
        raise InstrumentationError("run_end.status: must be complete")


def load_event_file(path: Path) -> list[dict[str, Any]]:
    """Load and validate one JSONL stream, including ordering and clock monotonicity."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise InstrumentationError(f"{path}:{line_number}: blank lines are not allowed")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise InstrumentationError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise InstrumentationError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    if not records:
        raise InstrumentationError(f"{path}: event stream is empty")
    validate_run_record(records[0])
    run = records[0]
    prior_end = run["timestamp_monotonic_ns"]
    prior_position_by_sequence: dict[int, int] = {}
    saw_end = False
    for expected_sequence, record in enumerate(records[1:], start=1):
        if record.get("sequence") != expected_sequence:
            raise InstrumentationError(
                f"{path}: sequence {record.get('sequence')!r} is not {expected_sequence}"
            )
        record_type = record.get("record_type")
        if record_type == "forward":
            if saw_end:
                raise InstrumentationError(f"{path}: forward appears after run_end")
            validate_forward_record(record, run)
            if record["started_monotonic_ns"] < prior_end:
                raise InstrumentationError(f"{path}: forward timestamps are not monotonic")
            prior_end = record["ended_monotonic_ns"]
            batch = record["batch"]
            for sequence_id, position in zip(
                batch["token_row_sequence_ids"], batch["token_positions"], strict=True
            ):
                if sequence_id is None:
                    continue
                prior_position = prior_position_by_sequence.get(sequence_id)
                if prior_position is not None and position != prior_position + 1:
                    raise InstrumentationError(
                        f"{path}: token positions are not contiguous for sequence {sequence_id}"
                    )
                prior_position_by_sequence[sequence_id] = position
        elif record_type == "run_end":
            if saw_end or expected_sequence != len(records) - 1:
                raise InstrumentationError(f"{path}: run_end must be the final record")
            validate_run_end_record(record, run)
            if record["timestamp_monotonic_ns"] < prior_end:
                raise InstrumentationError(f"{path}: run_end timestamp is not monotonic")
            saw_end = True
        else:
            raise InstrumentationError(f"{path}: unsupported record_type {record_type!r}")
    return records


def _metric(availability: str, value: int | None, unit: str, reason: str | None) -> dict[str, Any]:
    return {
        "availability": availability,
        "reason": reason,
        "unit": unit,
        "value": value,
    }


def _aggregate_optional(
    layers: Iterable[Mapping[str, Any]],
    path: tuple[str, ...],
    value_field: str,
    unit: str,
) -> dict[str, Any]:
    items: list[Mapping[str, Any]] = []
    for layer in layers:
        item: Any = layer
        for component in path:
            item = item[component]
        items.append(item)
    applicable = [item for item in items if item["availability"] != "not_applicable"]
    measured = [item for item in applicable if item["availability"] == "measured"]
    if applicable:
        if len(measured) != len(applicable):
            return _metric(
                "unavailable",
                None,
                unit,
                "applicable records include unavailable values",
            )
        return _metric("measured", sum(int(item[value_field]) for item in measured), unit, None)
    availability = "not_applicable" if items else "unavailable"
    reasons = sorted({str(item["reason"]) for item in items})
    return _metric(availability, None, unit, "; ".join(reasons) if reasons else "no records")


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate an already validated event stream without inventing missing metrics."""

    if not records:
        raise InstrumentationError("event stream is empty")
    run = records[0]
    validate_run_record(run)
    forwards = [record for record in records if record.get("record_type") == "forward"]
    layers = [layer for forward in forwards for layer in forward["layers"]]
    complete = bool(records[-1].get("record_type") == "run_end")
    totals = {
        "requested_expert_occurrences": _metric(
            "measured",
            sum(len(row) for layer in layers for row in layer["requested_expert_ids"]),
            "expert_routes",
            None,
        ),
        "unique_requested_expert_objects": _metric(
            "measured",
            sum(
                len({expert for row in layer["requested_expert_ids"] for expert in row})
                for layer in layers
            ),
            "unique_expert_objects_per_layer_forward",
            None,
        ),
        "residency_hits": _aggregate_optional(
            layers, ("residency",), "hit_count", "unique_expert_objects"
        ),
        "residency_misses": _aggregate_optional(
            layers, ("residency",), "miss_count", "unique_expert_objects"
        ),
        "expert_loads": _metric(
            "measured", sum(len(layer["loads"]) for layer in layers), "expert_objects", None
        ),
        "expert_evictions": _metric(
            "measured",
            sum(len(layer["evictions"]) for layer in layers),
            "expert_objects",
            None,
        ),
        "h2d_transfer_operations": _aggregate_optional(
            layers, ("transfers", "h2d"), "operation_count", "logical_operations"
        ),
        "h2d_transfer_objects": _aggregate_optional(
            layers, ("transfers", "h2d"), "object_count", "expert_objects"
        ),
        "h2d_transfer_bytes": _aggregate_optional(
            layers, ("transfers", "h2d"), "bytes", "bytes"
        ),
        "h2d_transfer_duration": _aggregate_optional(
            layers, ("transfers", "h2d"), "duration_ns", "nanoseconds"
        ),
        "d2h_transfer_bytes": _aggregate_optional(
            layers, ("transfers", "d2h"), "bytes", "bytes"
        ),
        "expert_compute_duration": _aggregate_optional(
            layers, ("compute",), "duration_ns", "nanoseconds"
        ),
    }
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "record_type": "aggregate",
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "run_id": run["run_id"],
        "complete": complete,
        "forward_count": len(forwards),
        "prefill_forward_count": sum(forward["phase"] == "prefill" for forward in forwards),
        "decode_forward_count": sum(forward["phase"] == "decode" for forward in forwards),
        "totals": totals,
        "source": dict(source or {}),
    }


def aggregate_event_file(events_path: Path, aggregate_path: Path) -> dict[str, Any]:
    """Validate an event file and create a checksum-bound aggregate without overwrite."""

    records = load_event_file(events_path)
    payload = events_path.read_bytes()
    aggregate = aggregate_records(
        records,
        source={
            "events_file": events_path.name,
            "events_sha256": hashlib.sha256(payload).hexdigest(),
            "events_size_bytes": len(payload),
        },
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with aggregate_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise InstrumentationError(f"refusing to overwrite aggregate: {aggregate_path}") from error
    return aggregate


class MoeEventWriter:
    """Append-only JSONL writer; the aggregate is produced only on a clean close."""

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        *,
        execution_mode: str,
        model: Mapping[str, Any],
        expert_cache: Mapping[str, Any],
    ) -> None:
        validate_run_id(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = output_dir / f"{run_id}.events.jsonl"
        self.aggregate_path = output_dir / f"{run_id}.aggregate.json"
        if self.aggregate_path.exists():
            raise InstrumentationError(f"refusing to overwrite aggregate: {self.aggregate_path}")
        try:
            self._stream: TextIO = self.events_path.open("x", encoding="utf-8", buffering=1)
        except FileExistsError as error:
            raise InstrumentationError(
                f"refusing to overwrite events: {self.events_path}"
            ) from error
        self.run_id = run_id
        self.execution_mode = execution_mode
        self._next_sequence = 1
        self._closed = False
        run = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "record_type": "run",
            "sequence": 0,
            "run_id": run_id,
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "created_at_utc": _utc_now(),
            "runtime": "freetoken",
            "execution_mode": execution_mode,
            "sequence_id_namespace": "freetoken-request-uid",
            "model": dict(model),
            "expert_cache": dict(expert_cache),
            "clock": {
                "timestamp_source": "time.monotonic_ns",
                "timestamp_unit": "nanosecond",
                "duration_source": "cuda_event",
                "duration_unit": "nanosecond",
            },
        }
        validate_run_record(run)
        self._run = run
        self._prior_end_ns = run["timestamp_monotonic_ns"]
        self._stream.write(_canonical_line(run))

    def write_forward(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise InstrumentationError("cannot write to a closed trace")
        value = dict(record)
        value.update(
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "record_type": "forward",
                "sequence": self._next_sequence,
                "run_id": self.run_id,
                "execution_mode": self.execution_mode,
            }
        )
        validate_forward_record(value, self._run)
        if value["started_monotonic_ns"] < self._prior_end_ns:
            raise InstrumentationError("forward timestamps are not monotonic")
        self._stream.write(_canonical_line(value))
        self._stream.flush()
        self._prior_end_ns = value["ended_monotonic_ns"]
        self._next_sequence += 1

    def close(self) -> dict[str, Any] | None:
        if self._closed:
            return None
        run_end = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "record_type": "run_end",
            "sequence": self._next_sequence,
            "run_id": self.run_id,
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "ended_at_utc": _utc_now(),
            "status": "complete",
        }
        validate_run_end_record(run_end, self._run)
        if run_end["timestamp_monotonic_ns"] < self._prior_end_ns:
            raise InstrumentationError("run_end timestamp is not monotonic")
        self._stream.write(_canonical_line(run_end))
        self._stream.close()
        self._closed = True
        return aggregate_event_file(self.events_path, self.aggregate_path)

    def __enter__(self) -> "MoeEventWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        elif not self._closed:
            self._stream.close()
            self._closed = True
