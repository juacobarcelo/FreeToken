"""Opt-in, versioned MoE evidence export.

The package deliberately keeps its record and aggregation helpers free of Torch so
their semantics can be tested on CPU-only hosts.  The live CUDA recorder is imported
by the engine only when ``--moe-instrumentation-dir`` is supplied.
"""

from .records import (
    AGGREGATE_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    InstrumentationError,
    MoeEventWriter,
    aggregate_event_file,
    aggregate_records,
    load_event_file,
    validate_run_id,
)

__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "EVENT_SCHEMA_VERSION",
    "InstrumentationError",
    "MoeEventWriter",
    "aggregate_event_file",
    "aggregate_records",
    "load_event_file",
    "validate_run_id",
]
