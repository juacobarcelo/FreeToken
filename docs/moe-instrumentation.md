# MoE instrumentation

FreeToken can emit opt-in, versioned evidence about GPT-OSS expert routing,
residency, transfers, and expert compute. Instrumentation is disabled by default.
It observes the existing decisions and does not select routes, cache entries, or
execution modes.

## Enable a run

Pass both flags to `ft serve`:

```bash
ft serve --model /models/gpt-oss-20b \
  --moe-backend offload --moe-cache-rate 0.25 \
  --moe-instrumentation-dir /evidence/run-001 \
  --moe-instrumentation-run-id offload-25-rep-01
```

Schema 1.1 supports GPT-OSS, tensor-parallel sizes 1 and 2, and an effective
`fused` or `offload` backend. It rejects hybrid/CPU execution, prefill hit-D2D,
and MoE cache resizing during a run because those paths do not yet have complete
evidence hooks. An `auto` request is accepted only when it resolves to `offload`.
Schema 1.0 TP1 event streams remain readable.

The output directory receives:

- TP1: `<run-id>.events.jsonl` and `<run-id>.aggregate.json`;
- TP2: one pair per rank, named
  `<run-id>.tp-rank-<rank>-of-02.{events.jsonl,aggregate.json}`.

Each stream is append-only and each deterministic aggregate is bound to its event
file by SHA-256 and written on clean shutdown. The run and aggregate records carry
the rank, world size, and the declaration that router choices are replicated.
Consumers must reconcile both TP2 streams and reject any route disagreement; a
single rank is not complete TP2 route evidence.

Existing files are never overwritten. Stop the server cleanly to obtain the
aggregate. An interrupted JSONL file remains useful as partial evidence, but its
summary must declare `complete: false`.

The machine-readable contracts are installed with the package:

- `instrumentation/schemas/moe-events-v1.schema.json`;
- `instrumentation/schemas/moe-aggregate-v1.schema.json`.

## Measurement semantics

Each forward record includes the phase and, for decode, a zero-based decode step.
Each layer records the raw requested expert ids before an offload kernel rewrites
them to cache slots.

The run declares the `freetoken-request-uid` sequence-id namespace. Each batch
records active `request_sequence_ids` and aligns every route row with a
`token_row_sequence_ids` entry and zero-based `token_positions` entry. These ids
remain stable for the lifetime of a request. FreeToken exposes the same uid in
the OpenAI response id as `chatcmpl-<uid>`, which allows a harness to join a
runtime sequence back to its workload request without relying on arrival order.

Decode CUDA graphs may execute padded dummy rows. `token_row_count` includes every
executed row because a padded route can affect the real cache state. The first
`active_token_row_count` rows correspond to active requests; the remainder are
padding and have null sequence ids and positions. Prefill rows are all active
tokens and appear in request order.

Residency hits and misses count unique `(layer, expert)` objects requested in one
layer forward, against the cache state immediately before admission. Loads,
evictions, and transitions carry layer, expert, and slot ids. A complete-layer
prefill overlap copy targets a transient device buffer. The buffers borrow cache
slots; any persistent entries invalidated on reuse are reported separately as
evictions.

Transfer bytes are the exact tensor-bank bytes copied per expert object. Transfer
durations and expert-kernel durations use CUDA events and nanoseconds. Forward and
run timestamps use `time.monotonic_ns`; the two clock domains must not be compared
directly. FreeToken keeps authoritative expert weights in host memory, so an
eviction discards a device copy and D2H expert transfer is `not_applicable`.
Fused execution has no expert cache, so cache residency and expert transfers are
also `not_applicable`, not zero.

The current offload recorder declares an empty `initial_resident_objects` set at
the `before_first_observed_forward` boundary. The schema also represents an
explicit non-empty starting set for fixture and future producers. Ordered layer
transitions can therefore reconstruct the cache state without assuming an
unreported warm state.

## Timing overhead

The recorder copies fixed-size observations on device and synchronizes once at each
forward boundary before writing evidence. That synchronization is part of the
instrumented runtime overhead. Compare matched warmed runs with instrumentation off
and on. If median makespan overhead exceeds the experiment threshold, retain the
functional evidence but do not use instrumented makespan or duration values as
representative performance measurements.
