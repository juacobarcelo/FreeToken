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

Schema 1.0 supports GPT-OSS, tensor-parallel size 1, and an effective `fused` or
`offload` backend. It rejects hybrid/CPU execution, prefill hit-D2D, and MoE cache
resizing during a run because those paths do not yet have complete evidence hooks.
An `auto` request is accepted only when it resolves to `offload`.

The output directory receives:

- `<run-id>.events.jsonl`: append-only run, forward, and clean-run-end records;
- `<run-id>.aggregate.json`: a deterministic summary bound to the event file by
  SHA-256, written on clean shutdown.

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

Decode CUDA graphs may execute padded dummy rows. `token_row_count` includes every
executed row because a padded route can affect the real cache state. The first
`active_token_row_count` rows correspond to active requests; the remainder are
padding. Prefill rows are all active tokens.

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

## Timing overhead

The recorder copies fixed-size observations on device and synchronizes once at each
forward boundary before writing evidence. That synchronization is part of the
instrumented runtime overhead. Compare matched warmed runs with instrumentation off
and on. If median makespan overhead exceeds the experiment threshold, retain the
functional evidence but do not use instrumented makespan or duration values as
representative performance measurements.
