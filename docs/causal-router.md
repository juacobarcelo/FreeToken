# Causal expert router adapter

The issue-38 optimization requires an InferenceSystemPlanner revision exporting
`ordered_lru_slots`. It retains the issue-36 prefill arithmetic correction and
depends on the companion FreeToken PR #3 before integration into the controlled
base. See [ISP #38](https://github.com/juacobarcelo/inference-system-planner/issues/38).

Each active layer constructs one private eviction order on its first missing
expert. Construction stays inside `_select_victim` timing. Later loads consume
the same order and skip protected slots; when all slots are protected, the
original first-enqueued event and copy-stream wait are preserved. Remaining
unprotected IDs and usage cannot change during this exclusive forward. The
order is discarded at the layer boundary and is never built for resident-only
or planning-only execution. Learned routes, group order, kernels and cache
geometry remain unchanged.

The CPU regression executes the real forward, load and victim methods with
cache and event doubles, comparing slot choices, mapping updates and waits to
the original one-shot selector over repeated layers:

```bash
PYTHONPATH=python:/path/to/issue-38-planner/src python -m pytest -q \
  tests/moe/test_causal_router_slot_order.py \
  tests/moe/test_router_diagnostic_offline.py \
  tests/instrumentation/test_records.py tests/test_runtime_layout.py
```

These tests do not execute CUDA copies or expert arithmetic. Optional GPU tests
require the pinned FreeToken runtime and supported NVIDIA hardware; the matched
TP2 full-model diagnostic additionally requires the approved checkpoint and
experiment resource envelope. No full-model run of this optimization is claimed.

The optional `--moe-router-config` flag connects the bounded Alternative A
policy from InferenceSystemPlanner issue 22 to FreeToken's existing GPT-OSS
routed-forward boundary. It is an experiment adapter, not a default serving
mode. Without the flag, the normal FreeToken path is unchanged.

The adapter receives only the current layer's `topk_ids`, current cache maps,
and costs frozen in the versioned planner YAML. It groups matching routes,
runs resident demand first, orders missing experts with the planner policy,
and pipelines one expert transfer ahead of compute. It uses the existing pinned
host banks, GPU slot cache, copy stream, MXFP4 kernels, and final top-k reduction.
No future layer route or complete trace crosses the adapter boundary.
Each scheduled group exposes only its selected cache slot to the MXFP4 kernel;
the full multi-layer cache is never treated as the kernel's expert axis.

## Requirements

- GPT-OSS MXFP4 expert banks;
- `--moe-backend offload` and tensor parallel size 2;
- an explicit expert cache size matching the aggregate byte capacity in the
  planner configuration;
- `max_running_req`, planner batch size, and planner microbatch size all equal;
- the issue-22 InferenceSystemPlanner package installed in both rank processes.

CUDA graphs are disabled because every layer exposes a new route to a causal
host decision. Performance instrumentation, CPU/hybrid expert execution, and
runtime expert-cache resizing are rejected. Stock prefill overlap is disabled;
the adapter schedules the routed prefill and decode work itself.

Example:

```bash
ft serve --model /model \
  --tensor-parallel-size 2 \
  --max-running-req 16 \
  --moe-backend offload \
  --moe-cache-size 2395 \
  --moe-router-config /experiment/alternative-a.yaml
```

Run TP2 route capture in a separate unmodified process with
`--moe-instrumentation-dir` and no router flag. Both rank files must be
reconciled by InferenceSystemPlanner before use.

The OpenAI chat response retains `chatcmpl-<freetoken-request-uid>` as its id
and also exposes sampled ids as `choices[0].message.token_ids` (or on the final
stream choice). This makes an otherwise empty control token identifiable
without retaining generated text.
# Exact decode regression prerequisite

InferenceSystemPlanner issue #41 combines the latest admission correction and
cache-slot optimization, then checks prefill and real decode separately with
one and two independent requests. Matching final tokens is insufficient.

The active decode path now freezes split-K counts from the complete forward
before grouping experts and reduces the original top-k columns using the stock
decode sum. For GPT-OSS-120B TP2, one token with four experts uses gate/up and
down split counts 45/18; two tokens use 23/9. Each smaller group retains those
counts. Prefill continues to retain its complete-forward configuration.

`tests/moe/test_gpt_oss.py::test_causal_router_preserves_mxfp4_result` requires
exact individual partials, their scatter positions and the final output for
both phases, including one/two-token decode shapes that expose the former
split-count difference. CPU checks do not qualify a full model for performance;
the ISP four-case capture, independent comparison and immutable-reference
receipt must also pass on the actual build before timing it.
