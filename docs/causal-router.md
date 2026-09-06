# Causal expert router adapter

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
