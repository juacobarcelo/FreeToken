# Matched router diagnostic

The experimental controls implement [planner issue 36](https://github.com/juacobarcelo/inference-system-planner/issues/36).
They require the matching InferenceSystemPlanner package. Default serving keeps the observer absent.

`--moe-router-mode planning-only` requires `--moe-router-config`. It reads the real current route/cache metadata,
runs the same Python planner as active execution, discards the plan, and invokes the original prefill/decode path.
The learned top-k selector and all Triton kernels are unchanged. `active` retains the existing Alternative A policy.
The serving boundary passes its actual prefill/decode phase to the active adapter. During prefill, every expert
group uses the original forward's kernel family and the configuration selected from the complete token count,
model expert count and top-k. A group with one or three rows must not silently switch to split-K decode or
choose a different arithmetic tile. Configuration selection occurs once per layer, before executing its groups.
This addresses the kernel-family mismatch found alongside a numerical divergence in the first full TP2
diagnostic. The corrected full-model comparison remains necessary; this change does not claim faster routing.

`--router-diagnostic-config FILE` is an opt-in experiment control, not a public request seed API.
The companion package validates the frozen GPT-OSS-120B MXFP4 TP2 configuration, seeds both workers,
and extends the existing idle cache rebuild with a verified cold expert reset. Exactly one reset and measured
batch are allowed per process, following a discarded actual-workload warm-up. No cache geometry change is allowed.
The scheduler's ordinary TP rebuild rejection has a narrow exception for that idle diagnostic reset on TP2.
It rejects repeated resets, changed capacities, other pool changes, other TP sizes and non-diagnostic serving.
This does not implement general TP resize recovery: the diagnostic's external owner terminates both workers
on a failed reset or timeout before any performance sample is accepted.
Performance mode installs no tensor/timing observer. Separate correctness and cost processes collect bounded evidence;
a failed or incomplete capture cannot pass the comparison gate. Only isolated replay replaces layer inputs with
captured baseline values; the A/B/C full-model executions propagate their own results.

The offline protocol tests execute the real preparation/dispatch methods with CPU doubles:

```bash
PYTHONPATH=/path/to/inference-system-planner/src python -m pytest -q tests/moe/test_router_diagnostic_offline.py
```

The optional GPU regression is `tests/moe/test_gpt_oss.py::test_causal_router_preserves_mxfp4_result`.
It checks planning-only operand/cache immutability and exact baseline output on tiny TP1 fixtures. The active
prefill cases require exact equality too, including a three-row prompt and a single-row expert group within a
32-row prompt. Decode retains its pre-existing approximate tolerance; its split counts and intermediate rounding
are not covered by this prefill correction. These fixtures are not a full-model or TP2 equivalence claim.
The issue-36 campaign must establish those boundaries separately before interpreting runtime differences.
