"""Exercise actual admission dispatch without Torch, GPU hardware, or model data."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode", ["disabled", "warmup", "performance", "correctness"])
def test_diagnostic_binds_complete_input_before_prefill_chunking(mode):
    root = Path(__file__).resolve().parents[2]
    source = ast.parse((root / "python/freetoken/scheduler/scheduler.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_process_one_msg")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {name: type(name, (), {}) for name in ("BatchBackendMsg", "ExitMsg", "UserMsg")}
    namespace["logger"] = SimpleNamespace(debug_rank0=lambda *args: None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                 "scheduler.py:admission", "exec"), namespace)
    calls = []
    msg = namespace["UserMsg"]()
    msg.uid, msg.input_ids = 73, [101, 102, 202, 2]
    msg.sampling_params = SimpleNamespace(max_tokens=1)

    def bind(uid, ids):
        assert uid == 73 and ids is msg.input_ids
        calls.append("bind-complete-input")

    def prefill(actual):
        assert actual is msg
        calls.append("prefill-admission")

    diagnostic = None if mode == "disabled" else SimpleNamespace(
        capture_active=mode == "correctness", admit_request=bind)
    scheduler = SimpleNamespace(engine=SimpleNamespace(max_seq_len=2048, router_diagnostic=diagnostic),
                                prefill_manager=SimpleNamespace(add_one_req=prefill))
    namespace["_process_one_msg"](scheduler, msg)
    assert calls == (["bind-complete-input"] if mode == "correctness" else []) + ["prefill-admission"]
