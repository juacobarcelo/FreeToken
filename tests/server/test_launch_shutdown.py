from __future__ import annotations

import signal
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from freetoken.server import launch


def test_scheduler_sigterm_uses_graceful_shutdown() -> None:
    events: list[str] = []

    class FakeScheduler:
        def __init__(self, _args) -> None:
            events.append("initialized")

        def sync_all_ranks(self) -> None:
            events.append("synchronized")

        def run_forever(self) -> None:
            events.append("running")
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)

        def shutdown(self) -> None:
            events.append("shutdown")

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    fake_scheduler_module = SimpleNamespace(Scheduler=FakeScheduler)
    args = SimpleNamespace(
        shell_mode=False,
        silent_output=False,
        tp_info=SimpleNamespace(is_primary=lambda: False),
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "freetoken.scheduler": fake_scheduler_module},
    ):
        launch._run_scheduler(args, SimpleNamespace())

    assert events == ["initialized", "synchronized", "running", "shutdown"]
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm
