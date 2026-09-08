"""Idle-only hot switch between the stock MoE offload path and the project router.

The MoE layer picks its path on every forward from two attributes of the shared
``OffloadMoeCache`` (``causal_router`` and ``router_mode``, see ``layers/moe.py``).
``AlternativeAAdapter`` plans on the host every layer (``.cpu()`` / ``.item()``),
which a captured CUDA graph would not replay, so every router mode -- including the
stock path it is compared against -- runs eagerly (``--cuda-graph-max-bs 0``). The
switch is therefore a metadata swap plus an expert-cache reset: no graph recapture,
no allocation, no collective. The caller (scheduler) must guarantee an idle engine
and all TP ranks must apply identical arguments in the same order. KV pages and
any prefix cache are left untouched.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

ROUTER_MODES = ("off", "planning-only", "active")


class RouterModeRejected(RuntimeError):
    """Recoverable refusal: the routing path is untouched and serving continues."""


def graphs_enabled(cuda_graph_bs: Sequence[int] | None, cuda_graph_max_bs: int | None) -> bool:
    """Mirror GraphRunner's resolution of the two CUDA-graph knobs (see engine/graph.py)."""
    if cuda_graph_bs is not None:
        return len(cuda_graph_bs) > 0
    return cuda_graph_max_bs is None or cuda_graph_max_bs >= 1


def validate_router_mode(
    mode: str,
    *,
    router_config: str | None,
    cuda_graph_bs: Sequence[int] | None,
    cuda_graph_max_bs: int | None,
) -> None:
    """Startup validation shared by the engine config adjuster."""
    if mode not in ROUTER_MODES:
        raise ValueError("moe_router_mode must be off, planning-only or active")
    if mode == "planning-only" and not router_config:
        raise ValueError("planning-only requires --moe-router-config")
    if router_config and graphs_enabled(cuda_graph_bs, cuda_graph_max_bs):
        raise ValueError(
            "--moe-router-config requires --cuda-graph-max-bs 0: the router plans on the host "
            "every layer, which a captured decode graph would not replay"
        )


def current_router_mode(cache: Any) -> str:
    """The path the MoE layer takes right now: "off" whenever no adapter is attached."""
    if cache is None or getattr(cache, "causal_router", None) is None:
        return "off"
    return getattr(cache, "router_mode", "active")


def apply_router_mode(
    cache: Any,
    adapter: Any,
    mode: str,
    *,
    graphs_active: bool,
    synchronize: Callable[[], None] | None = None,
) -> bool:
    """Swap the routing path on an idle engine. Returns True when the path changed.

    Refusals (``RouterModeRejected``) happen before anything is touched. Past that point
    the expert cache is reset so both paths start cold from identical residency, which is
    also what keeps the two TP ranks' slot maps symmetric.
    """
    if mode not in ROUTER_MODES:
        raise RouterModeRejected(f"unknown router mode {mode!r}")
    if cache is None:
        raise RouterModeRejected("this model has no MoE offload cache")
    if mode != "off" and adapter is None:
        raise RouterModeRejected(
            "no --moe-router-config was loaded at startup; restart with one to enable the router"
        )
    if mode == current_router_mode(cache):
        return False
    if graphs_active:
        raise RouterModeRejected(
            "CUDA graphs are captured for the current routing path; restart with "
            "--cuda-graph-max-bs 0 to switch at runtime"
        )
    if synchronize is not None:
        synchronize()
    cache.causal_router = adapter if mode != "off" else None
    cache.router_mode = mode
    cache.reset()
    return True
