"""
The one result type both routing paths return.

Kept in its own tiny module because citra_fast_path.py, the Smart Path
backends and jarvis_router.py all produce it, and none of them should
have to import the others just to get at a dataclass.
"""

from dataclasses import dataclass


@dataclass
class RouteResult:
    """
    Uniform return type for both paths, so __main__ doesn't need to know
    which path produced an answer to display it consistently — it just
    reads .path, .success, .message, and .latency_ms off whatever came back.

    match_latency_ms / dispatch_latency_ms are populated ONLY for FAST path
    results (None on SMART path results, where the distinction doesn't
    apply — there's no separate "match" phase, just one LLM call). See the
    detailed comment in JarvisRouter._try_fast_path for why these two are
    kept separate rather than folded into one number.
    """
    path: str            # "FAST" or "SMART"
    success: bool
    message: str
    latency_ms: float
    match_latency_ms: float | None = None
    dispatch_latency_ms: float | None = None

    def format_for_terminal(self) -> str:
        """Human-readable one-block summary for the interactive test loop."""
        status = "OK" if self.success else "FAILED"
        header = f"[{self.path} PATH | {status} | {self.latency_ms:.2f}ms total]"
        if self.path == "FAST" and self.match_latency_ms is not None:
            header += (
                f"\n  regex match+extract: {self.match_latency_ms:.4f}ms"
                f"  |  hardware dispatch: {self.dispatch_latency_ms:.4f}ms"
            )
        return f"{header}\n{self.message}"
