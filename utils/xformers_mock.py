"""
Surgical hardware bypass for Windows systems that lack xformers.

Instead of a full MagicMock, we expose only the minimal structure that
dinov2.py needs to import cleanly. The missing symbols (e.g.
`memory_efficient_attention`, `unbind`) are intentionally omitted so that
attention.py falls back to PyTorch's native attention engine automatically.
"""

import sys


class _MockFmha:
    pass


class _MockOps:
    fmha = _MockFmha()


class _MockXformers:
    ops = _MockOps()


def patch_xformers() -> None:
    """Inject stub modules only if xformers is not already installed."""
    if "xformers" not in sys.modules:
        sys.modules["xformers"] = _MockXformers()
        sys.modules["xformers.ops"] = _MockOps()
        sys.modules["xformers.ops.fmha"] = _MockFmha()