try:
    import nvtx

    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False


def nvtx_range(name, color="blue"):
    if _NVTX_AVAILABLE:
        return nvtx.annotate(name, color=color)
    import contextlib

    return contextlib.nullcontext()
