try:
    import nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False

def nvtx_range(name, color="blue"):
    """轻量 NVTX range 上下文管理器，nvtx 未安装时为 no-op。"""
    if _NVTX_AVAILABLE:
        return nvtx.annotate(name, color=color)
    import contextlib
    return contextlib.nullcontext()