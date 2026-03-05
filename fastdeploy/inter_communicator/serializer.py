"""
Serialization utility module.
Supports pickle (default) and Apache fory (optional, enabled by FD_USE_fory=1).
fory uses JIT compilation for custom classes, avoiding pickle's type dispatch overhead.
"""

import logging
from multiprocessing.reduction import ForkingPickler

logger = logging.getLogger("fastdeploy")

_fory_instance = None
_use_fory = False


def _init_fory():
    """Lazy-initialize fory instance and register all needed classes."""
    global _fory_instance
    if _fory_instance is not None:
        return _fory_instance

    import pyfory

    _fory_instance = pyfory.Fory()

    # Register all classes that go through ZMQ serialization
    from fastdeploy.engine.pooling_params import PoolingParams
    from fastdeploy.engine.request import (
        CompletionOutput,
        ControlRequest,
        ControlResponse,
        Request,
        RequestMetrics,
        RequestOutput,
    )
    from fastdeploy.engine.sampling_params import GuidedDecodingParams, SamplingParams

    for cls in [
        Request,
        RequestOutput,
        CompletionOutput,
        RequestMetrics,
        SamplingParams,
        PoolingParams,
        GuidedDecodingParams,
        ControlRequest,
        ControlResponse,
    ]:
        _fory_instance.register_type(cls)

    # Register nested types that may appear in fields
    try:
        from fastdeploy.worker.output import LogprobsLists, SpeculateMetrics

        _fory_instance.register_type(LogprobsLists)
        _fory_instance.register_type(SpeculateMetrics)
    except (ImportError, Exception):
        pass

    try:
        from fastdeploy.entrypoints.openai.protocol import DeltaMessage, ToolCall

        _fory_instance.register_type(ToolCall)
        _fory_instance.register_type(DeltaMessage)
    except (ImportError, Exception):
        pass

    logger.info("[Serializer] fory initialized with registered classes")
    return _fory_instance


def init(use_fory: bool = False):
    """Initialize the serializer. Call once at startup."""
    global _use_fory
    _use_fory = use_fory
    if _use_fory:
        _init_fory()
        logger.info("[Serializer] Using fory serialization")
    else:
        logger.info("[Serializer] Using pickle serialization")


def dumps(obj) -> bytes:
    """Serialize object to bytes."""
    if _use_fory:
        fory = _init_fory()
        return fory.serialize(obj)
    else:
        return ForkingPickler.dumps(obj)


def loads(data: bytes):
    """Deserialize bytes to object."""
    if _use_fory:
        fory = _init_fory()
        return fory.deserialize(data)
    else:
        return ForkingPickler.loads(data)
