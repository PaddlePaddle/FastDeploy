import paddle

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        cuda_host_alloc,
        cuda_host_free,
        set_data_ipc,
        share_external_data,
        swap_cache_all_layers,
        unset_data_ipc,
    )

    memory_allocated = paddle.device.cuda.memory_allocated
elif current_platform.is_xpu():
    from fastdeploy.model_executor.ops.xpu import (
        cuda_host_alloc,
        cuda_host_free,
        set_data_ipc,
        share_external_data,
        swap_cache_all_layers,
    )

    unset_data_ipc = None
    memory_allocated = paddle.device.xpu.memory_allocated

else:
    raise RuntimeError("Prefix cache ops only supported CUDA nor XPU platform ")


def set_device(device):
    if current_platform.is_cuda():
        paddle.set_device(f"gpu:{device}")
    elif current_platform.is_xpu():
        paddle.set_device(f"xpu:{device}")
    else:
        raise RuntimeError("No supported platform")


def share_external_data_(cache, cache_name, cache_shape, use_ipc):
    if current_platform.is_cuda():
        cache = share_external_data(cache, cache_name, cache_shape)
    elif current_platform.is_xpu():
        cache = share_external_data(cache, cache_name, cache_shape, use_ipc)
    else:
        raise RuntimeError("No supported platform")
    return cache


__all__ = [
    "cuda_host_alloc",
    "cuda_host_free",
    "set_data_ipc",
    "share_external_data_",
    "swap_cache_all_layers",
    "unset_data_ipc",  # XPU是 None
    "set_device",
    "memory_allocated",
]
