"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import paddle.jit.dy2static.utils as jit_utils
import paddle.nn.layer
from paddle.device.cuda import graphs

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import (
    capture_custom_allreduce,
    custom_ar_clear_ipc_handles,
)
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend", "cudagraph_piecewise_backend.log")


# ---------------------------------------------------------------------------
# c10::cuda stream synchronization
# ---------------------------------------------------------------------------
# DeepEP's C++ extension (deep_ep_cpp) calls c10::cuda::getCurrentCUDAStream()
# to determine which CUDA stream to use.  Paddle provides a compatibility
# implementation of this function in libphi_core.so, backed by a thread-local
# variable `tls_current_streams`.  However, paddle.device.stream_guard() only
# updates Paddle's GPUContext stream -- it does NOT call
# c10::cuda::setCurrentCUDAStream().  As a result, DeepEP's C++ code sees the
# default stream instead of the capture stream, and its operations are NOT
# captured in the CUDA graph.
#
# The helper below synchronises the c10 stream state so that
# getCurrentCUDAStream() returns the same stream that Paddle's stream_guard
# set.  It uses ctypes to call the C++ functions directly in libphi_core.so.
#
# ABI notes (x86_64, GCC, libstdc++):
#   c10::cuda::CUDAStream contains c10::Stream which contains c10::Device.
#   c10::Device has a std::string member, making CUDAStream non-POD.
#   Non-POD return/argument types are passed via a hidden pointer (rdi).
#   The raw cudaStream_t (StreamId) is stored at offset 40 in the buffer.

_c10_lib = None
_c10_set_fn = None
_c10_ext_fn = None
_c10_get_fn = None
_C10_STREAM_BUF_SIZE = 64  # sizeof(CUDAStream) <= 48, use 64 for safety
_C10_STREAM_ID_OFFSET = 40  # offset of StreamId within CUDAStream buffer


def _init_c10_stream_funcs():
    """Lazily resolve c10::cuda symbols from libphi_core.so."""
    global _c10_lib, _c10_set_fn, _c10_ext_fn, _c10_get_fn
    if _c10_lib is not None:
        return _c10_lib is not False
    try:
        import ctypes
        import glob

        # Locate libphi_core.so
        candidates = glob.glob(os.path.join(os.path.dirname(paddle.__file__), "libs", "libphi_core.so"))
        if not candidates:
            _c10_lib = False
            return False
        _c10_lib = ctypes.CDLL(candidates[0], mode=ctypes.RTLD_GLOBAL)

        # c10::cuda::getStreamFromExternal(cudaStream_t, c10::DeviceIndex)
        # Returns CUDAStream via hidden pointer (rdi), args: rsi=stream, edx=device_index
        _c10_ext_fn = _c10_lib._ZN3c104cuda21getStreamFromExternalEP11CUstream_sta
        _c10_ext_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        _c10_ext_fn.restype = None

        # c10::cuda::setCurrentCUDAStream(CUDAStream)
        # Takes CUDAStream via hidden pointer (rdi)
        _c10_set_fn = _c10_lib._ZN3c104cuda20setCurrentCUDAStreamENS0_10CUDAStreamE
        _c10_set_fn.argtypes = [ctypes.c_void_p]
        _c10_set_fn.restype = None

        # c10::cuda::getCurrentCUDAStream(c10::DeviceIndex)
        # Returns CUDAStream via hidden pointer (rdi), arg: esi=device_index
        _c10_get_fn = _c10_lib._ZN3c104cuda20getCurrentCUDAStreamEa
        _c10_get_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _c10_get_fn.restype = None

        return True
    except Exception as e:
        logger.debug(f"Failed to init c10 stream funcs: {e}")
        _c10_lib = False
        return False


def _set_c10_current_stream(raw_stream, device_index):
    """Set c10::cuda's current stream so that DeepEP uses the correct stream.

    Args:
        raw_stream: cudaStream_t as an integer.
        device_index: CUDA device index (int).
    """
    if not _init_c10_stream_funcs():
        return
    import ctypes

    buf = ctypes.create_string_buffer(_C10_STREAM_BUF_SIZE)
    _c10_ext_fn(buf, ctypes.c_void_p(int(raw_stream)), device_index)
    _c10_set_fn(buf)


def _get_c10_current_stream(device_index):
    """Get c10::cuda's current raw stream (cudaStream_t as integer).

    Returns the raw cudaStream_t or None on failure.
    """
    if not _init_c10_stream_funcs():
        return None
    import ctypes

    buf = ctypes.create_string_buffer(_C10_STREAM_BUF_SIZE)
    _c10_get_fn(buf, device_index)
    return int.from_bytes(buf.raw[_C10_STREAM_ID_OFFSET : _C10_STREAM_ID_OFFSET + 8], "little")


def _get_cuda_device_index():
    """Get the current CUDA device index as an integer."""
    try:
        return paddle.framework.core.get_cuda_current_device_id()
    except Exception:
        # Fallback: parse from paddle.device.get_device() which returns e.g. 'gpu:0'
        dev_str = str(paddle.device.get_device())
        if ":" in dev_str:
            return int(dev_str.split(":")[1])
        return 0


def _get_paddle_raw_stream(device_index):
    """Get Paddle's current raw CUDA stream for the given device."""
    try:
        from paddle.base import core

        return core._get_current_stream(int(device_index)).raw_stream
    except Exception:
        return None


class _DeepEPStreamGuard:
    """Context manager that sets both Paddle and c10::cuda current streams.

    Paddle's ``paddle.device.stream_guard`` only updates the GPUContext
    stream.  DeepEP's C++ extension reads the current stream via
    ``c10::cuda::getCurrentCUDAStream()`` which is backed by a separate
    thread-local variable (``tls_current_streams``).  Without also setting
    this variable, DeepEP runs on the wrong stream and its operations are
    not captured in the CUDA graph.

    This guard synchronises both so that DeepEP sees the same stream as
    Paddle.
    """

    def __init__(self, stream):
        self.stream = stream
        self._paddle_guard = None
        self._prev_c10_stream = None
        self._device_index = None

    def __enter__(self):
        if self.stream is None:
            return
        # 1. Set Paddle's current stream
        self._paddle_guard = paddle.device.stream_guard(self.stream)
        self._paddle_guard.__enter__()

        # 2. Set c10::cuda's current stream to match Paddle's
        try:
            self._device_index = _get_cuda_device_index()
            self._prev_c10_stream = _get_c10_current_stream(self._device_index)
            raw_stream = _get_paddle_raw_stream(self._device_index)
            if raw_stream is not None:
                _set_c10_current_stream(raw_stream, self._device_index)
        except Exception as e:
            logger.warning(f"Failed to set c10 current stream: {e}")

    def __exit__(self, *args):
        if self._paddle_guard is not None:
            self._paddle_guard.__exit__(*args)
        # Restore c10 stream to match Paddle's restored stream
        if self._device_index is not None:
            try:
                raw_stream = _get_paddle_raw_stream(self._device_index)
                if raw_stream is not None:
                    _set_c10_current_stream(raw_stream, self._device_index)
            except Exception:
                pass


def _clean_deepep_low_latency_buffer():
    """Clean DeepEP low-latency buffer before warmup/capture/replay.

    DeepEP's low-latency kernels require parts of the buffer to be
    zero-initialized.  After a normal-mode dispatch/combine run (or a
    previous low-latency run) the buffer is "dirty".  If the buffer is
    not cleaned before the next low-latency dispatch, the kernel reads
    stale metadata and can encounter illegal-instruction errors (CUDA
    error 715).

    This is the same approach used by SGLang's
    DeepEPCudaGraphRunnerAdapter, which calls
    clean_low_latency_buffer() before every forward pass that uses
    low-latency mode.
    """
    try:
        from fastdeploy.model_executor.layers.moe.ep import DeepEPBufferManager

        DeepEPBufferManager.clean_low_latency_buffer()
    except Exception:
        pass


@dataclass
class ConcreteSizeEntry:
    """Record the concrete information corresponding to the current shape(num_tokens)"""

    # Concrete shape
    real_shape: int
    # The size is in cudagraph_capture_sizes
    use_cudagraph: bool = True
    # Has runtime-bs been captured before
    captured: bool = False

    # Need to be captured callable object（dynamic graph or static graph backend）
    runnable: Callable = None  # type: ignore
    # Number of completed warmups
    num_finished_warmup: int = 0
    # Captured cuda graph object corresponding to the current real shape
    cuda_graph: Optional[graphs.CUDAGraph] = None
    # Output buffers of cudagraph
    output_buffers: List[Optional[paddle.Tensor]] = field(default_factory=list)


class Dy2StCudaGraphManager:
    def __init__(self):

        self.state = jit_utils.CUDAGraphState.DISABLE
        self.captured_batch_size = set()
        self.batch_size = -1

    def run_impl(self, original_run_impl, inputs, parameters, attrs):

        run_state = self.state
        prog_attrs, cuda_graph_attrs = attrs
        if run_state == jit_utils.CUDAGraphState.REPLAY:
            if self.batch_size not in self.captured_batch_size:
                run_state = jit_utils.CUDAGraphState.DISABLE
        elif run_state == jit_utils.CUDAGraphState.CAPTURE:
            self.captured_batch_size.add(self.batch_size)

        cuda_graph_attrs |= {
            "cuda_graph_state": run_state,
            "cuda_graph_dispatch_key": self.batch_size if run_state != jit_utils.CUDAGraphState.DISABLE else 0,
        }
        return original_run_impl(inputs, parameters, (prog_attrs, cuda_graph_attrs))

    @contextmanager
    def run_impl_guard(self):
        with paddle.jit.dy2static.pir_partial_program.replace_run_impl_guard(
            self.run_impl,
        ):
            yield


class CudaGraphPiecewiseBackend:
    """Manage the capture and replay of CUDA graphs at the subgraph level."""

    def __init__(
        self,
        fd_config: FDConfig,
        dy_runnable: Callable,
        runnable: Callable,
    ):
        self.fd_config = fd_config
        self.dy_runnable = dy_runnable
        self.runnable = runnable
        self.cudagraph_capture_sizes = fd_config.graph_opt_config.cudagraph_capture_sizes
        self.cudagraph_capture_sizes_prefill = fd_config.graph_opt_config.cudagraph_capture_sizes_prefill
        self.warm_up_size = fd_config.graph_opt_config.cudagraph_num_of_warmups
        self.real_shape_to_captured_size = fd_config.graph_opt_config.real_shape_to_captured_size
        self.real_shape_to_captured_size_prefill = fd_config.graph_opt_config.real_shape_to_captured_size_prefill
        self.full_cuda_graph = fd_config.graph_opt_config.full_cuda_graph
        self.dy2st = fd_config.graph_opt_config.graph_opt_level > 0
        self.unique_memory_pool_id = None
        if self.fd_config.graph_opt_config.use_unique_memory_pool:
            # TODO(gongshaotian): Optimize code
            if paddle.is_compiled_with_cuda():
                from paddle.base.core import CUDAGraph

                self.unique_memory_pool_id = CUDAGraph.gen_new_memory_pool_id()

        self._create_entry_dict()

        self.cuda_graph_manager = None
        if self.fd_config.graph_opt_config.graph_opt_level > 0:
            self.cuda_graph_manager = Dy2StCudaGraphManager()

        self.speculative_decoding = fd_config.speculative_config.method is not None
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.real_bsz_to_captured_size = fd_config.graph_opt_config.real_bsz_to_captured_size

        # Create a dedicated capture stream (same approach as SGLang).
        # DeepEP's low_latency_dispatch internally creates cross-stream dependencies
        # (communication stream <-> default/legacy stream). If CUDA graph capture
        # happens on the default stream, these dependencies cause:
        #   "operation would make the legacy stream depend on a capturing blocking stream"
        # By capturing on a separate non-default stream, the default stream is free
        # and DeepEP can create the required dependencies without conflict.
        self._capture_stream = paddle.device.Stream() if paddle.is_compiled_with_cuda() else None

    def run_static_model(self, entry: ConcreteSizeEntry, **kwargs):

        if not entry.captured:
            # Run warmup and capture on a dedicated non-default stream to avoid
            # "legacy stream depends on capturing blocking stream" errors when
            # DeepEP low_latency_dispatch creates cross-stream dependencies.
            # Clean the DeepEP buffer before warmup to ensure low-latency
            # kernels see a zero-initialized buffer.
            _clean_deepep_low_latency_buffer()
            with _DeepEPStreamGuard(self._capture_stream):
                # Warmup the model
                for n in range(entry.num_finished_warmup, self.warm_up_size):
                    entry.num_finished_warmup += 1
                    entry.runnable(**kwargs)
                    logger.debug(
                        f"[CUDA GRAPH][ID:{id(self)}] Warm up for batch size {entry.real_shape}, "
                        f"finished ({n + 1}/{entry.num_finished_warmup}) times"
                    )

                # Store input addresses for debug
                input_addresses = [x.data_ptr() for (_, x) in kwargs.items() if isinstance(x, paddle.Tensor)]
                entry.input_addresses = input_addresses

                # Capture
                self.cuda_graph_manager.state = jit_utils.CUDAGraphState.CAPTURE
                self.cuda_graph_manager.batch_size = entry.real_shape
                entry.captured = True
                with capture_custom_allreduce():
                    with self.cuda_graph_manager.run_impl_guard():
                        entry.runnable(**kwargs)

        # Replay on the same capture stream
        self.cuda_graph_manager.state = jit_utils.CUDAGraphState.REPLAY
        self.cuda_graph_manager.batch_size = entry.real_shape
        # NOTE: do NOT call _clean_deepep_low_latency_buffer() here.
        # The captured graph already contains the clean_low_latency_buffer kernel
        # from apply()'s is_moe_start_layer path.  Adding an extra external clean
        # would cause an nvshmemx_barrier_all_block() mismatch: worker 0 (replay)
        # would hit 2 barriers while empty-input workers hit only 1.
        with _DeepEPStreamGuard(self._capture_stream):
            with self.cuda_graph_manager.run_impl_guard():
                result = entry.runnable(**kwargs)
            paddle.device.synchronize()
        return result

    def __call__(self, **kwargs) -> List[paddle.Tensor] | paddle.Tensor:
        # Get real shape (total num tokens)
        if self.speculative_decoding and all(self.real_bsz_to_captured_size.values()):
            seq_lens_this_time: paddle.Tensor = kwargs["forward_meta"].seq_lens_this_time
            real_bsz = kwargs["forward_meta"].real_bsz
            num_running_requests = real_bsz if real_bsz > 0 else int((seq_lens_this_time.flatten() > 0).sum().item())
            num_running_requests = max(1, num_running_requests)
            real_shape = self.real_bsz_to_captured_size[num_running_requests]
        else:
            ids_remove_padding: paddle.Tensor = kwargs["forward_meta"].ids_remove_padding
            real_shape = ids_remove_padding.shape[0]
        exist_prefill = kwargs["forward_meta"].exist_prefill
        # Static split graph mode: use Static + CUDAGraph for prefill/mixed phase
        static_cudagraph_for_prefill = exist_prefill and not self.full_cuda_graph and self.dy2st
        # Static full graph mode: use Static + CUDAGraph for decode phase only
        static_cudagraph_for_decode = not exist_prefill and self.full_cuda_graph and self.dy2st

        if static_cudagraph_for_prefill:
            padding_real_shape = self.real_shape_to_captured_size_prefill[real_shape]
        else:
            padding_real_shape = self.real_shape_to_captured_size[real_shape]

        logger.debug(
            f"[CUDA GRAPH][ID:{id(self)}] The actual real shape obtained by CUDAGraph is :{real_shape}, "
            f"The padded shape is :{padding_real_shape}, If Padding :{real_shape != padding_real_shape}"
        )
        entry = self.concrete_size_entries.get((padding_real_shape, static_cudagraph_for_prefill))
        assert entry is not None, f"real shape:{padding_real_shape} is not in cuda graph capture list."
        if entry.runnable is None:
            # Static prefill uses static graph runnable, others use dynamic graph runnable
            entry.runnable = self.runnable if static_cudagraph_for_prefill else self.dy_runnable
            logger.debug(f"[CUDA GRAPH][ID:{id(self)}] New entry lazy initialize with real shape {padding_real_shape}")

        if not entry.use_cudagraph:
            return entry.runnable(**kwargs)

        # Execution modes with CUDAGraph:
        # - Static split graph mode: Static + CUDAGraph for prefill/mixed, Dynamic + CUDAGraph for decode
        # - Static full graph mode: Dynamic for prefill/mixed, Static + CUDAGraph for decode
        # - Dynamic mode: Dynamic + CUDAGraph for decode only
        if static_cudagraph_for_prefill or static_cudagraph_for_decode:
            return self.run_static_model(entry, is_decode=static_cudagraph_for_decode, **kwargs)

        # Capture a new cuda graph
        if entry.cuda_graph is None:
            assert (
                real_shape == padding_real_shape
            ), f"real_shape:{real_shape} is not equal to padding_real_shape:{padding_real_shape} when capture new graph."

            # Run warmup and capture on a dedicated non-default stream to avoid
            # "legacy stream depends on capturing blocking stream" errors when
            # DeepEP low_latency_dispatch creates cross-stream dependencies.
            # Clean the DeepEP buffer before warmup to ensure low-latency
            # kernels see a zero-initialized buffer.
            _clean_deepep_low_latency_buffer()
            with _DeepEPStreamGuard(self._capture_stream):
                # Warmup the model
                for n in range(entry.num_finished_warmup, self.warm_up_size):
                    entry.num_finished_warmup += 1
                    entry.runnable(**kwargs)
                    logger.info(
                        f"[CUDA GRAPH][ID:{id(self)}] Warm up for real shape {padding_real_shape}, "
                        f"finished ({n + 1}/{entry.num_finished_warmup}) times"
                    )

                # Store input addresses for debug
                input_addresses = [x.data_ptr() for (_, x) in kwargs.items() if isinstance(x, paddle.Tensor)]
                entry.input_addresses = input_addresses

                new_grpah = graphs.CUDAGraph(pool_id=self.unique_memory_pool_id)
                paddle.device.synchronize()

                # Capture
                with capture_custom_allreduce():
                    new_grpah.capture_begin()
                    outputs = entry.runnable(**kwargs)
                    if isinstance(outputs, paddle.Tensor):
                        assert outputs is not None
                        outputs = [outputs]
                    new_grpah.capture_end()

                # Store output buffer
                entry.cuda_graph = new_grpah
                for output in outputs:
                    if output is not None:
                        output_buffer = paddle.zeros_like(output)
                        output._share_buffer_to(output_buffer)
                        output._clear()
                        entry.output_buffers.append(output_buffer)
                    else:
                        entry.output_buffers.append(None)

                paddle.device.synchronize()

            # For CUDAGraph debug
            # self._save_cudagrpah_dot_files(entry)
            logger.info(f"[CUDA GRAPH][ID:{id(self)}] CUDAGraph captured for real shape {padding_real_shape}")

        # Replay on the same capture stream
        # NOTE: CUDAGraph::Replay() uses the stream saved during capture (stream_)
        # internally via cudaGraphLaunch(), so the graph is always launched on the
        # capture stream regardless of the current stream. We use stream_guard here
        # so that Paddle's internal stream state is consistent.
        # After replay, we need to synchronize the capture stream with the default
        # stream because the capture stream is non-blocking (kStreamNonBlocking)
        # and does not implicitly synchronize with the default stream.
        # NOTE: do NOT call _clean_deepep_low_latency_buffer() here.
        # The captured graph already contains the clean_low_latency_buffer kernel
        # from apply()'s is_moe_start_layer path.  Adding an extra external clean
        # would cause an nvshmemx_barrier_all_block() mismatch: worker 0 (replay)
        # would hit 2 barriers while empty-input workers hit only 1.
        with _DeepEPStreamGuard(self._capture_stream):
            entry.cuda_graph.replay()
            paddle.device.synchronize()
        logger.debug(f"[CUDA GRAPH][ID:{id(self)}] CUDAGraph replayed for real shape {padding_real_shape}")
        if len(entry.output_buffers) == 1:
            return entry.output_buffers[0]
        return entry.output_buffers

    def _create_entry_dict(self):
        """ """
        # Runtime real shape -> ConcreteSizeEntry
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        for shape in self.cudagraph_capture_sizes:
            self.concrete_size_entries[shape, False] = ConcreteSizeEntry(real_shape=shape)

        for shape in self.cudagraph_capture_sizes_prefill:
            self.concrete_size_entries[shape, True] = ConcreteSizeEntry(real_shape=shape)

        logger.info(
            f"[CUDA GRAPH][ID:{id(self)}] CUDAGraph capture list {self.cudagraph_capture_sizes}, "
            "Created all real shape entry."
        )

    def clear_graph(self):
        """ """
        # Clear graphs
        custom_ar_clear_ipc_handles()
        for (_id, _), entry in self.concrete_size_entries.items():
            if entry.cuda_graph:
                del entry.cuda_graph
                logger.debug(f"[CUDA GRAPH][ID:{id(self)}] The CUDAGraph with shape {_id} has been cleared.")

        del self.concrete_size_entries
        paddle.device.cuda.empty_cache()

        self._decode_capture_index = 0

        # Create new entrys
        self._create_entry_dict()

    def _save_cudagrpah_dot_files(self, entry):
        """Print CUDAGrpah to dot files"""
        log_dir = envs.FD_LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        if entry.cuda_graph:
            entry.cuda_graph.print_to_dot_files(
                f"{log_dir}/GraphDotFiles/backend{id(self)}_shape{entry.real_shape}",
                1 << 0,
            )

    def check_capture_successful(self):
        """Check whether the shapes are captured or not"""
        for (shape, _), entry in self.concrete_size_entries.items():
            if not entry.captured:
                raise ValueError(f"[CUDA GRAPH][ID:{id(self)}] Shape {shape} capture failed.")
