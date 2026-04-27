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
from fastdeploy.platforms import current_platform
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend", "worker_process.log")


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

        # Expected decode capture sequence (descending), consistent with capture_model() iteration order.
        # Used to validate that captures happen in the correct order.
        self._decode_expected_sequence: list[int] = sorted(self.cudagraph_capture_sizes, reverse=True)
        # Points to the next expected position in _decode_expected_sequence.
        self._decode_capture_index: int = 0

    def _validate_decode_capture_order(self, shape: int) -> None:
        """Validate that decode CUDA graph captures happen in expected descending order.

        Raises RuntimeError immediately if the actual capture order deviates from
        the order defined by cudagraph_capture_sizes (sorted descending).
        """
        if current_platform.is_xpu():
            return

        if self._decode_capture_index >= len(self._decode_expected_sequence):
            raise RuntimeError(
                f"[CUDA GRAPH][ID:{id(self)}] Unexpected CUDA graph capture: shape={shape}. "
                f"All {len(self._decode_expected_sequence)} expected captures have already completed. "
                f"Expected sequence: {self._decode_expected_sequence}"
            )
        expected = self._decode_expected_sequence[self._decode_capture_index]
        if shape != expected:
            raise RuntimeError(
                f"[CUDA GRAPH][ID:{id(self)}] CUDA graph capture order mismatch at index "
                f"{self._decode_capture_index}: expected shape={expected}, got shape={shape}. "
                f"Full expected sequence: {self._decode_expected_sequence}"
            )
        logger.debug(
            f"[CUDA GRAPH][ID:{id(self)}] Capture order validated: shape={shape} matches "
            f"expected sequence at index {self._decode_capture_index} "
            f"(sequence: {self._decode_expected_sequence})"
        )
        self._decode_capture_index += 1

    def run_static_model(self, entry: ConcreteSizeEntry, is_decode: bool = False, **kwargs):

        if not entry.captured:
            if is_decode:
                self._validate_decode_capture_order(entry.real_shape)
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

        # Replay
        self.cuda_graph_manager.state = jit_utils.CUDAGraphState.REPLAY
        self.cuda_graph_manager.batch_size = entry.real_shape
        with self.cuda_graph_manager.run_impl_guard():
            return entry.runnable(**kwargs)

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
            self._validate_decode_capture_order(padding_real_shape)
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

        # Replay
        entry.cuda_graph.replay()
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
