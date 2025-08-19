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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import paddle.nn.layer
from paddle.device.cuda import graphs
from paddle.jit.dy2static.utils import CUDAGraphState

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import capture_custom_allreduce
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend", "cudagraph_piecewise_backend.log")


@dataclass
class ConcreteSizeEntry:
    """Record the concrete information corresponding to the current batch size"""

    # Concrete batch size
    runtime_bs: int
    # The size is in cudagraph_capture_sizes
    use_cudagraph: bool = True
    # Has runtime-bs been captured before
    captured: bool = False

    # Need to be captured callable object（dynamic graph or static grpah backend）
    runnable: Callable = None  # type: ignore
    # Number of completed warmups
    num_finished_warmup: int = 0
    # Captured cuda graph object corresponding to the current batch size
    cuda_graph: Optional[graphs.CUDAGraph] = None
    # Output buffer of cudagraph
    output_buffer: Optional[paddle.Tensor] = None


class Dy2StCudaGraphManager:
    def __init__(self):
        self.state = CUDAGraphState.DISABLE
        self.captrued_batch_size = set()
        self.batch_size = -1

    def run_impl(self, original_run_impl, inputs, parameters, attrs):
        run_state = self.state
        prog_attrs, cuda_graph_attrs = attrs
        if run_state == CUDAGraphState.REPLAY:
            if self.batch_size not in self.captrued_batch_size:
                run_state = CUDAGraphState.DISABLE
        elif run_state == CUDAGraphState.CAPTURE:
            self.captrued_batch_size.add(self.batch_size)

        cuda_graph_attrs |= {
            "cuda_graph_state": run_state,
            "cuda_graph_dispatch_key": self.batch_size if run_state != CUDAGraphState.DISABLE else 0,
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
        runnable: Callable,
    ):
        self.fd_config = fd_config
        self.runnable = runnable
        self.cudagraph_capture_sizes = fd_config.graph_opt_config.cudagraph_capture_sizes
        self.warm_up_size = fd_config.graph_opt_config.cudagraph_num_of_warmups
        self.batch_size_to_captured_size = fd_config.graph_opt_config.batch_size_to_captured_size

        # Runtime batch size -> ConcreteSizeEntry
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        for shape in self.cudagraph_capture_sizes:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(runtime_bs=shape)

        self.cuda_graph_manager = None
        if self.fd_config.graph_opt_config.graph_opt_level > 0:
            self.cuda_graph_manager = Dy2StCudaGraphManager()

        logger.info(
            f"[CUDA GRAPH] CUDAGraph capture list {self.cudagraph_capture_sizes}, " "Created all batch sizes entry."
        )

    def run_static_model(self, entry: ConcreteSizeEntry, **kwargs):
        if not entry.captured:
            # Warmup the model
            for n in range(entry.num_finished_warmup, self.warm_up_size):
                entry.num_finished_warmup += 1
                entry.runnable(**kwargs)
                logger.debug(
                    f"[CUDA GRAPH] Warm up for batch size {entry.runtime_bs}, "
                    f"finished ({n + 1}/{entry.num_finished_warmup}) times"
                )

            # Store input addresses for debug
            input_addresses = [x.data_ptr() for (_, x) in kwargs.items() if isinstance(x, paddle.Tensor)]
            entry.input_addresses = input_addresses

            # Capture
            self.cuda_graph_manager.state = CUDAGraphState.CAPTURE
            self.cuda_graph_manager.batch_size = entry.runtime_bs
            with self.cuda_graph_manager.run_impl_guard():
                entry.runnable(**kwargs)

        # Replay
        self.cuda_graph_manager.state = CUDAGraphState.REPLAY
        self.cuda_graph_manager.batch_size = entry.runtime_bs
        with self.cuda_graph_manager.run_impl_guard():
            return entry.runnable(**kwargs)

    def __call__(self, **kwargs):
        # Get batch size
        ids_remove_padding: paddle.Tensor = kwargs["ids_remove_padding"]
        batch_size = ids_remove_padding.shape[0]
        padding_batch_size = self.batch_size_to_captured_size[batch_size]
        logger.debug(
            f"[CUDA GRAPH] The actual batch size obtained by CUDAGraph is :{batch_size}, "
            f"The padded batch size is :{padding_batch_size}"
        )

        entry = self.concrete_size_entries.get(padding_batch_size)
        assert entry is not None, f"Batch size:{padding_batch_size} is not in cuda graph capture list."
        if entry.runnable is None:
            entry.runnable = self.runnable
            logger.debug(f"[CUDA GRAPH] New entry lazy initialize with batch size {padding_batch_size}")

        if not entry.use_cudagraph:
            return entry.runnable(**kwargs)

        if self.fd_config.graph_opt_config.graph_opt_level > 0:
            return self.run_static_model(entry, **kwargs)

        # Capture a new cuda graph
        if entry.cuda_graph is None:
            # Warmup the model
            for n in range(entry.num_finished_warmup, self.warm_up_size):
                entry.num_finished_warmup += 1
                entry.runnable(**kwargs)
                logger.debug(
                    f"[CUDA GRAPH] Warm up for batch size {padding_batch_size}, "
                    f"finished ({n + 1}/{entry.num_finished_warmup}) times"
                )

            # Store input addresses for debug
            input_addresses = [x.data_ptr() for (_, x) in kwargs.items() if isinstance(x, paddle.Tensor)]
            entry.input_addresses = input_addresses

            new_grpah = graphs.CUDAGraph()
            paddle.device.synchronize()

            # Capture
            with capture_custom_allreduce():
                new_grpah.capture_begin()
                output = entry.runnable(**kwargs)
                new_grpah.capture_end()

            # Store output buffer
            entry.cuda_graph = new_grpah
            entry.output_buffer = paddle.zeros_like(output)
            output._share_buffer_to(entry.output_buffer)
            output._clear

            paddle.device.synchronize()
            logger.debug(f"[CUDA GRAPH] CUDAGraph captured for batch size {padding_batch_size}")

        # Replay
        entry.cuda_graph.replay()
        logger.debug(f"[CUDA GRAPH] CUDAGraph replayed for batch size {padding_batch_size}")
        return entry.output_buffer
