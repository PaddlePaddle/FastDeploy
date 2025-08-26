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

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import paddle.nn.layer
from paddle.device.cuda import graphs

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import capture_custom_allreduce
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend", "cudagraph_piecewise_backend.log")


@dataclass
class ConcreteSizeEntry:
    """Record the concrete information corresponding to the current shape(num_tokens)"""

    # Concrete shape
    real_shape: int
    # The size is in cudagraph_capture_sizes
    use_cudagraph: bool = True
    # Has runtime-bs been captured before
    captured: bool = False

    # Need to be captured callable object（dynamic graph or static grpah backend）
    runnable: Callable = None  # type: ignore
    # Number of completed warmups
    num_finished_warmup: int = 0
    # Captured cuda graph object corresponding to the current real shape
    cuda_graph: Optional[graphs.CUDAGraph] = None
    # Output buffer of cudagraph
    output_buffer: Optional[paddle.Tensor] = None


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
        self.real_shape_to_captured_size = fd_config.graph_opt_config.real_shape_to_captured_size

        self._create_entry_dict()

    def __call__(self, **kwargs):
        # Get real shape(all num tokens)
        ids_remove_padding: paddle.Tensor = kwargs["ids_remove_padding"]
        real_shape = ids_remove_padding.shape[0]
        padding_real_shape = self.real_shape_to_captured_size[real_shape]
        logger.debug(
            f"[CUDA GRAPH] The actual real shape obtained by CUDAGraph is :{real_shape}, "
            f"The padded shape is :{padding_real_shape}"
        )

        entry = self.concrete_size_entries.get(padding_real_shape)
        assert entry is not None, f"real shape:{padding_real_shape} is not in cuda graph capture list."
        if entry.runnable is None:
            entry.runnable = self.runnable
            logger.debug(f"[CUDA GRAPH] New entry lazy initialize with real shape {padding_real_shape}")

        if not entry.use_cudagraph:
            return entry.runnable(**kwargs)

        # Capture a new cuda graph
        if entry.cuda_graph is None:
            # Warmup the model
            for n in range(entry.num_finished_warmup, self.warm_up_size):
                entry.num_finished_warmup += 1
                entry.runnable(**kwargs)
                logger.debug(
                    f"[CUDA GRAPH] Warm up for real shape {padding_real_shape}, "
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

            # For CUDAGraph debug
            # self._save_cudagrpah_dot_files(entry)
            logger.debug(f"[CUDA GRAPH] CUDAGraph captured for real shape {padding_real_shape}")

        # Replay
        entry.cuda_graph.replay()
        logger.debug(f"[CUDA GRAPH] CUDAGraph replayed for real shape {padding_real_shape}")
        return entry.output_buffer

    def _create_entry_dict(self):
        """ """
        # Runtime real shape -> ConcreteSizeEntry
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        for shape in self.cudagraph_capture_sizes:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(real_shape=shape)

        logger.info(
            f"[CUDA GRAPH] CUDAGraph capture list {self.cudagraph_capture_sizes}, " "Created all real shape entry."
        )

    def clear_graph(self):
        """ """
        # Clear graphs
        for id, entry in self.concrete_size_entries.items():
            if entry.cuda_graph:
                del entry.cuda_graph
                logger.debug(f"[CUDA GRAPH] The CUDAGraph with shape {id} has been cleared.")

        del self.concrete_size_entries
        paddle.device.cuda.empty_cache()

        # Create new entrys
        self._create_entry_dict()

    def _save_cudagrpah_dot_files(self, entry):
        """Print CUDAGrpah to dot files"""
        if entry.cuda_graph:
            entry.cuda_graph.print_to_dot_files(
                f"./log/GraphDotFiles/backend{id(self)}_shape{entry.real_shape}",
                1 << 0,
            )
