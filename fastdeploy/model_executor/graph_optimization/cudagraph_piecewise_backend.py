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

import paddle.device.cuda.graphs as graphs
import paddle.nn.layer

from fastdeploy.config import LLMConfig
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend",
                    "cudagraph_piecewise_backend.log")


@dataclass
class ConcreteSizeEntry:
    """ Record the concrete information corresponding to the current batch size """
    # Concrete batch size
    runtime_bs: int
    # The size is in cudagraph_capture_sizes
    use_cuda_graph: bool = True
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

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class CudaGraphPiecewiseBackend:
    """ """

    def __init__(
        self,
        llm_config: LLMConfig,
        runnable: Callable,
    ):
        self.llm_config = llm_config
        self.runnable = runnable
        self.cuda_graph_capture_size = llm_config.graph_opt_config.cudagraph_capture_sizes
        # runtime_bs -> ConcreteSizeEntry
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        for shape in self.cuda_graph_capture_size:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_bs=shape)

        print("create all batch size entry")

    def __call__(self, **kwargs):
        # Get batch size
        input_ids: paddle.Tensor = kwargs['input_ids']
        batch_size = input_ids.shape[0]
        entry = self.concrete_size_entries.get(batch_size)
        if entry.runnable is None:
            entry.runnable = self.runnable
            print(
                f"[CUDA GRAPH] new entry lazy initialize with batch size {batch_size}"
            )

        if not entry.use_cuda_graph:
            return entry.runnable(**kwargs)

        # Capture a new cuda graph
        if entry.cuda_graph is None:
            # Warmup the model
            for n in range(entry.num_finished_warmup):
                entry.num_finished_warmup += 1
                entry.runnable(**kwargs)
                print(
                    f"[CUDA GRAPH] warm up for batch size "
                    f"{batch_size}, finished ({n+1}/{entry.num_finished_warmup}) times"
                )

            # Store input addresses for debug
            input_addresses = [
                x.data_ptr() for (_, x) in kwargs.items()
                if isinstance(x, paddle.Tensor)
            ]
            entry.input_addresses = input_addresses

            new_grpah = graphs.CUDAGraph()
            paddle.device.synchronize()

            # Capture
            new_grpah.capture_begin()
            output = entry.runnable(**kwargs)
            new_grpah.capture_end()

            # Store output buffer
            entry.cuda_graph = new_grpah
            entry.output_buffer = paddle.zeros_like(output)
            output._share_buffer_to(entry.output_buffer)
            output._clear

            paddle.device.synchronize()
            print(
                f"[CUDA GRAPH] cuda graph captured for batch size {batch_size}"
            )

        # Replay
        entry.cuda_graph.replay()
        print(f"[CUDA GRAPH] cuda graph replayed for batch size {batch_size}")
        return entry.output_buffer
