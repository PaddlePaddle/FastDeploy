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

from fastdeploy.config import FDConfig
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend",
                    "cudagraph_piecewise_backend.log")


@dataclass
class ConcreteSizeEntry:
    """ Record the concrete information corresponding to the current batch size """
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

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class CudaGraphPiecewiseBackend:
    """ """

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

        # runtime_bs -> ConcreteSizeEntry
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        for shape in self.cudagraph_capture_sizes:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_bs=shape)

        print("[CUDA GRAPH] Created all batch size entry ")

    def __call__(self, **kwargs):
        # Get batch size
        ids_remove_padding: paddle.Tensor = kwargs["ids_remove_padding"]
        batch_size = ids_remove_padding.shape[0]

        padding_batch_size = self.batch_size_to_captured_size[batch_size]
        # print(
        #     f"[CUDA GRAPH] The actual batch size obtained by CUDAGraph is :{batch_size}, ",
        #     f"The padded batch size is :{padding_batch_size}"
        # )

        entry = self.concrete_size_entries.get(padding_batch_size)
        assert entry is not None, f"Batch size:{padding_batch_size} is not in cuda graph capture list."
        if entry.runnable is None:
            entry.runnable = self.runnable
            # print(
            #     f"[CUDA GRAPH] New entry lazy initialize with batch size {padding_batch_size}"
            # )

        if not entry.use_cudagraph:
            return entry.runnable(**kwargs)

        # Capture a new cuda graph
        if entry.cuda_graph is None:
            # Warmup the model
            for n in range(entry.num_finished_warmup, self.warm_up_size):
                entry.num_finished_warmup += 1
                entry.runnable(**kwargs)
                # print(
                #     "[CUDA GRAPH] Warm up for batch size ",
                #     f"{padding_batch_size}, finished ({n+1}/{entry.num_finished_warmup}) times"
                # )

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
            # print(
            #     f"[CUDA GRAPH] CUDAGraph captured for batch size {padding_batch_size}"
            # )

        # Replay
        entry.cuda_graph.replay()
        # print(f"[CUDA GRAPH] CUDAGraph replayed for batch size {padding_batch_size}")
        return entry.output_buffer
