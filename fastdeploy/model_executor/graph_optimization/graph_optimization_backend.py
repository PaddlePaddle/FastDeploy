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

from typing import Callable, Optional

from paddle.jit.dy2static.utils import Backend

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.graph_optimization.cudagraph_piecewise_backend import (
    CudaGraphPiecewiseBackend,
)


class GraphOptBackend:
    """
    Integrated various graph optimization functions, including dynamic graph to static graph conversion,
    CINN compilation optimization, CudaGraph, and so on.
    """

    fd_config: FDConfig
    cudagraph_piecewise_backend: Optional[CudaGraphPiecewiseBackend] = None

    def __init__(self, runnable: Callable, fd_config: FDConfig):
        self.runnable = runnable
        self.fd_config = fd_config

        self.max_captre_batch = fd_config.graph_opt_config.cudagraph_capture_sizes[0]
        if self.fd_config.graph_opt_config.graph_opt_level > 0:
            # 1. Prepare cuda grpah input buffers (contain output of subgraphs)

            # 2. Convert dynamic grpah to static graph
            from paddle.jit import sot

            backend = Backend.CINN if self.fd_config.graph_opt_config.graph_opt_level > 1 else Backend.PHI
            self.runnable = sot.symbolic_translate(self.runnable, training=False, backend=backend)

    def __call__(self, **kwargs):
        if not self.fd_config.graph_opt_config.use_cudagraph:
            return self.runnable(**kwargs)
        if self.cudagraph_piecewise_backend is None:
            self.cudagraph_piecewise_backend = CudaGraphPiecewiseBackend(
                fd_config=self.fd_config, runnable=self.runnable
            )

        assert kwargs["forward_meta"].ids_remove_padding is not None
        batch_size = kwargs["forward_meta"].ids_remove_padding.shape[0]

        if (not kwargs["forward_meta"].step_use_cudagraph) or (batch_size > self.max_captre_batch):
            return self.runnable(**kwargs)
        else:
            return self.cudagraph_piecewise_backend.__call__(**kwargs)
