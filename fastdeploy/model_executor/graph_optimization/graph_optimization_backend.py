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

from fastdeploy.config import LLMConfig
from fastdeploy.model_executor.graph_optimization.cudagraph_piecewise_backend import \
    CudaGraphPiecewiseBackend


class GraphOptBackend:
    """ """

    llm_config: LLMConfig
    cudagraph_piecewise_backend: Optional[CudaGraphPiecewiseBackend] = None

    def __init__(self, runnable: Callable, llm_config: LLMConfig):
        self.runnable = runnable
        self.llm_config = llm_config

    def __call__(self, **kwargs):
        # 1. TODO(gongshaotian): Static graph
        if self.llm_config.graph_opt_config.graph_opt_level > 0:
            # 1. Prepare cuda grpah input buffers (contain output of subgraphs)

            # 2. Convert dynamic grpah to static graph
            if self.llm_config.graph_opt_config.graph_opt_level > 1:
                # with cinn
                pass
            else:
                # not use cinn
                pass

            # 3. Split the static graph and get a list of callable obj

            # 4. Get piecewise cuda grpah backend list

            return self.runnable  # Fake return value

        # 2. Dynamic graph
        else:
            print(self.cudagraph_piecewise_backend is None)
            if self.cudagraph_piecewise_backend is None:
                self.cudagraph_piecewise_backend = CudaGraphPiecewiseBackend(
                    llm_config=self.llm_config, runnable=self.runnable)
            # TODO(gongshaotian): handling kwargs
            assert kwargs["input_ids"] is not None
            return self.cudagraph_piecewise_backend.__call__(**kwargs)
