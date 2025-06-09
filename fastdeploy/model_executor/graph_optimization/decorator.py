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

from abc import abstractmethod
from typing import Callable, Optional, TypeVar

import paddle.nn.layer

from fastdeploy.config import LLMConfig
from fastdeploy.model_executor.graph_optimization.graph_optimization_backend import \
    GraphOptBackend

_T = TypeVar("_T", bound=type[paddle.nn.Layer])


def support_graph_opt(cls: Optional[_T] = None) -> _T:
    """
    A decorator for wrapping models or layers with CUDA graph support.
    This enables efficient kernel launch sequencing for improved GPU performance.

    Example usage:

    '''
    @support_graph_opt
    class ErnieBot(paddle.nn.Layer):
        def __init__(**kwargs):
            ...

        def forward(self, x: paddle.Tensor, y: paddle.Tensor):
            ...
    '''
    """
    if GraphOptWrapper in cls.__bases__:
        return cls
    else:
        cls.__bases__ = cls.__bases__ + (GraphOptWrapper, )
    origin_init = cls.__init__

    def __init__(self, llm_config: LLMConfig, **kwargs):
        """ Decorator model.__init__() func """
        origin_init(self, llm_config=llm_config, **kwargs)
        self.use_graph_opt = (
            not (llm_config.graph_opt_config.graph_opt_level == 0
                 and not llm_config.graph_opt_config.use_cudagraph))
        if self.use_graph_opt:
            GraphOptWrapper.__init__(self,
                                     llm_config=llm_config,
                                     graph_opt_backend=None)
        else:
            # Not use graph optimization
            return

    def __call__(self, **kwargs):
        """ Decorator model.__call__() func """
        if not self.use_graph_opt:
            return self.forward(**kwargs)

        return self.graph_opt_backend(**kwargs)

    cls.__init__ = __init__
    cls.__call__ = __call__
    return cls


class GraphOptWrapper:
    """ """

    def __init__(
        self,
        graph_opt_backend: Optional[Callable] = None,
        llm_config: LLMConfig = None,
    ):
        if graph_opt_backend is None:
            graph_opt_backend = GraphOptBackend(self.forward, llm_config)
        self.graph_opt_backend = graph_opt_backend

    @abstractmethod
    def forward(self, **kwargs):
        """ """
        pass

    def __call__(self, **kwargs):
        return self.graph_opt_backend(**kwargs)
