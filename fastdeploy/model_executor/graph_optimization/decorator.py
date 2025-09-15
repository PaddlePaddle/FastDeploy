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

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.graph_optimization.graph_optimization_backend import (
    GraphOptBackend,
)

_T = TypeVar("_T", bound=type[paddle.nn.Layer])


def support_graph_optimization(cls: Optional[_T] = None) -> _T:
    """
    A decorator for wrapping models or layers with static graph and CUDAGraph support.
    This enables efficient kernel launch sequencing for improved GPU performance.

    Example usage:

    '''
    @support_graph_optimization
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
        cls.__bases__ = cls.__bases__ + (GraphOptWrapper,)
    origin_init = cls.__init__

    def __init__(self, fd_config: FDConfig, **kwargs):
        """Decorator model.__init__() func"""
        origin_init(self, fd_config=fd_config, **kwargs)
        self.use_graph_opt = fd_config.graph_opt_config.graph_opt_level > 0 or fd_config.graph_opt_config.use_cudagraph
        if self.use_graph_opt:
            GraphOptWrapper.__init__(self, fd_config=fd_config, graph_opt_backend=None)
        else:
            # Not use graph optimization
            return

    def __call__(self, **kwargs):
        """Decorator model.__call__() func"""
        if not self.use_graph_opt:
            return self.forward(**kwargs)

        return self.graph_opt_backend(**kwargs)

    cls.__init__ = __init__
    cls.__call__ = __call__
    return cls


class GraphOptWrapper:
    """The wrapper for GraphOptBackend"""

    def __init__(
        self,
        graph_opt_backend: Optional[Callable] = None,
        fd_config: FDConfig = None,
    ):
        if graph_opt_backend is None:
            graph_opt_backend = GraphOptBackend(self.forward, fd_config)
        self.graph_opt_backend = graph_opt_backend

    @abstractmethod
    def forward(self, **kwargs):
        """Abstract methods for implementing model.forward()"""
        pass

    def __call__(self, **kwargs):
        return self.graph_opt_backend(**kwargs)

    def clear_grpah_opt_backend(self, fd_config):
        """ """
        # TODO(gongshaotian): Resolve the bug of static graphs not being able to update weights
        assert (
            fd_config.graph_opt_config.graph_opt_level < 1
        ), "Currently unable to update weights in static graph mode."
        self.graph_opt_backend.clear_cudagraph_piecewise_backend()


def cuda_graph_buffers(buffer_meta):
    def decorator(cls):
        original_init = cls.__init__

        def __init__(self, fd_config: FDConfig, **kwargs):
            original_init(self, fd_config=fd_config, **kwargs)

            def _resolve_path(root, path: str):
                cur = root
                for p in path.split("."):
                    cur = getattr(cur, p)
                return cur

            if not hasattr(self, "_cuda_graph_buffers"):
                self._cuda_graph_buffers = {}
                for name, meta in buffer_meta.items():
                    shape = [_resolve_path(fd_config, s) if isinstance(s, str) else s for s in meta["shape"]]
                    dtype = meta["dtype"]
                    if "." in meta["dtype"]:
                        dtype = _resolve_path(fd_config, meta["dtype"])
                    self._cuda_graph_buffers[name] = paddle.full(
                        shape=shape,
                        dtype=dtype,
                        fill_value=meta.get("value", 0),
                    )

        cls.__init__ = __init__
        return cls

    return decorator
