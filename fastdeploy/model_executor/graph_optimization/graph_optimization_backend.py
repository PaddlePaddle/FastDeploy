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

import functools
import inspect
import types
from typing import Callable, Optional, TypeVar, get_type_hints

from paddle.jit import sot
from paddle.jit.dy2static.utils import Backend as ToStaticBackend
from typing_extensions import ParamSpec

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.graph_optimization.cudagraph_piecewise_backend import (
    CudaGraphPiecewiseBackend,
)
from fastdeploy.model_executor.graph_optimization.dynamic_dims_marker import (
    resolve_dynamic_dims,
)
from fastdeploy.model_executor.graph_optimization.utils import in_profile_run_mode
from fastdeploy.model_executor.graph_optimization.utils import (
    in_sot_warmup_mode as in_warmup_mode,
)
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend", "cudagraph_piecewise_backend.log")


P = ParamSpec("P")
T = TypeVar("T")


def apply_to_static_optimization(fn: Callable[P, T], backend: ToStaticBackend) -> Callable[P, T]:
    forward_fn = fn
    forward_sig = inspect.signature(forward_fn)
    forward_type_hints = get_type_hints(forward_fn)
    static_forward_fn = sot.symbolic_translate(forward_fn, training=False, backend=backend)
    unsafe_static_forward_fn = None

    @functools.wraps(forward_fn)
    def warmup_impl(self, *args, **kwargs):
        nonlocal unsafe_static_forward_fn
        bound_args = forward_sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        for name, arg in bound_args.arguments.items():
            if name not in forward_type_hints:
                continue
            annotation = forward_type_hints[name]
            resolve_dynamic_dims(arg, name, annotation)

        result = static_forward_fn(self, *args, **kwargs)
        original_code = forward_fn.__code__
        (new_guarded_codes, _) = sot.opcode_translator.executor.executor_cache.OpcodeExecutorCache().cache[
            original_code
        ]
        # Check has only one graph
        if len(new_guarded_codes) > 1:
            raise RuntimeError("Model has multiple generated code, please check all dynamic dim has marked.")
        # Check generated code has no break graph
        new_code = new_guarded_codes[0][0][0]
        if any(name.startswith("$") for name in new_code.co_names):  # TODO(SigureMo): It's a internal impl
            raise RuntimeError("Model has breakgraph, please set env SOT_LOG_LEVEL=3 to check it.")
        unsafe_static_forward_fn = types.FunctionType(
            new_code,
            forward_fn.__globals__,
            forward_fn.__name__,
            forward_fn.__defaults__,
            forward_fn.__closure__,
        )
        return result

    @functools.wraps(forward_fn)
    def static_forward(self, *args, **kwargs):
        nonlocal unsafe_static_forward_fn
        if in_profile_run_mode():
            return forward_fn(self, *args, **kwargs)
        if in_warmup_mode():
            return warmup_impl(self, *args, **kwargs)
        assert unsafe_static_forward_fn is not None
        return unsafe_static_forward_fn(self, *args, **kwargs)

    return static_forward


class GraphOptBackend:
    """
    Integrated various graph optimization functions, including dynamic graph to static graph conversion,
    CINN compilation optimization, CudaGraph, and so on.
    """

    fd_config: FDConfig
    cudagraph_piecewise_backend: Optional[CudaGraphPiecewiseBackend] = None

    def __init__(self, runnable: Callable, fd_config: FDConfig):
        self.runnable = runnable
        self.dy_runnable = self.runnable
        self.fd_config = fd_config
        self.max_captre_size = fd_config.graph_opt_config.cudagraph_capture_sizes[0]
        self._debug_count_cudagraph_replay = 0
        self._debug_count_total_step = 0

        if self.fd_config.graph_opt_config.graph_opt_level > 0:
            # 1. Prepare cuda graph input buffers (contain output of subgraphs)

            # 2. Convert dynamic graph to static graph

            backend = (
                ToStaticBackend.CINN if self.fd_config.graph_opt_config.graph_opt_level > 1 else ToStaticBackend.PHI
            )
            self.runnable = apply_to_static_optimization(
                self.runnable.__func__,
                backend,
            ).__get__(self.runnable.__self__)

        self.cudagraph_switch_threshold = (
            1024 if self.fd_config.graph_opt_config.graph_opt_level > 0 else self.max_captre_size
        )

    def __call__(self, **kwargs):
        self._debug_count_total_step += 1
        if not self.fd_config.graph_opt_config.use_cudagraph:
            return self.runnable(**kwargs)
        if self.cudagraph_piecewise_backend is None:
            self.cudagraph_piecewise_backend = CudaGraphPiecewiseBackend(
                fd_config=self.fd_config, runnable=self.runnable
            )

        assert kwargs["forward_meta"].ids_remove_padding is not None
        real_shape = kwargs["forward_meta"].ids_remove_padding.shape[0]

        if (not kwargs["forward_meta"].step_use_cudagraph) or (real_shape > self.cudagraph_switch_threshold):
            return self.dy_runnable(**kwargs)
        else:
            self._debug_count_cudagraph_replay += 1
            logger.debug(
                f"[CUDA GRAPH][ID:{id(self.cudagraph_piecewise_backend)}] Total step count: {self._debug_count_total_step}, CUDAGraph replay count: {self._debug_count_cudagraph_replay}"
            )
            return self.cudagraph_piecewise_backend.__call__(**kwargs)

    def clear_cudagraph_piecewise_backend(self):
        """ """
        self.cudagraph_piecewise_backend.clear_graph()
