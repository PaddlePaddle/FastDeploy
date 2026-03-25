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
from typing import Callable, Optional, Sequence

import paddle

import fastdeploy


def block_wise_cuda_graph_wrap(
    inputs: Sequence[str],
    key_fn: Optional[Callable[..., tuple]] = None,
):
    """
    Method decorator that wraps a forward method with CUDA Graph capture/replay.

    On the first call for a given cache key (derived from tensor shapes/dtypes),
    the decorated method is captured into a CUDA Graph. Subsequent calls with the
    same key will replay the graph after updating input data pointers.

    Args:
        inputs: Names of parameters that are input tensors to be tracked for
            CUDA Graph pointer replacement. These must be parameter names of the
            decorated method. Only non-None tensor arguments are tracked.
        key_fn: Optional callable to generate the cache key from method arguments.
            Signature: key_fn(arg0, arg1, ...) with args in declaration order
            (excluding self). Defaults to a key based on tensor shapes/dtypes.

    Example:
        class MyLayer(nn.Layer):
            @cuda_graph_wrap(inputs=["x", "residual"])
            def forward(self, x, residual=None):
                return some_op(x, residual)
    """

    def decorator(method: Callable) -> Callable:
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())  # ["self", "x", "residual_input", ...]

        for name in inputs:
            if name not in params or name == "self":
                raise ValueError(
                    f"cuda_graph_wrap: input '{name}' is not a parameter of "
                    f"{method.__qualname__}. Available: {[p for p in params if p != 'self']}"
                )

        # ---- Pre-compute at decoration time (runs once) ----

        _EMPTY = inspect.Parameter.empty
        _Tensor = paddle.Tensor

        # For each non-self param: (name, args_index, default_value)
        # args_index is position in *args (0-based, since self is consumed by Python)
        _param_info = tuple((p, i - 1, sig.parameters[p].default) for i, p in enumerate(params) if p != "self")

        # For each declared input tensor: (name, args_index)
        _input_info = tuple((name, params.index(name) - 1) for name in inputs)

        # Instance attribute names (short to save getattr overhead)
        _g = f"_cg_{method.__name__}_g"
        _ci = f"_cg_{method.__name__}_ci"
        _co = f"_cg_{method.__name__}_co"

        _use_custom_key = key_fn is not None

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not fastdeploy.envs.FD_USE_BLOCK_WISE_CUDA_GRAPH:
                return method(self, *args, **kwargs)

            nargs = len(args)

            # Skip CUDA graph if any input tensor has a 0 in its shape
            for a in args:
                if isinstance(a, _Tensor) and 0 in a.shape:
                    return method(self, *args, **kwargs)
            for v in kwargs.values():
                if isinstance(v, _Tensor) and 0 in v.shape:
                    return method(self, *args, **kwargs)

            # === Key generation: inline, no sig.bind ===
            if _use_custom_key:
                # Resolve all args for custom key_fn
                resolved = []
                for pname, aidx, default in _param_info:
                    if pname in kwargs:
                        resolved.append(kwargs[pname])
                    elif aidx < nargs:
                        resolved.append(args[aidx])
                    elif default is not _EMPTY:
                        resolved.append(default)
                    else:
                        resolved.append(None)
                key = key_fn(*resolved)
            else:
                # Default: fast inline key from shapes/dtypes
                _kp = []
                for pname, aidx, default in _param_info:
                    if pname in kwargs:
                        v = kwargs[pname]
                    elif aidx < nargs:
                        v = args[aidx]
                    else:
                        v = default
                    if isinstance(v, _Tensor):
                        _kp.append((tuple(v.shape), v.dtype))
                    elif v is None:
                        _kp.append(None)
                    elif callable(v):
                        _kp.append(True)
                key = tuple(_kp)

            # === Lazy init via __dict__ (bypass nn.Layer.__getattr__) ===
            _d = self.__dict__
            try:
                graphs = _d[_g]
                cinputs = _d[_ci]
                coutputs = _d[_co]
            except KeyError:
                graphs = {}
                cinputs = {}
                coutputs = {}
                _d[_g] = graphs
                _d[_ci] = cinputs
                _d[_co] = coutputs

            if key not in graphs:
                # === First encounter: capture ===
                graph = paddle.device.cuda.graphs.CUDAGraph(enable_replace=True)
                graphs[key] = graph

                ci = {}
                for name, aidx in _input_info:
                    v = kwargs[name] if name in kwargs else (args[aidx] if aidx < nargs else None)
                    if v is not None and isinstance(v, _Tensor):
                        ci[name] = v
                cinputs[key] = ci

                graph.capture_begin()
                result = method(self, *args, **kwargs)
                graph.capture_end()

                graph.replay()

                coutputs[key] = result
                return result
            else:
                # === Replay path (HOT PATH) ===
                old_ptrs = []
                new_ptrs = []
                ci = cinputs[key]

                for name, aidx in _input_info:
                    v = kwargs[name] if name in kwargs else (args[aidx] if aidx < nargs else None)
                    if v is not None and name in ci:
                        old_ptrs.append(ci[name].data_ptr())
                        new_ptrs.append(v.data_ptr())
                        ci[name] = v

                if old_ptrs:
                    graphs[key].replace_input_ptrs(old_ptrs, new_ptrs)
                graphs[key].replay()

                return coutputs[key]

        return wrapper

    return decorator
