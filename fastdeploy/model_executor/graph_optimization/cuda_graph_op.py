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
from paddleformers.utils.log import logger as _LOGGER

import fastdeploy

# ---- Module-level state for pre-captured block-wise CUDA graphs ----

# When True, the wrapper is in the capture phase (during dummy_run) and
# will capture new graphs. When False, uncached keys fall back to eager.
_BLOCK_WISE_CAPTURING: bool = False

# Registry of all shared-mode graph caches, for bulk clearing.
_ALL_SHARED_CACHES: list = []

# Global counter / registry of all captured block-wise graphs (for logging).
# Each entry: (qualname, key, shared_mode)
_CAPTURED_GRAPH_LOG: list = []


def get_captured_graph_log():
    """Return the list of all captured (qualname, key, shared) triples."""
    return list(_CAPTURED_GRAPH_LOG)


def dump_captured_graph_summary():
    """Print a summary of all captured block-wise CUDA graphs."""
    from collections import Counter

    if not fastdeploy.envs.FD_BLOCK_WISE_DEBUG:
        return
    if not _CAPTURED_GRAPH_LOG:
        _LOGGER.info("[block_wise_cuda_graph] no graph captured")
        return
    counter = Counter(q for q, _, _ in _CAPTURED_GRAPH_LOG)
    _LOGGER.info(
        f"[block_wise_cuda_graph] total captured graphs={len(_CAPTURED_GRAPH_LOG)} "
        f"across {len(counter)} distinct methods:"
    )
    for qname, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        _LOGGER.info(f"  - {qname} : {cnt} graph(s)")


def set_block_wise_capturing(capturing: bool):
    """Toggle the capture phase flag. Only capture graphs when this is True."""
    global _BLOCK_WISE_CAPTURING
    _BLOCK_WISE_CAPTURING = capturing


def clear_all_block_wise_graphs():
    """Clear all shared block-wise graph caches (e.g. for RL weight updates)."""
    for graphs, cinputs, coutputs in _ALL_SHARED_CACHES:
        graphs.clear()
        cinputs.clear()
        coutputs.clear()


def block_wise_cuda_graph_wrap(
    inputs: Sequence[str],
    self_attrs: Sequence[str] = (),
    key_fn: Optional[Callable[..., tuple]] = None,
):
    """
    Method decorator that wraps a forward method with CUDA Graph capture/replay.

    On the first call for a given cache key (derived from tensor shapes/dtypes),
    the decorated method is captured into a CUDA Graph. Subsequent calls with the
    same key will replay the graph after updating input data pointers.

    When ``_BLOCK_WISE_CAPTURING`` is managed via ``set_block_wise_capturing``,
    new graphs are only captured during the capture phase (dummy_run). At runtime,
    uncached keys fall back to eager execution, avoiding expensive on-the-fly captures.

    When ``self_attrs`` is provided, the named tensor attributes of ``self``
    (e.g. ``weight``) are also tracked for pointer replacement, and the graph
    cache is **shared across all instances** (closure-level). This allows layers
    with identical computation but different weights to share a single captured
    graph, dramatically reducing the total number of graphs from O(num_layers)
    to O(num_unique_shapes).

    When ``self_attrs`` is empty (default), graphs are cached per instance.

    Output tensors from the capture phase are reused across replays — the graph
    always writes to the same output memory. This avoids per-replay allocation
    overhead. Callers must consume the output before the next replay of the same
    graph (which is naturally satisfied in sequential layer-by-layer forward).

    Args:
        inputs: Names of parameters that are input tensors to be tracked for
            CUDA Graph pointer replacement. These must be parameter names of the
            decorated method. Only non-None tensor arguments are tracked.
        self_attrs: Attribute names on ``self`` that are tensor parameters to be
            replaced via pointer replacement (e.g. ``["weight"]``). When non-empty,
            enables cross-instance graph sharing.
        key_fn: Optional callable to generate the cache key from method arguments.
            Signature: key_fn(arg0, arg1, ...) with args in declaration order
            (excluding self). Defaults to a key based on tensor shapes/dtypes.

    Example:
        class MyNorm(nn.Layer):
            @block_wise_cuda_graph_wrap(
                inputs=["x", "residual"],
                self_attrs=["weight"],  # all layers share one graph
            )
            def forward(self, x, residual=None):
                return rms_norm(x, self.weight), residual
    """

    def decorator(method: Callable) -> Callable:
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())  # ["self", "x", "residual_input", ...]
        _qualname = method.__qualname__

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

        _self_attr_names = tuple(self_attrs)
        _shared = len(_self_attr_names) > 0

        _use_custom_key = key_fn is not None

        # --- Cache storage ---
        # When self_attrs is provided: closure-level (shared across all instances)
        # When not: per-instance (stored in self.__dict__)
        if _shared:
            _shared_graphs = {}
            _shared_cinputs = {}
            _shared_coutputs = {}  # stores actual result tensors (reused across replays)
            _ALL_SHARED_CACHES.append((_shared_graphs, _shared_cinputs, _shared_coutputs))

        # Per-instance attribute key names
        _g = f"_cg_{method.__name__}_g"
        _ci = f"_cg_{method.__name__}_ci"
        _co = f"_cg_{method.__name__}_co"

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
                # Include self_attrs shapes/dtypes in key
                for attr_name in _self_attr_names:
                    attr = getattr(self, attr_name, None)
                    if attr is not None and isinstance(attr, _Tensor):
                        _kp.append((attr_name, tuple(attr.shape), attr.dtype))
                    else:
                        _kp.append((attr_name, None))
                key = tuple(_kp)

            # === Get cache (shared or per-instance) ===
            if _shared:
                graphs = _shared_graphs
                cinputs = _shared_cinputs
                coutputs = _shared_coutputs
            else:
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
                # === First encounter: only capture during capture phase ===
                if not _BLOCK_WISE_CAPTURING:
                    # Not in capture phase -- fall back to eager
                    return method(self, *args, **kwargs)

                # === Capture ===
                graph = paddle.device.cuda.graphs.CUDAGraph(enable_replace=True)
                graphs[key] = graph
                ci = {}
                for name, aidx in _input_info:
                    v = kwargs[name] if name in kwargs else (args[aidx] if aidx < nargs else None)
                    if v is not None and isinstance(v, _Tensor):
                        ci[name] = v.data_ptr()

                # Record self_attrs pointers for cross-instance replacement
                for attr_name in _self_attr_names:
                    attr = getattr(self, attr_name, None)
                    if attr is not None and isinstance(attr, _Tensor):
                        ci[f"__attr_{attr_name}"] = attr.data_ptr()

                cinputs[key] = ci

                graph.capture_begin()
                result = method(self, *args, **kwargs)
                graph.capture_end()

                # --- Log which op just entered the CUDA graph ---
                _CAPTURED_GRAPH_LOG.append((_qualname, key, _shared))
                if fastdeploy.envs.FD_BLOCK_WISE_DEBUG:
                    _LOGGER.info(
                        f"[block_wise_cuda_graph] captured #{len(_CAPTURED_GRAPH_LOG)} "
                        f"op={_qualname} shared={_shared} key={key}"
                    )

                graph.replay()

                # Store the actual result for reuse. The graph always writes to
                # the same output memory, so we return the same tensors on replay.
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
                        old_ptrs.append(ci[name])
                        new_ptr = v.data_ptr()
                        new_ptrs.append(new_ptr)
                        ci[name] = new_ptr

                # Replace self_attrs pointers (e.g. weight)
                for attr_name in _self_attr_names:
                    attr_key = f"__attr_{attr_name}"
                    if attr_key in ci:
                        attr = getattr(self, attr_name, None)
                        if attr is not None:
                            old_ptrs.append(ci[attr_key])
                            new_ptr = attr.data_ptr()
                            new_ptrs.append(new_ptr)
                            ci[attr_key] = new_ptr

                if old_ptrs:
                    graphs[key].replace_input_ptrs(old_ptrs, new_ptrs)
                graphs[key].replay()

                # Reuse the output tensors from capture — graph wrote fresh
                # data to the same memory, no allocation needed.
                return coutputs[key]

        return wrapper

    return decorator
