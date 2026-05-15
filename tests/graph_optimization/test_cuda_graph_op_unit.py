# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
In-process unit tests for
``fastdeploy.model_executor.graph_optimization.cuda_graph_op``.

These tests exercise helper functions and the non-capturing branches of the
``block_wise_cuda_graph_wrap`` decorator to improve coverage. They intentionally
avoid invoking the actual CUDA Graph capture path so that they can run without
spinning up a serving process or recording on a CUDA stream.
"""

import paddle
import pytest

import fastdeploy
from fastdeploy.model_executor.graph_optimization import cuda_graph_op


class _ToyLayer(paddle.nn.Layer):
    """Minimal layer used to exercise the wrap decorator paths."""

    def __init__(self):
        super().__init__()
        self.weight = paddle.ones([2, 2], dtype="float32")


def _set_env(monkeypatch, **kwargs):
    """Helper to override FD env vars dynamically via monkeypatch."""
    for key, value in kwargs.items():
        monkeypatch.setattr(fastdeploy.envs, key, value, raising=False)


def test_get_captured_graph_log_returns_copy(monkeypatch):
    """get_captured_graph_log should return a list copy of the registry."""
    monkeypatch.setattr(cuda_graph_op, "_CAPTURED_GRAPH_LOG", [("op", ("k",), False)])
    log = cuda_graph_op.get_captured_graph_log()
    assert log == [("op", ("k",), False)]
    log.append(("other", (), True))
    assert cuda_graph_op._CAPTURED_GRAPH_LOG == [("op", ("k",), False)]


def test_dump_captured_graph_summary_paths(monkeypatch):
    """Cover early-returns and the summary-print branch of dump_captured_graph_summary."""
    # Debug disabled: early return.
    _set_env(monkeypatch, FD_BLOCK_WISE_DEBUG=False)
    monkeypatch.setattr(cuda_graph_op, "_CAPTURED_GRAPH_LOG", [("op", (), False)])
    cuda_graph_op.dump_captured_graph_summary()

    # Debug enabled, empty log.
    _set_env(monkeypatch, FD_BLOCK_WISE_DEBUG=True)
    monkeypatch.setattr(cuda_graph_op, "_CAPTURED_GRAPH_LOG", [])
    cuda_graph_op.dump_captured_graph_summary()

    # Debug enabled, non-empty log.
    monkeypatch.setattr(
        cuda_graph_op,
        "_CAPTURED_GRAPH_LOG",
        [("a.fwd", ("k1",), True), ("a.fwd", ("k2",), True), ("b.fwd", ("k1",), False)],
    )
    cuda_graph_op.dump_captured_graph_summary()


def test_clear_all_block_wise_graphs():
    """clear_all_block_wise_graphs should empty every registered shared cache."""
    g, ci, co = {"k": object()}, {"k": object()}, {"k": object()}
    snapshot = list(cuda_graph_op._ALL_SHARED_CACHES)
    try:
        cuda_graph_op._ALL_SHARED_CACHES.clear()
        cuda_graph_op._ALL_SHARED_CACHES.append((g, ci, co))
        cuda_graph_op.clear_all_block_wise_graphs()
        assert g == {} and ci == {} and co == {}
    finally:
        cuda_graph_op._ALL_SHARED_CACHES.clear()
        cuda_graph_op._ALL_SHARED_CACHES.extend(snapshot)


def test_block_wise_wrap_invalid_input_raises():
    """Decorator should raise ValueError when 'inputs' name is not a parameter."""
    with pytest.raises(ValueError):

        @cuda_graph_op.block_wise_cuda_graph_wrap(inputs=["nonexistent"])
        def forward(self, x):
            return x


def test_block_wise_wrap_disabled_passthrough(monkeypatch):
    """When FD_USE_BLOCK_WISE_CUDA_GRAPH is off, wrapper should call eager."""
    _set_env(monkeypatch, FD_USE_BLOCK_WISE_CUDA_GRAPH=False)

    class M(_ToyLayer):
        @cuda_graph_op.block_wise_cuda_graph_wrap(inputs=["x"])
        def forward(self, x, residual=None):
            return x + 1

    m = M()
    x = paddle.zeros([2, 2], dtype="float32")
    out = m.forward(x)
    assert paddle.all(out == 1).item()


def test_block_wise_wrap_zero_shape_skips(monkeypatch):
    """A zero-dim tensor input (positional or keyword) should bypass capture."""
    _set_env(monkeypatch, FD_USE_BLOCK_WISE_CUDA_GRAPH=True)
    cuda_graph_op.set_block_wise_capturing(False)

    class M(_ToyLayer):
        @cuda_graph_op.block_wise_cuda_graph_wrap(inputs=["x"])
        def forward(self, x, residual=None):
            return x

    m = M()
    empty = paddle.zeros([0, 4], dtype="float32")
    # Positional zero-shape arg: hits the `for a in args` branch.
    assert m.forward(empty).shape == [0, 4]
    # Keyword zero-shape arg: hits the `for v in kwargs.values()` branch.
    assert m.forward(paddle.ones([2, 2]), residual=empty).shape == [2, 2]


def test_block_wise_wrap_per_instance_cache_eager_fallback(monkeypatch):
    """Per-instance cache init + eager fallback when not in capture phase."""
    _set_env(monkeypatch, FD_USE_BLOCK_WISE_CUDA_GRAPH=True)
    cuda_graph_op.set_block_wise_capturing(False)

    class M(_ToyLayer):
        @cuda_graph_op.block_wise_cuda_graph_wrap(inputs=["x"])
        def forward(self, x, hook=None):
            return x * 2

    m = M()
    x = paddle.ones([2, 2], dtype="float32")
    # First call: initializes per-instance cache dicts on self.__dict__.
    out1 = m.forward(x, hook=lambda v: v)  # callable arg covers callable-key branch
    # Second call reuses the existing per-instance cache (try/except hit path).
    out2 = m.forward(x)
    assert paddle.all(out1 == 2).item()
    assert paddle.all(out2 == 2).item()
    # The decorator should have stashed cache attributes on the instance.
    assert any(name.startswith("_cg_forward_") for name in m.__dict__)


def test_block_wise_wrap_custom_key_fn(monkeypatch):
    """Custom key_fn path is used to compute the cache key."""
    _set_env(monkeypatch, FD_USE_BLOCK_WISE_CUDA_GRAPH=True)
    cuda_graph_op.set_block_wise_capturing(False)

    seen_keys = []

    def key_fn(x, residual):
        k = ("custom", tuple(x.shape) if x is not None else None)
        seen_keys.append(k)
        return k

    class M(_ToyLayer):
        @cuda_graph_op.block_wise_cuda_graph_wrap(inputs=["x"], key_fn=key_fn)
        def forward(self, x, residual=None):
            return x

    m = M()
    m.forward(paddle.ones([3, 3], dtype="float32"))
    assert seen_keys and seen_keys[0][0] == "custom"


def test_set_block_wise_capturing_toggle():
    """set_block_wise_capturing should mutate the module-level flag."""
    cuda_graph_op.set_block_wise_capturing(True)
    assert cuda_graph_op._BLOCK_WISE_CAPTURING is True
    cuda_graph_op.set_block_wise_capturing(False)
    assert cuda_graph_op._BLOCK_WISE_CAPTURING is False
