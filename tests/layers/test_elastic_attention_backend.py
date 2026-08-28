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
"""Layer-level tests for ``Qwen3ElasticAttention`` and its backend.

These tests are intentionally **construction-time / contract-level**:
end-to-end ``forward_mixed`` requires a fully initialised distributed env
(``init_distributed_environment``), KV-cache buffers and a built BSA op.
That heavy path is covered by the integration smoke
``models/qwen3_elastic/run_elastic_qwen3_4b.py``. Here we guard the
documented invariants that are easy to break and hard to notice:

1. Elastic config knobs are mirrored onto ``self.attn`` so the backend can
   read them (§4.1 "key trick").
2. ``block_size`` is read from ``pretrained_config``, NOT from
   ``model_config``, so a leaking ``cache_config.block_size = 64`` cannot
   silently halve sink_blocks / local_blocks.
3. The router decision caches (``_z_kv_cache`` / ``_head_mask_type_cache``)
   exist with the right shape / dtype.
4. ``Qwen3ElasticAttentionBackend`` is the class that ``PawQwen3ForCausalLM``
   advertises via ``_get_attn_backend_cls``.
"""

import unittest
from unittest import mock

import paddle

# Like test_elastic_qwen3_patches, this test cannot side-step the parent
# fastdeploy package init: ``Qwen3ElasticAttention`` constructs FD's real
# ``Attention`` layer (mocked here, but the import path must still resolve)
# and the elastic backend module imports FD's attention base classes. So
# the same fully-built fastdeploy_ops requirement applies.
try:
    import fastdeploy.model_executor.models.qwen3_elastic  # noqa: F401
    from fastdeploy.model_executor.models.qwen3_elastic.modeling_elastic_qwen3 import (  # noqa: F401
        Qwen3ElasticAttention,
    )

    _FD_FULLY_BUILT = True
    _FD_IMPORT_ERR = None
except Exception as _e:  # noqa: BLE001
    _FD_FULLY_BUILT = False
    _FD_IMPORT_ERR = _e


class _Cfg:
    """Bag-of-attrs stand-in for FDConfig sub-configs."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _make_minimal_fd_config():
    """Build the smallest FDConfig-like object that ``Qwen3ElasticAttention``
    needs at __init__ time (we never run forward in this test)."""
    pc = _Cfg(block_size=128)  # elastic granularity (the §4.1 trick)
    mc = _Cfg(
        head_dim=64,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        # elastic fields populated by populate_elastic_fields with defaults
        pretrained_config=pc,
    )
    parallel = _Cfg(tensor_parallel_size=1)
    return _Cfg(model_config=mc, parallel_config=parallel)


@unittest.skipUnless(
    _FD_FULLY_BUILT,
    f"fastdeploy custom-ops not fully built (got: {_FD_IMPORT_ERR!r})",
)
@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda(),
    "Qwen3ElasticAttention pulls in FD layers that require a CUDA build.",
)
class TestQwen3ElasticAttentionConstruction(unittest.TestCase):
    """Layer __init__ contract; no forward."""

    def setUp(self):
        # We only need the construction path. Patch the heavy children with
        # ``MagicMock`` so we don't depend on a fully-initialised distributed
        # env or KV cache pool.
        self._patches = []
        for path in (
            "fastdeploy.model_executor.models.qwen3_elastic.modeling_elastic_qwen3.QKVParallelLinear",
            "fastdeploy.model_executor.models.qwen3_elastic.modeling_elastic_qwen3.RowParallelLinear",
            "fastdeploy.model_executor.models.qwen3_elastic.modeling_elastic_qwen3.QKRMSNorm",
            "fastdeploy.model_executor.models.qwen3_elastic.modeling_elastic_qwen3.Attention",
        ):
            p = mock.patch(path)
            self._patches.append(p)
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_elastic_attrs_mirrored_onto_self_attn(self):
        fd = _make_minimal_fd_config()
        layer = Qwen3ElasticAttention(fd_config=fd, layer_id=0, prefix="model.layers.0.self_attn")

        # The backend's ``forward_mixed`` receives ``self.attn`` (NOT this
        # parent), so every elastic knob must be reachable from there.
        for name in (
            "mask_allocator",
            "toggle_type",
            "retrieval_mode",
            "enable_ada_sparsity",
            "pooling_mode",
            "block_size",
            "sink_blocks",
            "local_blocks",
            "xattn_stride",
            "xattn_threshold",
            "xattn_norm",
            "_z_kv_cache",
            "_head_mask_type_cache",
        ):
            self.assertTrue(
                hasattr(layer.attn, name),
                msg=f"layer.attn is missing mirrored attr {name!r}",
            )
            self.assertEqual(getattr(layer.attn, name), getattr(layer, name), msg=f"layer.attn.{name} != layer.{name}")

    def test_block_size_comes_from_pretrained_config(self):
        """Regression test for §4.1: ``cache_config.block_size`` (e.g. 64)
        leaking onto model_config must NOT win over ``pretrained_config.block_size``.
        """
        fd = _make_minimal_fd_config()
        # Simulate cache_config.block_size leaking onto model_config.
        fd.model_config.block_size = 64
        # ckpt elastic block_size (the source of truth) is 128.
        fd.model_config.pretrained_config.block_size = 128

        layer = Qwen3ElasticAttention(fd_config=fd, layer_id=0, prefix="x")
        self.assertEqual(layer.block_size, 128, "block_size MUST be read from pretrained_config")
        # Derived counters use elastic block_size (128), not 64.
        self.assertEqual(layer.sink_blocks, (layer.sink_size + 128 - 1) // 128)
        self.assertEqual(layer.local_blocks, (layer.local_window_size + 128 - 1) // 128)

    def test_router_cache_shapes(self):
        fd = _make_minimal_fd_config()
        layer = Qwen3ElasticAttention(fd_config=fd, layer_id=0, prefix="x")
        self.assertEqual(list(layer._z_kv_cache.shape), [layer.num_kv_heads_local])
        self.assertEqual(layer._z_kv_cache.dtype, paddle.int32)
        self.assertEqual(list(layer._head_mask_type_cache.shape), [layer.num_heads_local])
        self.assertEqual(layer._head_mask_type_cache.dtype, paddle.int32)

    def test_mask_allocator_is_router(self):
        from fastdeploy.model_executor.models.qwen3_elastic.utils import AttentionRouter

        fd = _make_minimal_fd_config()
        layer = Qwen3ElasticAttention(fd_config=fd, layer_id=0, prefix="x")
        self.assertIsInstance(layer.mask_allocator, AttentionRouter)
        self.assertEqual(layer.mask_allocator.num_kv_heads, layer.num_kv_heads_local)
        self.assertEqual(layer.mask_allocator.d_feature, layer.head_dim)


@unittest.skipUnless(
    _FD_FULLY_BUILT,
    f"fastdeploy custom-ops not fully built (got: {_FD_IMPORT_ERR!r})",
)
@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda(),
    "Qwen3ElasticAttentionBackend imports CUDA-only modules.",
)
class TestModelDeclaresElasticBackend(unittest.TestCase):
    def test_paw_qwen3_backend_class(self):
        from fastdeploy.model_executor.layers.attention.elastic_attn_backend import (
            Qwen3ElasticAttentionBackend,
        )
        from fastdeploy.model_executor.models.qwen3_elastic.modeling_elastic_qwen3 import (
            PawQwen3ForCausalLM,
        )

        # The model class advertises the elastic backend via the public hook
        # that FastDeploy's selector consults.
        self.assertIs(
            PawQwen3ForCausalLM._get_attn_backend_cls(),
            Qwen3ElasticAttentionBackend,
        )


if __name__ == "__main__":
    unittest.main()
