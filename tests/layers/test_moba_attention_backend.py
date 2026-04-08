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

import unittest
from unittest.mock import patch

import numpy as np
import paddle

from fastdeploy.model_executor.layers.attention.moba_attention_backend import (
    PlasAttentionBackend,
    PlasAttentionMetadata,
)


class DummyFDConfig:
    def __init__(self):
        self.cache_config = type("CacheConfig", (), {"block_size": 4})()
        self.model_config = type("ModelConfig", (), {"max_model_len": 16, "head_dim": 8, "num_hidden_layers": 2})()
        self.scheduler_config = type("SchedulerConfig", (), {"max_num_seqs": 2})()
        self.plas_attention_config = type(
            "PlasConfig",
            (),
            {
                "plas_block_size": 4,
                "plas_encoder_top_k_left": 1,
                "plas_encoder_top_k_right": 1,
                "plas_use_encoder_seq_limit": 1,
                "plas_decoder_top_k_left": 1,
                "plas_decoder_top_k_right": 1,
                "plas_use_decoder_seq_limit": 1,
                "plas_max_seq_length": 32,
            },
        )()
        self.graph_opt_config = type("GraphOptConfig", (), {"cudagraph_capture_sizes": None})()
        self.parallel_config = type("ParallelConfig", (), {"block_size": 4})()


class DummyForwardMeta:
    def __init__(self, enc_seq=[4, 4], dec_seq=[2, 2]):
        self.seq_lens_encoder = paddle.to_tensor(enc_seq, dtype="int64")
        self.seq_lens_decoder = paddle.to_tensor(dec_seq, dtype="int64")
        self.seq_lens_this_time = sum(dec_seq)
        self.cu_seqlens_q = paddle.to_tensor([0] + list(np.cumsum(dec_seq)), dtype="int64")
        self.caches = [paddle.zeros([2, 4, 8])] * 4
        self.block_tables = None
        self.rotary_embs = None


class DummyLayer:
    def __init__(self, layer_id=0, cache_quant_type_str=None, plas_use_mlp=True):
        self.layer_id = layer_id
        self.qkv_bias = None
        self.cache_k_block_means = None
        self.cache_quant_type_str = cache_quant_type_str
        self.plas_use_mlp = plas_use_mlp


class TestPlasAttentionBackend(unittest.TestCase):
    @patch(
        "fastdeploy.model_executor.layers.attention.moba_attention_backend.get_cur_cu_seq_len_k",
        return_value=(paddle.to_tensor([1, 2]), paddle.to_tensor([1, 2]), paddle.to_tensor([2])),
    )
    def test_init_attention_metadata(self, mock_get_cu_seq):
        # Test initialization of attention metadata
        fd_config = DummyFDConfig()
        backend = PlasAttentionBackend(fd_config, kv_num_heads=2, num_heads=2, head_dim=8)
        forward_meta = DummyForwardMeta()
        backend.init_attention_metadata(forward_meta)

        self.assertIsInstance(backend.attention_metadata, PlasAttentionMetadata)
        self.assertTrue(backend.attention_metadata.q_input.shape[0] > 0)

    @patch(
        "fastdeploy.model_executor.layers.attention.moba_attention_backend.get_cur_cu_seq_len_k",
        return_value=(
            paddle.to_tensor([0]),  # cu_seq_q_pack
            paddle.to_tensor([0]),  # cu_seqlens_k
            paddle.to_tensor([0]),  # q_pack_tokens
        ),
    )
    def test_init_attention_metadata_empty_seq(self, mock_get_cu_seq):
        # Test metadata init with empty sequences
        fd_config = DummyFDConfig()
        backend = PlasAttentionBackend(fd_config, kv_num_heads=2, num_heads=2, head_dim=8)
        forward_meta = DummyForwardMeta()
        forward_meta.seq_lens_encoder = paddle.to_tensor([0])
        forward_meta.seq_lens_decoder = paddle.to_tensor([0])
        forward_meta.cu_seqlens_q = paddle.to_tensor([0, 0])
        backend.init_attention_metadata(forward_meta)

    def test_get_kv_cache_shape(self):
        # Test KV cache shape calculation under different quant types
        fd_config = DummyFDConfig()
        backend = PlasAttentionBackend(fd_config, kv_num_heads=2, num_heads=2, head_dim=8)

        # Default
        key_shape, value_shape = backend.get_kv_cache_shape(max_num_blocks=2)
        self.assertEqual(key_shape, [2, 2, 4, 8])

        # int4_zp quant
        key_shape_int4, value_shape_int4 = backend.get_kv_cache_shape(max_num_blocks=2, kv_cache_quant_type="int4_zp")
        self.assertEqual(key_shape_int4, [2, 2, 4, 4])

        # Other quant types
        key_shape_other, value_shape_other = backend.get_kv_cache_shape(max_num_blocks=2, kv_cache_quant_type="int8")
        self.assertEqual(key_shape_other, [2, 2, 4, 8])

    @patch(
        "fastdeploy.model_executor.layers.attention.moba_attention_backend.moba_attention",
        return_value=(paddle.ones([4, 4]), None),
    )
    @patch(
        "fastdeploy.model_executor.layers.attention.moba_attention_backend.get_cur_cu_seq_len_k",
        return_value=(paddle.to_tensor([1, 2]), paddle.to_tensor([1, 2]), paddle.to_tensor([2])),
    )
    def test_forward_mixed(self, mock_get_cu_seq, mock_moba_attention):
        # Test mixed forward path with various layer configurations
        fd_config = DummyFDConfig()
        backend = PlasAttentionBackend(fd_config, kv_num_heads=2, num_heads=2, head_dim=8)
        forward_meta = DummyForwardMeta()
        backend.init_attention_metadata(forward_meta)

        # Complete layer attributes
        layer = DummyLayer()
        qkv = paddle.zeros([4, 4])
        compressed_kv = paddle.zeros([4, 4])
        k_pe = paddle.zeros([4, 4])

        out = backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=qkv,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            layer=layer,
            forward_meta=forward_meta,
        )
        self.assertTrue((out.numpy() == 1).all())

        # Layer with missing attributes, no cache quant
        layer_missing = DummyLayer(layer_id=1, cache_quant_type_str=None)
        out2 = backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=qkv,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            layer=layer_missing,
            forward_meta=forward_meta,
        )
        self.assertTrue((out2.numpy() == 1).all())

        # Layer with int4_zp cache quant
        layer_int4 = DummyLayer(layer_id=1, cache_quant_type_str="int4_zp")
        out3 = backend.forward_mixed(
            q=None,
            k=None,
            v=None,
            qkv=qkv,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            layer=layer_int4,
            forward_meta=forward_meta,
        )
        self.assertTrue((out3.numpy() == 1).all())


if __name__ == "__main__":
    unittest.main()
