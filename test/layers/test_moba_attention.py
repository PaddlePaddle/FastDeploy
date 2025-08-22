# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

try:
    from fastdeploy.model_executor.ops.gpu import get_cur_cu_seq_len_k, moba_attention
except:
    moba_attention = None
    get_cur_cu_seq_len_k = None
import os

from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.moba_attention_backend import (
    MobaAttentionBackend,
)
from fastdeploy.platforms import _Backend, current_platform


class ModelConfig:
    def __init__(self):
        self.num_hidden_layers = 12
        self.head_dim = 128
        self.num_attention_heads = 8
        self.num_key_value_heads = 1


class ParallelConfig:
    def __init__(self):
        self.block_size = 128
        self.max_model_len = 128 * 1024
        self.max_num_seqs = 1
        self.tensor_parallel_size = 1


class ForwardMode:
    def __init__(self):
        pass


class FDConfig:
    def __init__(self):
        self.parallel_config = ParallelConfig()
        self.model_config = ModelConfig()
        self.quant_config = {}


def test_moba_attention(seq_len, num_heads, num_kv_heads, head_dim):
    max_seq_len = int(128 * 1024)
    moba_encoder_top_k_left = int(5)
    moba_encoder_top_k_right = int(10)
    moba_use_encoder_seq_limit = int(20 * 128)
    moba_decoder_top_k_left = int(20)
    moba_decoder_top_k_right = int(20)
    moba_use_decoder_seq_limit = int(20 * 128)
    os.environ["FD_ATTENTION_BACKEND"] = "MOBA_ATTN"
    os.environ["FD_MOBA_MLP_WEIGHT_PATH"] = "None"
    os.environ["FD_MOBA_ENCODER_TOP_K_LEFT"] = str(moba_encoder_top_k_left)
    os.environ["FD_MOBA_ENCODER_TOP_K_RIGHT"] = str(moba_encoder_top_k_right)
    os.environ["FD_MOBA_DECODER_TOP_K_LEFT"] = str(moba_decoder_top_k_left)
    os.environ["FD_MOBA_DECODER_TOP_K_RIGHT"] = str(moba_decoder_top_k_right)
    os.environ["FD_MOBA_USE_ENCODER_SEQ_LIMIT"] = str(moba_use_encoder_seq_limit)
    os.environ["FD_MOBA_USE_DECODER_SEQ_LIMIT"] = str(moba_use_decoder_seq_limit)
    os.environ["FD_MOBA_BLOCK_SIZE"] = str(128)
    os.environ["FD_MOBA_MAX_SEQ_LENGTH"] = str(max_seq_len)

    max_dec_len_this_time = int(0)
    qkv = paddle.randn([1, 4 * seq_len, num_heads + 2 * num_kv_heads, head_dim], dtype="bfloat16")
    q_input = qkv[:, :, :num_heads, :].reshape([-1, num_heads, head_dim])
    k_input = qkv[:, :, num_heads : num_heads + num_kv_heads, :].reshape([-1, num_kv_heads, head_dim])
    v_input = qkv[:, :, num_heads + num_kv_heads :, :].reshape([-1, num_kv_heads, head_dim])
    cu_seqlens_q = paddle.to_tensor([0, seq_len], dtype="int32")

    seq_lens_encoder = paddle.to_tensor([seq_len], dtype="int32")

    seq_lens_decoder = paddle.to_tensor([0], dtype="int32")

    cachesk = paddle.zeros([(seq_len + 63) // 64 * 256, num_kv_heads, 64, head_dim], dtype="bfloat16")
    cachesv = paddle.zeros([(seq_len + 63) // 64 * 256, num_kv_heads, 64, head_dim], dtype="bfloat16")

    block_tables = paddle.arange((seq_len + 63) // 64).astype("int32")

    rotary_embs = paddle.ones([seq_len, head_dim], dtype="float32")

    cache_k_block_means = paddle.zeros([(seq_len + 63) // 64 + 10, num_kv_heads, 64, head_dim], dtype="bfloat16")

    fd_config = FDConfig()
    forward_meta = ForwardMode()
    forward_meta.seq_lens_encoder = seq_lens_encoder
    forward_meta.seq_lens_decoder = seq_lens_decoder
    forward_meta.seq_lens_this_time = seq_lens_encoder
    forward_meta.cu_seqlens_q = cu_seqlens_q
    moba_attention_backend = MobaAttentionBackend(fd_config, 8, 1, 128)
    moba_attention_backend.init_attention_metadata(forward_meta)
    moba_attention_backend.get_kv_cache_shape(100)

    if moba_attention is None:
        return
    if get_cur_cu_seq_len_k is None:
        return

    cu_seq_q_pack, cu_seqlens_k, q_pack_tokens = get_cur_cu_seq_len_k(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_encoder,
        int(128),
    )

    out = moba_attention(
        qkv,
        q_input,
        k_input,
        v_input,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seq_q_pack,
        q_pack_tokens,
        seq_lens_encoder,
        seq_lens_encoder,
        cachesk,
        cachesv,
        block_tables,
        rotary_embs,
        cache_k_block_means,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        int(num_heads),
        int(num_kv_heads),
        int(head_dim),
        max_seq_len,
        int(seq_len),
        max_dec_len_this_time,
        moba_encoder_top_k_left,
        moba_encoder_top_k_right,
        moba_use_encoder_seq_limit,
        moba_decoder_top_k_left,
        moba_decoder_top_k_right,
        moba_use_decoder_seq_limit,
        False,
        "none",
    )[0]

    attention = Attention(fd_config, 0)

    selected_backend = _Backend.__members__.get(os.environ["FD_ATTENTION_BACKEND"])
    attention_cls = current_platform.get_attention_backend_cls(selected_backend)

    return out, attention, attention_cls


if __name__ == "__main__":
    if paddle.is_compiled_with_cuda():
        seq_len = int(2 * 1024)
        num_heads = int(8)
        num_kv_heads = int(1)
        head_dim = int(128)
        test_moba_attention(seq_len, num_heads, num_kv_heads, head_dim)
