# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random

import numpy as np
import paddle
from utils import init_inplace_tensor

# block_attn_fused is deprecated and should be removed in the future
from fastdeploy.model_executor.ops.xpu import (
    block_attn,
    block_attn_fused,
    get_infer_param,
)


def print_all_not_equal_elements_info(k, x, y):
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    index = paddle.nonzero(x_flatten != y_flatten)
    x_not_equal = x_flatten[index]
    y_not_equal = y_flatten[index]
    print(f"reference not equal element of {k}: {x_not_equal}")
    print(f"calculated result not equal element of {k}: {y_not_equal}")
    xy_diff = x - y
    xy_mean_diff = paddle.mean(xy_diff)
    xy_max_abs_diff = paddle.max(paddle.abs(xy_diff))
    xy_min_abs_diff = paddle.min(paddle.abs(xy_diff))
    print(f"{k} mean diff: {xy_mean_diff}, max abs diff: {xy_max_abs_diff}, min abs diff: {xy_min_abs_diff}")


def run_prefix_cache_block_attn(
    block_attn_func,
    qkv,
    seq_len,
    seq_lens_this_time,
    hit_prefix_len,
    key_cache,
    value_cache,
    rotary_embs,
    block_tables,
    attn_out,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale,
    v_dequant_scale,
    k_zp,
    v_zp,
    shift,
    smooth,
    q_norm_weight,
    k_norm_weight,
    kv_signal_data_cpu,
    cachekv_signal_thread_cpu,
    use_neox_rotary_style,
    rope_3d,
    num_speculative_tokens,
):
    if key_cache.dtype == paddle.int8:
        rtol = 1e-1
        atol = 1e-2
    else:
        rtol = 1e-2
        atol = 1e-3
    # prefix cache block attn
    seq_lens_encoder = paddle.to_tensor([seq_len - hit_prefix_len, 0, 0, 0, 0], dtype="int32")
    seq_lens_decoder = paddle.to_tensor([hit_prefix_len, 0, 0, 0, 0], dtype="int32")

    (
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        prefix_len,
        decoder_context_len,
        decoder_context_len_cache,
        prefix_block_tables,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        prefix_len_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        len_info_cpu,
    ) = init_inplace_tensor(seq_lens_encoder.shape[0], block_tables.shape)
    (
        slot_mapping_enc,
        slot_mapping_dec,
    ) = get_infer_param(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        block_tables,
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        prefix_len,
        decoder_context_len,
        decoder_context_len_cache,
        prefix_block_tables,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        prefix_len_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        len_info_cpu,
        64,
        num_speculative_tokens,
    )
    qkv_prefix = qkv[hit_prefix_len:]
    attn_out_prefix_cache = block_attn_func(
        qkv_prefix,
        key_cache,
        value_cache,
        rotary_embs,
        block_tables,
        prefix_block_tables,
        len_info_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        encoder_batch_map_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        decoder_batch_map_cpu,
        prefix_len_cpu,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        encoder_batch_map,
        decoder_context_len,
        decoder_context_len_cache,
        decoder_batch_map,
        prefix_len,
        slot_mapping_enc,
        slot_mapping_dec,
        k_quant_scale,
        v_quant_scale,
        k_dequant_scale,
        v_dequant_scale,
        k_zp,
        v_zp,
        shift,
        smooth,
        q_norm_weight,
        k_norm_weight,
        kv_signal_data_cpu,
        cachekv_signal_thread_cpu,
        use_neox_rotary_style,
        rope_3d,
    )
    attn_out_np = attn_out[hit_prefix_len:].astype("float32").numpy()
    attn_out_prefix_cache_np = attn_out_prefix_cache.astype("float32").numpy()
    is_passed = np.allclose(attn_out_np, attn_out_prefix_cache_np, rtol=rtol, atol=atol)
    if not is_passed:
        print(f"block_attn_func: {block_attn_func}")
        print("prefix_cache block_attn check failed!")
        print(f"origin block_attn_out: {attn_out[hit_prefix_len:]}")
        print(f"prefix_cache block_attn_out: {attn_out_prefix_cache}")
        print("not equal elements are listed below:")
        print_all_not_equal_elements_info("block_attn_out", attn_out[hit_prefix_len:], attn_out_prefix_cache)
    else:
        print(f"prefix_cache check of {block_attn_func} PASSED!")
    assert is_passed
    return attn_out_prefix_cache


def run_block_attn(
    seed,
    is_fused,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    mode,  # 1 for split kvcache encoder only, 2 for split kvcache decoder only, 3 for mixed
    hit_prefix_len,
    kvcache_dtype,
    has_zp,
    use_neox_rotary_style,
    rotary_embs_shape,
    num_speculative_tokens,
):
    assert mode == 0 or mode == 1, "mixed mode not supported yet!"
    if mode == 0:
        encoder_seq_len = seq_len
        decoder_seq_len = 0
    elif mode == 1:
        encoder_seq_len = 0
        decoder_seq_len = seq_len
    else:
        pass
    seq_lens_encoder = paddle.to_tensor([encoder_seq_len, 0, 0, 0, 0], dtype="int32")
    seq_lens_decoder = paddle.to_tensor([decoder_seq_len, 0, 0, 0, 0], dtype="int32")
    seq_lens_this_time = paddle.to_tensor([seq_len, 0, 0, 0, 0], dtype="int32")
    block_tables = paddle.arange(0, block_batch * max_block_per_seq, dtype="int32")
    block_tables = block_tables.reshape((block_batch, max_block_per_seq))

    (
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        prefix_len,
        decoder_context_len,
        decoder_context_len_cache,
        prefix_block_tables,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        prefix_len_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        len_info_cpu,
    ) = init_inplace_tensor(seq_lens_encoder.shape[0], block_tables.shape)
    (
        slot_mapping_enc,
        slot_mapping_dec,
    ) = get_infer_param(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        block_tables,
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        prefix_len,
        decoder_context_len,
        decoder_context_len_cache,
        prefix_block_tables,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        prefix_len_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        len_info_cpu,
        64,
        num_speculative_tokens,
    )
    qkv = paddle.uniform(
        shape=[seq_len, (head_num + 2 * kv_head_num) * head_dim], dtype="bfloat16", min=-1.0, max=1.0, seed=seed
    )

    rotary_embs = paddle.uniform(shape=rotary_embs_shape, dtype="float32", min=-1.0, max=1.0, seed=seed)
    key_cache = paddle.zeros(
        shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
        dtype=kvcache_dtype,
    )
    value_cache = paddle.zeros(
        shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
        dtype=kvcache_dtype,
    )

    scale_tensor_k = None
    scale_tensor_v = None
    k_quant_scale = None
    v_quant_scale = None
    k_dequant_scale = None
    v_dequant_scale = None
    k_zp = None
    v_zp = None
    if kvcache_dtype == "int8":
        scale_tensor_k = paddle.uniform(
            shape=[kv_head_num * head_dim], dtype="bfloat16", min=1.0, max=1.0, seed=seed
        )  # max
        scale_tensor_v = paddle.uniform(
            shape=[kv_head_num * head_dim], dtype="bfloat16", min=1.0, max=1.0, seed=seed
        )  # max
        k_quant_scale = 127.0 / scale_tensor_k  # for C8 per channel means 127 / max
        v_quant_scale = 127.0 / scale_tensor_v  # for C8 per channel means 127 / max
        if has_zp:
            k_dequant_scale = 1 / k_quant_scale  # for C8 per channel zp means max
            v_dequant_scale = 1 / v_quant_scale  # for C8 per channel zp means max
            k_zp = paddle.zeros(shape=[kv_head_num * head_dim], dtype="bfloat16")
            v_zp = paddle.zeros(shape=[kv_head_num * head_dim], dtype="bfloat16")
        else:
            k_dequant_scale = paddle.cast(scale_tensor_k, dtype="float32")  # for C8 per channel means max
            v_dequant_scale = paddle.cast(scale_tensor_v, dtype="float32")  # for C8 per channel means max
    # variable below are not yet used
    shift = None
    smooth = None
    q_norm_weight = None
    k_norm_weight = None
    kv_signal_data_cpu = None
    cachekv_signal_thread_cpu = None
    rope_3d = False

    if is_fused:
        block_attn_func = block_attn_fused
    else:
        block_attn_func = block_attn
    attn_out = block_attn_func(
        qkv,
        key_cache,
        value_cache,
        rotary_embs,
        block_tables,
        prefix_block_tables,
        len_info_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        encoder_batch_map_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        decoder_batch_map_cpu,
        prefix_len_cpu,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        encoder_batch_map,
        decoder_context_len,
        decoder_context_len_cache,
        decoder_batch_map,
        prefix_len,
        slot_mapping_enc,
        slot_mapping_dec,
        k_quant_scale,
        v_quant_scale,
        k_dequant_scale,
        v_dequant_scale,
        k_zp,
        v_zp,
        shift,
        smooth,
        q_norm_weight,
        k_norm_weight,
        kv_signal_data_cpu,
        cachekv_signal_thread_cpu,
        use_neox_rotary_style,
        rope_3d,
    )
    result = {
        "block_attn_out": attn_out,
        "key_cache": key_cache,
        "value_cache": value_cache,
    }

    # prefix cache
    if mode == 0 and hit_prefix_len > 0:
        assert hit_prefix_len < seq_len
        attn_out_prefix_cache = run_prefix_cache_block_attn(
            block_attn_func,
            qkv,
            seq_len,
            seq_lens_this_time,
            hit_prefix_len,
            key_cache,
            value_cache,
            rotary_embs,
            block_tables,
            attn_out,
            k_quant_scale,
            v_quant_scale,
            k_dequant_scale,
            v_dequant_scale,
            k_zp,
            v_zp,
            shift,
            smooth,
            q_norm_weight,
            k_norm_weight,
            kv_signal_data_cpu,
            cachekv_signal_thread_cpu,
            use_neox_rotary_style,
            rope_3d,
            num_speculative_tokens,
        )
        result["prefix_cache_block_attn_out"] = attn_out_prefix_cache
    return result


def run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    hit_prefix_len=0,
    kvcache_dtype="bfloat16",
    has_zp=False,
    use_neox_rotary_style=False,
    only_run_spliced=False,
    num_speculative_tokens=0,
):
    rtol = 1e-3
    atol = 1e-2
    # 0 for prefill only, 1 for decode only, 2 for mixed
    # TODO: mixed mode not supported yet, get_infer_param should be modified first
    mode_name = ["prefill only", "decode only", "mixed"]

    if use_neox_rotary_style:
        embedding_type = "neox"
    else:
        embedding_type = "rope"
    for mode in [0, 1]:
        if mode == 0:
            seq_len_list = [seq_len]
        elif mode == 1:
            # seq_len > 1 goes into mtp branch, which only supports seq_len <= 31
            # TODO: mtp mode need further adaption
            # seq_len_list = [1, random.randint(2, 31)]
            seq_len_list = [1]
        for idx, seqlen in enumerate(seq_len_list):
            if idx == 0:
                branch_name = "non mtp branch"
            elif idx == 1:
                branch_name = "mtp branch"
            print(
                f"runnning block attention of mode {mode_name[mode]} ({branch_name}), is_prefix_cache: {hit_prefix_len > 0}, kvcache type: {kvcache_dtype}, has_zp: {has_zp}, rotary_style: {embedding_type}"
            )
            if not only_run_spliced:
                fused_result = run_block_attn(
                    seed,
                    True,  # is_fused
                    head_num,
                    kv_head_num,
                    head_dim,
                    seqlen,
                    block_batch,
                    max_block_per_seq,
                    block_size,
                    mode,
                    hit_prefix_len,
                    kvcache_dtype,
                    has_zp,
                    use_neox_rotary_style,
                    rotary_embs_shape,
                    num_speculative_tokens,
                )
            spliced_result = run_block_attn(
                seed,
                False,  # is_fused
                head_num,
                kv_head_num,
                head_dim,
                seqlen,
                block_batch,
                max_block_per_seq,
                block_size,
                mode,
                hit_prefix_len,
                kvcache_dtype,
                has_zp,
                use_neox_rotary_style,
                rotary_embs_shape,
                num_speculative_tokens,
            )
            if "fused_result" in locals() and "spliced_result" in locals():
                for k in fused_result.keys():
                    if paddle.is_integer(fused_result[k]):
                        fused_v = fused_result[k].astype("int32")
                        spliced_v = spliced_result[k].astype("int32")
                        fused_v_np = fused_v.numpy()
                        splice_v_np = spliced_v.numpy()
                        # is_passed = np.allclose(fused_v_np, splice_v_np, rtol=1e-1, atol=1e-1)
                        is_passed = np.allclose(fused_v_np, splice_v_np, rtol=1e-2, atol=rtol)
                    else:
                        fused_v = fused_result[k].astype("float32")
                        spliced_v = spliced_result[k].astype("float32")
                        fused_v_np = fused_v.numpy()
                        splice_v_np = spliced_v.numpy()
                        is_passed = np.allclose(fused_v_np, splice_v_np, rtol=rtol, atol=atol, equal_nan=True)
                    if not is_passed:
                        print(f"{k} in mode {mode_name[mode]} check FAILED!")
                        print(f"fused {k}: {fused_v}")
                        print(f"spliced {k}: {spliced_v}")
                        print("not equal elements are listed below:")
                        print_all_not_equal_elements_info(k, fused_v, spliced_v)
                    else:
                        print(f"{k} in mode {mode_name[mode]} check PASSED!")
                    assert is_passed
                print("")
            else:
                if "fused_result" not in locals():
                    print("fused_result not found.")
                if "spliced_result" not in locals():
                    print("spliced_result not found.")
                print("skip comparison.")


seed = random.randint(0, 2026)
paddle.seed(seed)
head_num = 64
kv_head_num = 8
head_dim = 128
rotary_embs_shape = [2, 1, 8192, 1, head_dim]
seq_len = 128
block_batch = 5
max_block_per_seq = 128
block_size = 64
# TODO: if hit_prefix_len has a small value, e.g. hit_prefix_len == 2, block_attn_out and prefix_cache_block_attn_out will have greater diff
hit_prefix_len = 71

# no prefix cache
# block_attn fused vs spliced
use_neox_rotary_style = False
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    0,
    kvcache_dtype="bfloat16",
    has_zp=False,
    use_neox_rotary_style=use_neox_rotary_style,
)
# c8 quantization block_attn fused vs spliced
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    0,
    kvcache_dtype="int8",
    has_zp=False,
    use_neox_rotary_style=use_neox_rotary_style,
)
# c8 zp quantization block_attn fused vs spliced
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    0,
    kvcache_dtype="int8",
    has_zp=True,
    use_neox_rotary_style=use_neox_rotary_style,
)

# prefix cache
# block_attn fused vs spliced
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    hit_prefix_len,
    kvcache_dtype="bfloat16",
    has_zp=False,
    use_neox_rotary_style=use_neox_rotary_style,
)
# c8 quantization block_attn fused vs spliced
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    hit_prefix_len,
    kvcache_dtype="int8",
    has_zp=False,
    use_neox_rotary_style=use_neox_rotary_style,
)
# c8 zp quantization block_attn fused vs spliced
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    hit_prefix_len,
    kvcache_dtype="int8",
    has_zp=True,
    use_neox_rotary_style=use_neox_rotary_style,
)

# # neox
# # block_attn fused vs spliced
# # no prefix cache
use_neox_rotary_style = True
only_run_spliced = False
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    0,
    kvcache_dtype="bfloat16",
    has_zp=False,
    use_neox_rotary_style=use_neox_rotary_style,
    only_run_spliced=only_run_spliced,
)
# prefix cache
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    hit_prefix_len,
    kvcache_dtype="bfloat16",
    has_zp=False,
    use_neox_rotary_style=use_neox_rotary_style,
    only_run_spliced=only_run_spliced,
)

# neox glm 4.5 air debug
head_num = 24
kv_head_num = 2
head_dim = 128
seq_len = 128
block_batch = 64
max_block_per_seq = 2050
block_size = 64
rotary_embs_shape = [2, 1, 131072, 1, head_dim // 2]

use_neox_rotary_style = True
only_run_spliced = False
run_compare_block_attn(
    seed,
    head_num,
    kv_head_num,
    head_dim,
    seq_len,
    block_batch,
    max_block_per_seq,
    block_size,
    rotary_embs_shape,
    0,
    kvcache_dtype="bfloat16",
    has_zp=False,
    use_neox_rotary_style=use_neox_rotary_style,
    only_run_spliced=only_run_spliced,
)

print("\nALL PASSED!")
