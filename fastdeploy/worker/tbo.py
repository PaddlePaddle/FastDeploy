"""
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
"""

import threading

import paddle

from fastdeploy.model_executor.forward_meta import ForwardMeta

event0 = threading.Event()
event1 = threading.Event()


GLOBAL_THREAD_INFO = {}

GLOBAL_THREAD_INFO["thread0"] = [event0, event1]
GLOBAL_THREAD_INFO["thread1"] = [event1, event0]


GLOBAL_ATTN_BUFFERS = {}


def let_another_thread_run():
    thread_name = threading.current_thread().name

    if thread_name in GLOBAL_THREAD_INFO:
        GLOBAL_THREAD_INFO[thread_name][1].set()
        GLOBAL_THREAD_INFO[thread_name][0].wait()
        GLOBAL_THREAD_INFO[thread_name][0].clear()


def is_last_thread():
    thread_name = threading.current_thread().name

    return thread_name == "thread1"


def creat_empty_forward_meta(forward_meta: ForwardMeta):

    res = ForwardMeta(
        ids_remove_padding=forward_meta.ids_remove_padding[0:0],
        rotary_embs=forward_meta.rotary_embs,
        attn_backend=forward_meta.attn_backend,
        caches=forward_meta.caches,
    )

    res.hidden_states = forward_meta.hidden_states[0:0]
    res.decode_states = forward_meta.decode_states[0:0]

    return res


def split_batch_decoder_layers(forward_meta: ForwardMeta, fd_config):
    split_num = 2
    res = [creat_empty_forward_meta(forward_meta), forward_meta]
    res[0].tbo_microbatch_id = 0
    res[1].tbo_microbatch_id = 1
    total_token_num = forward_meta.ids_remove_padding.shape[0]

    if total_token_num < 1024:
        return res

    chunk_token_num = (total_token_num + split_num - 1) // split_num

    split_sections = []
    for i in range(0, split_num):
        start_token_id = i * chunk_token_num
        end_token_id = start_token_id + chunk_token_num
        end_token_id = min(total_token_num, end_token_id)
        split_sections.append(end_token_id)

    # 由于多模的图片理解，需要将多模拟的token聚集在一起！
    # 所以需要将split_sections[0]适当的偏移一下！

    special_tokens = [
        fd_config.model_config.image_patch_id,
    ]

    ids_remove_padding_cpu = forward_meta.ids_remove_padding.numpy().tolist()
    detect_pos = split_sections[0]
    while ids_remove_padding_cpu[detect_pos] in special_tokens:
        detect_pos += 1
        if detect_pos >= len(ids_remove_padding_cpu):
            return res
    split_sections[0] = detect_pos

    for i in range(0, split_num):
        start_token_id = 0 if i == 0 else split_sections[i - 1]
        end_token_id = split_sections[i]

        res[i] = ForwardMeta(
            ids_remove_padding=None,
            rotary_embs=forward_meta.rotary_embs,
            attn_backend=forward_meta.attn_backend,
            caches=forward_meta.caches,
        )

        # 我们需要处理的这一段token位于[start_bs, end_bs)里面！
        start_bs = forward_meta.batch_id_per_token[start_token_id]
        end_bs = forward_meta.batch_id_per_token[end_token_id - 1]
        end_bs += 1

        if len(forward_meta.rotary_embs.shape) == 6:
            max_bs = forward_meta.rotary_embs.shape[0]
            assert max_bs == forward_meta.block_tables.shape[0]
            assert forward_meta.rotary_embs.shape[1:3] == [2, 1]
            assert forward_meta.rotary_embs.shape[4] == 1
            res[i].rotary_embs = forward_meta.rotary_embs[start_bs:end_bs]
        res[i].block_tables = forward_meta.block_tables[start_bs:end_bs]
        res[i].ids_remove_padding = forward_meta.ids_remove_padding[start_token_id:end_token_id]
        res[i].batch_id_per_token = forward_meta.batch_id_per_token[start_token_id:end_token_id] - start_bs

        # 下面这三个要好好弄，小心出错！
        # 我需要记录下  start_bs 他被left chunk 瓜分了多少了！
        # 我需要记录下  (end_bs-1) 他被 right chunk 瓜分了多少了！
        start_bs_s_token_by_left_chunk = start_token_id - forward_meta.cu_seqlens_q[start_bs].item()
        end_bs_s_token_by_right_chunk = forward_meta.cu_seqlens_q[end_bs].item() - end_token_id

        res[i].seq_lens_this_time = forward_meta.seq_lens_this_time[start_bs:end_bs] + 0
        res[i].seq_lens_this_time[0] -= start_bs_s_token_by_left_chunk
        res[i].seq_lens_this_time[-1] -= end_bs_s_token_by_right_chunk

        res[i].seq_lens_encoder = forward_meta.seq_lens_encoder[start_bs:end_bs] + 0
        if res[i].seq_lens_encoder[0].item() > 0:
            res[i].seq_lens_encoder[0] -= start_bs_s_token_by_left_chunk
        if res[i].seq_lens_encoder[-1].item() > 0:
            res[i].seq_lens_encoder[-1] -= end_bs_s_token_by_right_chunk

        res[i].seq_lens_decoder = forward_meta.seq_lens_decoder[start_bs:end_bs] + 0
        res[i].seq_lens_decoder[0] += start_bs_s_token_by_left_chunk

        cu_seqlens_q = [0] + paddle.cumsum(res[i].seq_lens_this_time).numpy().tolist()
        res[i].cu_seqlens_q = paddle.to_tensor(cu_seqlens_q).cast("int32")

        # res[i].cu_seqlens_k = res[i].cu_seqlens_q

        for key in GLOBAL_ATTN_BUFFERS[i]:
            setattr(res[i], key, GLOBAL_ATTN_BUFFERS[i][key])

        if forward_meta.attn_mask_offsets is not None:
            mask_num = forward_meta.attn_mask_offsets.shape[0]
            if mask_num == total_token_num * 2:
                res[i].attn_mask_offsets = forward_meta.attn_mask_offsets[start_token_id * 2 : end_token_id * 2]
            elif mask_num == total_token_num:
                res[i].attn_mask_offsets = forward_meta.attn_mask_offsets[start_token_id:end_token_id]
            else:
                assert False, "Invalid attn_mask_offsets shape"

        # This is adapt 5.0
        if hasattr(forward_meta, "hidden_states"):
            res[i].hidden_states = forward_meta.hidden_states[start_token_id:end_token_id]
            # 下面这个其实不需要，因为纯文不需要这个！
            res[i].decode_states = forward_meta.decode_states[start_bs:end_bs]

        res[i].tbo_microbatch_id = i
    return res
