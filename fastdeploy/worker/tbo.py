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


def split_batch_decoder_layers(forward_meta: ForwardMeta):
    split_num = 2
    real_bs = forward_meta.seq_lens_this_time.shape[0]

    res = [forward_meta] * split_num

    if real_bs < split_num or forward_meta.ids_remove_padding.shape[0] == 0:
        return res

    mc_bs = (real_bs + split_num - 1) // split_num

    for i in range(0, split_num):
        start_bs = i * mc_bs

        end_bs = start_bs + mc_bs
        end_bs = min(end_bs, real_bs)

        if start_bs >= end_bs:
            continue

        start_token_id = forward_meta.cu_seqlens_q[start_bs].item()
        end_token_id = forward_meta.cu_seqlens_q[end_bs].item()

        if start_token_id >= end_token_id:
            continue

        res[i] = ForwardMeta(
            ids_remove_padding=None,
            rotary_embs=forward_meta.rotary_embs,
            attn_backend=forward_meta.attn_backend,
            caches=forward_meta.caches,
        )

        res[i].rotary_embs = forward_meta.rotary_embs[start_bs:end_bs]

        res[i].ids_remove_padding = forward_meta.ids_remove_padding[start_token_id:end_token_id]
        res[i].batch_id_per_token = forward_meta.batch_id_per_token[start_token_id:end_token_id] - start_bs

        res[i].seq_lens_encoder = forward_meta.seq_lens_encoder[start_bs:end_bs]
        res[i].seq_lens_decoder = forward_meta.seq_lens_decoder[start_bs:end_bs]
        res[i].seq_lens_this_time = forward_meta.seq_lens_this_time[start_bs:end_bs]

        res[i].block_tables = forward_meta.block_tables[start_bs:end_bs]

        res[i].cu_seqlens_q = forward_meta.cu_seqlens_q[start_bs : end_bs + 1] - start_token_id
        res[i].cu_seqlens_k = forward_meta.cu_seqlens_k[start_bs : end_bs + 1] - start_token_id

        for key in GLOBAL_ATTN_BUFFERS[i]:
            setattr(res[i], key, GLOBAL_ATTN_BUFFERS[i][key])

        if forward_meta.attn_mask_offsets is not None:
            mask_num = forward_meta.attn_mask_offsets.shape[0]
            token_num = forward_meta.ids_remove_padding.shape[0]
            if mask_num == token_num * 2:
                res[i].attn_mask_offsets = forward_meta.attn_mask_offsets[start_token_id * 2 : end_token_id * 2]
            elif mask_num == token_num:
                res[i].attn_mask_offsets = forward_meta.attn_mask_offsets[start_token_id:end_token_id]
            else:
                assert False, "Invalid attn_mask_offsets shape"

        # This is to  adapt 5
        if hasattr(forward_meta, "hidden_states"):
            res[i].hidden_states = forward_meta.hidden_states[start_token_id:end_token_id]
            res[i].decode_states = forward_meta.decode_states[start_bs:end_bs]

    return res
