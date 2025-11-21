import threading

from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.attention.append_attn_backend import (
    allocate_launch_related_buffer,
)

event0 = threading.Event()
event1 = threading.Event()

GLOBAL_THREAD_INFO = {}

GLOBAL_THREAD_INFO["thread0"] = [event0, event1]
GLOBAL_THREAD_INFO["thread1"] = [event1, event0]


def split_batch(forward_meta: ForwardMeta, tmp_dict):
    split_num = 2
    real_bs = forward_meta.seq_lens_this_time.shape[0]

    if forward_meta.ids_remove_padding.shape[0] > 0:
        print(real_bs)

    res = [forward_meta] * split_num

    if real_bs < split_num:
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

        res[i].ids_remove_padding = forward_meta.ids_remove_padding[start_token_id:end_token_id]
        res[i].batch_id_per_token = forward_meta.batch_id_per_token[start_token_id:end_token_id] - start_bs

        res[i].seq_lens_encoder = forward_meta.seq_lens_encoder[start_bs:end_bs]
        res[i].seq_lens_decoder = forward_meta.seq_lens_decoder[start_bs:end_bs]
        res[i].seq_lens_this_time = forward_meta.seq_lens_this_time[start_bs:end_bs]

        res[i].block_tables = forward_meta.block_tables[start_bs:end_bs]

        res[i].cu_seqlens_q = forward_meta.cu_seqlens_q[start_bs : end_bs + 1] - start_token_id
        res[i].cu_seqlens_k = forward_meta.cu_seqlens_k[start_bs : end_bs + 1] - start_token_id

        attention_buffer = allocate_launch_related_buffer(**tmp_dict)
        for key in attention_buffer:
            setattr(res[i], key, attention_buffer[key])

    return res
