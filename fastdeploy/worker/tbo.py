import threading

from fastdeploy.model_executor.forward_meta import ForwardMeta

event0 = threading.Event()
event1 = threading.Event()

GLOBAL_THREAD_INFO = {}

GLOBAL_THREAD_INFO["thread0"] = [event0, event1]
GLOBAL_THREAD_INFO["thread1"] = [event1, event0]


GLOBAL_ATTN_BUFFERS = {}

from mm_custom_ops import calculate_decode_states_token_num
def split_batch(forward_meta: ForwardMeta, inputs):
    split_num = 2
    real_bs = forward_meta.seq_lens_this_time.shape[0]

    if forward_meta.ids_remove_padding.shape[0] > 0:
        print(real_bs)

    res = [forward_meta] * split_num

    inputs_res = [inputs] * split_num

    if real_bs < split_num:
        return res, inputs_res

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
            res[i].attn_mask_offsets = forward_meta.attn_mask_offsets[start_token_id*2:end_token_id*2]
        
        inputs_res[i] = {}
        inputs_res[i]["ids_remove_padding"] = inputs["ids_remove_padding"][start_token_id:end_token_id]
        inputs_res[i]["decode_states"] = inputs["decode_states"][start_bs:end_bs]

        inputs_res[i]["image_features"] = None
        inputs_res[i]["video_features"] = None
        inputs_res[i]["audio_features"] = None
        inputs_res[i]["image_grid_thws"] = None
        inputs_res[i]["video_grid_thws"] = None

        out = calculate_decode_states_token_num(
            inputs_res[i]["decode_states"], 
            res[i].seq_lens_this_time
        )
        text_token_num, audio_token_num, vision_token_num = out[0].item(), out[1].item(), out[2].item()

        assert audio_token_num == 0
        assert vision_token_num == 0

        res[i].text_token_num = text_token_num
        res[i].audio_token_num = 0
        res[i].vision_token_num = 0
        res[i].num_speculative_token = 0
        res[i].image_scale_idx = forward_meta.image_scale_idx[start_bs:end_bs]

    return res, inputs_res


def let_another_thread_run():
    thread_name = threading.current_thread().name

    if thread_name in GLOBAL_THREAD_INFO:
        GLOBAL_THREAD_INFO[thread_name][1].set()
        GLOBAL_THREAD_INFO[thread_name][0].wait()
        GLOBAL_THREAD_INFO[thread_name][0].clear()


def split_batch_eb5_layers(forward_meta: ForwardMeta):
    split_num = 2
    real_bs = forward_meta.seq_lens_this_time.shape[0]

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
            res[i].attn_mask_offsets = forward_meta.attn_mask_offsets[start_token_id*2:end_token_id*2]

        res[i].hidden_states = forward_meta.hidden_states[start_token_id:end_token_id]
        res[i].decode_states = forward_meta.decode_states[start_bs:end_bs]

    return res
