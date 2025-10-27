import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.ops.gpu import update_attn_mask_offsets


def run_update_attn_mask_offsets_case(
    seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, is_block_step, max_model_len=8, decode_states_len=4
):
    bsz = len(seq_lens_this_time)

    # cu_seqlens_q: 累积和
    cu_seqlens_q = np.zeros(bsz, dtype="int32")
    cu_seqlens_q[1:] = np.cumsum(seq_lens_this_time[:-1])
    cu_seqlens_q = paddle.to_tensor(cu_seqlens_q, dtype="int32")
    print("cu_seqlens_q", cu_seqlens_q)
    # ids_remove_padding 只是用来确定 batch_seq_lens
    ids_remove_padding = paddle.randint(low=0, high=10, shape=[sum(seq_lens_this_time)], dtype="int32")

    # attention_mask: (bsz, max_model_len)
    attention_mask = paddle.arange(bsz * max_model_len, dtype="int32").reshape([bsz, max_model_len])

    # 每个 batch 一个 decoder offset
    attention_mask_decoder = paddle.zeros([bsz], dtype="int32")

    attention_mask_decoder[:] = paddle.to_tensor(seq_lens_decoder, dtype="int32")

    # decode_states: (bsz, decode_states_len)
    decode_states = paddle.full([bsz, decode_states_len], -1, dtype="int32")

    mask_rollback = paddle.full([bsz, 1], 0, dtype="int32")

    # 调用 op
    attn_mask_offsets = update_attn_mask_offsets(
        ids_remove_padding,
        paddle.to_tensor(seq_lens_this_time, dtype="int32"),
        paddle.to_tensor(seq_lens_encoder, dtype="int32"),
        paddle.to_tensor(seq_lens_decoder, dtype="int32"),
        cu_seqlens_q,
        attention_mask,
        attention_mask_decoder,
        paddle.to_tensor(is_block_step, dtype="bool"),
        decode_states,
        mask_rollback,
    )
    if isinstance(attn_mask_offsets, list):
        attn_mask_offsets = attn_mask_offsets[0]
    return attn_mask_offsets.numpy(), decode_states.numpy()


def test_stop_case():
    attn_mask_offsets, _ = run_update_attn_mask_offsets_case(
        seq_lens_this_time=[2],
        seq_lens_encoder=[0],
        seq_lens_decoder=[0],
        is_block_step=[False],
    )
    # stop 场景不应更新
    assert np.all(attn_mask_offsets == 0) or np.allclose(attn_mask_offsets, 0)


def test_prefill_case():
    attn_mask_offsets, _ = run_update_attn_mask_offsets_case(
        seq_lens_this_time=[5],
        seq_lens_encoder=[5],
        seq_lens_decoder=[0],
        is_block_step=[False],
    )
    # 应该拷贝了 attention_mask 的一部分，不全是 0
    assert np.allclose(attn_mask_offsets, np.arange(0, 5))


def test_decoder_case():
    attn_mask_offsets, decode_states_out = run_update_attn_mask_offsets_case(
        seq_lens_this_time=[3],
        seq_lens_encoder=[0],
        seq_lens_decoder=[2],
        is_block_step=[False],
    )
    # decoder 场景 attn_mask_offsets 应该有非零更新
    assert np.allclose(attn_mask_offsets, np.array([2, 3, 4]))
    # decode_states 前面部分应该被重置为 0
    assert np.any(decode_states_out == 0)


def test_non_block_step_case():
    attn_mask_offsets, _ = run_update_attn_mask_offsets_case(
        seq_lens_this_time=[0, 2],
        seq_lens_encoder=[0, 0],
        seq_lens_decoder=[0, 20],
        is_block_step=[True, False],
    )
    # 进入 block step，Query 1 不应该被写入
    assert np.allclose(attn_mask_offsets, np.array([20, 21]))


def test_mixed_batch_case():
    attn_mask_offsets, decode_states_out = run_update_attn_mask_offsets_case(
        seq_lens_this_time=[2, 5, 1],
        seq_lens_encoder=[0, 5, 0],
        seq_lens_decoder=[2, 0, 2],
        is_block_step=[False, False, False],
    )
    # batch 混合场景，至少部分更新
    assert attn_mask_offsets.shape[0] == sum([2, 5, 1])
    assert np.allclose(attn_mask_offsets, np.array([2, 3, 8, 9, 10, 11, 12, 2]))
    assert decode_states_out.shape[1] == 4


if __name__ == "__main__":
    pytest.main([__file__])
