"""Paddle port of ``elasticattn.src.utils.find_blocks_chunked``.

Selects which key blocks each query block needs to attend to, based on a
threshold on cumulative attention mass.  Inference path: ``mode='prefill'``,
``decoding=False``, ``causal=True``.
"""

from __future__ import annotations

import paddle


def find_blocks_chunked(
    input_tensor: paddle.Tensor,
    current_index: int,
    threshold,
    num_to_choose,
    decoding: bool,
    mode: str = "both",
    causal: bool = True,
) -> paddle.Tensor:
    """Threshold-cumulative block selector.

    Args:
        input_tensor: [B, H, Qchunk, Kblocks] attention sums per block.
        current_index: index of the first query block w.r.t. K.
        threshold: float in (0,1] -- min cumulative mass to keep.
        num_to_choose: alternative to threshold (unsupported here).
        decoding: True if running decode path.
        mode: 'both' / 'prefill' / 'decode'.
        causal: apply causal block mask.

    Returns:
        bool tensor [B, H, Qchunk, Kblocks].
    """
    assert threshold is None or num_to_choose is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape

    if mode == "prefill" and decoding:
        return paddle.ones_like(input_tensor, dtype="bool")
    if mode == "decode" and not decoding:
        mask = paddle.ones_like(input_tensor, dtype="bool")
        return mask

    input_tensor = input_tensor.astype("float32")

    if threshold is None:
        raise NotImplementedError("block num chunk prefill not implemented")

    total_sum = input_tensor.sum(axis=-1, keepdim=True)
    if isinstance(threshold, paddle.Tensor):
        thr = threshold.astype("float32")
        required_sum = total_sum * thr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            [batch_size, head_num, chunk_num, 1]
        )
    else:
        required_sum = total_sum * float(threshold)

    if causal:
        mask = paddle.zeros_like(input_tensor, dtype="bool")
        # Always keep block 0 (sink) and the diagonal block.
        mask[:, :, :, 0] = True
        eye = paddle.eye(chunk_num, dtype="int32").astype("bool").unsqueeze(0).unsqueeze(0)
        eye = eye.expand([batch_size, head_num, chunk_num, chunk_num])
        # set mask[:, :, :, current_index : current_index + chunk_num] = eye
        diag_slice = paddle.zeros_like(input_tensor, dtype="bool")
        diag_slice[:, :, :, current_index : current_index + chunk_num] = eye
        mask = mask | diag_slice

        # zero-out mass that's already covered by `mask`, then sort the rest.
        other_values = paddle.where(mask, paddle.zeros_like(input_tensor), input_tensor)
        sorted_values = paddle.sort(other_values, axis=-1, descending=True)

        # Prepend a column of zeros and the mass already retained.
        retained_mass = paddle.where(mask, input_tensor, paddle.zeros_like(input_tensor)).sum(
            axis=-1, keepdim=True
        )
        zeros_col = paddle.zeros(
            [batch_size, head_num, chunk_num, 1], dtype="float32"
        )
        sorted_values = paddle.concat(
            [zeros_col, retained_mass, sorted_values[:, :, :, :-2]], axis=-1
        )

        # Argsort indices: force already-selected entries to the front.
        boosted = paddle.where(mask, 100000.0 * (1.0 + input_tensor), input_tensor)
        index = paddle.argsort(boosted, axis=-1, descending=True)

        cumulative_sum = paddle.concat(
            [zeros_col, sorted_values[:, :, :, :-1]], axis=-1
        ).cumsum(axis=-1)
        index_mask = cumulative_sum < required_sum
        # zero out indices we don't keep -> default to block 0 (already True)
        index = paddle.where(index_mask, index, paddle.zeros_like(index))

        # Scatter: mask[b,h,q,index[b,h,q,:]] = True.
        # NOTE: paddle GPU put_along_axis has no bool kernel; do the scatter
        # in int32 then cast back.
        flat_mask = mask.reshape([batch_size, head_num * chunk_num, block_num]).astype("int32")
        flat_idx = index.reshape([batch_size, head_num * chunk_num, block_num])
        true_vals = paddle.ones_like(flat_idx, dtype="int32")
        flat_mask = paddle.put_along_axis(
            flat_mask, flat_idx, true_vals, axis=-1, reduce="assign"
        )
        mask = flat_mask.reshape([batch_size, head_num, chunk_num, block_num]).astype("bool")
    else:
        mask = paddle.zeros_like(input_tensor, dtype="bool")
        sorted_values = paddle.sort(input_tensor, axis=-1, descending=True)
        index = paddle.argsort(input_tensor, axis=-1, descending=True)
        zeros_col = paddle.zeros(
            [batch_size, head_num, chunk_num, 1], dtype="float32"
        )
        cumulative_sum = paddle.concat(
            [zeros_col, sorted_values[:, :, :, :-1]], axis=-1
        ).cumsum(axis=-1)
        index_mask = cumulative_sum < required_sum
        index = paddle.where(index_mask, index, paddle.zeros_like(index))

        flat_mask = mask.reshape([batch_size, head_num * chunk_num, block_num]).astype("int32")
        flat_idx = index.reshape([batch_size, head_num * chunk_num, block_num])
        true_vals = paddle.ones_like(flat_idx, dtype="int32")
        flat_mask = paddle.put_along_axis(
            flat_mask, flat_idx, true_vals, axis=-1, reduce="assign"
        )
        mask = flat_mask.reshape([batch_size, head_num, chunk_num, block_num]).astype("bool")

    if causal:
        # any out-of-causal entries set to False
        if current_index + chunk_num < block_num:
            zero_pad = paddle.zeros(
                [batch_size, head_num, chunk_num, block_num - (current_index + chunk_num)],
                dtype="bool",
            )
            mask = paddle.concat(
                [mask[:, :, :, : current_index + chunk_num], zero_pad], axis=-1
            )
    return mask
