import paddle


def init_inplace_tensor(bsz, block_tables_shape):
    encoder_batch_map = paddle.empty(bsz, dtype="int32")
    decoder_batch_map = paddle.empty(bsz, dtype="int32")
    encoder_batch_idx = paddle.empty(bsz, dtype="int32")
    decoder_batch_idx = paddle.empty(bsz, dtype="int32")
    encoder_seq_lod = paddle.empty(bsz + 1, dtype="int32")
    decoder_seq_lod = paddle.empty(bsz + 1, dtype="int32")
    encoder_kv_lod = paddle.empty(bsz + 1, dtype="int32")
    prefix_len = paddle.empty(bsz, dtype="int32")
    decoder_context_len = paddle.empty(bsz, dtype="int32")
    decoder_context_len_cache = paddle.empty(bsz, dtype="int32")

    prefix_block_tables = paddle.empty(block_tables_shape, dtype="int32")

    encoder_batch_map_cpu = paddle.empty(bsz, dtype="int32", device="cpu")
    decoder_batch_map_cpu = paddle.empty(bsz, dtype="int32", device="cpu")
    encoder_batch_idx_cpu = paddle.empty(bsz, dtype="int32", device="cpu")
    decoder_batch_idx_cpu = paddle.empty(bsz, dtype="int32", device="cpu")
    encoder_seq_lod_cpu = paddle.empty(bsz + 1, dtype="int32", device="cpu")
    decoder_seq_lod_cpu = paddle.empty(bsz + 1, dtype="int32", device="cpu")
    encoder_kv_lod_cpu = paddle.empty(bsz + 1, dtype="int32", device="cpu")
    prefix_len_cpu = paddle.empty(bsz, dtype="int32", device="cpu")
    decoder_context_len_cpu = paddle.empty(bsz, dtype="int32", device="cpu")
    decoder_context_len_cache_cpu = paddle.empty(bsz, dtype="int32", device="cpu")

    len_info_cpu = paddle.empty(7, dtype="int32", device="cpu")

    return (
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
    )
