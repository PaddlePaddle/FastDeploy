import paddle


def convert_float32_uint8(tensor):
    assert tensor.dtype == paddle.float32
    last_dim = tensor.shape[-1]
    return tensor.view("uint8")

    res = paddle.randn(tensor.shape[:-1] + [last_dim * 4], dtype="float32").cast("uint8")

    res0 = tensor.view(paddle.uint32).numpy()
    res1 = tensor.view(paddle.uint32).numpy()
    res2 = tensor.view(paddle.uint32).numpy()
    res3 = tensor.view(paddle.uint32).numpy()

    res0 = res0 & 0xF
    res1 = (res1 >> 8) & 0xF
    res2 = (res2 >> 16) & 0xF
    res3 = (res3 >> 24) & 0xF

    res0 = paddle.to_tensor(res0).cast(paddle.uint8)
    res1 = paddle.to_tensor(res1).cast(paddle.uint8)
    res2 = paddle.to_tensor(res2).cast(paddle.uint8)
    res3 = paddle.to_tensor(res3).cast(paddle.uint8)

    res[:, 0::4] = res0
    res[:, 1::4] = res1
    res[:, 2::4] = res2
    res[:, 3::4] = res2
    return res


def convert_bfloat16_uint8(tensor):
    assert tensor.dtype == paddle.bfloat16
    last_dim = tensor.shape[-1]

    return tensor.view("uint8")

    output_shape = tensor.shape[:-1] + [last_dim * 2]
    res = paddle.randn(output_shape, dtype="float32").cast("uint8")

    res0 = tensor.view(paddle.uint16).numpy()
    res1 = tensor.view(paddle.uint16).numpy()

    res0 = (res0 & 0x0F).astype("uint8")
    res1 = ((res1 >> 8) & 0x0F).astype("uint8")
    res0 = paddle.to_tensor(res0)
    res1 = paddle.to_tensor(res1)

    res[:, 0::2] = res0
    res[:, 1::2] = res1

    return res

    res0 = res0.unsqueeze(-1)
    res1 = res1.unsqueeze(-1)
    res = paddle.concat([res0, res1], axis=-1)

    res = res.reshape(output_shape)
    return res


def convert_uint8_float32(tensor):
    assert tensor.dtype == paddle.uint8
    last_dim = tensor.shape[-1]
    assert last_dim % 4 == 0

    tmp0 = tensor[:, 0::4].contiguous().numpy().astype("int32")
    tmp1 = tensor[:, 1::4].contiguous().numpy().astype("int32")
    tmp2 = tensor[:, 2::4].contiguous().numpy().astype("int32")
    tmp3 = tensor[:, 3::4].contiguous().numpy().astype("int32")
    tmp = (tmp3 << 24) | (tmp2 << 16) | (tmp1 << 8) | tmp0
    tmp = paddle.to_tensor(tmp).view(paddle.float32)

    return tmp


def convert_uint8_bfloat16(tensor):
    assert tensor.dtype == paddle.uint8
    last_dim = tensor.shape[-1]
    assert last_dim % 2 == 0

    tmp0 = tensor[:, 0::2].contiguous().numpy().astype("uint16")
    tmp1 = tensor[:, 1::2].contiguous().numpy().astype("uint16")
    tmp = (tmp1 << 8) | tmp0
    tmp = paddle.to_tensor(tmp).view(paddle.bfloat16)

    return tmp


def dsk_attn_write_cache_prefill(
    compressed_kv,
    kv_pe,
    kv_cache,
    slot_mapping,
    seq_lens_encoder,
    seq_lens_decoder,
    batch_id_per_token,
    cu_seqlens_q,
    block_tables,
    kv_signal_data,
    scale,
    cache_quant_type_str,
    max_seq_len,
    is_prefill,
):

    page_size = 64
    real_bs = cu_seqlens_q.shape[0] - 1
    assert real_bs == 1, "now only support bs == 1"
    token_num = compressed_kv.shape[0]
    assert compressed_kv.shape == [token_num, 512]
    assert kv_pe.shape == [token_num, 1, 64]
    zkk_kv_pe = kv_pe.reshape([token_num, 64])

    assert len(kv_cache.shape) == 4
    assert kv_cache.shape[1:] == [1, page_size, 656]

    assert kv_cache.dtype == paddle.uint8
    compressed_kv = compressed_kv.cast("float32").reshape([token_num, 4, 128])
    zkk_scale_max = compressed_kv.abs().max(axis=-1) + 0.00001
    assert zkk_scale_max.shape == [token_num, 4]
    zkk_quant_compressed_kv = compressed_kv / zkk_scale_max[:, :, None] * 448
    zkk_quant_compressed_kv = zkk_quant_compressed_kv.cast(paddle.float8_e4m3fn)
    zkk_quant_compressed_kv.reshape_([0, -1])

    zkk_scale_max = zkk_scale_max / 448.0
    for bs_id in range(real_bs):
        token_num_this_bs = seq_lens_encoder[bs_id].item()
        num_blocks = (token_num_this_bs + page_size - 1) // page_size
        for j in range(num_blocks):
            blockidx = block_tables[bs_id, j].item()

            want_token_num = min(page_size, token_num_this_bs - j * page_size)

            baseline = (
                zkk_quant_compressed_kv[j * page_size : j * page_size + want_token_num, :]
                .contiguous()
                .view(paddle.uint8)
            )
            kv_cache[blockidx, 0, :want_token_num, :512] = baseline

            baseline = zkk_scale_max[j * page_size : j * page_size + want_token_num, :]
            baseline = convert_float32_uint8(baseline)
            kv_cache[blockidx, 0, :want_token_num, 512:528] = baseline

            baseline = zkk_kv_pe[j * page_size : j * page_size + want_token_num, :]
            baseline = convert_bfloat16_uint8(baseline)
            kv_cache[blockidx, 0, :want_token_num, 528:656] = baseline
