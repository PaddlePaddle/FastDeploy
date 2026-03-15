import paddle


def convert_float32_uint8(tensor):
    assert tensor.dtype == paddle.float32
    # last_dim = tensor.shape[-1]
    return tensor.view("uint8")


def convert_bfloat16_uint8(tensor):
    assert tensor.dtype == paddle.bfloat16
    return tensor.view("uint8")


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


def dsk_attn_write_cache(
    compressed_kv,
    kv_pe,
    kv_cache,
    slot_mapping,
    scale,
    cache_quant_type_str,
):

    token_num = slot_mapping.shape[0]

    page_size = 64
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

    for token_id in range(token_num):
        dst_physical_pos = slot_mapping[token_id].item()
        dst_block = dst_physical_pos // page_size
        dst_offset = dst_physical_pos % page_size

        # write quant uint8
        baseline = zkk_quant_compressed_kv[token_id, :].contiguous().view(paddle.uint8)
        kv_cache[dst_block, 0, dst_offset, :512] = baseline

        baseline = zkk_scale_max[token_id, :]
        baseline = baseline + 0
        baseline = convert_float32_uint8(baseline).cast("uint8")
        kv_cache[dst_block, 0, dst_offset, 512:528] = baseline

        baseline = zkk_kv_pe[token_id, :]
        baseline = baseline + 0
        baseline = convert_bfloat16_uint8(baseline).contiguous()
        kv_cache[dst_block, 0, dst_offset, 528:656] = baseline
