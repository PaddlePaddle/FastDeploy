import paddle

try:
    from fastdeploy.model_executor.ops.iluvatar import wi4a16_weight_quantize_cuda
except:
    wi4a16_weight_quantize_cuda = None


def _get_weight_by_group_size(w, group_size):
    assert w.dim() == 2
    assert group_size in (-1, 32, 64, 128)
    if group_size == -1:
        quant_weight = w
    else:
        assert w.shape[-1] % group_size == 0
        quant_weight = w.reshape(-1, group_size)
    assert paddle.isnan(quant_weight).sum() == 0
    return quant_weight


def _pack_int4_to_int8(weight):
    return ((weight[:, 1::2] & 0xF) << 4) | (weight[:, 0::2] & 0xF)


def wi4a16_weight_quantize(w, group_size=128):
    """Quantize [k, n] weight to packed int4, scales, zeros (MoE wi4a16)."""
    k, n = w.shape
    assert k % group_size == 0 and n % 2 == 0
    if wi4a16_weight_quantize_cuda is not None:
        return wi4a16_weight_quantize_cuda(w.contiguous(), group_size)
    else:
        # [k, n] -> [n, k]
        w = w.T.contiguous()
        quant_weight = _get_weight_by_group_size(w, group_size)

        wmax = quant_weight.abs().max(axis=1, keepdim=True)
        scales = wmax / 7
        out = paddle.round(quant_weight.to(paddle.float32) / scales).clamp(-8, 7).to(paddle.int8)

        out = _pack_int4_to_int8(
            # NOTE: conver to numpy since paddle cannot support &
            out.view(w.shape[0], -1)
            .T.contiguous()
            .cpu()
            .numpy(),
        )
        out = paddle.from_numpy(out).T.contiguous()

        scales = scales.view(w.shape[0], -1).T.contiguous()
        zeros = paddle.zeros_like(scales)
        return out, scales, zeros
