from paddle.base import core
import paddle

def fused_linear_op(
        x,
        weight,
        weight_scale,
):  

    if weight_scale.dtype != x.dtype:
        weight_scale = paddle.cast(weight_scale, x.dtype)

    out = core.eager._run_custom_op("weight_only_linear_npu", x, weight, weight_scale)

    return out[0]