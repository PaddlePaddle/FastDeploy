import inspect

import paddle
import paddlenlp_ops
from paddle.base import core


# npu interface refer to gpu interface
def fused_sparse_moe(
    input,
    gate_weight,
    ffn1_weight,
    ffn2_weight,
    ffn1_bias,
    ffn1_scale,
    ffn2_bias,
    ffn2_scale,
    quant_method,
    moe_topk,
    tp_size:int
):
    """
    call npu func to implement this function
    """
    ffn1_weight = paddle.cast(ffn1_weight, paddle.bfloat16)
    ffn2_weight = paddle.cast(ffn2_weight, paddle.bfloat16)


    gate_weight = gate_weight.transpose([1, 0]).astype(input.dtype)

    temp = paddle.zeros([1]).astype(input.dtype)

    expert_array = paddle.arange(moe_topk * input.shape[0]).astype("int32")
    expert_group = paddle.ones([1]).astype("int32")
    one_hot = paddle.ones([1]).astype("int32")
    zero_hot = paddle.zeros([1]).astype("int32")

    # define quant mapping: may modify
    if quant_method == "weight_int4_only":
        quanttype = 11
    elif quant_method == "weight_int8_only":
        quanttype = 6
    else:
        quanttype = 1
    y = paddlenlp_ops.sparse_moe(
        input,
        gate_weight,
        temp,
        temp,
        temp,
        temp,
        temp,
        ffn1_weight,
        ffn1_bias if ffn1_bias else temp,
        temp,
        temp,
        ffn1_scale,
        temp,
        ffn2_weight,
        ffn2_bias if ffn2_bias else temp,
        temp,
        temp,
        ffn2_scale,
        temp,
        expert_array,
        expert_group,
        one_hot,
        zero_hot,
        moe_topk,
        input.dtype == paddle.bfloat16,
        tp_size,  
        quanttype,
    )
    return y


