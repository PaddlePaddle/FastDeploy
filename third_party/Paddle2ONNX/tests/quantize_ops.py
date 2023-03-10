import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from paddle import _C_ops, in_dynamic_mode, _legacy_C_ops


@paddle.jit.not_to_static
def quantize_linear(x,
                    scale,
                    zero_point,
                    bit_length=8,
                    quant_axis=-1,
                    name=None):
    helper = LayerHelper('quantize_linear', **locals())

    attrs = ('bit_length', bit_length, 'quant_axis', quant_axis)
    if in_dygraph_mode():
        return _legacy_C_ops.quantize_linear(x, scale, zero_point, *attrs)
    else:
        output = helper.create_variable_for_type_inference(dtype=x.dtype)

        inputs = {'X': x, 'Scale': scale, "ZeroPoint": zero_point}
        outputs = {'Y': output}

        helper.append_op(
            type="quantize_linear",
            inputs=inputs,
            attrs={'bit_length': bit_length,
                   'quant_axis': quant_axis},
            outputs=outputs)
        output.stop_gradient = True
        return output


@paddle.jit.not_to_static
def dequantize_linear(x,
                      scale,
                      zero_point,
                      bit_length=8,
                      quant_axis=-1,
                      name=None):
    helper = LayerHelper('dequantize_linear', **locals())

    attrs = ('bit_length', bit_length, 'quant_axis', quant_axis)
    if in_dygraph_mode():
        return _legacy_C_ops.dequantize_linear(x, scale, zero_point, *attrs)
    else:
        output = helper.create_variable_for_type_inference(dtype=x.dtype)

        inputs = {'X': x, 'Scale': scale, "ZeroPoint": zero_point}
        outputs = {'Y': output}

        helper.append_op(
            type="dequantize_linear",
            inputs=inputs,
            attrs={'bit_length': bit_length,
                   'quant_axis': quant_axis},
            outputs=outputs)
        output.stop_gradient = True
        return output
