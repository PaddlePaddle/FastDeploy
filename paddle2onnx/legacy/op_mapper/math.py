#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import numpy as np
from paddle2onnx.legacy.constant import dtypes
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.legacy.op_mapper import mapper_helper
import paddle


@op_mapper('matmul')
class MatMul():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', idx=0)
        y = node.input('Y', idx=0)
        if node.attr('transpose_X'):
            perm = list(range(len(node.input_shape('X', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node.input_dtype('X', 0) == paddle.float64:
                x = graph.make_node('Cast', inputs=x, to=dtypes.ONNX.FLOAT)
            x = graph.make_node('Transpose', inputs=[x], perm=perm)
        if node.attr('transpose_Y'):
            perm = list(range(len(node.input_shape('Y', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node.input_dtype('Y', 0) == paddle.float64:
                y = graph.make_node('Cast', inputs=y, to=dtypes.ONNX.FLOAT)
            y = graph.make_node('Transpose', inputs=[y], perm=perm)
        if node.attr('alpha') == 1.0:
            if node.input_dtype('X', 0) == paddle.float64:
                output_node = graph.make_node('MatMul', inputs=[x, y])
                graph.make_node(
                    'Cast',
                    inputs=output_node,
                    to=dtypes.ONNX.DOUBLE,
                    outputs=node.output('Out'))
            else:
                graph.make_node(
                    'MatMul', inputs=[x, y], outputs=node.output('Out'))
        else:
            if node.input_dtype('X', 0) == paddle.float64:
                output_node = graph.make_node('MatMul', inputs=[x, y])
                matmul = graph.make_node(
                    'Cast', inputs=output_node, to=dtypes.ONNX.DOUBLE)
                scale = graph.make_node(
                    'Constant',
                    dtype=dtypes.ONNX.DOUBLE,
                    value=node.attr('alpha'))
            else:
                matmul = graph.make_node('MatMul', inputs=[x, y])
                scale = graph.make_node(
                    'Constant',
                    dtype=dtypes.ONNX.FLOAT,
                    value=node.attr('alpha'))

            onnx_node = graph.make_node(
                'Mul', inputs=[matmul, scale], outputs=node.output('Out'))


@op_mapper('matmul_v2')
class MatMul():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', idx=0)
        y = node.input('Y', idx=0)
        out = node.output('Out')
        ## TODO(wangjunjie06): The current addition of cast op is only for onnxruntime optimization, after onnxruntime is repaired, remove this logic
        if node.attr('trans_x'):
            perm = list(range(len(node.input_shape('X', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node.input_dtype('X', 0) == paddle.float64:
                x = graph.make_node('Cast', inputs=x, to=dtypes.ONNX.FLOAT)
            x = graph.make_node('Transpose', inputs=[x], perm=perm)
        if node.attr('trans_y'):
            perm = list(range(len(node.input_shape('Y', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node.input_dtype('Y', 0) == paddle.float64:
                y = graph.make_node('Cast', inputs=y, to=dtypes.ONNX.FLOAT)
            y = graph.make_node('Transpose', inputs=[y], perm=perm)
        if node.input_dtype('X', 0) == paddle.float64:
            output_node = graph.make_node('MatMul', inputs=[x, y])
            graph.make_node(
                'Cast', inputs=output_node, to=dtypes.ONNX.DOUBLE, outputs=out)
        else:
            graph.make_node('MatMul', inputs=[x, y], outputs=out)


@op_mapper('exp')
class Exp():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Exp', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('abs')
class Abs:
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Abs', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('erf')
class Erf():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        x_dtype = node.input_dtype('X', 0)
        x = node.input('X', 0)
        # onnxruntime only support float32 Erf
        if x_dtype != paddle.float32:
            x = graph.make_node('Cast', inputs=x, to=dtypes.ONNX.FLOAT)
            erf_node = graph.make_node('Erf', inputs=[x])
            graph.make_node(
                'Cast',
                inputs=[erf_node],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                outputs=node.output('Out'))
        else:
            graph.make_node('Erf', inputs=[x], outputs=node.output('Out'))


@op_mapper('acos')
class Acos():
    supports_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Acos', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('asin')
class Asin():
    supports_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Asin', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('sinh')
class Sinh():
    supports_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'Sinh', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('sin')
class Sin():
    supports_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Sin', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('atan')
class Atan():
    supports_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Atan', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('tan')
class Tan():
    supports_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Tan', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('ceil')
class Ceil():
    supports_opset_version_range = (7, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        graph.make_node(
            'Ceil', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('cos')
class Cos():
    supports_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Cos', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('cosh')
class Cosh():
    supports_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'Cosh', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('log2')
class Log2():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        _ln2 = 0.693147180559945309
        dtype = dtypes.ONNX.FLOAT
        if node.input_dtype('X', 0) == paddle.float64:
            dtype = dtypes.ONNX.DOUBLE
        _ln2 = graph.make_node('Constant', dtype=dtype, value=_ln2)
        lnx = graph.make_node('Log', inputs=node.input('X'))
        graph.make_node('Div', inputs=[lnx, _ln2], outputs=node.output('Out'))


@op_mapper('logsumexp')
class LogSumExp():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):

        if node.attr('reduce_all'):
            if not node.attr('keepdim'):
                reduce_node = graph.make_node(
                    'ReduceLogSumExp',
                    inputs=node.input('X'),
                    keepdims=node.attr('keepdim'))
                mapper_helper.unsqueeze_helper(graph, reduce_node, [0],
                                               node.output('Out'))
            else:
                graph.make_node(
                    'ReduceLogSumExp',
                    inputs=node.input('X'),
                    keepdims=node.attr('keepdim'),
                    outputs=node.output('Out'))
        else:
            graph.make_node(
                'ReduceLogSumExp',
                inputs=node.input('X'),
                keepdims=node.attr('keepdim'),
                axes=node.attr('axis'),
                outputs=node.output('Out'))


@op_mapper(
    [
        'elementwise_add', 'elementwise_sub', 'elementwise_div',
        'elementwise_mul', 'elementwise_min', 'elementwise_max',
        'elementwise_pow'
    ],
    mapper_dict={
        'elementwise_add': 'Add',
        'elementwise_sub': 'Sub',
        'elementwise_div': 'Div',
        'elementwise_mul': 'Mul',
        'elementwise_min': 'Min',
        'elementwise_max': 'Max',
        'elementwise_pow': 'Pow'
    })
class ElementwiseOps():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        if node.type in ["elementwise_min", "elementwise_max"]:
            assert False, "{} op is not supported when opset version < 8".format(
                node.type)

        op_type = kw['mapper_dict'][node.type]
        axis = node.attr('axis')
        x = node.input('X', 0)
        y = node.input('Y', 0)

        if axis == -1 or axis == (len(x_shape) - 1
                                  ) or len(x_shape) == len(y_shape):
            onnx_node = graph.make_node(
                op_type, inputs=[x, y], outputs=node.output('Out'))
        else:
            broadcast_shape = [1] * len(x_shape)
            broadcast_shape[axis:axis + len(y_shape)] = y_shape
            broadcast_shape_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=list(broadcast_shape))
            y_node = graph.make_node(
                'Reshape', inputs=[y, broadcast_shape_node])
            onnx_node = graph.make_node(
                op_type, inputs=[x, y_node], outputs=node.output('Out'))

    @classmethod
    def opset_8(cls, graph, node, **kw):
        op_type = kw['mapper_dict'][node.type]
        x = node.input('X', 0)
        y = node.input('Y', 0)
        axis = node.attr('axis')
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        if axis == -1 or axis == (len(x_shape) - 1
                                  ) or len(x_shape) == len(y_shape):
            onnx_node = graph.make_node(
                op_type, inputs=[x, y], outputs=node.output('Out'))
        else:
            broadcast_shape = [1] * len(x_shape)
            broadcast_shape[axis:axis + len(y_shape)] = y_shape
            broadcast_shape_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=list(broadcast_shape))
            y_node = graph.make_node(
                'Reshape', inputs=[y, broadcast_shape_node])
            onnx_node = graph.make_node(
                op_type, inputs=[x, y_node], outputs=node.output('Out'))


@op_mapper('elementwise_mod')
class ElementWiseMod():
    support_opset_version_range = (10, 15)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        axis = node.attr('axis')
        x = node.input('X', 0)
        y = node.input('Y', 0)

        if node.input_dtype('Y', 0) == paddle.int32 or node.input_dtype(
                'Y', 0) == paddle.int64:
            onnx_node = graph.make_node(
                "Mod", inputs=[x, y], outputs=node.output('Out'))
            return

        fmod = 1

        abs_x_node = graph.make_node("Abs", inputs=[x])
        abs_y_node = graph.make_node("Abs", inputs=[y])

        dtype = dtypes.ONNX.FLOAT
        val_0 = [0.0]
        val_1 = [-1.0]
        if node.input_dtype('Y', 0) == paddle.float64:
            dtype = dtypes.ONNX.DOUBLE
        if node.input_dtype('Y', 0) == paddle.int32:
            dtype = dtypes.ONNX.INT32
            val_0 = [0]
            val_1 = [-1]
        if node.input_dtype('Y', 0) == paddle.int64:
            dtype = dtypes.ONNX.INT64
            val_0 = [0]
            val_1 = [-1]
        zero_node = graph.make_node('Constant', dtype=dtype, value=val_0)
        one_node = graph.make_node('Constant', dtype=dtype, value=val_1)

        mod_node = graph.make_node(
            "Mod", inputs=[abs_x_node, abs_y_node], fmod=fmod)

        minus_node = graph.make_node("Mul", inputs=[mod_node, one_node])

        condition_dtype = graph.make_node("Less", inputs=[x, zero_node])
        condition = graph.make_node(
            'Cast', inputs=[condition_dtype], to=dtypes.ONNX.BOOL)

        mod_res = graph.make_node(
            "Where", inputs=[condition, minus_node, mod_node])

        add_node = graph.make_node("Add", inputs=[mod_res, y])

        mod_y_mul_node = graph.make_node("Mul", inputs=[mod_res, y])
        condition_dtype_1 = graph.make_node(
            "Less", inputs=[mod_y_mul_node, zero_node])
        condition_1 = graph.make_node(
            'Cast', inputs=[condition_dtype_1], to=dtypes.ONNX.BOOL)

        graph.make_node(
            "Where",
            inputs=[condition_1, add_node, mod_res],
            outputs=node.output('Out'))


@op_mapper('elementwise_floordiv')
class ElementWiseFloorDiv():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        axis = node.attr('axis')
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        x_dtype = node.input_dtype('X', 0)
        y_dtype = node.input_dtype('Y', 0)
        x_dtype = dtypes.DTYPE_PADDLE_STR_MAP[x_dtype]
        y_dtype = dtypes.DTYPE_PADDLE_STR_MAP[y_dtype]
        is_int = False
        if x_dtype.count('int') > 0 and y_dtype.count('int') > 0:
            is_int = True
        if axis == -1 or axis == (len(x_shape) - 1
                                  ) or len(x_shape) == len(y_shape):
            if is_int:
                graph.make_node(
                    'Div', inputs=[x, y], outputs=node.output('Out'))
            else:
                div_node = graph.make_node('Div', inputs=[x, y])
                graph.make_node(
                    'Floor', inputs=[div_node], outputs=node.output('Out'))
        else:
            broadcast_shape = [1] * len(x_shape)
            broadcast_shape[axis:axis + len(y_shape)] = y_shape
            broadcast_shape_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=list(broadcast_shape))
            y_node = graph.make_node(
                'Reshape', inputs=[y, broadcast_shape_node])
            if is_int:
                div_node = graph.make_node(
                    'Div', inputs=[x, y_node], outputs=node.output('Out'))
            else:
                div_node = graph.make_node('Div', inputs=[x, y_node])
                graph.make_node(
                    'Floor', inputs=[div_node], outputs=node.output('Out'))


@op_mapper('pow')
class Pow():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        x = node.input('X', 0)
        x_dtype = node.input_dtype('X', 0)
        factor = node.attr('factor')
        # Pow-7 Only support input type as float and double
        if x_dtype == paddle.int32 or x_dtype == paddle.int64:
            x = graph.make_node('Cast', inputs=[x], to=dtypes.ONNX.FLOAT)
            factor_node = graph.make_node(
                'Constant',
                inputs=[],
                dims=[1],
                dtype=dtypes.ONNX.FLOAT,
                value=factor)
        else:
            factor_node = graph.make_node(
                'Constant',
                inputs=[],
                dims=[1],
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                value=factor)
        if x_dtype == paddle.int32 or x_dtype == paddle.int64:
            pow_node = graph.make_node('Pow', inputs=[x, factor_node])
            graph.make_node(
                'Cast',
                inputs=[pow_node],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                outputs=node.output('Out'))
        else:
            graph.make_node(
                'Pow', inputs=[x, factor_node], outputs=node.output('Out'))

    @classmethod
    def opset_12(cls, graph, node, **kw):
        x = node.input('X', 0)
        factor = node.attr('factor')
        factor_node = graph.make_node(
            'Constant',
            inputs=[],
            dims=[1],
            dtype=dtypes.ONNX.FLOAT,
            value=factor)
        pow_node = graph.make_node(
            'Pow', inputs=[x, factor_node], outputs=node.output('Out'))


@op_mapper('square')
class Square():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        x = node.input('X', 0)
        onnx_node = graph.make_node(
            'Mul', inputs=[x, x], outputs=node.output('Out'))


@op_mapper('cumsum')
class CumSum():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):

        axis = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=node.attr('axis'))
        graph.make_node(
            'CumSum',
            inputs=[node.input('X', 0), axis],
            outputs=node.output('Out'))


@op_mapper('mul')
class Mul():
    support_opset_version_range = (5, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        out = node.output('Out', 0)
        x_num_col_dims = node.attr('x_num_col_dims')
        y_num_col_dims = node.attr('y_num_col_dims')
        flatten_x = graph.make_node(
            'Flatten', inputs=node.input('X'), attrs={'axis': x_num_col_dims})
        flatten_y = graph.make_node(
            'Flatten', inputs=node.input('Y'), attrs={'axis': y_num_col_dims})
        mul_node = graph.make_node('MatMul', inputs=[flatten_x, flatten_y])

        x_shape = graph.make_node('Shape', inputs=[x])
        l_shape = mapper_helper.slice_helper(
            graph, x_shape, axes=[0], starts=[0], ends=[x_num_col_dims])
        y_shape = graph.make_node('Shape', inputs=[y])
        y_rank = len(node.input_shape('Y', 0))
        r_shape = mapper_helper.slice_helper(
            graph, y_shape, axes=[0], starts=[y_num_col_dims], ends=[y_rank])

        out_shape = graph.make_node('Concat', inputs=[l_shape, r_shape], axis=0)
        graph.make_node('Reshape', [mul_node, out_shape], node.output('Out'))


@op_mapper('affine_channel')
class AffineChannel():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if "data_layout" in node.attrs.keys():
            assert node.attrs['data_layout'] == 'NCHW' or node.attrs['data_layout'] == "AnyLayout",  \
                                "The affine_channel data format should be 'NCHW', but received data format " \
                                "is %s." % node.attrs['data_layout']
        x = node.input('X', 0)
        bias = node.input('Bias', 0)
        scale = node.input('Scale', 0)
        scale = mapper_helper.unsqueeze_helper(graph, scale, [0, 2, 3])
        bias = mapper_helper.unsqueeze_helper(graph, bias, [0, 2, 3])
        x = graph.make_node('Mul', inputs=[x, scale])
        x = graph.make_node('Add', inputs=[x, bias], outputs=node.output('Out'))


@op_mapper('bmm')
class BMM():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        mul_node = graph.make_node(
            'MatMul', inputs=[x, y], outputs=node.output('Out'))


@op_mapper('p_norm')
class PNorm():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        axis = node.attr('axis')
        if isinstance(axis, (int, float)):
            axis = [axis]
        p = node.attr('porder')
        keepdim = node.attr('keepdim')
        dtype = dtypes.ONNX.FLOAT
        if node.input_dtype('X', 0) == paddle.float64:
            dtype = dtypes.ONNX.DOUBLE

        pnode = graph.make_node('Constant', dtype=dtype, value=[p])

        abs_node = graph.make_node('Abs', inputs=[x])
        pow_node = graph.make_node('Pow', inputs=[abs_node, pnode])
        reduce_sum = graph.make_node(
            'ReduceSum', inputs=[pow_node], axes=axis, keepdims=keepdim)
        pnode1 = graph.make_node('Constant', dtype=dtype, value=[1.0 / p])
        graph.make_node(
            'Pow', inputs=[reduce_sum, pnode1], outputs=node.output('Out'))

    @classmethod
    def opset_13(cls, graph, node, **kw):
        x = node.input('X', 0)
        axis = node.attr('axis')
        if isinstance(axis, (int, float)):
            axis = [axis]
        p = node.attr('porder')
        keepdim = node.attr('keepdim')
        pnode = graph.make_node('Constant', dtype=dtypes.ONNX.FLOAT, value=[p])
        abs_node = graph.make_node('Abs', inputs=[x])
        pow_node = graph.make_node('Pow', inputs=[abs_node, pnode])
        axes = graph.make_node('Constant', dtype=dtypes.ONNX.INT64, value=axis)
        reduce_sum = graph.make_node(
            'ReduceSum', inputs=[pow_node, axes], keepdims=keepdim)
        pnode1 = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[1.0 / p])
        graph.make_node(
            'Pow', inputs=[reduce_sum, pnode1], outputs=node.output('Out'))


@op_mapper('sum')
class Sum():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Sum', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('floor')
class Floor():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Floor', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('log10')
class Log10():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        _ln10 = 2.30258509299404568401
        dtype = dtypes.ONNX.FLOAT
        if node.input_dtype('X', 0) == paddle.float64:
            dtype = dtypes.ONNX.DOUBLE
        _ln10 = graph.make_node('Constant', dtype=dtype, value=_ln10)
        lnx = graph.make_node('Log', inputs=node.input('X'))
        graph.make_node('Div', inputs=[lnx, _ln10], outputs=node.output('Out'))


@op_mapper('log1p')
class Log1p():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        dtype = dtypes.ONNX.FLOAT
        if node.input_dtype('X', 0) == paddle.float64:
            dtype = dtypes.ONNX.DOUBLE
        one = graph.make_node('Constant', attrs={'dtype': dtype, 'value': [1]})
        add_node = graph.make_node('Add', inputs=[node.input('X', 0), one])
        graph.make_node('Log', inputs=add_node, outputs=node.output('Out'))


@op_mapper(
    ['reduce_all', 'reduce_any'],
    mapper_dict={'reduce_all': 'ReduceMin',
                 'reduce_any': 'ReduceMax'})
class ReduceAll():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        op_type = kw['mapper_dict'][node.type]
        input_dtype = node.block.vars[node.input('X', 0)].dtype
        input_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[input_dtype]
        all_node = graph.make_node(
            'Cast', inputs=[node.input('X', 0)], to=dtypes.ONNX.INT32)

        attrs = {'keepdims': node.attr('keep_dim'), }
        if not node.attr('reduce_all'):
            attrs['axes'] = node.attr('dim')
        output_node = graph.make_node(op_type, inputs=[all_node], attrs=attrs)

        if node.attr('reduce_all') and not node.attr('keep_dim'):
            output_node = mapper_helper.unsqueeze_helper(graph, output_node,
                                                         [0])
        graph.make_node(
            'Cast',
            inputs=[output_node],
            to=input_dtype,
            outputs=node.output('Out'))


@op_mapper(
    ['reduce_mean', 'reduce_sum', 'reduce_min', 'reduce_max', 'reduce_prod'],
    mapper_dict={
        'reduce_mean': 'ReduceMean',
        'reduce_sum': 'ReduceSum',
        'reduce_min': 'ReduceMin',
        'reduce_max': 'ReduceMax',
        'reduce_prod': 'ReduceProd'
    })
class ReduceMean():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        op_type = kw['mapper_dict'][node.type]

        output_shape = node.output_shape('Out', 0)
        reduce_all = node.attr("reduce_all")
        axes = node.attr("dim")
        if reduce_all:
            axes = list(range(len(node.input_shape("X", 0))))
        if len(axes) == len(node.input_shape("X", 0)):
            reduce_all = True
        keepdims = node.attr('keep_dim')
        if keepdims:
            cls.create_reduce_node(graph, op_type,
                                   node.input("X"), node.output("Out"), axes, 1)
        else:
            if reduce_all:
                shape = graph.make_node(
                    "Constant", dtype=dtypes.ONNX.INT64, value=[-1])
                flatten_node = graph.make_node(
                    "Reshape", inputs=[node.input("X")[0], shape])
                cls.create_reduce_node(graph, op_type, [flatten_node],
                                       node.output("Out"), [0], 1)
            else:
                cls.create_reduce_node(graph, op_type,
                                       node.input("X"),
                                       node.output("Out"), axes, 0)

    @classmethod
    def create_reduce_node(cls, graph, op_type, inputs, outputs, axes,
                           keepdims):
        if graph.opset_version >= 13 and op_type == "ReduceSum":
            axes = graph.make_node(
                "Constant", dtype=dtypes.ONNX.INT64, value=axes)
            output = graph.make_node(
                "ReduceSum",
                inputs=inputs + [axes],
                outputs=outputs,
                keepdims=keepdims)
        else:
            output = graph.make_node(
                op_type,
                inputs=inputs,
                outputs=outputs,
                axes=axes,
                keepdims=keepdims)


@op_mapper('mean')
class Mean():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        shape = graph.make_node("Constant", dtype=dtypes.ONNX.INT64, value=[-1])
        flatten_node = graph.make_node(
            "Reshape", inputs=[node.input("X")[0], shape])
        mean_node = graph.make_node(
            'ReduceMean',
            inputs=flatten_node,
            outputs=node.output("Out"),
            keepdims=1)


@op_mapper('arg_max')
class ArgMax():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if node.attr('dtype') and node.attr('dtype') == 2:
            arg_node = graph.make_node(
                'ArgMax',
                inputs=node.input('X'),
                attrs={
                    'axis': node.attr('axis'),
                    'keepdims': node.attr('keepdims')
                })
            graph.make_node(
                'Cast',
                inputs=arg_node,
                attrs={'to': dtypes.ONNX.INT32},
                outputs=node.output('Out'))
        else:
            graph.make_node(
                'ArgMax',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                attrs={
                    'axis': node.attr('axis'),
                    'keepdims': node.attr('keepdims')
                })


@op_mapper('arg_min')
class ArgMin():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if node.attr('flatten'):
            flatten = graph.make_node('Flatten', inputs=node.input('X'), axis=0)
            squeeze_node = graph.make_node('Squeeze', inputs=flatten)
            graph.make_node(
                'ArgMin', inputs=squeeze_node, outputs=node.output('Out'))
        else:
            if node.attr('keepdims'):
                graph.make_node(
                    'ArgMin',
                    inputs=node.input('X'),
                    outputs=node.output('Out'),
                    axis=node.attr('axis'),
                    keepdims=1)
            else:
                graph.make_node(
                    'ArgMin',
                    inputs=node.input('X'),
                    outputs=node.output('Out'),
                    axis=node.attr('axis'),
                    keepdims=0)


@op_mapper('brelu')
class Hardtanh():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        mapper_helper.clip_helper(graph, node,
                                  node.input('X', 0),
                                  node.attr('t_max'),
                                  node.attr('t_min'), node.output('Out', 0))


@op_mapper('mv')
class Mv():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'MatMul',
            inputs=[node.input('X', 0), node.input('Vec', 0)],
            outputs=node.output('Out'))


@op_mapper('dot')
class Dot():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        mul_node = graph.make_node(
            'Mul', inputs=[node.input('X', 0), node.input('Y', 0)])
        graph.make_node(
            'ReduceSum',
            inputs=[mul_node],
            axes=[len(node.input_shape('X', 0)) - 1],
            outputs=node.output('Out'))

    @classmethod
    def opset_13(cls, graph, node, **kw):
        mul_node = graph.make_node(
            'Mul', inputs=[node.input('X', 0), node.input('Y', 0)])
        one = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT64,
            value=[len(node.input_shape('X', 0)) - 1])
        graph.make_node(
            'ReduceSum', inputs=[mul_node, one], outputs=node.output('Out'))


@op_mapper('dist')
class Dist():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        sub_node = graph.make_node(
            'Sub', inputs=[node.input('X', 0), node.input('Y', 0)])
        abs_node = graph.make_node('Abs', inputs=sub_node)
        if node.attr('p') == 0:
            assert graph.opset_version >= 9, "When p is 0, onnx opset should be (onnx_opset>=9)."
            sign_node = graph.make_node('Sign', inputs=abs_node)
            sum_node = graph.make_node(
                'ReduceSum', inputs=sign_node, keepdims=0)
            mapper_helper.unsqueeze_helper(graph, sum_node, [0],
                                           node.output('Out'))
        elif node.attr('p') == float('inf'):
            max_node = graph.make_node('ReduceMax', inputs=abs_node, keepdims=0)
            mapper_helper.unsqueeze_helper(graph, max_node, [0],
                                           node.output('Out'))
        elif node.attr('p') == float('-inf'):
            min_node = graph.make_node('ReduceMin', inputs=abs_node, keepdims=0)
            mapper_helper.unsqueeze_helper(graph, min_node, [0],
                                           node.output('Out'))
        else:
            x_dtype = node.input_dtype('X', 0)
            p = graph.make_node(
                'Constant',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                value=node.attr('p'))
            pow_node = graph.make_node(
                'Pow',
                inputs=[abs_node, p], )
            sum_node = graph.make_node('ReduceSum', inputs=pow_node, keepdims=0)
            sum_node = mapper_helper.unsqueeze_helper(graph, sum_node, [0])
            p_1 = graph.make_node('Reciprocal', inputs=p)
            graph.make_node(
                'Pow', inputs=[sum_node, p_1], outputs=node.output('Out'))


@op_mapper('round')
class Round():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        graph.make_node(
            'Round', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('rsqrt')
class Rsqrt():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        sqrt_node = graph.make_node('Sqrt', inputs=node.input('X'))
        graph.make_node(
            'Reciprocal', inputs=sqrt_node, outputs=node.output('Out'))


@op_mapper('sign')
class Sign():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'Sign', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('scale')
class Scale():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        scale = node.attr('scale')
        bias = node.attr('bias')
        if len(node.input('ScaleTensor')) == 0 and np.fabs(
                scale - 1.0) < 1e-06 and np.fabs(bias - 0.0) < 1e-06:
            graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        else:
            input_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)]
            if input_dtype in [
                    dtypes.ONNX.INT16, dtypes.ONNX.INT32, dtypes.ONNX.INT64
            ]:
                outputs = None
                data_type = dtypes.ONNX.FLOAT
                cast_node = graph.make_node(
                    'Cast', inputs=node.input('X'), attrs={'to': data_type})
            else:
                outputs = node.output('Out')
                data_type = input_dtype
                cast_node = node.input('X')[0]

            if len(node.input('ScaleTensor')) > 0:
                scale_node = node.input('ScaleTensor')[0]
                scale_type = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype(
                    'ScaleTensor', 0)]
                if scale_type != data_type:
                    scale_node = graph.make_node(
                        'Cast', inputs=[scale_node], attrs={'to': data_type})
            else:
                scale_node = graph.make_node(
                    'Constant', attrs={'dtype': data_type,
                                       'value': [scale]})
            bias_node = graph.make_node(
                'Constant', attrs={'dtype': data_type,
                                   'value': [bias]})

            if node.attr('bias_after_scale'):
                node1 = graph.make_node('Mul', inputs=[cast_node, scale_node])
                node2 = graph.make_node(
                    'Add', inputs=[node1, bias_node], outputs=outputs)
            else:
                node1 = graph.make_node('Add', inputs=[cast_node, bias_node])
                node2 = graph.make_node(
                    'Mul', inputs=[node1, scale_node], outputs=outputs)

            if input_dtype in [
                    dtypes.ONNX.INT16, dtypes.ONNX.INT32, dtypes.ONNX.INT64
            ]:
                cast_node = graph.make_node(
                    'Cast',
                    inputs=node2,
                    outputs=node.output('Out'),
                    attrs={'to': input_dtype})


@op_mapper('softmax')
class Softmax():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axis = node.attr('axis')
        shape = node.output_shape('Out', 0)
        if axis is None:
            axis = -1
        if axis < 0:
            axis += len(shape)
        if axis == len(shape) - 1:
            node = graph.make_node(
                'Softmax',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                attrs={'axis': axis})
        else:
            perm = [i for i in range(len(shape))]
            perm[-1] = axis
            perm[axis] = len(shape) - 1
            transpose_node = graph.make_node(
                'Transpose', inputs=node.input('X'), attrs={'perm': perm})
            softmax_node = graph.make_node(
                'Softmax', inputs=[transpose_node], axis=-1)
            transpose_node1 = graph.make_node(
                'Transpose',
                inputs=[softmax_node],
                outputs=node.output('Out'),
                attrs={'perm': perm})

    @classmethod
    def opset_13(cls, graph, node, **kw):
        graph.make_node(
            'Softmax',
            inputs=node.input('X'),
            axis=node.attr('axis'),
            outputs=node.output('Out'))


@op_mapper('unfold')
class Unfold():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):

        strides = node.attr('strides')
        stride_h = strides[0]
        stride_w = strides[1]

        paddings = node.attr('paddings')
        padding_h_1 = paddings[0]
        padding_w_1 = paddings[1]
        padding_h_2 = paddings[2]
        padding_w_2 = paddings[3]

        dilations = node.attr('dilations')
        dilation_h = dilations[0]
        dilation_w = dilations[1]

        kernel_sizes = node.attr('kernel_sizes')
        kernel_h = kernel_sizes[0]
        kernel_w = kernel_sizes[1]

        input_w = mapper_helper.shape_helper(graph, node.input('X', 0), 3)
        blocks_row_indices_node = cls._get_im2col_indices_along_dim(
            graph, node, 2, kernel_h, dilation_h, padding_h_1, padding_h_2,
            stride_h)
        blocks_col_indices_node = cls._get_im2col_indices_along_dim(
            graph, node, 3, kernel_w, dilation_w, padding_w_1, padding_w_2,
            stride_w)

        output_shape = cls._get_im2col_output_shape(graph, node, kernel_h,
                                                    kernel_w)
        padded_input = cls._get_im2col_padded_input(
            graph, node, padding_h_1, padding_h_2, padding_w_1, padding_w_2)

        output = graph.make_node(
            'Gather', inputs=[padded_input, blocks_row_indices_node], axis=2)

        output = graph.make_node(
            'Gather', inputs=[output, blocks_col_indices_node], axis=4)
        output = graph.make_node(
            'Transpose', inputs=[output], perm=[0, 1, 2, 4, 3, 5])

        graph.make_node(
            'Reshape', inputs=[output, output_shape], outputs=node.output('Y'))

    @classmethod
    def _get_im2col_indices_along_dim(cls, graph, node, index, kernel_size_d,
                                      dilation_d, padding_d_1, padding_d_2,
                                      stride_d):
        input_shape = node.input_shape('X', 0)
        if input_shape[index] == -1:
            input_d_node = mapper_helper.shape_helper(graph,
                                                      node.input('X', 0), index)

            padding_d_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=[padding_d_1 + padding_d_2])
            blocks_d_node = graph.make_node(
                'Add', inputs=[input_d_node, padding_d_node])

            dilation_kernel_size_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=[dilation_d * (kernel_size_d - 1)])
            blocks_d_node = graph.make_node(
                'Sub', inputs=[blocks_d_node, dilation_kernel_size_node])

            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[0])
            stride_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[stride_d])
            blocks_d_indices_node = graph.make_node(
                'Range', inputs=[zero_node, blocks_d_node, stride_node])
        else:
            end = input_shape[
                index] + padding_d_1 + padding_d_2 - dilation_d * (kernel_size_d
                                                                   - 1)
            stride = stride_d
            blocks_d_indices = np.arange(0, end, stride)
            blocks_d_indices_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=blocks_d_indices.flatten().tolist())

        kernel_grid = np.arange(0, kernel_size_d * dilation_d, dilation_d)
        kernel_grid_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT64,
            value=kernel_grid.flatten().tolist())

        shape_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[-1, 1])
        kernel_mask_node = graph.make_node(
            'Reshape', inputs=[kernel_grid_node, shape_node])

        block_mask_node = graph.make_node(
            'Add', inputs=[blocks_d_indices_node, kernel_mask_node])
        return block_mask_node

    @classmethod
    def _get_im2col_output_shape(cls, graph, node, kernel_h, kernel_w):
        batch_dim = mapper_helper.shape_helper(graph, node.input('X', 0), 0)
        channel_dim = mapper_helper.shape_helper(graph, node.input('X', 0), 1)

        constant_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[kernel_h * kernel_w])
        channel_unfolded = graph.make_node(
            'Mul', inputs=[channel_dim, constant_node])

        concat_const_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[-1])
        result_node = graph.make_node(
            'Concat',
            inputs=[batch_dim, channel_unfolded, concat_const_node],
            axis=0)

        return result_node

    @classmethod
    def _get_im2col_padded_input(cls, graph, node, padding_h_1, padding_h_2,
                                 padding_w_1, padding_w_2):
        pad_const_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT64,
            value=[
                0, 0, padding_h_1, padding_w_1, 0, 0, padding_h_2, padding_w_2
            ])
        result_node = graph.make_node(
            'Pad', inputs=[node.input('X', 0), pad_const_node])
        return result_node


@op_mapper('softmax_with_cross_entropy')
class SoftmaxCrossEntropyLoss():
    support_opset_version_range = (12, 15)

    @classmethod
    def opset_12(cls, graph, node, **kw):
        if node.attr('soft_label'):
            raise Exception(
                "SoftmaxCrossEntropyLoss in onnx not support soft label.")
        scores = node.input('Logits', 0)
        labels = node.input('Label', 0)
        # Whether return_softmax is True or False, the model will have two outputs
        outputs = [node.output('Loss', 0), node.output('Softmax', 0)]

        shape = node.input_shape('Logits', 0)
        if len(shape) < 2:
            raise Exception(
                "SoftmaxCrossEntropyLoss in onnx not support 1D logits.")
        axis = node.attr('axis')
        if axis < 0:
            axis += len(shape)
        if axis == 1:
            squeeze_node = mapper_helper.squeeze_helper(graph, labels, [axis])
            loss_node, softmax_node = graph.make_node(
                'SoftmaxCrossEntropyLoss',
                inputs=[scores, squeeze_node],
                outputs=2,
                ignore_index=node.attr('ignore_index'),
                reduction='none')
            loss_node = mapper_helper.unsqueeze_helper(graph, loss_node,
                                                       [axis], outputs[0])
            # onnx output is log(softmax), but paddle output is softmax
            graph.make_node('Exp', inputs=[softmax_node], outputs=outputs[1])
        else:
            perm = [i for i in range(len(shape))]
            perm[1] = axis
            perm[axis] = 1
            transpose_scores = graph.make_node(
                'Transpose', inputs=[scores], perm=perm)
            transpose_labels = graph.make_node(
                'Transpose', inputs=[labels], perm=perm)
            squeeze_labels = mapper_helper.squeeze_helper(
                graph, transpose_labels, [1])

            loss_node, softmax_node = graph.make_node(
                'SoftmaxCrossEntropyLoss',
                inputs=[transpose_scores, squeeze_labels],
                ignore_index=node.attr('ignore_index'),
                outputs=2,
                reduction='none')
            output_node = mapper_helper.unsqueeze_helper(graph, loss_node, [1])
            graph.make_node(
                'Transpose', inputs=output_node, outputs=outputs[0], perm=perm)
            softmax_node = graph.make_node(
                'Transpose', inputs=softmax_node, perm=perm)
            # onnx output is log(softmax), but paddle output is softmax
            graph.make_node('Exp', inputs=[softmax_node], outputs=outputs[1])
