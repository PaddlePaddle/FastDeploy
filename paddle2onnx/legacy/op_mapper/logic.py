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
import paddle
from paddle2onnx.utils import logging
from paddle2onnx.legacy.op_mapper import mapper_helper


@op_mapper('greater_equal')
class GreaterOrEqual():
    support_opset_version_range = (12, 15)

    @classmethod
    def opset_12(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'GreaterOrEqual',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('equal')
class Equal():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        if node.input_dtype('X', 0) in [paddle.float32, paddle.float64]:
            warning_info = "Operator 'Equal' only support input with dtype of int/bool, now the dtype of input is {}, this may cause wrong results, it is more recommend converting this model with opset version >= 11.".format(
                node.input_dtype('X', 0))
            logging.warning(warning_info)
            x_node = graph.make_node(
                'Cast', inputs=node.input('X'), to=dtypes.ONNX.INT32)
            y_node = graph.make_node(
                'Cast', inputs=node.input('Y'), to=dtypes.ONNX.INT32)
            onnx_node = graph.make_node(
                'Equal', inputs=[x_node, y_node], outputs=node.output('Out'))
        else:
            onnx_node = graph.make_node(
                'Equal',
                inputs=[node.input('X', 0), node.input('Y', 0)],
                outputs=node.output('Out'))

    @classmethod
    def opset_11(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'Equal',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('not_equal')
class NotEqual():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        equal_val = None
        if node.input_dtype('X', 0) in [paddle.float32, paddle.float64]:
            warning_info = "Operator 'not_equal' only support input with dtype of int/bool, now the dtype of input is {}, this may cause wrong results, it is more recommend converting this model with opset version >= 11.".format(
                node.input_dtype('X', 0))
            logging.warning(warning_info)
            x_node = graph.make_node(
                'Cast', inputs=node.input('X'), to=dtypes.ONNX.INT32)
            y_node = graph.make_node(
                'Cast', inputs=node.input('Y'), to=dtypes.ONNX.INT32)
            equal_val = graph.make_node(
                'Equal', inputs=[x_node, y_node], outputs=node.output('Out'))
        else:
            equal_val = graph.make_node(
                'Equal',
                inputs=[node.input('X', 0), node.input('Y', 0)],
                outputs=node.output('Out'))
        k_node = graph.make_node(
            'Cast', inputs=[equal_val], to=dtypes.ONNX.INT64)
        const = graph.make_node('Constant', dtype=dtypes.ONNX.INT64, value=1)
        sub_ = graph.make_node('Sub', inputs=[const, k_node])
        graph.make_node(
            'Cast',
            inputs=[sub_],
            outputs=node.output('Out'),
            to=dtypes.ONNX.BOOL)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        equal_val = graph.make_node(
            'Equal', inputs=[node.input('X', 0), node.input('Y', 0)])
        k_node = graph.make_node(
            'Cast', inputs=[equal_val], to=dtypes.ONNX.INT64)
        const = graph.make_node('Constant', dtype=dtypes.ONNX.INT64, value=1)
        sub_ = graph.make_node('Sub', inputs=[const, k_node])
        graph.make_node(
            'Cast',
            inputs=[sub_],
            outputs=node.output('Out'),
            to=dtypes.ONNX.BOOL)


@op_mapper('greater_than')
class GreaterThan():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        if node.input_dtype('X', 0) in [paddle.int32, paddle.int64]:
            warning_info = "Operator 'greater_than' only support input with dtype of float/double, now the dtype of input is {}, this may cause wrong results, it is more recommend converting this model with opset version >= 11.".format(
                node.input_dtype('X', 0))
            logging.warning(warning_info)
            x_node = graph.make_node(
                'Cast', inputs=node.input('X'), to=dtypes.ONNX.INT32)
            y_node = graph.make_node(
                'Cast', inputs=node.input('Y'), to=dtypes.ONNX.INT32)
            graph.make_node(
                'Greater',
                inputs=[node.input('X', 0), node.input('Y', 0)],
                outputs=node.output('Out'))
        else:
            graph.make_node(
                'Greater',
                inputs=[node.input('X', 0), node.input('Y', 0)],
                outputs=node.output('Out'))

    @classmethod
    def opset_11(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'Greater',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('logical_and')
class LogicalAnd():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'And',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('logical_not')
class LogicalNot():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Not', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('logical_or')
class LogicalOr():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Or',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('logical_xor')
class LogicalXOr():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Xor',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('less_equal')
class LessOrEqual():
    support_opset_version_range = (12, 15)

    @classmethod
    def opset_12(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'LessOrEqual',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('less_than')
class Less_than():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        if node.input_dtype('X', 0) in [paddle.int32, paddle.int64]:
            warning_info = "Operator 'less_than' only support input with dtype of float/double, now the dtype of input is {}, this may cause wrong results, it is more recommend converting this model with opset version >= 11.".format(
                node.input_dtype('X', 0))
            logging.warning(warning_info)
            x_node = graph.make_node(
                'Cast', inputs=node.input('X'), to=dtypes.ONNX.INT32)
            y_node = graph.make_node(
                'Cast', inputs=node.input('Y'), to=dtypes.ONNX.INT32)
            graph.make_node(
                'Less',
                inputs=[node.input('X', 0), node.input('Y', 0)],
                outputs=node.output('Out'))
        else:
            graph.make_node(
                'Less',
                inputs=[node.input('X', 0), node.input('Y', 0)],
                outputs=node.output('Out'))

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'Less',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'), )


@op_mapper('isfinite_v2')
class Isfinite():
    support_opset_version_range = (10, 15)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        is_inf = graph.make_node('IsInf', inputs=node.input('X', 0))
        is_nan = graph.make_node('IsNaN', inputs=node.input('X', 0))
        finite = graph.make_node('Or', inputs=[is_inf, is_nan])
        graph.make_node('Not', inputs=[finite], outputs=node.output('Out'))


@op_mapper('isinf_v2')
class IsInf():
    support_opset_version_range = (10, 15)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        graph.make_node(
            'IsInf', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('isnan_v2')
class IsNaN():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'IsNaN', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('isnan')
class IsNaN():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        isnan = graph.make_node('IsNaN', inputs=node.input('X'))
        cast_node = graph.make_node(
            'Cast', inputs=isnan, attrs={'to': dtypes.ONNX.FLOAT})
        reduce_node = graph.make_node(
            'ReduceMax', inputs=[cast_node], keepdims=False)
        mapper_helper.unsqueeze_helper(graph, reduce_node, [0],
                                       node.output('Out'))
