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


@op_mapper('where_index')
class WhereIndex():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        nonzero_node = graph.make_node(
            'NonZero', inputs=node.input('Condition'))
        graph.make_node(
            'Transpose',
            inputs=[nonzero_node],
            outputs=node.output('Out'),
            perm=[1, 0])


@op_mapper('top_k_v2')
class TopKV2():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        sorted = node.attr('sorted')
        # for paddle, In gpu device, it always return the sorted value
        # if not sorted:
        #     sorted = True
        if 'K' in node.inputs and len(node.input('K')) > 0:
            k_node = node.input('K', 0)
            k_node_dtype = node.input_dtype('K', 0)
            if dtypes.DTYPE_PADDLE_STR_MAP[k_node_dtype] != 'int64':
                k_node = graph.make_node(
                    'Cast', inputs=[k_node], to=dtypes.ONNX.INT64)
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)],
                largest=node.attr('largest'),
                sorted=sorted,
                axis=node.attr('axis'))
        else:
            k = node.attr('k')
            k_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [k]})
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)],
                largest=node.attr('largest'),
                sorted=sorted,
                axis=node.attr('axis'))


@op_mapper('top_k')
class TopK():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        if 'K' in node.inputs and len(node.input('K')) > 0:
            k_node = node.input('K', 0)
            k_node_dtype = node.input_dtype('K', 0)
            if dtypes.DTYPE_PADDLE_STR_MAP[k_node_dtype] != 'int64':
                k_node = graph.make_node(
                    'Cast', inputs=[k_node], to=dtypes.ONNX.INT64)
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)])
        else:
            k = node.attr('k')
            k_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [k]})
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)])


@op_mapper('argsort')
class ArgSort():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        shape = graph.make_node('Shape', inputs=node.input('X', 0))
        from paddle2onnx.legacy.op_mapper import mapper_helper
        axis = node.attr('axis')
        if axis < 0:
            axis = axis + len(node.input_shape('X', 0))
        dim_size = mapper_helper.slice_helper(
            graph, shape, axes=[0], starts=[axis], ends=[axis + 1])
        if graph.opset_version > 10:
            if not node.attr('descending'):
                graph.make_node(
                    'TopK',
                    inputs=[node.input('X', 0), dim_size],
                    outputs=[node.output('Out', 0), node.output('Indices', 0)],
                    axis=node.attr('axis'),
                    largest=0)
            else:
                graph.make_node(
                    'TopK',
                    inputs=[node.input('X', 0), dim_size],
                    outputs=[node.output('Out', 0), node.output('Indices', 0)],
                    axis=node.attr('axis'),
                    largest=1)
        else:
            if not node.attr('descending'):
                raise Exception(
                    "descending=False only support opset version>=11.")
            else:
                graph.make_node(
                    'TopK',
                    inputs=[node.input('X', 0), dim_size],
                    outputs=[node.output('Out', 0), node.output('Indices', 0)],
                    axis=node.attr('axis'))

    @classmethod
    def opset_6(cls, graph, node, **kw):
        shape = node.input_shape('X', 0)
        k = shape[node.attr('axis')]
        assert k > 0, "while input shape is dynamic, it only support opset version>=10."
        input_dtype = node.input_dtype('X', 0)
        dtype = dtypes.DTYPE_PADDLE_STR_MAP[input_dtype]
        inputs = node.input('X', 0)
        if dtype in ["int32", "int64"]:
            inputs = graph.make_node(
                'Cast', inputs=inputs, to=dtypes.ONNX.FLOAT)
        if not node.attr('descending'):
            raise Exception("descending=False only support opset version>=11.")
        else:
            output_node = node.output('Out', 0)
            graph.make_node(
                'TopK',
                inputs=[inputs],
                outputs=[output_node, node.output('Indices', 0)],
                axis=node.attr('axis'),
                k=k)
            if dtype in ["int32", "int64"]:
                graph.make_node(
                    'Cast',
                    inputs=[output_node],
                    to=dtypes.DTYPE_PADDLE_ONNX_MAP[input_dtype],
                    outputs=[output_node])


@op_mapper('index_select')
class IndexSelect():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Gather',
            inputs=[node.input('X', 0), node.input('Index', 0)],
            axis=node.attr('dim'),
            outputs=node.output('Out'))


@op_mapper('unique')
class Unique():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        if node.attr('axis') == []:
            graph.make_node(
                'Unique',
                inputs=node.input('X'),
                outputs=[
                    node.output('Out', 0), node.output('Indices', 0),
                    node.output('Index', 0), node.output('Counts', 0)
                ])
        else:
            graph.make_node(
                'Unique',
                inputs=node.input('X'),
                axis=node.attr('axis')[0],
                outputs=[
                    node.output('Out', 0), node.output('Indices', 0),
                    node.output('Index', 0), node.output('Counts', 0)
                ])


@op_mapper('where')
class Where():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'Where',
            inputs=[
                node.input('Condition', 0), node.input('X', 0),
                node.input('Y', 0)
            ],
            outputs=node.output('Out'))


@op_mapper('masked_select')
class MaskSelect():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        index = graph.make_node('NonZero', inputs=node.input('Mask', 0))
        index = graph.make_node('Transpose', inputs=[index], perm=[1, 0])
        graph.make_node(
            'GatherND',
            inputs=[node.input('X', 0), index],
            outputs=node.output('Y'))
