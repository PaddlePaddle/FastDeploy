#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle2caffe.constant.op_mapping_status import *
from paddle2caffe.op_mapper import OpMapper as op_mapper
from paddle2caffe.op_mapper import mapper_helper

from paddle2caffe.graph import Graph, Node
from paddle2caffe.utils import logging


@op_mapper(
    ['relu', 'tanh', 'log', 'sigmoid', 'sqrt'],
    mapper_dict={
        'relu': 'ReLU',
        'tanh': 'Tanh',
        'log': 'Log',
        'sigmoid': 'Sigmoid',
        'sqrt': 'Sqrt',
    })
class ActivationOps:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        caffe_type = kw['mapper_dict'][source_node.op_type]
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type=caffe_type, attrs=None)
        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('leaky_relu')
class LeakyRelu:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/leaky_relu_cn.html#leaky-relu"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        attrs = {'negative_slope': source_node.attr('alpha')}
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type='ReLU', attrs=attrs)
        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('softplus')
class Softplus:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('prelu')
class PRelu:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/prelu_cn.html#prelu"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        attrs = {'channel_shared': source_node.attr('channel_shared')}
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type='PReLU', attrs=attrs)
        if len(source_node.params) > 0:
            raise NotImplementedError('To be continued')

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('relu6')
class Relu6:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type='ReLU6', attrs=None)
        return OP_MAPPING_IDENTITY, [caffe_node.name]

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('standard caffe have no ReLU6 support, try use custom caffe')


@op_mapper('hard_sigmoid')
class HardSigmoid:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """
        https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Hardsigmoid_cn.html#hardsigmoid
        change to caffe as:
            out = max(0, min(1, slope·x + offset))
            ==> out = max(0, min(6, 6·slope·x + 6·offset)) / 6
            ==> out = relu6(6·slope·x + 6·offset) / 6
        input  -> Scale -> relu6 -> Scale ->  output
        """
        new_slope = 6. * source_node.attr('slope')
        new_offset = 6. * source_node.attr('offset')

        input_blob_name = source_node.input(0)
        input_blob = caffe_graph.get_blob(input_blob_name)
        output_blob_name = source_node.output(0)
        input_shape = source_graph.blob_map[source_node.input(0)].shape
        channel_num = input_shape[1]

        extra_blob_name_scale_0 = input_blob_name + '_extra_scale_0'
        extra_blob_name_relu6 = input_blob_name + '_extra_relu6'

        source_name = source_node.name  # final scale will use source name
        extra_scale_0_name = source_name + '_extra_scale_0'
        extra_relu6_name = source_name + '_extra_relu6'

        # connect input
        assert extra_scale_0_name not in input_blob.dst_nodes_names
        input_blob.dst_nodes_names.append(extra_scale_0_name)
        # add scale 0
        scale_node_0 = caffe_graph.make_node('Scale', source_node.raw_name + '_extra_scale_0', extra_scale_0_name,
                                             [input_blob_name], [extra_blob_name_scale_0],
                                             attrs=dict(), do_insert=True)
        extra_blob_scale_0 = caffe_graph.make_blob(input_shape, extra_blob_name_scale_0, extra_blob_name_scale_0,
                                                   scale_node_0.name, [extra_relu6_name], do_insert=True)
        mapper_helper.creat_params(scale_node_0, caffe_graph, 'scale', new_slope, channel_num)
        mapper_helper.creat_params(scale_node_0, caffe_graph, 'bias', new_offset, channel_num)
        # add relu6
        relu6_node = caffe_graph.make_node('ReLU6', source_node.raw_name + '_extra_relu6', extra_relu6_name,
                                           [extra_blob_scale_0.name], [extra_blob_name_relu6],
                                           attrs=dict(), do_insert=True)
        extra_blob_relu6 = caffe_graph.make_blob(input_shape, extra_blob_name_relu6, extra_blob_name_relu6,
                                                 relu6_node.name, [source_name], do_insert=True)
        # add scale 1
        assert output_blob_name not in input_blob.dst_nodes_names
        input_blob.dst_nodes_names.append(output_blob_name)
        scale_node_1 = caffe_graph.make_node('Scale', source_node.raw_name, source_node.name,
                                             [extra_blob_relu6.name], [output_blob_name],
                                             attrs=dict(), do_insert=True)
        mapper_helper.creat_params(scale_node_1, caffe_graph, 'scale', 1. / 6, channel_num)
        # connect output
        _ = caffe_graph.transfer_op_output(source_node, source_graph)

        return OP_MAPPING_WITH_EXTRA, [scale_node_0.name, relu6_node.name, scale_node_1.name]


@op_mapper('swish')
class Swish:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """
        https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Swish_cn.html#swish
        change to caffe as: 𝑓(𝑥) = 𝑥·sigmoid(β𝑥)
        input  -> (Scale) -> Sigmoid -> Eltwise(MUL) ->  output
              |-> ------------------- ->|
        """
        beta = source_node.attrs.get('beta')

        input_blob_name = source_node.input(0)
        output_blob_name = source_node.output(0)
        extra_blob_name_scale = input_blob_name + '_extra_scale'
        extra_blob_name_sigmoid = input_blob_name + '_extra_sigmoid'

        input_shape = source_graph.blob_map[source_node.input(0)].shape

        source_name = source_node.name  # Eltwise will use source name
        extra_scale_name = source_name + '_extra_scale'
        extra_sigmoid_name = source_name + '_extra_sigmoid'

        input_blob = caffe_graph.get_blob(input_blob_name)
        scale_node = None
        if beta - 1.0 < 1e-5:  # consider as beta == 1
            # connect input
            assert extra_sigmoid_name not in input_blob.dst_nodes_names
            input_blob.dst_nodes_names.append(extra_sigmoid_name)
            # add sigmoid
            sigmoid_node = caffe_graph.make_node('Sigmoid', source_node.raw_name + '_extra_sigmoid', extra_sigmoid_name,
                                                 [input_blob_name], [extra_blob_name_sigmoid],
                                                 attrs=dict(), do_insert=True)
            extra_blob_sigmoid = caffe_graph.make_blob(input_shape, extra_blob_name_sigmoid, extra_blob_name_sigmoid,
                                                       sigmoid_node.name, [source_name], do_insert=True)
        else:
            # connect input
            assert extra_scale_name not in input_blob.dst_nodes_names
            input_blob.dst_nodes_names.append(extra_scale_name)
            # add scale
            scale_node = caffe_graph.make_node('Scale', source_node.raw_name + '_extra_scale', extra_scale_name,
                                               [input_blob_name], [extra_blob_name_scale],
                                               attrs=dict(), do_insert=True)
            extra_blob_scale = caffe_graph.make_blob(input_shape, extra_blob_name_scale, extra_blob_name_scale,
                                                     scale_node.name, [source_name], do_insert=True)
            # add sigmoid
            sigmoid_node = caffe_graph.make_node('Sigmoid', source_node.raw_name + '_extra_sigmoid', extra_sigmoid_name,
                                                 [extra_blob_scale.name], [extra_blob_name_sigmoid],
                                                 attrs=dict(), do_insert=True)
            extra_blob_sigmoid = caffe_graph.make_blob(input_shape, extra_blob_name_sigmoid, extra_blob_name_sigmoid,
                                                       sigmoid_node.name, [source_name], do_insert=True)

        # add eltwise
        # connect input as input B
        assert output_blob_name not in input_blob.dst_nodes_names
        input_blob.dst_nodes_names.append(output_blob_name)
        eltwise_node = caffe_graph.make_node('Eltwise', source_node.raw_name, source_node.name,
                                             [input_blob_name, extra_blob_sigmoid.name], [output_blob_name],
                                             attrs={'operation': 'PROD'}, do_insert=True)
        # connect output
        _ = caffe_graph.transfer_op_output(source_node, source_graph)

        if scale_node is None:
            return OP_MAPPING_WITH_EXTRA, [sigmoid_node.name, eltwise_node.name]
        else:
            return OP_MAPPING_WITH_EXTRA, [scale_node.name, sigmoid_node.name, eltwise_node.name]


@op_mapper('hard_swish')
class HardSwish:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """
        https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Hardswish_cn.html#hardswish
        only valid when offset = 3.0，threshold = scale = 6.0
        change to caffe as: 𝑥·relu6(x + 3.) / 6.
        input  -> Scale -> ReLU6 -> Scale -> Eltwise(MUL) ->  output
              |-> --------------------------- ->|
        """
        offset = source_node.attrs.get('offset')
        threshold = source_node.attrs.get('threshold')
        scale = source_node.attrs.get('scale')
        if abs(offset - 3.) > 1e-5 or abs(threshold - 6.) > 1e-5 or abs(scale - 6.) > 1e-5:
            raise ValueError('hardswish with current attrs can not convert to caffe')

        input_blob_name = source_node.input(0)
        input_blob = caffe_graph.get_blob(input_blob_name)
        output_blob_name = source_node.output(0)
        input_shape = source_graph.blob_map[source_node.input(0)].shape
        channel_num = input_shape[1]

        extra_blob_name_scale_0 = input_blob_name + '_extra_scale_0'
        extra_blob_name_relu6 = input_blob_name + '_extra_relu6'
        extra_blob_name_scale_1 = input_blob_name + '_extra_scale_1'

        source_name = source_node.name  # Eltwise will use source name
        extra_scale_0_name = source_name + '_extra_scale_0'
        extra_relu6_name = source_name + '_extra_relu6'
        extra_scale_1_name = source_name + '_extra_scale_1'

        # connect input
        assert extra_scale_0_name not in input_blob.dst_nodes_names
        input_blob.dst_nodes_names.append(extra_scale_0_name)
        # add scale 0
        scale_node_0 = caffe_graph.make_node('Scale', source_node.raw_name + '_extra_scale_0', extra_scale_0_name,
                                             [input_blob_name], [extra_blob_name_scale_0],
                                             attrs=dict(), do_insert=True)
        extra_blob_scale_0 = caffe_graph.make_blob(input_shape, extra_blob_name_scale_0, extra_blob_name_scale_0,
                                                   scale_node_0.name, [extra_relu6_name], do_insert=True)
        mapper_helper.creat_params(scale_node_0, caffe_graph, 'scale', 1., channel_num)
        mapper_helper.creat_params(scale_node_0, caffe_graph, 'bias', 3., channel_num)
        # add relu6
        relu6_node = caffe_graph.make_node('ReLU6', source_node.raw_name + '_extra_relu6', extra_relu6_name,
                                           [extra_blob_scale_0.name], [extra_blob_name_relu6],
                                           attrs=dict(), do_insert=True)
        extra_blob_relu6 = caffe_graph.make_blob(input_shape, extra_blob_name_relu6, extra_blob_name_relu6,
                                                 relu6_node.name, [extra_scale_1_name], do_insert=True)
        # add scale 1
        scale_node_1 = caffe_graph.make_node('Scale', source_node.raw_name + '_extra_scale_1', extra_scale_1_name,
                                             [extra_blob_relu6.name], [extra_blob_name_scale_1],
                                             attrs=dict(), do_insert=True)
        extra_blob_scale_1 = caffe_graph.make_blob(input_shape, extra_blob_name_scale_1, extra_blob_name_scale_1,
                                                   scale_node_1.name, [source_name], do_insert=True)
        mapper_helper.creat_params(scale_node_1, caffe_graph, 'scale', 1. / 6, channel_num)

        # add eltwise
        # connect input as input B
        assert output_blob_name not in input_blob.dst_nodes_names
        input_blob.dst_nodes_names.append(output_blob_name)
        eltwise_node = caffe_graph.make_node('Eltwise', source_node.raw_name, source_node.name,
                                             [input_blob_name, extra_blob_scale_1.name], [output_blob_name],
                                             attrs={'operation': 'PROD'}, do_insert=True)
        # connect output
        _ = caffe_graph.transfer_op_output(source_node, source_graph)

        return OP_MAPPING_WITH_EXTRA, [scale_node_0.name, relu6_node.name, scale_node_1.name, eltwise_node.name]
