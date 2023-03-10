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

import logging

import numpy as np

from paddle2caffe.constant.op_mapping_status import *
from paddle2caffe.op_mapper import OpMapper as op_mapper

from paddle2caffe.graph import Graph, Node
from paddle2caffe.op_mapper import mapper_helper

"""
    1. paddle matmul/mul参数排列为 input_chn, output_chn, caffe对应InnerProduct顺序相反
"""


@op_mapper(['matmul', 'matmul_v2'])
class MatMul:
    have_custom = True

    @classmethod
    def _single_input(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, x_shape=None):
        # before: input  -->   MatMul  -->  output
        input_blob_name = source_node.input_blobs_names[0]
        node_name = source_node.name
        output_blob_name = source_node.output_blobs_names[0]

        fc_params = source_graph.get_parameters(source_node.params['Y'])
        fc_attr = {'num_output': fc_params.shape[1]}

        if len(fc_params.shape) > 2:
            raise NotImplementedError('this mul can not be cvt into FC')

        extra_node = None
        if len(x_shape) > 2:
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 1, extra_node: 1
            # after:  input  -->   Flatten  --> new_blob -->  InnerProduct  -->  output
            extra_blob_name = output_blob_name + '_extra_x'
            extra_node_name = node_name + '_extra_x_flatten'
            # connect input -> extra(Flatten)
            caffe_graph.blob_map[input_blob_name].dst_nodes_names.append(extra_node_name)
            extra_node = caffe_graph.make_node('Flatten', 'extra_node_for_matmul', extra_node_name,
                                               [input_blob_name], [extra_blob_name],
                                               attrs={'axis': 1}, do_insert=True)
            flatten_shape = [x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]]
            extra_blob = caffe_graph.make_blob(flatten_shape, 'extra_blob_for_matmul', extra_blob_name,
                                               extra_node.name, [node_name],
                                               domain=None, do_insert=True)
            caffe_node = caffe_graph.make_node('InnerProduct', source_node.raw_name, node_name,
                                               [extra_blob.name], [output_blob_name],
                                               attrs=fc_attr, do_insert=True)
            # copy output connection from source graph
            caffe_graph.transfer_op_output(source_node, source_graph)
        else:
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
            caffe_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'InnerProduct', attrs=fc_attr, input_blob_num=1)

        mapper_helper.inherit_params(source_node, source_graph, 'Y', caffe_node, caffe_graph, 'weights')
        # change params order
        paddle_like_weights = caffe_graph.parameters[caffe_node.params['weights']]
        assert len(paddle_like_weights.shape) == 2, 'MatMul with 3-dim(>2-dim) weights, which is illegal'
        caffe_graph.parameters[caffe_node.params['weights']] = np.transpose(paddle_like_weights, axes=[1, 0])

        if extra_node is None:
            return OP_MAPPING_IDENTITY, [caffe_node.name]
        else:
            return OP_MAPPING_WITH_EXTRA, [extra_node.name, caffe_node.name]

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        if len(source_node.input_blobs_names) == 1:
            # in fact is fully connection
            transpose_X = source_node.attr('transpose_X')
            transpose_Y = source_node.attr('transpose_Y')
            assert transpose_X is False and transpose_Y is False, 'do not support transpose operation'

            x_shape = source_graph.blob_map.get(source_node.input(0)).shape
            return cls._single_input(caffe_graph, source_node, source_graph, x_shape)
        else:
            raise ModuleNotFoundError('standard caffe do not have matmul layer')

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        if len(source_node.input_blobs_names) == 1:
            return cls.standard_mapping(caffe_graph, source_node, source_graph)
        else:
            raise NotImplementedError('to be continued')


@op_mapper('mul')
class Mul:
    have_custom = False

    @classmethod
    def _single_input(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, x_shape=None):
        # before: input  -->   MatMul  -->  output
        input_blob_name = source_node.input_blobs_names[0]
        node_name = source_node.name
        output_blob_name = source_node.output_blobs_names[0]

        fc_params = source_graph.get_parameters(source_node.params['Y'])
        fc_attr = {'num_output': fc_params.shape[1]}

        if len(fc_params.shape) > 2:
            raise NotImplementedError('this mul can not be cvt into FC')

        extra_node = None
        if len(x_shape) > 2:
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 1, extra_node: 1
            # after:  input  -->   Flatten  --> new_blob -->  InnerProduct  -->  output
            extra_blob_name = output_blob_name + '_extra_x'
            extra_node_name = node_name + '_extra_x_flatten'
            # connect input -> extra(Flatten)
            caffe_graph.blob_map[input_blob_name].dst_nodes_names.append(extra_node_name)
            extra_node = caffe_graph.make_node('Flatten', 'extra_node_for_matmul', extra_node_name,
                                               [input_blob_name], [extra_blob_name],
                                               attrs={'axis': 1}, do_insert=True)
            flatten_shape = [x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]]
            extra_blob = caffe_graph.make_blob(flatten_shape, 'extra_blob_for_matmul', extra_blob_name,
                                               extra_node.name, [node_name],
                                               domain=None, do_insert=True)
            caffe_node = caffe_graph.make_node('InnerProduct', source_node.raw_name, node_name,
                                               [extra_blob.name], [output_blob_name],
                                               attrs=fc_attr, do_insert=True)
            # copy output connection from source graph
            caffe_graph.transfer_op_output(source_node, source_graph)
        else:
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
            caffe_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'InnerProduct', attrs=fc_attr, input_blob_num=1)

        mapper_helper.inherit_params(source_node, source_graph, 'Y', caffe_node, caffe_graph, 'weights')
        # change params order
        paddle_like_weights = caffe_graph.parameters[caffe_node.params['weights']]
        assert len(paddle_like_weights.shape) == 2, 'MatMul with 3-dim(>2-dim) weights, which is illegal'
        caffe_graph.parameters[caffe_node.params['weights']] = np.transpose(paddle_like_weights, axes=[1, 0])

        if extra_node is None:
            return OP_MAPPING_IDENTITY, [caffe_node.name]
        else:
            return OP_MAPPING_WITH_EXTRA, [extra_node.name, caffe_node.name]

    @classmethod
    def _double_input(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph):
        raise NotImplementedError

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/mul_cn.html#mul"""
        if source_node.attr('x_num_col_dims') != 1 or source_node.attr('y_num_col_dims') != 1:
            raise NotImplementedError
        x_shape = source_graph.blob_map.get(source_node.input(0)).shape
        weight = None
        if len(source_node.params) > 0:
            weight = source_graph.parameters.get(source_node.params['Y'])

        if weight is not None:
            return cls._single_input(caffe_graph, source_node, source_graph, x_shape)
        else:
            return cls._double_input(caffe_graph, source_node, source_graph)


@op_mapper('exp')
class Exp:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/exp_cn.html#exp"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        caffe_node = mapper_helper.convert_identity_operation(
            caffe_graph, source_node, source_graph, 'Exp', dict())

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('abs')
class Abs:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        caffe_node = mapper_helper.convert_identity_operation(
            caffe_graph, source_node, source_graph, 'AbsVal', dict())

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('pow')
class Pow:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/pow_cn.html#pow"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        attrs = {'power': source_node.attr('factor')}
        caffe_node = mapper_helper.convert_identity_operation(
            caffe_graph, source_node, source_graph, 'Power', attrs)

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper(
    ['reduce_mean', 'reduce_sum', 'reduce_min', 'reduce_max', 'reduce_prod'],
    mapper_dict={
        'reduce_mean': 'ReduceMean',
        'reduce_sum': 'ReduceSum',
        'reduce_min': 'ReduceMin',
        'reduce_max': 'ReduceMax',
        'reduce_prod': 'ReduceProd'
    })
class ReduceMean:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('arg_max')
class ArgMax:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmax_cn.html#argmax"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        if source_node.attrs.get('keepdim', True) is False:
            raise NotImplementedError('To be continued')

        attrs = {'axis': source_node.attr('axis')}
        caffe_node = mapper_helper.convert_identity_operation(
            caffe_graph, source_node, source_graph, 'ArgMax', attrs)

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('scale')
class Scale:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # fluid output may ends with scale layer
        is_output_node = False
        for output_node in source_graph.output_nodes:
            if source_node.output(0) == output_node.input(0):
                is_output_node = True
                break
        if is_output_node:
            output_node = Node('Output', source_node.input(0), 'output',
                               [source_node.input(0)], [], attrs=dict())
            caffe_graph.output_nodes.append(output_node)
            return OP_MAPPING_AMMEND, []
        else:
            scale = source_node.attrs.get('scale', 1.0)
            bias = source_node.attrs.get('bias', 0.0)
            input_shape = source_graph.blob_map.get(source_node.input(0)).shape
            input_chn = input_shape[1]

            scale_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'Scale', {})
            mapper_helper.creat_params(scale_node, caffe_graph, 'scale', scale, input_chn)
            if abs(bias) > 1e-5:
                mapper_helper.creat_params(scale_node, caffe_graph, 'bias', bias, input_chn)
            return OP_MAPPING_IDENTITY, [scale_node.name]


@op_mapper('softmax')
class Softmax:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        ori_input_shape = source_graph.get_blob(source_node.input(0)).shape
        attrs = {'axis': source_node.attrs.get('axis', 1)}
        if source_node.attrs.get('axis', 1) < 0:
            attrs['axis'] = source_node.attrs.get('axis') + len(ori_input_shape)
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        softmax_node = mapper_helper.convert_identity_operation(
            caffe_graph, source_node, source_graph, 'Softmax', attrs)

        return OP_MAPPING_IDENTITY, [softmax_node.name]


@op_mapper(
    [
        'elementwise_add',
        'elementwise_sub',
        'elementwise_mul',
        'elementwise_max',
    ],
    mapper_dict={
        'elementwise_add': 'SUM',
        'elementwise_sub': 'Sub',
        'elementwise_mul': 'PROD',
        'elementwise_max': 'MAX',
    })
class ElementwiseOps:
    have_custom = False

    @classmethod
    def _multi_input_with_boardcast(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # change input shape into N,C,H,W + N,C and use Scale layer
        # only works for elementwise_mul
        assert source_node.op_type == 'elementwise_mul', 'boardcast hack can only used in Mul'
        input_shape_x = source_graph.get_blob(source_node.input(0)).shape
        input_shape_y = source_graph.get_blob(source_node.input(1)).shape

        if len(input_shape_x) == len(input_shape_y):
            assert input_shape_y[-2:] == (1, 1), 'shape mismatch can not boardcast using scale'
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 1, extra_node: 1
            input_blob_x_name = source_node.input_blobs_names[0]
            input_blob_y_name = source_node.input_blobs_names[1]
            output_blob_name = source_node.output_blobs_names[0]
            ori_output_shape = source_graph.blob_map[output_blob_name].shape

            node_name = source_node.name
            extra_blob_name = input_blob_y_name + '_extra_flatten'
            extra_node_name = node_name + '_extra_flatten'
            extra_output_shape = ori_output_shape[:-2]
            flatten_node = caffe_graph.make_node('Flatten', source_node.raw_name + '_extra', extra_node_name,
                                                 [input_blob_y_name], [extra_blob_name],
                                                 attrs=dict(), do_insert=True)
            extra_blob = caffe_graph.make_blob(extra_output_shape, 'extra_blob_for_boardcast', extra_blob_name,
                                               flatten_node.name, [node_name],
                                               domain=None, do_insert=True)
            eltwise_node = caffe_graph.make_node('Scale', source_node.raw_name, source_node.name,
                                                 [input_blob_x_name, extra_blob_name], [output_blob_name],
                                                 attrs={'axis': 0}, do_insert=True)

            # fix input connect from source graph
            assert input_blob_x_name in caffe_graph.blob_map.keys(), '{} not found in blob'.format(input_blob_x_name)
            assert input_blob_y_name in caffe_graph.blob_map.keys(), '{} not found in blob'.format(input_blob_y_name)
            # fix output connect from source graph
            # check graph is in typo-sort
            assert output_blob_name not in caffe_graph.blob_map.keys()
            output_blob = source_graph.get_blob(output_blob_name, do_copy=True)
            assert output_blob.src_node_name == node_name

            output_blob.dst_nodes_names = []  # cut out source_graph output connection which may change in transfer
            caffe_graph.blob_map[output_blob_name] = output_blob

            return OP_MAPPING_WITH_EXTRA, [flatten_node.name, eltwise_node.name]
        else:
            if len(input_shape_y) == 2:
                # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
                caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                                      'Scale', attrs={'axis': 0}, input_blob_num=2)
                return OP_MAPPING_IDENTITY, [caffe_node.name]
            else:
                raise NotImplementedError('need to blob swap')

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        attrs = {'operation': kw['mapper_dict'][source_node.op_type]}
        if len(source_node.params) == 0 and len(source_node.input_blobs_names) == 2:
            input_shape_x = source_graph.get_blob(source_node.input(0)).shape
            input_shape_y = source_graph.get_blob(source_node.input(1)).shape

            if len(input_shape_x) == len(input_shape_y):
                if input_shape_x[1:] == input_shape_y[1:]:
                    assert kw['mapper_dict'][source_node.op_type] != 'Sub', 'Caffe Eltwise do not support sub'
                    # transfer info: input blob: n, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
                    multi_input_num = len(source_node.input_blobs_names)
                    caffe_node = mapper_helper.convert_identity_operation(
                        caffe_graph, source_node, source_graph, 'Eltwise', attrs, input_blob_num=multi_input_num)
                    return OP_MAPPING_IDENTITY, [caffe_node.name]
                elif input_shape_x[-2:] == (1, 1) or input_shape_y[-2:] == (1, 1):
                    return cls._multi_input_with_boardcast(caffe_graph, source_node, source_graph)
                else:
                    raise NotImplementedError('caffe do not support elementwise with multi boardcast')
            elif abs(len(input_shape_x) - len(input_shape_y)) == 2:
                return cls._multi_input_with_boardcast(caffe_graph, source_node, source_graph)
            else:
                raise NotImplementedError('caffe do not support elementwise with multi boardcast')
        elif len(source_node.params) == 0 and len(source_node.input_blobs_names) > 2:
            raise NotImplementedError('To be continued')
        else:
            ori_input_shape = source_graph.get_blob(source_node.input(0)).shape
            # change single input elementwise into Scale
            # transfer info: input blob: 1, output blob: 1, node: 0, extra_blob: 0, extra_node: 1
            assert kw['mapper_dict'][source_node.op_type] != 'MAX', 'Caffe do not support MAX with constant'
            src_value = source_graph.get_parameters(source_node.params['Y'])
            caffe_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'Scale', attrs=dict())
            chn_num = ori_input_shape[1]
            if kw['mapper_dict'][source_node.op_type] == 'SUM':
                # 1 * input + src_value
                mapper_helper.creat_params(caffe_node, caffe_graph, 'weights', 1., chn_num)
                weight_name = caffe_node.name + '.weights'  # from creat_params()
                bias_name = source_node.params['Y']
                caffe_graph.parameters[bias_name] = src_value
            elif kw['mapper_dict'][source_node.op_type] == 'Sub':
                # 1 * input + -src_value
                mapper_helper.creat_params(caffe_node, caffe_graph, 'weights', 1., chn_num)
                weight_name = caffe_node.name + '.weights'  # from creat_params()
                bias_name = source_node.params['Y']
                caffe_graph.parameters[bias_name] = -src_value
            else:
                weight_name = source_node.params['Y']
                caffe_graph.parameters[weight_name] = src_value
                mapper_helper.creat_params(caffe_node, caffe_graph, 'bias', 0., chn_num)
                bias_name = caffe_node.name + '.bias'  # from creat_params()

            caffe_node.params = {
                'weights': weight_name,
                'bias': bias_name
            }

            return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('elementwise_div')
class ElementwiseDivOps:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        if len(source_node.params) == 0 and len(source_node.input_blobs_names) == 2:
            input_shape_x = source_graph.get_blob(source_node.input(0)).shape
            input_shape_y = source_graph.get_blob(source_node.input(1)).shape
            if len(input_shape_x) != 2 or len(input_shape_y) != 2:
                logging.warning('currently only support pending elementwise_div,'
                                ' which is common in SSD/YOLO/FasterRCNNs')
            # this node is a dummy node, normally will removed when processing SSD/YOLO/FasterRCNNs
            dummy_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'Eltwise', dict(), input_blob_num=2)
        else:
            raise NotImplementedError('To be continued')

        return OP_MAPPING_PENDDING, [dummy_node.name]