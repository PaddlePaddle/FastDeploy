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
import math
import collections

from paddle2caffe.constant.op_mapping_status import *
from paddle2caffe.op_mapper import OpMapper as op_mapper
from paddle2caffe.op_mapper import mapper_helper

from paddle2caffe.graph import Graph, Node
from paddle2caffe.graph.caffe_graph.graph_helper import compute_caffe_output_shape, get_symmetric_padding
from paddle2caffe.utils import logging

"""
    1. paddle weights参数排列为 output_chn, input_chn, kernel_h, kernel_w
    2. pool、conv、deconv在mapping过程中需视参数进行shape的对齐，补充额外的dummy layer
"""


@op_mapper(['conv2d', 'depthwise_conv2d'])
class Conv:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv2D_cn.html#conv2d"""
        # transfer paddle conv2d attr
        input_shape = source_graph.blob_map[source_node.input(0)].shape
        kernel_params_shape = source_graph.parameters[source_node.params['Filter']]['shape']
        kernel_shape = kernel_params_shape[-2:]
        num_output = kernel_params_shape[0]
        dilations = source_node.attr('dilations')
        strides = source_node.attr('strides')
        group = source_node.attr('groups')
        paddings = source_node.attr('paddings')
        padding_algorithm = source_node.attr('padding_algorithm')

        strides, dilations, paddings = mapper_helper.conv2d_attr_helper(
            strides, dilations, paddings, padding_algorithm, kernel_shape, input_shape)

        attrs = {
            'num_output': num_output,
            'group': group,
            # flowing attr need further process in emitter
            'strides': strides,
            'dilations': dilations,
            'paddings': paddings,
            'kernels': kernel_shape,
        }
        pad_h, pad_w, is_over_sized = get_symmetric_padding(strides, paddings, 'Convolution')
        new_paddings = [pad_h, pad_h, pad_w, pad_w]
        attrs['paddings'] = new_paddings
        caffe_ho, caffe_wo = compute_caffe_output_shape(input_shape, strides, paddings, kernel_shape,
                                                        dilations=dilations, op_type='Convolution')
        output_shape = source_graph.blob_map[source_node.output(0)].shape
        offset_h = caffe_ho - output_shape[2]
        offset_w = caffe_wo - output_shape[3]

        # in those cases do not need dummy fix
        if offset_h == offset_w == 0:
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
            conv_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'Convolution', attrs)
            mapper_helper.inherit_params(source_node, source_graph, 'Filter',
                                         conv_node, caffe_graph, 'weights', method='conv.weights')
            # append bias property
            if len(source_node.params) > 1:
                conv_node.attrs['bias_term'] = True
                raise NotImplementedError('what is conv2d bias key?')
            else:
                conv_node.attrs['bias_term'] = False
            return OP_MAPPING_IDENTITY, [conv_node.name]
        else:
            # use dummy to fix offset mismatch

            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 1, extra_node: 1
            input_blob_name = source_node.input_blobs_names[0]
            output_blob_name = source_node.output_blobs_names[0]
            node_name = source_node.name
            extra_blob_name = input_blob_name + '_dummy'
            extra_node_name = node_name + '_dummy'
            _ = source_node.output_blobs_names[0]

            # copy input connect from source graph
            _ = caffe_graph.transfer_op_input(source_node, source_graph)
            ori_input_shape = source_graph.blob_map[input_blob_name].shape
            ori_output_shape = source_graph.blob_map[output_blob_name].shape
            extra_output_shape = list(ori_output_shape)
            extra_output_shape[2] -= offset_h
            extra_output_shape[3] -= offset_w
            conv_node = caffe_graph.make_node('Convolution', source_node.raw_name, node_name,
                                              [input_blob_name], [extra_blob_name],
                                              attrs=attrs, do_insert=True)
            mapper_helper.inherit_params(source_node, source_graph, 'Filter',
                                         conv_node, caffe_graph, 'weights', method='conv.weights')
            # append bias property
            if len(source_node.params) > 1:
                conv_node.attrs['bias_term'] = True
                raise NotImplementedError('what is conv2d bias key?')
            else:
                conv_node.attrs['bias_term'] = False
            extra_blob = caffe_graph.make_blob(extra_output_shape, 'extra_blob_for_conv', extra_blob_name,
                                               node_name, [extra_node_name],
                                               domain=None, do_insert=True)
            kernel_shape = [ori_output_shape[1], ori_input_shape[1], kernel_shape[0], kernel_shape[1]]
            dummy_node = mapper_helper.creat_dummy_dwconv(caffe_graph, source_node, source_graph,
                                                          extra_blob_name, extra_node_name,
                                                          kernel_shape=kernel_shape, offset=[offset_h, offset_w],
                                                          is_right_bottom=not is_over_sized)

            return OP_MAPPING_WITH_EXTRA, [conv_node.name, dummy_node.name]


@op_mapper(['pool2d', 'max_pool2d_with_index'])
class Pool:
    have_contribute = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/pool2d_cn.html#pool2d"""
        # transfer paddle pool2d attr
        input_shape = source_graph.blob_map[source_node.input(0)].shape
        k_size = source_node.attr('ksize')
        strides = source_node.attr('strides')
        paddings = source_node.attr('paddings')
        padding_algorithm = source_node.attr('padding_algorithm')
        pooling_type = source_node.attr('pooling_type')

        global_pooling = source_node.attrs.get('global_pooling', False)
        ceil_mode = source_node.attrs.get('ceil_mode', True)
        exclusive = source_node.attrs.get('exclusive', False)
        if exclusive:
            logging.warning('caffe do not support exclusive param, will get ignored')
        # adaptive logic is referenced from XXX
        adaptive = source_node.attrs.get('adaptive', False)
        if adaptive:
            k_size[0] = int(input_shape[1] / k_size[0])
            k_size[1] = int(input_shape[2] / k_size[1])
            strides = k_size

        # fix kernel size if is over-sized
        if input_shape[2] > 0 and input_shape[2] + paddings[0] < k_size[0]:
            k_size[0] = input_shape[2] + paddings[0]
        if input_shape[3] > 0 and input_shape[3] + paddings[1] < k_size[1]:
            k_size[1] = input_shape[3] + paddings[1]
        if k_size[0] >= input_shape[2] and k_size[1] >= input_shape[3]:
            global_pooling = True

        strides, k_size, paddings = mapper_helper.pool2d_attr_helper(
            strides, paddings, padding_algorithm, k_size, input_shape)

        pool_attrs = {
            'strides': strides,
            'paddings': paddings,
            'kernels': k_size,
            'pooling_type': pooling_type,
            'global_pooling': global_pooling,
            'ceil_mode': ceil_mode,
        }

        caffe_ho, caffe_wo = compute_caffe_output_shape(input_shape, strides, paddings, k_size,
                                                        op_type='Pooling')
        output_shape = source_graph.blob_map[source_node.output(0)].shape
        offset_h = caffe_ho - output_shape[2]
        offset_w = caffe_wo - output_shape[3]

        # in those cases do not need dummy fix
        if offset_h == offset_w == 0 or global_pooling is True:
            pool_attrs.pop('ceil_mode')
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
            pool_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'Pooling', pool_attrs)
            return OP_MAPPING_IDENTITY, [pool_node.name]

        # use dummy to fix offset mismatch
        pool_attrs.pop('ceil_mode')  # remove ceil_mode
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 1, extra_node: 1
        input_blob_name = source_node.input_blobs_names[0]
        node_name = source_node.name
        extra_blob_name = input_blob_name + '_dummy'
        extra_node_name = node_name + '_dummy'
        _ = source_node.output_blobs_names[0]

        # copy input connect from source graph
        _ = caffe_graph.transfer_op_input(source_node, source_graph)
        ori_input_shape = source_graph.blob_map[input_blob_name].shape
        pool_node = caffe_graph.make_node('Pooling', source_node.raw_name, node_name,
                                          [input_blob_name], [extra_blob_name],
                                          attrs=pool_attrs, do_insert=True)
        extra_blob = caffe_graph.make_blob(ori_input_shape, 'extra_blob_for_batch_norm', extra_blob_name,
                                           node_name, [extra_node_name],
                                           domain=None, do_insert=True)
        kernel_shape = [ori_input_shape[1], ori_input_shape[1], k_size[0], k_size[1]]  # pooling inout have same chn
        dummy_node = mapper_helper.creat_dummy_dwconv(caffe_graph, source_node, source_graph,
                                                      extra_blob_name, extra_node_name,
                                                      kernel_shape=kernel_shape, offset=[offset_h, offset_w])
        return OP_MAPPING_WITH_EXTRA, [pool_node.name, dummy_node.name]


@op_mapper('batch_norm')
class BatchNorm:
    have_contribute = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm_cn.html#paddle.nn.BatchNorm"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 1, extra_node: 1
        # before: input  -->   batch_norm  -->  output
        # after:  input  -->   BatchNorm  --> new_blob -->  Scale  -->  output
        input_blob_name = source_node.input_blobs_names[0]
        node_name = source_node.name
        output_blob_name = source_node.output_blobs_names[0]
        extra_blob_name = output_blob_name + '_extra'
        extra_node_name = node_name + '_extra'

        # copy input connect from source graph
        _ = caffe_graph.transfer_op_input(source_node, source_graph, 1)
        ori_input_shape = source_graph.blob_map[input_blob_name].shape

        attrs_bn = {'eps': source_node.attr('epsilon')}
        attrs_scale = {'bias_term': True, 'num_axes': 1}
        caffe_node = caffe_graph.make_node('BatchNorm', source_node.raw_name, node_name,
                                           [input_blob_name], [extra_blob_name],
                                           attrs=attrs_bn, do_insert=True)
        extra_blob = caffe_graph.make_blob(ori_input_shape, 'extra_blob_for_batch_norm', extra_blob_name,
                                           node_name, [extra_node_name],
                                           domain=None, do_insert=True)
        extra_node = caffe_graph.make_node('Scale', 'extra_node_for_batch_norm', extra_node_name,
                                           [extra_blob], [output_blob_name],
                                           attrs=attrs_scale, do_insert=True)
        # inherit fluid params into BN and Scale
        mapper_helper.inherit_params(source_node, source_graph, 'Mean', caffe_node, caffe_graph, 'mean')
        mapper_helper.inherit_params(source_node, source_graph, 'Variance', caffe_node, caffe_graph, 'variance')
        mapper_helper.inherit_params(source_node, source_graph, 'Scale', extra_node, caffe_graph, 'scale')
        mapper_helper.inherit_params(source_node, source_graph, 'Bias', extra_node, caffe_graph, 'bias')

        # connect output_blob with extra_node
        assert output_blob_name not in caffe_graph.blob_map.keys()
        output_blob = source_graph.get_blob(output_blob_name, do_copy=True)
        output_blob.src_node_name = extra_node
        output_blob.dst_nodes_names = []  # cut out source_graph output connection which may change in later transfer
        caffe_graph.blob_map[output_blob_name] = output_blob

        return OP_MAPPING_WITH_EXTRA, [caffe_node.name, extra_node.name]


@op_mapper('norm')
class Normalize:
    have_contribute = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """API link unknown"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        if source_node.attr('epsilon') is not None:
            logging.warning('\tcaffe Normalize do not support epsilon params, will get ignored')
        attrs = {'across_spatial': False,
                 'channel_shared': False}
        caffe_node = mapper_helper.convert_identity_operation(
            caffe_graph, source_node, source_graph, 'Normalize', attrs)
        if len(source_node.params) > 0:
            raise NotImplementedError

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('dropout')
class Dropout:
    have_contribute = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Dropout_cn.html#dropout"""
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        attrs = {'dropout_ratio': source_node.attrs.get('dropout_prob', 0.5)}
        if source_node.attrs.get('axis', None) is not None:
            logging.warning('caffe do not support axis in dropout, will get ignore')
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        caffe_node = mapper_helper.convert_identity_operation(
            caffe_graph, source_node, source_graph, 'Dropout', attrs)

        return OP_MAPPING_IDENTITY, [caffe_node.name]
