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
import logging

import paddle.fluid.core as core
import numpy as np
import copy

from paddle2caffe.graph import Graph, Node
from paddle2caffe.constant import dtypes


def _calculate_pads_if_algorithm_same(input_shape, strides, kernel_shape):
    """
    # reference for Function UpdatePaddingAndDilation in：
    # https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/conv_op.h
    """
    _, in_c, in_h, in_w = input_shape

    out_h = (in_h + strides[0] - 1) // strides[0]
    out_w = (in_w + strides[1] - 1) // strides[1]

    k_size_h = kernel_shape[0]
    k_size_w = kernel_shape[1]

    pad_h_sum = max((out_h - 1) * strides[0] + k_size_h - in_h, 0)
    pad_h_0 = pad_h_sum // 2
    pad_h_1 = pad_h_sum - pad_h_0
    pad_w_sum = max((out_w - 1) * strides[1] + k_size_w - in_w, 0)
    pad_w_0 = pad_w_sum // 2
    pad_w_1 = pad_w_sum - pad_w_0

    pads = [pad_h_0, pad_h_1, pad_w_0, pad_w_1]
    return pads


def inherit_params(source_node: Node, source_graph: Graph, source_key,
                   self_node: Node, self_graph: Graph, self_key,
                   method='default'):
    name = source_node.params[source_key]
    value = source_graph.get_parameters(name)
    if method == 'conv.weights':
        # change order from (OC, IC, H, W) into XXX
        assert len(value.shape) == 4
    else:
        if method != 'default':
            logging.warning('not default inherit method:{} but get default process'.format(method))
    self_node.params[self_key] = name
    self_graph.parameters[name] = value


def creat_params(source_node: Node, source_graph: Graph,
                 param_key, param_value, tensor_shape):
    weights = np.full(tensor_shape, param_value, dtype=np.float32)
    weights_name = source_node.name + '.' + param_key
    source_node.params[param_key] = weights_name
    source_graph.parameters[weights_name] = weights


def convert_identity_operation(caffe_graph: Graph, source_node: Node, source_graph: Graph,
                               caffe_type, attrs=None, input_blob_num=1, output_blob_num=1) -> Node:
    """
    transfer identity node (transfer process will maintain the same uname for node and blob)
    when to use convert_identity_operation:
        trans all(input blob, output blob, node according to their uname)
    when to use transfer_op_input:
        when you trans only input connection(input blob, node according to their uname)
    when to use transfer_op_output:
        when you trans only output connection(input blob, node according to their uname)
    ** notice that there is a difference between trans connection input and connection output **
    """
    node_name = source_node.name

    # copy node input connection from source graph
    input_blob_name_list = caffe_graph.transfer_op_input(source_node, source_graph, input_blob_num)
    # copy node output connection from source graph
    output_blob_name_list = caffe_graph.transfer_op_output(source_node, source_graph, output_blob_num)

    caffe_node = caffe_graph.make_node(caffe_type, source_node.raw_name, node_name,
                                       input_blob_name_list, output_blob_name_list,
                                       attrs=attrs, do_insert=True)

    # params transfer need be done manually
    # caffe_node.params = source_node.params

    return caffe_node


def creat_dummy_dwconv(caffe_graph: Graph, source_node: Node, source_graph: Graph,
                       dummy_blob_name, dummy_node_name,
                       kernel_shape, offset, is_right_bottom=True) -> Node:
    """creat_dummy_dwconv for pool/conv/deconv shape mismatch in tail from source_node"""

    node_name = dummy_node_name
    output_chn, input_chn, _, _ = kernel_shape
    offset_h, offset_w = offset
    # several attr is fixed
    dw_conv_attrs = {
        'num_output': output_chn,
        'group': output_chn,
        'strides': [1, 1],
        'paddings': [0, 0, 0, 0],
        'kernels': [offset_h + 1, offset_w + 1],
        'dilations': [1, 1],
        'bias_term': False
    }

    input_blob_name = dummy_blob_name
    # connect dummy node with dummy_blob as its input
    assert input_blob_name in caffe_graph.blob_map.keys(), '{} not found in blob'.format(input_blob_name)
    if node_name not in caffe_graph.blob_map[input_blob_name].dst_nodes_names:
        caffe_graph.blob_map[input_blob_name].dst_nodes_names.append(node_name)
    input_blob_name_list = [input_blob_name]
    # copy node output connection from source graph
    output_blob_name_list = caffe_graph.transfer_op_output(source_node, source_graph)

    caffe_node = caffe_graph.make_node("Convolution", source_node.raw_name + '_dummy', node_name,
                                       input_blob_name_list, output_blob_name_list,
                                       attrs=dw_conv_attrs, do_insert=True)

    # creat params for dummy node
    dummy_kernel_shape = (output_chn, 1, offset_h + 1, offset_w + 1)
    weights = np.zeros(dummy_kernel_shape)
    if is_right_bottom:  # pool/conv mismatch offset always happen in bottom + right sides
        weights[:, :, 0, 0] = 1.
    else:
        weights[:, :, -1, -1] = 1.
    weights_name = node_name + '.w_0_dummy'
    caffe_node.params['weights'] = weights_name
    caffe_graph.parameters[weights_name] = weights

    return caffe_node


def conv2d_attr_helper(strides, dilation, padding, padding_algorithm, kernel_shape=None, input_shape=None):
    """
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv2D_cn.html#conv2d
    strides -> [stride_h, stride_w]
    dilation -> [dilation_h, dilation_w]
    padding -> [padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]
    """
    # processing strides
    if isinstance(strides, int):
        strides = [strides, strides]
    elif isinstance(strides, list) or isinstance(strides, tuple):
        assert len(strides) == 2
    else:
        raise ValueError("可以为单个整数或包含两个整数的元组或列表，分别表示卷积沿着高和宽的步长。"
                         "如果为单个整数，表示沿着高和宽的步长都等于该整数")
    # processing dilation
    if isinstance(dilation, int):
        dilation = [dilation, dilation]
    elif isinstance(dilation, list) or isinstance(dilation, tuple):
        assert len(dilation) == 2
    else:
        raise ValueError("可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。"
                         "如果为单个整数，表示高和宽的空洞都等于该整数")
    # processing padding
    if padding_algorithm == 'SAME':
        # need to use input_shape to calculate padding
        assert isinstance(input_shape, tuple) and len(input_shape) == 4
        assert kernel_shape is not None
        padding = _calculate_pads_if_algorithm_same(input_shape, strides, kernel_shape)
    elif padding_algorithm == 'VALID':
        padding = [0, 0, 0, 0]
    else:
        # padding_algorithm = 'EXPLICIT'
        if isinstance(padding, int):
            padding = [padding] * 4
        elif isinstance(padding, list) or isinstance(padding, tuple) and len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]
        elif isinstance(padding, list) or isinstance(padding, tuple) and len(padding) == 4:
            pass
        else:
            raise NotImplementedError('unknown padding attr')

    return strides, dilation, padding


def pool2d_attr_helper(strides, padding, padding_algorithm, kernel_size, input_shape=None):
    """
    https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/pool2d_cn.html#pool2d
    strides -> [stride_h, stride_w]
    kernel_size -> [kernel_h, kernel_w]
    padding -> [padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]
    """
    # processing strides
    if isinstance(strides, int):
        strides = [strides, strides]
    elif isinstance(strides, list) or isinstance(strides, tuple):
        assert len(strides) == 2
    else:
        raise ValueError("可以为单个整数或包含两个整数的元组或列表，分别表示卷积沿着高和宽的步长。"
                         "如果为单个整数，表示沿着高和宽的步长都等于该整数")

    # processing kernel_size
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2
    else:
        raise ValueError("如果它是一个元组或列表，那么它包含两个整数值：(pool_size_Height, pool_size_Width)。"
                         "若为一个整数，则表示H和W维度上均为该值")

    # processing padding
    if padding_algorithm == 'SAME':
        # need to use input_shape to calculate padding
        assert isinstance(input_shape, tuple) and len(input_shape) == 4
        padding = _calculate_pads_if_algorithm_same(input_shape, strides, kernel_size)
    elif padding_algorithm == 'VALID':
        padding = [0, 0, 0, 0]
    elif padding_algorithm == 'adaptive':
        raise NotImplementedError('TODO')
    else:
        padding_algorithm = 'EXPLICIT'
        if isinstance(padding, int):
            padding = [padding] * 4
        elif isinstance(padding, list) or isinstance(padding, tuple) and len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]
        elif isinstance(padding, list) or isinstance(padding, tuple) and len(padding) == 4:
            pass
        else:
            raise NotImplementedError('unknown padding attr')

    return strides, kernel_size, padding


def resize_shape_helper(input_shape, output_shape, size, scales, align_special=False):
    """
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/interpolate_cn.html#interpolate
    input_shape -> [in_h, in_w]
    output_shape -> [out_h, out_w]
    size -> [size_h, size_w] or None
    scales -> [scale_h, scale_w] or None
    """
    in_h, in_w = input_shape
    out_h, out_w = output_shape

    # processing scales
    is_scale_valid = True
    if isinstance(scales, int) or isinstance(scales, float):
        if scales <= 1e-5:
            is_scale_valid = False
        else:
            scales = [scales, scales]
    elif isinstance(scales, list) or isinstance(scales, tuple):
        assert len(scales) == 2
        if scales[0] <= 1e-5 or scales[1]:
            is_scale_valid = False
    elif scales is None:
        is_scale_valid = False
    else:
        raise ValueError("输入的高度或宽度的乘数因子。如果scale_factor是一个list或tuple，它必须与输入的shape匹配")
    if not is_scale_valid:
        scales = [out_h / in_h, out_w / in_w]
    # change scales into int if can be div in integer
    if out_h % in_h == 0:
        scales[0] = int(scales[0])
    if out_w % in_w == 0:
        scales[1] = int(scales[1])

    # processing size
    is_size_valid = True
    if isinstance(size, int):
        if size <= 1e-5:
            is_size_valid = False
        else:
            size = [size, size]
    elif isinstance(size, list) or isinstance(size, tuple):
        assert len(size) == 2
        if size[0] <= 1e-5 or size[1] <= 1e-5:
            is_size_valid = False
    elif size is None:
        is_size_valid = False
    else:
        raise ValueError("输入为4D张量时，形状为为(out_h, out_w)的2-D Tensor")
    if not is_size_valid:
        size = [out_h, out_w]
    if align_special is True:
        # can not get integer scale, so use size instead
        assert size[0] == out_h
        assert size[1] == out_w

    return size, scales
