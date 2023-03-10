#   Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import math
import numpy as np

from paddle2caffe.utils import logging
# from paddle2caffe.graph.caffegraph.shape_helper import *
from paddle2caffe.caffe_helper.caffe_pb2 import *

layer_id_to_info = {
    0: {'type': 'None', 'attr': None},
    1: {'type': 'Accuracy', 'attr': AccuracyParameter},
    2: {'type': 'BNLL', 'attr': None},
    3: {'type': 'Concat', 'attr': ConcatParameter},
    4: {'type': 'Convolution', 'attr': ConvolutionParameter},
    5: {'type': 'Data', 'attr': DataParameter},
    6: {'type': 'Dropout', 'attr': DropoutParameter},
    7: {'type': 'EuclideanLoss', 'attr': None},
    8: {'type': 'Flatten', 'attr': FlattenParameter},
    9: {'type': 'HDF5Data', 'attr': HDF5DataParameter},
    10: {'type': 'HDF5Output', 'attr': HDF5OutputParameter},
    11: {'type': 'Im2col', 'attr': None},
    12: {'type': 'ImageData', 'attr': ImageDataParameter},
    13: {'type': 'InfogainLoss', 'attr': None},
    14: {'type': 'InnerProduct', 'attr': InnerProductParameter},
    15: {'type': 'LRN', 'attr': LRNParameter},
    16: {'type': 'MultinomialLogisticLoss', 'attr': None},
    17: {'type': 'Pooling', 'attr': PoolingParameter},
    18: {'type': 'ReLU', 'attr': ReLUParameter},
    19: {'type': 'Sigmoid', 'attr': SigmoidParameter},
    20: {'type': 'Softmax', 'attr': SoftmaxParameter},
    21: {'type': 'SoftmaxWithLoss', 'attr': None},
    22: {'type': 'Split', 'attr': None},
    23: {'type': 'TanH', 'attr': TanHParameter},
    24: {'type': 'WindowData', 'attr': WindowDataParameter},
    25: {'type': 'Eltwise', 'attr': EltwiseParameter},
    26: {'type': 'Power', 'attr': PowerParameter},
    27: {'type': 'SigmoidCrossEntropyLoss', 'attr': None},
    28: {'type': 'HingeLoss', 'attr': None},
    29: {'type': 'MemoryData', 'attr': MemoryDataParameter},
    30: {'type': 'ArgMax', 'attr': ArgMaxParameter},
    31: {'type': 'Threshold', 'attr': ThresholdParameter},
    32: {'type': 'DummyData', 'attr': DummyDataParameter},
    33: {'type': 'Slice', 'attr': SliceParameter},
    34: {'type': 'MVN', 'attr': MVNParameter},
    35: {'type': 'AbsVal', 'attr': None},
    36: {'type': 'Silence', 'attr': None},
    37: {'type': 'ContrastiveLoss', 'attr': None},
    38: {'type': 'Exp', 'attr': ExpParameter},
    39: {'type': 'BatchNorm', 'attr': BatchNormParameter},
    40: {'type': 'Scale', 'attr': ScaleParameter},
    41: {'type': 'ReLU', 'attr': ReLUParameter},
    42: {'type': 'Pooling', 'attr': PoolingParameter},
    43: {'type': 'Add', 'attr': None},
    44: {'type': 'MatMul', 'attr': MatMulParameter},
    45: {'type': 'Clip', 'attr': ClipParameter},
    46: {'type': 'Reshape', 'attr': ReshapeParameter},
    47: {'type': 'PReLU', 'attr': PReLUParameter},

    # following appears only in custom caffe
    100: {'type': 'Deconvolution', 'attr': ConvolutionParameter},
    101: {'type': 'ELU', 'attr': ELUParameter},
    102: {'type': 'Axpy', 'attr': AxpyParameter},
    103: {'type': 'Upsample', 'attr': UpsampleParameter},
    104: {'type': 'Interp', 'attr': InnerProductParameter},
    105: {'type': 'ReLU6', 'attr': ReLU6Parameter},
    106: {'type': 'Permute', 'attr': PermuteParameter},
    107: {'type': 'PriorBox', 'attr': PriorBoxParameter},
    108: {'type': 'DetectionOutput', 'attr': DetectionOutputParameter},
    109: {'type': 'Normalize', 'attr': NormalizeParameter},
    110: {'type': 'Yolov3DetectionOutput', 'attr': Yolov3DetectionOutputParameter},
}


LAYER_TYPES = [info['type'] for info in layer_id_to_info.values()]
LayerType = type('LayerType', (), {t: t for t in LAYER_TYPES})
# KernelParameters = namedtuple('KernelParameters', ['global_pooling', 'k_h', 'k_w', 's_h', 's_w', 'p_h', 'p_w'])


class NodeMap(LayerType):

    @staticmethod
    def map_raw_type(node_kind):
        if isinstance(node_kind, int):
            node_kind = layer_id_to_info[node_kind]['type']
        else:
            node_kind = str(node_kind)
        if node_kind in LAYER_TYPES:
            return node_kind
        return None

    @staticmethod
    def map_raw_attr(node_kind):
        return layer_id_to_info[node_kind]['attr']


def get_symmetric_padding(strides, paddings, op_type):
    """

    :param strides: [stride_h, stride_w]
    :param paddings: [padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]
    :param op_type:
    :return:
    """
    stride_h = strides[0]
    stride_w = strides[1]

    is_over_sized = False

    if paddings[0] != paddings[1] or paddings[2] != paddings[3]:
        logging.warning('unequal padding begin and end which caffe do not support'
                        ', caffe will modify padding to close this different')
        is_over_sized = True

    if op_type == "Unpool":  # TODO 待触发
        pad_h = 0
        pad_w = 0
    else:
        # sometimes happened while padding is SAME
        # this over size will cause to use dummy with left and top type
        pad_h = paddings[0] + (0 if paddings[0] == paddings[1] else stride_h)
        pad_w = paddings[2] + (0 if paddings[2] == paddings[3] else stride_w)

    return [pad_h, pad_w, is_over_sized]


def compute_caffe_output_shape(input_shape, strides, paddings, kernels,
                               dilations=(1, 1), op_type='Pooling', ceil_mode=True):
    # TODO 补充dilation
    assert len(input_shape) == 4
    h_i, w_i = input_shape[2:4]
    pad_h, pad_w, _ = get_symmetric_padding(strides, paddings, op_type)
    stride_h, stride_w = strides
    kernel_h, kernel_w = kernels
    dilation_h, dilation_w = dilations

    if op_type == 'Pooling':
        if ceil_mode:
            h_o = math.ceil((h_i + 2 * pad_h - kernel_h) / stride_h) + 1
            w_o = math.ceil((w_i + 2 * pad_w - kernel_w) / stride_w) + 1
        else:
            h_o = math.floor((h_i + 2 * pad_h - kernel_h) / stride_h) + 1
            w_o = math.floor((w_i + 2 * pad_w - kernel_w) / stride_w) + 1
        # this is interesting
        if pad_h + pad_w > 0:
            h_o -= 1 if (h_o - 1) * stride_h >= h_i + pad_h else 0
            w_o -= 1 if (w_o - 1) * stride_w >= w_i + pad_w else 0

    elif op_type == 'Convolution':
        k_h_ext = dilation_h * (kernel_h - 1) + 1
        k_w_ext = dilation_w * (kernel_w - 1) + 1
        h_o = int((h_i + 2 * pad_h - k_h_ext) / stride_h + 1)
        w_o = int((w_i + 2 * pad_w - k_w_ext) / stride_w + 1)

    elif op_type == 'Unpool':
        h_o = (h_i - 2 * pad_h - kernel_h + stride_h) * stride_h
        w_o = (w_i - 2 * pad_w - kernel_w + stride_w) * stride_w

    else:
        logging.warning('unknown op type:{}, use default compute'.format(op_type))
        h_o = int((h_i + 2 * pad_h - kernel_h) / stride_h + 1)
        w_o = int((w_i + 2 * pad_w - kernel_w) / stride_w + 1)

    return h_o, w_o
