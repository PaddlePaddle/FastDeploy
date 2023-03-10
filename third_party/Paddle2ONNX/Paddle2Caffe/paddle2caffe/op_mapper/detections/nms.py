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

from paddle2caffe.graph import Graph, Node
from paddle2caffe.op_mapper import mapper_helper
from .ssd_bbox import mapping_nms_ssd
from .yolo_bbox import mapping_nms_yolo

"""
_transfer_dict:
    'state': OP_MAPPING_XXX
    'info': extra info
_network_helper:
    'SSD': details in ssd_bbox
    'YOLO': details in yolo_bbox
"""


@op_mapper(['multiclass_nms', 'multiclass_nms2', 'multiclass_nms3'])
class MulticlassNMS:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        if caffe_graph._network_helper.get('SSD') is not None:
            return mapping_nms_ssd(caffe_graph, source_node, source_graph)
        elif caffe_graph._network_helper.get('YOLO') is not None:
            return mapping_nms_yolo(caffe_graph, source_node, source_graph)
        else:
            raise NotImplementedError('currently only network with priorbox(SSD) or yolobox(YOLO) '
                                      'can convert multiclass_nms(as detection/yolodetection')
