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

"""
yolo_process use caffe_graph._network_helper['YOLO'] to share process across different op_mappers

caffe_graph._network_helper['YOLO']:
    class_num: int, e.g. 80 for COCO
    clear_input: bool, only need clear input info once
    scales_list: [32, 16, ...], also known as downsample_ratio in paddle
    anchors_all_list: [anchors_list_each, ...]
    sample_node_name_list: [node_name, ...], with node that generate yolo_box layer and in caffe_graph

1. all list above idx is corresponded
2. class_num must equal between all yolo box layer
"""
from __future__ import absolute_import

import numpy as np

from paddle2caffe.constant.op_mapping_status import *
from paddle2caffe.op_mapper import OpMapper as op_mapper
from paddle2caffe.op_mapper import mapper_helper

from paddle2caffe.graph import Graph, Node
from paddle2caffe.utils import logging


def init_yolo_conf():
    yolo_cfg = dict()
    yolo_cfg['class_num'] = 0
    yolo_cfg['clear_input'] = False
    yolo_cfg['scales_list'] = list()
    yolo_cfg['anchors_all_list'] = list()
    yolo_cfg['sample_node_name_list'] = list()
    return yolo_cfg


@op_mapper('yolo_box')
class YoloBox:
    have_custom = True

    @classmethod
    def _clear_node_bottomup(cls, start_blob_name, caffe_graph):
        """ e.g.
        img_scale   img_shape
            |        |
           element_div
                |
              cast
        """
        max_process_node = 10   # keep while loop safe
        node_name_list = [start_blob_name.src_node_name]
        processd_node_num = 0
        while len(node_name_list) > 0 and processd_node_num < max_process_node:
            cur_node_name = node_name_list.pop()
            processd_node_num += 1
            if caffe_graph.node_map.get(cur_node_name) is not None:
                # may already deleted in formal mapping
                cur_node: Node = caffe_graph.node_map[cur_node_name]
                caffe_graph.remove_node(cur_node_name)
                logging.info('\tdelete node:{}, which is not needed in caffe Yolo'.format(cur_node_name))
                is_input = False
                for blob_name in cur_node.input_blobs_names:
                    # already locate input blob
                    for node in caffe_graph.input_nodes:
                        if node.output(0) == blob_name:
                            caffe_graph.input_nodes.remove(node)
                            is_input = True
                            break
                    # if blob in graph
                    if not is_input and caffe_graph.blob_map.get(blob_name) is not None:
                        blob = caffe_graph.blob_map[blob_name]
                        caffe_graph.remove_blob(blob_name)
                        node_name_list += [blob.src_node_name]

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # [attention] this mapping load yolo config into _network_helper and amend ops between yolo box and nms
        # change : image + sample node -> yolo -> transpose/concat -> (yolo detection later)
        # into   : sample node -> (yolo detection later)
        img_size_tensor = source_graph.blob_map[source_node.input(0)]    # input_0 should be net input or input+math
        input_tensor = source_graph.blob_map[source_node.input(1)]
        scale_x_y = source_node.attr('scale_x_y')
        if scale_x_y is not None and scale_x_y != 1:
            raise ValueError('caffe yolo do not support scale setting')
        # get node attr
        anchors = source_node.attr('anchors')
        class_num = source_node.attr('class_num')
        downsample_ratio = source_node.attr('downsample_ratio')

        # process _network_helper
        if caffe_graph._network_helper.get('YOLO') is None:
            caffe_graph._network_helper['YOLO'] = init_yolo_conf()
        yolo_config = caffe_graph._network_helper['YOLO']

        if yolo_config['class_num'] != 0 and yolo_config['class_num'] != class_num:
            raise ValueError('class_num must equal between all yolo box layer')
        else:
            yolo_config['class_num'] = class_num
        yolo_config['scales_list'].append(downsample_ratio)
        yolo_config['anchors_all_list'].append(anchors)
        yolo_config['sample_node_name_list'].append(input_tensor.src_node_name)

        # make ops between yolo box and nms skipped(currently process transpose and concat)
        for node in source_graph.get_successor_nodes_with_node(source_node):
            caffe_graph._transfer_dict[node.name]['state'] = OP_MAPPING_SKIPPED
            caffe_graph._transfer_dict[node.name]['info'] = source_node.name
            if node.op_type in ['transpose', 'transpose2']:
                for son_node in source_graph.get_successor_nodes_with_node(node):
                    caffe_graph._transfer_dict[son_node.name]['state'] = OP_MAPPING_SKIPPED
                    caffe_graph._transfer_dict[son_node.name]['info'] = source_node.name

        # remove img_size_tensor path, this process just need do once
        if yolo_config['clear_input'] is False:
            logging.info('clear yolo box ImgSize path since caffe do not need')
            caffe_input_name_list = [node.name for node in caffe_graph.input_nodes]
            if str(img_size_tensor.src_node_name) in caffe_input_name_list:
                # img_size_tensor is input, just remove input node
                for node in caffe_graph.input_nodes:
                    if node.name == img_size_tensor.src_node_name:
                        caffe_graph.input_nodes.remove(node)
            else:
                # img_size_tensor is from node, need to remove bottom up to input img_size
                cls._clear_node_bottomup(img_size_tensor, caffe_graph)
            yolo_config['clear_input'] = True

        return OP_MAPPING_PENDDING, [source_node.name]


def mapping_nms_yolo(caffe_graph: Graph, source_node: Node, source_graph: Graph):
    yolo_config = caffe_graph._network_helper['YOLO']
    branch_num = len(yolo_config['scales_list'])
    assert len(yolo_config['anchors_all_list']) == len(yolo_config['sample_node_name_list']) == branch_num

    # get nms params
    background_label = source_node.attr('background_label')
    if background_label is not None and background_label != -1 and \
            background_label != yolo_config['class_num']:
        logging.warning('background_label != class_num, caffe yolo detection do not support, may cause problem')
    nms_threshold = source_node.attr('nms_threshold')
    score_threshold = source_node.attr('score_threshold')

    # init yolo detection output attrs
    biases, mask = list(), list()
    mask_anchor_num = 0
    for idx in range(branch_num):
        biases += yolo_config['anchors_all_list'][idx]
        assert len(yolo_config['anchors_all_list'][idx]) % 2 == 0, 'anchors should keep in pair'
        mask_anchor_num += len(yolo_config['anchors_all_list'][idx]) // 2
    # mask: stands for anchors merged order, if is in order, value just like 0,1,2,3...
    mask = [idx for idx in range(mask_anchor_num)]
    detection_attrs = {
        'anchors_scale': yolo_config['scales_list'],
        'biases': biases,
        'mask': mask,
        'mask_group_num': branch_num,
        'nms_threshold': nms_threshold,
        'num_classes': yolo_config['class_num'],
        'confidence_threshold': score_threshold
    }

    # connection input, in sample_node_name_list order
    input_blob_name_list = []
    for node_name in yolo_config['sample_node_name_list']:
        if len(caffe_graph.node_map[node_name].output_blobs_names) > 1:
            logging.warning('yolo box node should not have no brother node')
        input_blob_name = caffe_graph.node_map[node_name].output(0)
        input_blob_name_list.append(input_blob_name)
        input_blob = caffe_graph.blob_map[input_blob_name]
        if source_node.name not in input_blob.dst_nodes_names:
            input_blob.dst_nodes_names.append(source_node.name)
    # connection output and creat node
    output_blob_name_list = caffe_graph.transfer_op_output(source_node, source_graph)
    caffe_node = caffe_graph.make_node('Yolov3DetectionOutput', source_node.raw_name, source_node.name,
                                       input_blob_name_list, output_blob_name_list,
                                       attrs=detection_attrs, do_insert=True)

    return OP_MAPPING_WITH_FUSED, [caffe_node.name]
