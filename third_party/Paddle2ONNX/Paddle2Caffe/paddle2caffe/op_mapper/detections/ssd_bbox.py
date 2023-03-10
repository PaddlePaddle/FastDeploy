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
ssd_process use caffe_graph._network_helper['SSD'] to share process across different op_mappers

caffe_graph._network_helper['SSD']:
    min_max_aspect_ratios_order: Bool, which stands for
        [min, ..., aspect_ratios_n, ... max] or [min, max, ..., aspect_ratios_n, ...]
    sample_node_name_list: [node_name, ...], with node that generate (priorbox + conf + loc) and in caffe_graph
    sub_group_list: list of dict as follow (idx is the same as sample_node_name_list)
        {'conf': conf_node_name,
         'loc': loc_node_name,
         'priorbox': priorbox_node_name,
         'pbox_num': priorbox_num, used to tell conf/loc shape
         'have_max': bool, when priorbox have no max, also do not need chn-swap}

1. different sub_group may have the same sample_node, so sample node list may have duplicated value
2. sample_node_name_list and sub_group_list idx is corresponded since its created by priorbox mapper
"""
from __future__ import absolute_import

import numpy as np

from paddle2caffe.constant.op_mapping_status import *
from paddle2caffe.op_mapper import OpMapper as op_mapper
from paddle2caffe.op_mapper import mapper_helper

from paddle2caffe.graph import Graph, Node
from paddle2caffe.utils import logging


def find_conf_and_loc(caffe_graph, num_class):
    """
    find conf and loc node based on source node which located by prior box
        search every single-path road which begin with conv,
        until reached max_conv_depth or not single-path road
    """
    max_conv_depth = 2
    ssd_conf = caffe_graph._network_helper['SSD']
    sample_node_name_list = ssd_conf['sample_node_name_list']

    for idx, sample_node_name in enumerate(sample_node_name_list):
        son_nodes = caffe_graph.get_successor_nodes_with_node(sample_node_name)
        sub_config = ssd_conf['sub_group_list'][idx]
        pbox_num = sub_config['pbox_num']
        for begin_node in son_nodes:
            if begin_node.op_type == 'Convolution':
                # begin locate
                locate_success = False
                search_depth = 0
                cur_node = begin_node
                while locate_success is False and search_depth < max_conv_depth:
                    # logging.info(cur_node.output_blobs_names[0], cur_node.op_type)
                    # check node is single-path road
                    if len(cur_node.output_blobs_names) != 1 or \
                            len(caffe_graph.blob_map[cur_node.output(0)].dst_nodes_names) > 1:
                        logging.warning('locate SSD conf/loc path failed, not a single-path road')
                    # check cur_node is located or not
                    if cur_node.op_type == 'Convolution':
                        chn_num = caffe_graph.blob_map[cur_node.output(0)].shape[1]
                        if chn_num == pbox_num * 4:  # consider as loc node
                            assert sub_config.get('loc') is None, 'locate SSD loc node failed'
                            sub_config['loc'] = cur_node.name
                            locate_success = True
                        elif chn_num == pbox_num * num_class:  # consider as conf node
                            assert sub_config.get('conf') is None, 'locate SSD conf node failed'
                            sub_config['conf'] = cur_node.name
                            locate_success = True
                        search_depth += 1
                    cur_node = caffe_graph.get_successor_nodes_with_node(cur_node.name)[0]

        assert sub_config.get('conf') is not None and sub_config.get('loc') is not None


def swap_conf_and_loc(caffe_graph, num_class):
    """
    when min_max_aspect_ratios_order=False
    priorbox along with its conf and loc box order is [min, aspect_ratios, max]
    but caffe is fixed order [min, max, aspect_ratios]
    so we need to swap conf and loc params channel order to rematch
    """
    def _swap_params(params, pbox_num):
        ori_weight = params
        if len(params.shape) == 4:
            o_c, i_c, k_h, i_w = ori_weight.shape

            ori_weight_reshape = ori_weight.reshape((pbox_num, -1, i_c, k_h, i_w))
            min_part = ori_weight_reshape[0, :, :, :, :][np.newaxis, :]
            aspect_ratios_part = ori_weight_reshape[1:-1, :, :, :, :]
            max_part = ori_weight_reshape[-1, :, :, :, :][np.newaxis, :]

            new_weight = np.concatenate((min_part, max_part, aspect_ratios_part), axis=0)
            new_weight = new_weight.reshape((o_c, i_c, k_h, i_w))

        elif len(params.shape) == 1:
            # print(params)
            o_c = ori_weight.shape

            ori_weight_reshape = ori_weight.reshape((pbox_num, -1))
            min_part = ori_weight_reshape[0, :][np.newaxis, :]
            aspect_ratios_part = ori_weight_reshape[1:-1, :]
            max_part = ori_weight_reshape[-1, :][np.newaxis, :]

            new_weight = np.concatenate((min_part, max_part, aspect_ratios_part), axis=0)
            new_weight = new_weight.reshape(o_c)
        else:
            print(params.shape)
            raise NotImplementedError('Unknown params type')

        return new_weight

    ssd_conf = caffe_graph._network_helper['SSD']
    for idx, sample_node_name in enumerate(ssd_conf['sample_node_name_list']):
        sub_group_conf = ssd_conf['sub_group_list'][idx]
        assert sub_group_conf.get('conf') is not None and sub_group_conf.get('loc') is not None

        pbox_num = sub_group_conf['pbox_num']
        loc_node = caffe_graph.node_map[sub_group_conf['loc']]
        conf_node = caffe_graph.node_map[sub_group_conf['conf']]

        if sub_group_conf['have_max']:
            for node in [loc_node, conf_node]:
                # swap conv
                conv_node: Node = node
                # logging.info(conv_node.output(0))
                new_weight = _swap_params(caffe_graph.parameters[conv_node.params['weights']], pbox_num)
                caffe_graph.parameters[conv_node.params['weights']] = new_weight

                # swap son node if they are scale/batchnorm
                son_nodes = caffe_graph.get_successor_nodes_with_node(conv_node)
                if len(son_nodes) > 1:
                    logging.warning('SSD conf/loc node have multiply successor nodes')
                son_node = son_nodes[0]
                if son_node.op_type in ['BatchNorm', 'Scale']:
                    for bn_key in son_node.params.keys():
                        new_weight = _swap_params(caffe_graph.parameters[son_node.params[bn_key]], pbox_num)
                        caffe_graph.parameters[son_node.params[bn_key]] = new_weight


@op_mapper(['prior_box', 'density_prior_box'])
class PriorBox:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        input_img_shape = source_graph.blob_map[source_node.input(0)].shape    # input_0 should be net/image input
        input_stage_shape = source_graph.blob_map[source_node.input(1)].shape
        # get node attr
        pbox_attr = dict()
        pbox_attr['flip'] = source_node.attr('flip')
        pbox_attr['clip'] = source_node.attr('clip')
        pbox_attr['min_size'] = source_node.attr('min_sizes')
        pbox_attr['max_size'] = source_node.attr('max_sizes')
        pbox_attr['aspect_ratio'] = source_node.attr('aspect_ratios')
        pbox_attr['variance'] = source_node.attr('variances')

        if source_node.op_type == 'density_prior_box':
            pbox_attr['fixed_sizes'] = source_node.attr('fixed_sizes')
            pbox_attr['fixed_ratios'] = source_node.attr('fixed_ratios')
            pbox_attr['densities'] = source_node.attr('densities')

        # caffe aspect_ratio will automatic use 1.0 as default
        # need to remove while fluid node have this value
        if pbox_attr.get('aspect_ratio') is not None and 1.0 in pbox_attr['aspect_ratio']:
            pbox_attr['aspect_ratio'].remove(1.0)
        if source_node.attr('step_w') != 0:
            pbox_attr['step_w'] = source_node.attr('step_w')
        else:
            pbox_attr['step_w'] = float(input_img_shape[-1]) / input_stage_shape[-1]
        if source_node.attr('step_h') != 0:
            pbox_attr['step_h'] = source_node.attr('step_h')
        else:
            pbox_attr['step_h'] = float(input_img_shape[-2]) / input_stage_shape[-2]
        pbox_attr['offset'] = source_node.attr('offset')

        # process _network_helper
        # 1.min_max_aspect_ratios_order
        if caffe_graph._network_helper.get('SSD') is None:
            caffe_graph._network_helper['SSD'] = dict()
        graph_ssd_config = caffe_graph._network_helper['SSD']
        graph_ssd_config['min_max_aspect_ratios_order'] = \
            source_node.attrs.get('min_max_aspect_ratios_order', False)
        # 2.sample_node_name_list
        input_node_name = caffe_graph.blob_map[source_node.input(1)].src_node_name

        if graph_ssd_config.get('sample_node_name_list') is None:
            graph_ssd_config['sample_node_name_list'] = [input_node_name]
        else:
            graph_ssd_config['sample_node_name_list'].append(input_node_name)
        sub_group = dict()
        if graph_ssd_config.get('sub_group_list') is None:
            graph_ssd_config['sub_group_list'] = list()
        # 3.1 sub_group_list -- priorbox
        sub_group['priorbox'] = source_node.name
        # 3.2 sub_group_list -- pbox_num
        if pbox_attr.get('flip', False):
            pbox_num = len(pbox_attr['min_size']) + len(pbox_attr['max_size']) + 2 * len(pbox_attr['aspect_ratio'])
        else:
            pbox_num = len(pbox_attr['min_size']) + len(pbox_attr['max_size']) + len(pbox_attr['aspect_ratio'])
        sub_group['pbox_num'] = pbox_num
        sub_group['have_max'] = True if len(pbox_attr['max_size']) > 0 else False
        graph_ssd_config['sub_group_list'].append(sub_group)

        output_num = len(source_node.output_blobs_names)
        if output_num == 1:
            caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                                  caffe_type="PriorBox", attrs=pbox_attr,
                                                                  input_blob_num=2)
        else:
            # output including pbox and its variance, need skip variance since caffe is one output
            assert output_num == 2
            caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                                  caffe_type="PriorBox", attrs=pbox_attr,
                                                                  input_blob_num=2, output_blob_num=1)
            # fluid input order is opposite
            caffe_node.input_blobs_names.reverse()
            # skip paddle prior box variance path (flatten + concat)
            out_variance_node = source_graph.get_successor_nodes_with_node(source_node.name)[1]
            out_variance_concat_node = source_graph.get_successor_nodes_with_node(out_variance_node.name)[0]
            if out_variance_node.op_type not in ['flatten', 'flatten2'] \
                    and out_variance_concat_node.op_type != 'concat':
                logging.error('\t skip paddle prior box variance path failed, this may cause trouble')
            caffe_graph._transfer_dict[out_variance_node.name]['state'] = OP_MAPPING_SKIPPED
            caffe_graph._transfer_dict[out_variance_node.name]['info'] = caffe_node.name
            caffe_graph._transfer_dict[out_variance_concat_node.name]['state'] = OP_MAPPING_SKIPPED
            caffe_graph._transfer_dict[out_variance_concat_node.name]['info'] = caffe_node.name

        return OP_MAPPING_WITH_FUSED, [caffe_node.name]


@op_mapper('box_coder')
class BoxCoder:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # [attention] this mapping only process former caffe_graph nodes
        # change : prior box -> flatten -> concat
        # into   : prior box -> concat(axis=2), since caffe need to keep prior box shape
        priorbox_concat_blob_name = source_node.input(0)  # boxcoder input is in [pbox, (pbox_var), bbox] order
        concat_output_blob = caffe_graph.blob_map[priorbox_concat_blob_name]
        concat_node = caffe_graph.node_map[concat_output_blob.src_node_name]
        flatten_output_blobs = concat_node.input_blobs_names
        new_concat_input_blobs_new = []
        for flatten_name in flatten_output_blobs:
            flatten_out_blob = caffe_graph.blob_map[flatten_name]
            flatten_node = caffe_graph.node_map[flatten_out_blob.src_node_name]
            flatten_in_blob = caffe_graph.blob_map[flatten_node.input_blobs_names[0]]
            # flatten_in/out_blob should have only 1 output node, as a signal path sub-graph
            assert len(flatten_out_blob.dst_nodes_names) == 1 and len(flatten_in_blob.dst_nodes_names) == 1

            # re-connect flatten_in_blob to concat, delete flatten_out_blob + flatten_node
            new_concat_input_blobs_new.append(flatten_in_blob.name)
            flatten_in_blob.dst_nodes_name = [concat_node.name]

            caffe_graph.node_map.pop(flatten_node.name)
            caffe_graph.blob_map.pop(flatten_out_blob.name)
        # change concat node info
        concat_node.input_blobs_names = new_concat_input_blobs_new
        concat_node.attrs['axis'] = 2

        return OP_MAPPING_WITH_FUSED, []


def mapping_nms_ssd(caffe_graph: Graph, source_node: Node, source_graph: Graph):
    assert caffe_graph._network_helper.get('SSD') is not None
    ssd_conf = caffe_graph._network_helper['SSD']

    # box_coder do not exist in caffe_graph, connection is captured in source_graph
    box_coder_fake = source_graph.get_precursor_nodes_with_node(source_node)[0]
    assert box_coder_fake.op_type == 'box_coder', 'unrecognized ssd structure!'
    if len(box_coder_fake.input_blobs_names) == 3:
        # [pbox, (pbox_var), bbox]
        pbox_blob = caffe_graph.blob_map[box_coder_fake.input(0)]
        pbox_node = caffe_graph.node_map[pbox_blob.src_node_name]
        bbox_blob = caffe_graph.blob_map[box_coder_fake.input(2)]
        bbox_node = caffe_graph.node_map[bbox_blob.src_node_name]
    else:
        pbox_blob = caffe_graph.blob_map[box_coder_fake.input(0)]
        pbox_node = caffe_graph.node_map[pbox_blob.src_node_name]
        bbox_blob = caffe_graph.blob_map[box_coder_fake.input(1)]
        bbox_node = caffe_graph.node_map[bbox_blob.src_node_name]
    conf_blob = caffe_graph.blob_map[source_node.input(1)]
    conf_node = caffe_graph.node_map[conf_blob.src_node_name]

    # step1. for conf_node, get class_num, change final transpose -> flatten
    conf_father_node = caffe_graph.get_precursor_nodes_with_node(conf_node)[0]
    while conf_father_node.op_type != "Softmax":
        conf_father_node = caffe_graph.get_precursor_nodes_with_node(conf_father_node)[0]
    softmax_input_shape = caffe_graph.blob_map[conf_father_node.input(0)].shape
    class_num = softmax_input_shape[conf_father_node.attr('axis')]
    if source_node.attr('num_classes') is not None:
        class_num = source_node.attr('num_classes')

    if conf_node.op_type == 'Permute':
        conf_node.op_type = 'Flatten'
        conf_node.attrs = {'axis': 1}
        ori_shape = caffe_graph.blob_map[conf_node.output(0)].shape
        new_shape = [ori_shape[0], np.prod(ori_shape) / ori_shape]
        caffe_graph.blob_map[conf_node.output(0)].shape = new_shape
    else:
        logging.warning('Unknown SSD conf sub-path, may cause convert problem')

    # step2. if min_max_aspect_ratios_order is False, do conf/loc params swap
    if ssd_conf['min_max_aspect_ratios_order'] is False:
        logging.info('\t doing conf/loc params swap')
        find_conf_and_loc(caffe_graph, num_class=class_num)
        swap_conf_and_loc(caffe_graph, num_class=class_num)

    # step3. for loc_node, amend its reshape layer / or add flatten if shape > 2dims
    if bbox_node.op_type == 'Reshape':
        new_bbox_node = caffe_graph.get_precursor_nodes_with_node(bbox_node)[0]
        if new_bbox_node.op_type != 'Concat':
            logging.warning('Unknown SSD loc sub-path, may cause convert problem')
        # remove reshape on loc path, clear its successor dst
        caffe_graph.remove_node(bbox_node)
        caffe_graph.remove_blob(bbox_node.output(0))
        caffe_graph.blob_map[new_bbox_node.output(0)].dst_nodes_names = []
        bbox_node = new_bbox_node
    output_shape = caffe_graph.blob_map[bbox_node.output(0)].shape
    if len(output_shape) > 2:
        extra_blob_name = bbox_node.output(0) + '_extra_flatten'
        extra_node_name = bbox_node.name + '_extra_flatten'
        new_output_shape = [output_shape[0], np.prod(output_shape) / output_shape[0]]
        extra_blob = caffe_graph.make_blob(new_output_shape, extra_blob_name, extra_blob_name,
                                           extra_node_name, [source_node.name], do_insert=True)
        _ = caffe_graph.make_node('Flatten', extra_node_name, extra_node_name,
                                  [bbox_node.output(0)], [extra_blob.name],
                                  attrs={'axis': 1}, do_insert=True)
        bbox_final_blob = extra_blob.name
    else:
        bbox_final_blob = bbox_node.output(0)
    # step4. get detection output layer params
    det_attrs = {
        'background_label_id': source_node.attrs.get('background_label', 0),
        'nms_param.nms_threshold': source_node.attr('nms_threshold'),
        'nms_param.top_k': source_node.attr('nms_top_k'),
        'code_type': 'CENTER_SIZE',
        'keep_top_k': source_node.attr('keep_top_k'),
        'confidence_threshold': source_node.attr('score_threshold'),
        'share_location': True,
        'num_classes': class_num
    }

    # step5. creat node and connection
    # 5.1 input connection
    node_name = source_node.name
    input_blobs_names = [bbox_final_blob, conf_node.output(0), pbox_node.output(0)]  # this order is fixed
    for input_blob_name in input_blobs_names:
        # check graph is in typo-sort
        assert input_blob_name in caffe_graph.blob_map.keys(), '{} not found in blob'.format(input_blob_name)
        if node_name not in caffe_graph.blob_map[input_blob_name].dst_nodes_names:
            caffe_graph.blob_map[input_blob_name].dst_nodes_names.append(node_name)
    # 5.2 output connection
    output_blob_name_list = caffe_graph.transfer_op_output(source_node, source_graph)
    # 5.2 creat node
    caffe_node = caffe_graph.make_node('DetectionOutput', source_node.raw_name, node_name,
                                       input_blobs_names, output_blob_name_list,
                                       attrs=det_attrs, do_insert=True)

    return OP_MAPPING_WITH_FUSED, [caffe_node.name]
