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

import os
import copy
import numpy as np

from paddle2caffe.caffe_helper.caffe_pb2 import NetParameter, LayerParameter

from paddle2caffe.graph import Blob, Node, Graph
from paddle2caffe.constant import NodeDomain
from paddle2caffe.constant.op_mapping_status import *

from paddle2caffe.utils import logging

from paddle2caffe.graph.caffe_graph.graph_helper import NodeMap, LAYER_TYPES
from paddle2caffe.op_mapper import OpMapper


class CaffeNode(Node):
    def __init__(self, caffe_layer, layer_name, raw_name, input_blobs, output_blobs, attrs, domain=NodeDomain.CAFFE):
        if isinstance(caffe_layer, LayerParameter):
            # use raw caffe LayerParameter init
            super(CaffeNode, self).__init__(NodeMap.map_raw_type(caffe_layer),
                                            layer_name, caffe_layer.name,
                                            caffe_layer.bottom, caffe_layer.top,
                                            attrs, domain)
            self.caffe_layer = caffe_layer  # raw op saved
        elif isinstance(caffe_layer, str):
            # create symbolic CaffeNode
            super(CaffeNode, self).__init__(NodeMap.map_raw_type(caffe_layer),
                                            layer_name, raw_name,
                                            input_blobs, output_blobs,
                                            attrs, domain)
            self.caffe_layer = None
        else:
            raise ValueError('CaffeNode can be init by only LayerParameter or TypeStr')

        self.type = NodeMap.map_raw_type(caffe_layer)

    def __str__(self):
        node_str = ''
        attrs = ''
        for key, value in self.attrs.items():
            attrs += ', ' + key + '=' + str(value)
        node_str += "  {} = {}::{}(inputs={}{}) \n".format(
            self.output_blobs_names, self.domain, self.op_type, self.input_blobs_names, attrs)
        return node_str

    def attr(self, name, default=None):
        if name in self.attrs:
            return self.attrs[name]
        return default


class CaffeGraph(Graph):
    def __init__(self, caffe_net: NetParameter, use_caffe_custom=True):
        super(CaffeGraph, self).__init__()

        self.use_caffe_custom = use_caffe_custom
        self._net = caffe_net
        self._skip_tensor_list = []

        # following params is used for graph transfer mapping
        self._transfer_dict = dict()
        self._network_helper = dict()

    def make_node(self, type_str_or_layer_param, layer_name, node_name=None,
                  input_blobs=None, output_blobs=None,
                  attrs=None, do_insert=False,
                  domain=NodeDomain.CAFFE,
                  **kw) -> CaffeNode:
        if isinstance(type_str_or_layer_param, str):
            if type_str_or_layer_param in LAYER_TYPES:
                return self._make_node_symbolic(type_str_or_layer_param, layer_name, node_name,
                                                input_blobs, output_blobs, attrs, do_insert, domain, **kw)
            else:
                raise ValueError('un-support layer type:{:s} in caffe'.format(type_str_or_layer_param))
        elif isinstance(type_str_or_layer_param, LayerParameter):
            raise NotImplementedError
        else:
            raise ValueError('expect type_str or layer_param but get {:s}'.format(type(type_str_or_layer_param)))

    def _make_node_symbolic(self, type_str, layer_name, node_name=None,
                            input_blobs=None, output_blobs=None,
                            attrs=None, do_insert=False,
                            domain=NodeDomain.CAFFE,
                            **kw) -> CaffeNode:
        raw_name = layer_name
        if node_name is None:
            node_name = self.generate_unique_node_name(raw_name)

        # attach layer attrs
        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if input_blobs is None:
            input_blobs = []
        if output_blobs is None:
            output_blobs = []

        node = CaffeNode(type_str, node_name, raw_name,
                         input_blobs, output_blobs, attrs, domain)
        if do_insert is True:
            self.insert_node(node)

        return node

    def transfer_input_node(self, source_graph: Graph):
        """ transfer source graph input nodes into caffe graph input nodes """
        for node in source_graph.input_nodes:
            caffe_input_node = copy.deepcopy(node)
            caffe_input_blob = source_graph.get_blob(node.output(0), do_copy=True)
            caffe_input_blob.dst_nodes_names = []

            caffe_input_node.op_type = 'Data'
            caffe_input_node.domain = NodeDomain.CAFFE
            self.input_nodes.append(caffe_input_node)

            self.blob_map[caffe_input_blob.name] = caffe_input_blob

    def transfer_op_node(self, source_graph: Graph, use_caffe_custom=True):
        caffe_mapper = OpMapper(None, use_caffe_custom=use_caffe_custom)
        caffe_mapper.check_support_status(source_graph.node_map)
        # init mapping info
        for name, node in list(source_graph.node_map.items()):
            self._transfer_dict[name] = {'state': OP_MAPPING_WAITTING, 'info': None}
        # build op nodes
        for name, node in list(source_graph.node_map.items()):
            # print(name, node.op_type)
            if self._transfer_dict[name]['state'] == OP_MAPPING_SKIPPED:
                logging.info('mapping skip since current node:{} is already processed during mapping node:{}'
                             .format(name, self._transfer_dict[name]['info']))
            elif self._transfer_dict[name]['state'] == OP_MAPPING_WAITTING:
                mapping_state = caffe_mapper.mapping(self, node, source_graph)
                self._transfer_dict[name]['state'] = mapping_state
            else:
                raise ValueError('Unknown mapping status')
            # self.report_graph()

    def transfer_graph(self, source_graph: Graph):
        # check transfer mapping data is init
        assert len(self._transfer_dict) == len(self._network_helper) == 0
        self.transfer_input_node(source_graph)
        self.transfer_op_node(source_graph, self.use_caffe_custom)

    def check_if_need_skip(self, op):
        """

        :param op:
        :return:
        """
        is_skip = False
        for ipt in op.input_names:
            for tensor_name in op.input(ipt):
                if tensor_name in self._skip_tensor_list:
                    logging.info('current node will get early stop')
                    is_skip = True
                    break
        if is_skip:
            for opt in op.output_names:
                for tensor_name in op.output(opt):
                    self._skip_tensor_list.append(tensor_name)
        return is_skip


def build_from_graph(src_graph: Graph, use_caffe_custom=True):
    """

    :param src_graph:
    :return:
    """
    caffe_graph = CaffeGraph(None, use_caffe_custom)
    caffe_graph.transfer_graph(src_graph)

    return caffe_graph
