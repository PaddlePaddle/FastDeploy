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
import six
import re
import collections
from typing import List, Tuple, Dict, Union

import paddle2caffe.constant as constant
from paddle2caffe.graph import Node, Blob
from paddle2caffe.utils import logging


def normalize_name(input_str):
    """

    :param input_str:
    :return:
    """
    pattern = re.compile(r'[^a-zA-Z0-9]')
    return re.sub(pattern, '_', input_str)


class Graph(object):
    """
    common graph(network) abstraction of deep-learning DAG
    """
    def __init__(self, name=None):
        self.name = name if name is not None else 'Default'  # this name should be unique

        # node and blob map, stand for node and edge in graph
        self.blob_map = collections.OrderedDict()
        self.node_map = collections.OrderedDict()
        # input and output node list
        self.input_nodes: List[Node] = list()
        self.output_nodes: List[Node] = list()
        # params and const tensor, mainly is numpy array
        self.parameters = dict()
        self.const_tensor = dict()
        # several counter for graph info
        self.op_type_count = dict()
        self.node_num = 0
        self.blob_num = 0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __str__(self):
        graph_str = 'graph { \n'
        for node in self.input_nodes:
            graph_str += " input: {} \n".format(node.name)
        for node in self.output_nodes:
            graph_str += " output: {} \n \n".format(node.name)
        for name, node in self.node_map.items():
            graph_str += node.__str__()
        graph_str += ' }'
        return graph_str

    ################################################################
    # set property in node/blob level
    ################################################################
    def set_input_nodes(self, node_list):
        if isinstance(node_list, list):
            self.input_nodes = node_list
        else:
            raise TypeError(
                'input_nodes of Graph must be type: list, but got {}'.format(type(node_list)))

    def set_output_nodes(self, node_list):
        if isinstance(node_list, list):
            self.output_nodes = node_list
        else:
            raise TypeError(
                'output_nodes of Graph must be type: list, but got {}'.format(type(node_list)))

    def set_parameters(self, parameters):
        if isinstance(parameters, dict):
            self.parameters = parameters
        else:
            raise TypeError(
                'parameters of Graph must be type: dict, but got {}'.format(type(parameters)))

    def get_parameters(self, parameter_names):
        raise NotImplementedError('this should set in derived graph(e.g. paddle graph)')

    def generate_unique_node_name(self, op_raw_name):
        op_normed_name = normalize_name(op_raw_name)
        op_normed_name += '_idx' + str(self.node_num)
        return op_normed_name

    def generate_unique_blob_name(self, tensor_raw_name):
        tensor_normed_name = normalize_name(tensor_raw_name)
        tensor_normed_name += '_idx' + str(self.blob_num)
        return tensor_normed_name

    ################################################################
    # in node/blob level
    ################################################################
    def insert_node(self, node: Node):
        self.node_map[node.name] = node
        # renew counter
        self.node_num += 1
        cur_count = self.op_type_count.get(node.op_type)
        if cur_count is None:
            self.op_type_count[node.op_type] = 1
        else:
            self.op_type_count[node.op_type] = cur_count + 1

    def insert_blob(self, blob: Blob):
        self.blob_map[blob.name] = blob
        # renew counter
        self.blob_num += 1

    def make_blob(self, tensor_shape, tensor_name, blob_name=None,
                  src_node=None, dst_nodes=None,
                  domain=None,
                  do_insert=False) -> Blob:
        raw_name = tensor_name
        if blob_name is None:
            blob_name = self.generate_unique_blob_name(raw_name)

        src_node = src_node
        if dst_nodes is None:
            dst_nodes = []

        blob = Blob(blob_name, raw_name, src_node, dst_nodes, tensor_shape, domain)
        if do_insert is True:
            self.insert_blob(blob)

        return blob

    def make_node(self, op_type, layer_name, node_name=None,
                  input_blobs=None, output_blobs=None,
                  attrs=None,
                  domain=None,
                  do_insert=False,
                  **kw) -> Node:
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

        node = Node(op_type, node_name, input_blobs, output_blobs, attrs, domain)
        if do_insert is True:
            self.insert_node(node)

        return node

    def report_graph(self):
        logging.info('input: {}, output: {}, info: {}'.format(
            [node.name for node in self.input_nodes],
            [node.name for node in self.output_nodes],
            self.op_type_count))
        logging.debug(self)
        # for name, node in self.node_map.items():
        #     if len(node.params) > 0:
        #         logging.debug('node:{} type:{} params:{}'.format(node.name, node.op_type, node.params))
        # for name, blob in self.blob_map.items():
        #     logging.debug(blob)

    ################################################################
    # in node level
    ################################################################
    def get_node(self, name, do_copy=False) -> Node:
        if name not in self.node_map:
            raise TypeError('Node with name:{} not in graph'.format(name))
        if do_copy:
            node = copy.copy(self.node_map[name])
        else:
            node = self.node_map[name]
        return node

    def get_nodes_by_raw(self, raw_name, do_copy=False) -> List[Node]:
        nodes = []
        for name, node in self.node_map.items():
            if node.raw_name == raw_name:
                if do_copy:
                    temp_node = copy.copy(self.node_map[name])
                    nodes.append(temp_node)
                else:
                    nodes.append(node)
        return nodes

    def remove_node(self, node: Union[str, Node]) -> Node:
        if isinstance(node, Node):
            name = node.name
        else:
            name = node
        if name in self.node_map:
            removed_node = self.node_map.pop(name)
            # renew counter
            self.node_num -= 1
            cur_count = self.op_type_count.get(removed_node.op_type)
            if cur_count is None:
                logging.warning('remove node {} :: op_type {} with counter mismatch'
                                .format(removed_node.name, removed_node.op_type))
            else:
                self.op_type_count[removed_node.op_type] = cur_count - 1
            return removed_node
        raise TypeError('Node with name:{} not in graph'.format(name))

    ################################################################
    # in blob level
    ################################################################
    def get_blob(self, name, do_copy=False) -> Blob:
        if name not in self.blob_map:
            raise TypeError('Blob with name:{} not in graph'.format(name))
        if do_copy:
            blob = copy.copy(self.blob_map[name])
        else:
            blob = self.blob_map[name]
        return blob

    def get_blobs_by_raw(self, raw_name, do_copy=False) -> List[Blob]:
        blobs = []
        for name, blob in self.blob_map.items():
            if blob.raw_name == raw_name:
                if do_copy:
                    temp_blob = copy.copy(self.blob_map[name])
                    blobs.append(temp_blob)
                else:
                    blobs.append(blob)
        return blobs

    def remove_blob(self, blob: Union[str, Blob]) -> Blob:
        if isinstance(blob, Blob):
            name = blob.name
        else:
            name = blob
        if name in self.blob_map:
            removed_blob = self.blob_map.pop(name)
            # renew counter
            self.blob_num -= 1
            return removed_blob
        raise TypeError('Blob with name:{} not in graph'.format(name))

    ################################################################
    # in graph transfer level
    ################################################################
    def transfer_op_input(self, source_node: Node, source_graph, input_blob_num=1):
        """
        transfer source graph node's input blob into current graph
        (only creat symbolic connection)
        """
        node_name = source_node.name
        input_blob_name_list = []
        for in_idx in range(input_blob_num):
            input_blob_name = source_node.input_blobs_names[in_idx]
            # check graph is in typo-sort
            assert input_blob_name in self.blob_map.keys(), '{} not found in blob'.format(input_blob_name)
            if node_name not in self.blob_map[input_blob_name].dst_nodes_names:
                self.blob_map[input_blob_name].dst_nodes_names.append(node_name)

            input_blob_name_list.append(input_blob_name)

        return input_blob_name_list

    def transfer_op_output(self, source_node: Node, source_graph, output_blob_num=1):
        """
        transfer source graph node's output blob into current graph
        (will creat new blob in current graph)
        """
        node_name = source_node.name

        output_blob_name_list = []
        for out_idx in range(output_blob_num):
            output_blob_name = source_node.output_blobs_names[out_idx]
            # check graph is in typo-sort
            assert output_blob_name not in self.blob_map.keys()
            output_blob = source_graph.get_blob(output_blob_name, do_copy=True)
            assert output_blob.src_node_name == node_name

            output_blob.dst_nodes_names = []  # cut out source_graph output connection which may change in transfer
            self.blob_map[output_blob_name] = output_blob

            output_blob_name_list.append(output_blob_name)

        return output_blob_name_list

    ################################################################
    # in graph connection level
    ################################################################
    def get_precursor_nodes_with_node(self, name_or_node: Union[Node, str], idx=None) -> List[Node]:
        # get node
        if isinstance(name_or_node, six.string_types):
            node = self.get_node(name_or_node)
        else:
            node = name_or_node
        assert node is not None, ('Node with name {} not in graph.node_map'.format(node.name))

        # locate input_blobs (all in list format)
        input_blobs_name = node.input(idx)
        if input_blobs_name is None or (isinstance(input_blobs_name, list) and len(input_blobs_name) == 0):
            return []
        else:
            assert isinstance(input_blobs_name, list) and len(input_blobs_name) > 0
            input_blobs = list()
            for blob_name in input_blobs_name:
                input_blobs.append(self.blob_map[blob_name])

        # find precursor nodes name
        precursor_nodes_names = list()
        for blob in input_blobs:
            if blob.src_node_name is None:
                logging.warning('input blobs have no precursor_node')
                continue
            elif blob.src_node_name not in precursor_nodes_names:
                precursor_nodes_names.append(blob.src_node_name)

        # turn into node list
        precursor_nodes = list()
        for name in precursor_nodes_names:
            precursor_nodes.append(self.get_node(name))
        return precursor_nodes

    def get_successor_nodes_with_node(self, name_or_node: Union[Node, str], idx=None) -> List[Node]:
        # get node
        if isinstance(name_or_node, six.string_types):
            node = self.get_node(name_or_node)
        else:
            node = name_or_node
        assert node is not None, ('Node with name {} not in graph.node_map'.format(node.name))

        # locate output_blobs (all in list format)
        output_blobs_name = node.output(idx)
        if output_blobs_name is None or (isinstance(output_blobs_name, list) and len(output_blobs_name) == 0):
            return []
        else:
            assert isinstance(output_blobs_name, list) and len(output_blobs_name) > 0
            output_blobs = list()
            for blob_name in output_blobs_name:
                output_blobs.append(self.blob_map[blob_name])

        # find successor nodes name
        successor_nodes_names = list()
        for blob in output_blobs:
            if blob.dst_nodes_names is None:
                logging.warning('output blobs have no successor_nodes')
                continue
            for node_name in blob.dst_nodes_names:
                if node_name not in successor_nodes_names:
                    successor_nodes_names.append(node_name)

        # turn into node list
        successor_nodes = list()
        for name in successor_nodes_names:
            successor_nodes.append(self.get_node(name))
        return successor_nodes

    ################################################################
    # in graph level
    ################################################################
    def get_adjacency_map(self):
        adjacency_map = {}
        for current_layer_name, current_node in self.node_map.items():
            input_nodes = self.get_precursor_nodes_with_node(current_node)
            for ipt in input_nodes:
                for layer_name, node in self.node_map.items():
                    if current_node == node:
                        continue
                    outputs = self.get_successor_nodes_with_node(node)
                    if ipt in outputs:
                        if node not in adjacency_map:
                            adjacency_map[node] = {current_node}  # init set
                        else:
                            adjacency_map[node].add(current_node)
        return adjacency_map

    def get_topo_sort_list(self):
        topo_sort_list = list()
        adjacency_map = self.get_adjacency_map()
        for layer_name, node in self.node_map.items():
            if node not in adjacency_map:
                topo_sort_list.append(node)
        idx = 0
        while idx < len(topo_sort_list):
            current_node = topo_sort_list[idx]
            for input_node, output_nodes in adjacency_map.items():
                if current_node in output_nodes:
                    adjacency_map[input_node].remove(current_node)
                    if len(adjacency_map[input_node]) == 0:
                        topo_sort_list.append(input_node)
            idx += 1
        return topo_sort_list[::-1]

    def set_node_map(self, node_map):
        if isinstance(node_map, dict):
            self.node_map = node_map
            self.get_topo_sort_list()
        else:
            raise TypeError('node_map of Graph must be type: list, but got {}'.format(type(node_map)))
