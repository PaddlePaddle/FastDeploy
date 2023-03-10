#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import collections
from paddle2onnx.legacy.constant import NodeDomain


class Node(object):
    def __init__(self,
                 op_type,
                 inputs,
                 outputs,
                 attrs,
                 layer_name,
                 domain=NodeDomain.RAW):
        self.domain = domain
        self.type = op_type
        self.attrs = attrs
        self.layer_name = layer_name
        self.set_inputs(inputs)
        self.set_outputs(outputs)

    def __hash__(self):
        return hash(self.layer_name)

    def __eq__(self, other):
        if self.layer_name == other.layer_name:
            return True
        return False

    def __str__(self):
        node_str = ''
        attrs = ''
        for key, value in self.attrs.items():
            attrs += ', ' + key + '=' + str(value)
        node_str += "  {} = {}::{}(inputs={}{}) \n".format(
            self.outputs, self.domain, self.type, self.inputs, attrs)
        return node_str

    def input(self, idx=None):
        if idx is None:
            return self.inputs
        return self.inputs[idx]

    def output(self, idx=None):
        if idx is None:
            return self.outputs
        return self.outputs[idx]

    def attr(self, name):
        if name in self.attrs:
            return self.attrs[name]
        return None

    def set_inputs(self, inputs):
        if isinstance(inputs, list):
            self.inputs = [
                ipt.layer_name if isinstance(ipt, Node) else ipt
                for ipt in inputs
            ]
        elif isinstance(inputs, six.string_types):
            self.inputs = [inputs]
        elif isinstance(inputs, Node):
            self.inputs = [inputs.layer_name]
        else:
            raise TypeError(
                'Inputs of node must be type: list, Node, or String but got {}'.
                format(type(inputs)))

    def set_outputs(self, outputs):
        if isinstance(outputs, list):
            self.outputs = [
                opt.layer_name if isinstance(opt, Node) else opt
                for opt in outputs
            ]
        elif isinstance(outputs, six.string_types):
            self.outputs = [outputs]
        elif isinstance(outputs, Node):
            self.outputs = [outputs.layer_name]
        else:
            raise TypeError(
                'Outputs of node must be type: list, Node, or String but got {}'.
                format(type(outputs)))


class Graph(object):
    def __init__(self):
        self.parameters = {}
        self.node_map = collections.OrderedDict()
        self.input_nodes = list()
        self.output_nodes = list()
        self.op_type_count = dict()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

    def __str__(self):
        graph_str = 'graph { \n'
        for node in self.input_nodes:
            graph_str += " input: {} \n".format(node.layer_name)
        for node in self.output_nodes:
            graph_str += " output: {} \n \n".format(node.layer_name)
        for name, node in self.node_map.items():
            graph_str += node.__str__()
        graph_str += ' }'
        return graph_str

    def set_output_nodes(self, node_list):
        if isinstance(node_list, list):
            self.output_nodes = node_list
        else:
            raise TypeError(
                'output_nodes of Graph must be type: list, but got {}'.format(
                    type(node_list)))

    def set_node_map(self, node_map):
        if isinstance(node_map, dict):
            self.node_map = node_map
            self.generate_topo_sort()
        else:
            raise TypeError('node_map of Graph must be type: list, but got {}'.
                            format(type(node_map)))

    def set_input_nodes(self, node_list):
        if isinstance(node_list, list):
            self.input_nodes = node_list
        else:
            raise TypeError(
                'input_nodes of Graph must be type: list, but got {}'.format(
                    type(node_list)))

    def set_parameters(self, parameters):
        if isinstance(parameters, dict):
            self.parameters = parameters
        else:
            raise TypeError(
                'parameters of Graph must be type: dict, but got {}'.format(
                    type(parameters)))

    def generate_node_name(self, op_type):
        if op_type in self.op_type_count:
            self.op_type_count[op_type] += 1
        else:
            self.op_type_count[op_type] = 1
        # layer_name need follow https://github.com/onnx/onnx/blob/master/docs/OpConventions.md
        layer_name = op_type + '_' + str(self.op_type_count[op_type] - 1)
        return layer_name

    def insert_node(self, node):
        if node.type not in ['feed', 'fetch']:
            self.node_map[node.layer_name] = node

    def make_node(self,
                  op_type,
                  inputs=None,
                  outputs=None,
                  attrs=None,
                  layer_name=None,
                  domain=None,
                  **kw):
        if layer_name is None:
            layer_name = self.generate_node_name(op_type)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = [layer_name]
        node = Node(op_type, layer_name, inputs, outputs, attrs, domain)
        self.insert_node(node)
        return node

    def update_node(self,
                    node,
                    op_type=None,
                    inputs=None,
                    outputs=None,
                    attrs=None,
                    block=None,
                    move_to_end=True,
                    domain=None,
                    **kw):
        if op_type is not None:
            node.type = op_type
        if inputs is not None:
            node.set_inputs(inputs)
        if outputs is not None:
            node.set_outputs(outputs)
        if attrs is None:
            attrs = kw
        attrs.update(kw)
        node.attrs = attrs
        if domain is not None:
            node.domain = domain
        if move_to_end:
            self.node_map.pop(node.layer_name)
        self.node_map[node.layer_name] = node
        return node

    def get_node(self, name, copy=False):
        if name not in self.node_map:
            raise TypeError('Node with name:{} not in graph'.format(name))
        if copy:
            node = copy.copy(self.node_map[name])
        else:
            node = self.node_map[name]
        return node

    def remove_node_by_name(self, name):
        if name in self.node_map:
            node = self.node_map.pop(name)
            return node
        raise TypeError('Node with name:{} not in graph'.format(name))

    def remove_node(self, node):
        if isinstance(node, Node):
            node = self.remove_node_by_name(node.layer_name)
            return node
        else:
            node = self.remove_node_by_name(node)
            return node

    def get_output_nodes_of_node(self, node):
        if node in self.edge_map:
            return self.edge_map[node]
        elif self.get_node(node.layer_name, copy=False):
            return []
        else:
            raise KeyError('Node with layer_name {} not in graph.egde_map'.
                           format(node.layer_name))

    def get_adjacency_map(self):
        adjacency_map = {}
        for layer_name, current_node in self.node_map.items():
            inputs = current_node.inputs
            for ipt in inputs:
                for layer_name, node in self.node_map.items():
                    if current_node == node:
                        continue
                    outputs = node.outputs
                    if ipt in outputs:
                        if node not in adjacency_map:
                            adjacency_map[node] = set([current_node])
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
