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
import six
import collections

from paddle2caffe.constant import NodeDomain
from paddle2caffe.constant import DataDomain

from typing import List, Dict, Union, Optional, Any
from paddle2caffe.utils import logging


class Blob(object):
    """
    common Blob(Tensor) abstraction of deep-learning DAG
    """
    def __init__(self, blob_name: str, raw_name: str,
                 src_node, dst_nodes,
                 # src_node: Union[str, Node], dst_nodes: List[Union[str, Node]],
                 shape: List[int],
                 dtype=DataDomain.NONE):
        self.name = blob_name  # this name should be unique
        self.raw_name = raw_name
        self.dtype = dtype
        self.shape = shape
        self.src_node_name: Optional[str] = None
        self.set_src_node(src_node)
        self.dst_nodes_names: Optional[List[str]] = None
        self.set_dst_nodes(dst_nodes)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __str__(self):
        node_str = ''
        node_str += "  {} == ({}::{}) => {}) \n".format(
            self.src_node_name, self.dtype, self.shape, self.dst_nodes_names)
        return node_str

    def src(self) -> str:
        return self.src_node_name

    def dst(self, idx=None) -> Union[List[str], str]:
        if idx is None:
            if self.dst_nodes_names is None:
                return []
            else:
                return self.dst_nodes_names
        return self.dst_nodes_names[idx]

    def set_src_node(self, src_node):
        if isinstance(src_node, list):
            raise TypeError('Port can only get one src node')
        elif isinstance(src_node, Node):
            self.src_node_name = src_node.name
        elif isinstance(src_node, six.string_types) or src_node is None:
            self.src_node_name = src_node
        else:
            raise TypeError(
                'Src node of a port must be type: Node, or String but got {}'.format(type(src_node)))

    def set_dst_nodes(self, dst_nodes):
        if isinstance(dst_nodes, list):
            self.dst_nodes_names = [
                node.name if isinstance(node, Node) else node
                for node in dst_nodes
            ]
        elif dst_nodes is None:
            self.dst_nodes_names = None
        elif isinstance(dst_nodes, six.string_types):
            self.dst_nodes_names = [dst_nodes]
        elif isinstance(dst_nodes, Node):
            self.dst_nodes_names = [dst_nodes.name]
        else:
            raise TypeError(
                'Dst nodes of a port must be type: list, Node, or String but got {}'.format(type(dst_nodes)))


class Node(object):
    """
    common node(layer) abstraction of deep-learning DAG
    """
    def __init__(self, op_type: str,
                 layer_name: str, raw_name: str,
                 input_blobs: Union[Blob, str, List[Union[Blob, str]]],
                 output_blobs: Union[Blob, str, List[Union[Blob, str]]],
                 attrs: Dict[str, Any],
                 domain=NodeDomain.NONE):
        self.name = layer_name  # this name should be unique
        self.raw_name = raw_name
        self.op_type = op_type
        self.attrs = attrs
        self.input_blobs_names: List[Optional[str]] = list()
        self.set_inputs(input_blobs)
        self.output_blobs_names: List[Optional[str]] = list()
        self.set_outputs(output_blobs)
        self.params: Dict[str, Optional[str, np.ndarray]] = dict()
        self.domain = domain

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __str__(self):
        node_str = ''
        attrs = ''
        for key, value in self.attrs.items():
            attrs += ', ' + key + '=' + str(value)
        node_str += "  {} = {}::{}(inputs={}{}) \n".format(
            self.output_blobs_names, self.domain, self.op_type, self.input_blobs_names, attrs)
        return node_str

    def input(self, idx=None):
        """ safely get input(blob list) into [0~n] or specific idx blob """
        if self.input_blobs_names is None:
            self.input_blobs_names = []
        if idx is None:
            return self.input_blobs_names
        return self.input_blobs_names[idx]

    def output(self, idx=None):
        """ safely get output(blob list) into [0~n] or specific idx blob """
        if self.output_blobs_names is None:
            self.output_blobs_names = []
        if idx is None:
            return self.output_blobs_names
        return self.output_blobs_names[idx]

    def attr(self, name):
        if name in self.attrs:
            return self.attrs[name]
        return None

    def set_inputs(self, inputs):
        if isinstance(inputs, list):
            self.input_blobs_names = [
                ipt.name if isinstance(ipt, Blob) else ipt
                for ipt in inputs
            ]
        elif inputs is None:
            self.input_blobs_names = None
        elif isinstance(inputs, six.string_types):
            self.input_blobs_names = [inputs]
        elif isinstance(inputs, Blob):
            self.input_blobs_names = [inputs.name]
        else:
            raise TypeError(
                'Inputs of node must be type: list, Blob, or String but got {}'.
                format(type(inputs)))

    def set_outputs(self, outputs):
        if isinstance(outputs, list):
            self.output_blobs_names = [
                opt.name if isinstance(opt, Blob) else opt
                for opt in outputs
            ]
        elif outputs is None:
            self.output_blobs_names = None
        elif isinstance(outputs, six.string_types):
            self.output_blobs_names = [outputs]
        elif isinstance(outputs, Blob):
            self.output_blobs_names = [outputs.name]
        else:
            raise TypeError(
                'Outputs of node must be type: list, Blob, or String but got {}'.
                format(type(outputs)))
