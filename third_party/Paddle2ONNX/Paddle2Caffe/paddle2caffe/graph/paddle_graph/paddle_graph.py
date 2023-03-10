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
import collections
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import dygraph, Layer, layers
from paddle.fluid.framework import Operator

from paddle2caffe.graph import Blob, Node, Graph
from paddle2caffe.constant import NodeDomain
from paddle2caffe.utils import logging

from paddle2caffe.graph.paddle_graph.graph_helper import is_skip_orphan_blob, is_static_shape


class PaddleNode(Node):
    def __init__(self, paddle_op, layer_name, raw_name, input_blobs, output_blobs, attrs, block):
        super(PaddleNode, self).__init__(paddle_op.type,
                                         layer_name, raw_name,
                                         input_blobs, output_blobs,
                                         attrs, NodeDomain.PADDLE)
        self.paddle_op = paddle_op  # raw op saved
        self.block = block

    def __str__(self):
        node_str = ''
        attrs = ''
        for key, value in self.attrs.items():
            if key == 'op_callstack':
                continue
            attrs += ', ' + key + '=' + str(value)
        node_str += "  {} = {}::{}(inputs={}{}) \n".format(
            self.output_blobs_names, self.domain, self.op_type, self.input_blobs_names, attrs)
        return node_str

    def attr(self, name, default=None):
        if name in self.attrs:
            return self.attrs[name]
        return default


class PaddleGraph(Graph):
    def __init__(self, program, parameters, feed_var_names, fetch_vars):
        super(PaddleGraph, self).__init__()

        self._program = program
        self._blocks = program.blocks
        self._skip_tensor_list = []

        self.build_graph(parameters, feed_var_names, fetch_vars)

    def get_parameters(self, parameter_name):
        if self.parameters.get(parameter_name) is not None:
            return self.parameters.get(parameter_name)['data']
        else:
            return None

    def make_node(self, op, layer_name, node_name=None,
                  input_blobs=None, output_blobs=None,
                  attrs=None,
                  do_insert=False,
                  domain=NodeDomain.PADDLE,
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

        node = PaddleNode(op, node_name, raw_name,
                          input_blobs, output_blobs, attrs, self._blocks)
        if do_insert is True:
            self.insert_node(node)

        return node

    def add_input_node(self, inputs, block=None):
        """ change feed node into input node """
        for ipt in inputs:
            # parse tensor through feed_names
            tensor_name = ipt
            shape = block.var(ipt).shape
            dtype = block.var(ipt).dtype
            # also add output blob for input node
            blob_name = self.generate_unique_blob_name(tensor_name)
            node_name = self.generate_unique_node_name(tensor_name)
            node = Node('feed', node_name, tensor_name, [], [blob_name],
                        {}, NodeDomain.PADDLE)
            self.input_nodes.append(node)
            blob = Blob(blob_name, tensor_name, node_name, [], shape, dtype)
            self.blob_map[blob_name] = blob

    def add_graph_node(self, op):
        input_blobs_raw = []
        output_blobs_raw = []
        node_name = self.generate_unique_node_name(op.type)
        node_params_dict = {}

        # logging.debug(op.type)
        # check whether inputs/outputs is valid
        for ipt_key in op.input_names:
            names = op.input(ipt_key)
            # cut out several useless in/out port
            if is_skip_orphan_blob(op.type, ipt_key):
                continue
            # if len(names) > 0:
            #     logging.debug('\t{}'.format(','.join(names)))
            assert isinstance(names, list)
            if len(names) > 0:
                for name in names:
                    # case 1. input is params
                    if name in self.parameters.keys():
                        # case 1. input is params
                        node_params_dict[ipt_key] = name
                    else:
                        # case 2. input is blob
                        input_blobs_raw.append(name)
            else:
                continue
        for opt in op.output_names:
            names = op.output(opt)
            # cut out several useless in/out port
            if is_skip_orphan_blob(op.type, opt):
                continue
            # if len(names) > 0:
            #     logging.debug('\t{}'.format(','.join(names)))
            assert isinstance(names, list)
            if len(names) > 0:
                output_blobs_raw += names
            else:
                continue

        input_blobs_name = []
        # for input blobs, current op should in their dest nodes
        for blob_name_raw in input_blobs_raw:
            blob_list = self.get_blobs_by_raw(blob_name_raw)
            if len(blob_list) > 0:
                # if have multi blob(inplace blob), try connect with only the latest blob
                input_blobs_name.append(blob_list[-1].name)
                dst_list = self.blob_map[blob_list[-1].name].dst()
                if node_name not in dst_list:
                    dst_list.append(node_name)
                else:
                    logging.warning('multi tensor transfer from same node')
            else:
                raise ValueError('can not find input blob:{} for node:{}'.format(blob_name_raw, node_name))

        output_blobs_name = []
        # for output blobs, current op should be their src node
        # and those blobs should not in exist map
        for blob_name_raw in output_blobs_raw:
            cur_candidate = self.get_blobs_by_raw(blob_name_raw)
            if len(cur_candidate) == 1:
                # several fluid node have inplace tensor, just check
                assert op.type in ['relu', 'set_value'] and blob_name_raw in input_blobs_raw
            elif len(cur_candidate) > 1:
                raise ValueError('duplicate blob name and not inplace case')
            blob_name = self.generate_unique_blob_name(blob_name_raw)
            blob = self.make_blob(op.block.var(blob_name_raw).shape, blob_name_raw, blob_name,
                                  node_name, [],
                                  NodeDomain.PADDLE,
                                  do_insert=True)
            output_blobs_name.append(blob.name)

        node = self.make_node(op, "{:s}_idx{:d}".format(op.type, self.node_num), node_name,
                              input_blobs_name, output_blobs_name,
                              op.all_attrs(), do_insert=True)
        node.params = node_params_dict
        # logging.debug(node)

    def add_output_node(self, output_vars, block=None):
        for opt_var in output_vars:
            # parse tensor through fetch_target_vars
            tensor_name = opt_var.name
            node_name = self.generate_unique_node_name(tensor_name)
            # input blob for output node should already in graph and only one
            output_blob = self.get_blobs_by_raw(tensor_name)
            assert len(output_blob) == 1
            output_blob = output_blob[0]
            node = Node('fetch', node_name, tensor_name, [output_blob.name], [],
                        {}, NodeDomain.PADDLE)
            self.output_nodes.append(node)

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

    def build_graph(self,
                    parameters,
                    feed_var_names=None,
                    target_vars=None):

        self.set_parameters(parameters)

        # step1. init feed_var as input node
        self.add_input_node(feed_var_names, self._program.global_block())
        # step2. init other nodes
        for block in self._blocks:
            for i, op in enumerate(block.ops):
                if op.type in ['feed', 'fetch']:
                    continue
                elif self.check_if_need_skip(op):
                    continue
                else:
                    self.add_graph_node(op)
        # step3. init target_var as output node
        self.add_output_node(target_vars, self._program.global_block())


def shape_check(program, feed_var_names) -> None:
    """

    :param program:
    :param feed_var_names:
    :return:
    """
    for name in feed_var_names:
        var = program.global_block().var(name)
        if not is_static_shape(var.shape):
            raise ValueError('caffe only support static shape')
    return


def build_from_program(program,
                       feed_var_names=None,
                       fetch_vars=None,
                       scope=None) -> PaddleGraph:
    """

    :param program:
    :param feed_var_names:
    :param fetch_vars:
    :param scope:
    :return:
    """
    parameters_dict = {}
    vars = program.global_block().vars
    for name in vars:
        var = program.global_block().var(name)
        if name.endswith('feed') or name.endswith('fetch'):
            continue
        if not var.persistable:
            continue
        parameters_dict[name] = {
            'data': np.array(scope.var(name).get_tensor()),
            'dtype': var.dtype,
            'shape': var.shape
        }

    graph = PaddleGraph(program, parameters_dict, feed_var_names,
                        fetch_vars)

    return graph


def build_from_dygraph(layer, input_spec=None, output_spec=None):
    from paddle2caffe.graph.paddle_graph import graph_helper as dg_helper

    if isinstance(layer, dygraph.TranslatedLayer):
        program = layer.program()
        parameters_dict = {}
        pruned_vars = program.global_block().vars
        for param in layer.parameters():
            if param.name.endswith('feed') or param.name.endswith('fetch'):
                continue
            if not param.persistable:
                continue
            if param.name in pruned_vars:
                parameters_dict[param.name] = {
                    'data': np.array(param.value().get_tensor()),
                    'dtype': param.dtype,
                    'shape': param.shape
                }
        if input_spec is not None:
            logging.warning("Although input_spec is specified, TranslatedLayer is not support prune. "
                            "An Complete network will be exported.")
            input_spec = layer._input_spec()
        if output_spec is not None:
            logging.warning("Although output_spec is specified, TranslatedLayer is not support prune. "
                            "An Complete network will be exported.")
        feed_var_names = [ipt.name for ipt in layer._input_spec()]
        fetch_vars = [
            program.global_block().var(opt.name)
            for opt in layer._output_spec()
        ]
        graph = PaddleGraph(program, parameters_dict, feed_var_names,
                            fetch_vars)
        return graph
    elif isinstance(layer, Layer):
        program, feed_var_names, fetch_vars = dg_helper.get_program(
            layer, input_spec, output_spec)
        parameters_dict = {}
        pruned_vars = program.global_block().vars
        for param in layer.parameters():
            if param.name.endswith('feed') or param.name.endswith('fetch'):
                continue
            if not param.persistable:
                continue
            if param.name in pruned_vars:
                parameters_dict[param.name] = {
                    'data': np.array(param.value().get_tensor()),
                    'dtype': param.dtype,
                    'shape': param.shape
                }
        graph = PaddleGraph(program, parameters_dict, feed_var_names,
                            fetch_vars)
        return graph
    else:
        raise TypeError(
            "The input Layer should be 'Layer' or 'TranslatedLayer', but received  type is %s."
            % type(layer))
