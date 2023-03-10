# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

"""
op_mapper use caffe_graph._transfer_dict and caffe_graph._network_helper['network_name']
to save graph level information across different op_mappers

_transfer_dict:
    'state': OP_MAPPING_XXX
    'info': extra info
_network_helper:
    'SSD': details in ssd_bbox

* this is vital important while processing detection network *
*  which changes net graph structure rapidly during convert  *
"""
from __future__ import absolute_import

import inspect
from paddle2caffe.utils import logging

import six
import numpy as np

from paddle2caffe.constant.op_mapping_status import *

REGISTER_CUSTOM_PADDLE_OP = {}


def get_mapping_type(mapping_list, use_caffe_custom):
    if 'standard' not in mapping_list and 'custom' not in mapping_list:
        raise Exception('the specific method of operator mapper must have standard or custom, '
                        'but none achieved.')

    if use_caffe_custom:
        if 'custom' in mapping_list:
            return 'custom'
        else:
            return 'standard'
    else:
        if 'standard' not in mapping_list:
            raise ValueError('none standard caffe layer, you can try use custom caffe!')
        else:
            return 'standard'


def register_op_mapper(paddle_op, mapper_obj):
    paddle_op_list = []

    if isinstance(paddle_op, six.string_types):
        paddle_op_list.append(paddle_op)
    elif isinstance(paddle_op, list):
        paddle_op_list = paddle_op
    else:
        raise ValueError('paddle_op must be List or string, but got type {}.'.
                         format(type(paddle_op)))

    if not isinstance(mapper_obj, six.class_types):
        raise ValueError('mapper_obj must be Class, but got type {}.'.format(
            type(mapper_obj)))

    valid_register_list = []
    for k, v in inspect.getmembers(mapper_obj, inspect.ismethod):
        if k.endswith("_mapping"):
            version = k.replace("_mapping", "")
            if version != 'standard' or version != 'custom':
                raise Exception('the specific method of operator mapper must be named in standard or custom'
                                ', but got {}.'.format(k))
            valid_register_list.append(version)

    if 'standard' not in valid_register_list and 'custom' not in valid_register_list:
        raise Exception('the specific method of operator mapper must be in standard or custom, '
                        'but none achieved.')

    mapper = OpMapper(paddle_op_list)
    mapper(mapper_obj)


class OpMapper(object):
    OP_FACTORY = {}
    REGISTER_CUSTOM_PADDLE_OP = {}

    def __init__(self, paddle_op, use_caffe_custom=True, **kwargs):
        if not isinstance(paddle_op, list):
            paddle_op = [paddle_op]
        self.use_caffe_custom = use_caffe_custom
        self.paddle_op = paddle_op
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.endswith("_mapping"):
                version = k.replace("_mapping", "")
                for op in self.paddle_op:
                    if op not in OpMapper.OP_FACTORY:
                        OpMapper.OP_FACTORY[op] = {}
                    opset_dict = OpMapper.OP_FACTORY[op]
                    opset_dict[version] = (v, self.kwargs)

    def mapping(self, graph, node, source_graph=None):
        try:
            if node.op_type in OpMapper.REGISTER_CUSTOM_PADDLE_OP:
                custom_paddle_op = OpMapper.REGISTER_CUSTOM_PADDLE_OP[
                    node.op_type](node)
                custom_paddle_graph = custom_paddle_op.get_paddle_graph()
                OpMapper.check_support_status(custom_paddle_graph.node_map,
                                              graph.opset_version)
                graph.build_op_nodes(custom_paddle_graph.node_map)
            else:
                opsets = OpMapper.OP_FACTORY[node.op_type]
                versions = list(opsets.keys())
                version = get_mapping_type(versions, self.use_caffe_custom)
                mapper_func, kw = opsets[version]
                state, name_list = mapper_func(graph, node, source_graph, **kw)
                if state != OP_MAPPING_IDENTITY and node.op_type != 'batch_norm':
                    logging.info('un-identical mapping with node name list ({})'.format(name_list))
                return state
        except Exception as e:
            raise Exception(
                "Error happened when mapping node ['{}'] to caffe, "
                "which op_type is '{}' with inputs: {} and outputs: {}, specific error: ".
                format(node.raw_name, node.op_type, node.input_blobs_names,
                       node.output_blobs_names) + str(e))

    def check_support_status(self, node_map):
        op_mapping_status = {
            OP_MAPPING_NO_REGISTER: [],
            OP_MAPPING_NOT_STANDARD: [],
        }
        for name, node in list(node_map.items()):
            if node.op_type in OpMapper.REGISTER_CUSTOM_PADDLE_OP:
                continue
            if node.op_type not in OpMapper.OP_FACTORY:
                op_mapping_status[OP_MAPPING_NO_REGISTER].append(node)
            else:
                opsets = OpMapper.OP_FACTORY[node.op_type]
                versions = list(opsets.keys())
                if 'standard' not in versions and 'custom' not in versions:
                    logging.warning('illegal op mapper setting in {}'.format(node.op_type))
                    op_mapping_status[OP_MAPPING_NO_REGISTER].append(node)
                elif 'standard' not in versions:
                    op_mapping_status[OP_MAPPING_NOT_STANDARD].append(node)
                else:
                    continue

        if len(op_mapping_status[OP_MAPPING_NO_REGISTER]) > 0:
            unsupported_op_types = set([
                node.op_type for node in op_mapping_status[OP_MAPPING_NO_REGISTER]
            ])
            error_info = "\nThere's {} ops are not supported yet\n".format(
                len(unsupported_op_types))
            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)

        if len(op_mapping_status[OP_MAPPING_NOT_STANDARD]) > 0 and not self.use_caffe_custom:
            unsupported_op_types = set([
                node.op_type for node in op_mapping_status[OP_MAPPING_NOT_STANDARD]
            ])

            error_info = "\nThere's {} ops are only supported in custom caffe , " \
                         "please try use caffe custom mode.\n".format(len(unsupported_op_types))

            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)


# class CustomPaddleOp(object):
#     CREATE_TIMES = {}
#
#     def __init__(self, node):
#         self.main_program = paddle.static.Program()
#         self.startup_program = paddle.static.Program()
#         self.inputs = self.create_place_holder(node)
#         self.node = node
#
#     def generate_scope_name(self, node):
#         if node.op_type in CustomPaddleOp.CREATE_TIMES:
#             CustomPaddleOp.CREATE_TIMES[node.op_type] += 1
#         else:
#             CustomPaddleOp.CREATE_TIMES[node.op_type] = 1
#         scope_prefix = node.op_type + str(CustomPaddleOp.CREATE_TIMES[node.op_type] -
#                                        1) + '_'
#         return scope_prefix
#
#     def create_place_holder(self, node):
#         place_holders = {}
#         with paddle.static.program_guard(self.main_program,
#                                          self.startup_program):
#             for arg_name, idxs in node.inputs.items():
#                 place_holders[arg_name] = []
#                 for idx in range(len(idxs)):
#                     shape = node.input_shape(arg_name, idx)
#                     dtype = node.input_dtype(arg_name, idx)
#                     name = node.input(arg_name, idx)
#                     data = paddle.static.data(
#                         name=name, shape=shape, dtype=dtype)
#                     place_holders[arg_name].append(data)
#         return place_holders
#
#     def input(self, name, idx=None):
#         if name not in self.inputs:
#             return None
#         if idx is None:
#             return self.inputs[name]
#         if len(self.inputs[name]) <= idx:
#             return None
#         return self.inputs[name][idx]
#
#     def rename_node_output(self, graph, old_name, new_name):
#         output_idx = None
#         for idx in range(len(graph.output_nodes)):
#             if graph.output_nodes[idx].layer_name == old_name:
#                 output_idx = idx
#                 break
#         graph.output_nodes[output_idx].layer_name = new_name
#         need_rename_nodes = []
#         for name, node in graph.node_map.items():
#             for arg_name, outputs in node.outputs.items():
#                 for idx in range(len(outputs)):
#                     if outputs[idx] == old_name:
#                         node.outputs[arg_name][idx] = new_name
#                         need_rename_nodes.append(node)
#         for node in need_rename_nodes:
#             graph.node_map[node.layer_name] = node
#         return graph
#
#     def get_paddle_graph(self):
#         scope_prefix = self.generate_scope_name(self.node)
#         scope = paddle.static.Scope()
#         with paddle.static.scope_guard(scope):
#             with paddle.static.program_guard(self.main_program,
#                                              self.startup_program):
#                 with paddle.utils.unique_name.guard(scope_prefix):
#                     res = self.forward()
#                     feed_var_names = [
#                         var.name for vars in self.inputs.values()
#                         for var in vars
#                     ]
#                     fetch_vars = [var for vars in res.values() for var in vars]
#                     inference_program = graph_helper.get_program(
#                         self.main_program, feed_var_names, fetch_vars)
#                     paddle_graph = PaddleGraph.build_from_program(
#                         inference_program,
#                         feed_var_names,
#                         fetch_vars,
#                         scope=scope)
#
#         for arg_name, opts in res.items():
#             for idx in range(len(opts)):
#                 new_name = self.node.output(arg_name, idx)
#                 old_name = opts[idx].name
#                 paddle_graph = self.rename_node_output(paddle_graph, old_name,
#                                                        new_name)
#
#         return paddle_graph
#
#
# def register_custom_paddle_op(paddle_op, custom_op):
#     paddle_op_list = []
#
#     if isinstance(paddle_op, six.string_types):
#         paddle_op_list.append(paddle_op)
#     elif isinstance(paddle_op, list):
#         paddle_op_list = paddle_op
#     else:
#         raise ValueError("paddle_op' must be List or string, but got type {}.".
#                          format(type(paddle_op)))
#
#     if not isinstance(custom_op, six.class_types):
#         raise ValueError("'custom_op' must be Class, but got type {}.".format(
#             type(custom_op)))
#
#     forward = getattr(custom_op, "forward", None)
#     if not callable(forward):
#         raise Exception(
#             "Custom paddle operators must be implemented in function named 'forward'."
#         )
#
#     for op in paddle_op_list:
#         if op not in OpMapper.REGISTER_CUSTOM_PADDLE_OP:
#             OpMapper.REGISTER_CUSTOM_PADDLE_OP[op] = custom_op
