# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import six
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import layers

from paddle2onnx.legacy.graph import graph_helper, PaddleGraph
from paddle2onnx.utils import logging
from paddle2onnx.legacy.constant.op_mapping_status import *


REGISTER_CUSTOM_PADDLE_OP = {}


def get_max_support_version(versions, opset_version):
    max_version = -1
    for vs in sorted(versions):
        if vs <= opset_version:
            max_version = vs
    return max_version


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

    valid_register_func = 0
    for k, v in inspect.getmembers(mapper_obj, inspect.ismethod):
        if k.startswith("opset_"):
            version = int(k.replace("opset_", ""))
            if version > 13 or version < 1:
                raise Exception(
                    'the specific method of operator mapper must be named opset_[number](1<=number<=13), such as opset_9, but got {}.'.
                    format(k))
            valid_register_func += 1

    if valid_register_func == 0:
        raise Exception(
            'the specific method of operator mapper must be classmethod, which named opset_[number](1<=number<=13), such as opset_9, but none achieved.'
        )

    mapper = OpMapper(paddle_op_list)
    mapper(mapper_obj)


class OpMapper(object):
    OPSETS = {}
    REGISTER_CUSTOM_PADDLE_OP = {}

    def __init__(self, paddle_op, **kwargs):
        if not isinstance(paddle_op, list):
            paddle_op = [paddle_op]
        self.paddle_op = paddle_op
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("opset_"):
                version = int(k.replace("opset_", ""))
                for op in self.paddle_op:
                    if op not in OpMapper.OPSETS:
                        OpMapper.OPSETS[op] = {}
                    opset_dict = OpMapper.OPSETS[op]
                    opset_dict[version] = (v, self.kwargs)

    @staticmethod
    def mapping(graph, node, operator_export_type="ONNX"):
        try:
            if node.type in OpMapper.REGISTER_CUSTOM_PADDLE_OP:
                if operator_export_type in ["PaddleFallback"]:
                    opsets = OpMapper.OPSETS[node.type]
                    versions = list(opsets.keys())
                    convert_version = get_max_support_version(
                        versions, graph.opset_version)
                    mapper_func, kw = opsets[convert_version]
                    mapper_func(graph, node, **kw)
                else:
                    custom_paddle_op = OpMapper.REGISTER_CUSTOM_PADDLE_OP[
                        node.type](node)
                    custom_paddle_graph, output_results = custom_paddle_op.get_paddle_graph(
                    )
                    OpMapper.check_support_status(custom_paddle_graph.node_map,
                                                  graph.opset_version)
                    graph.build_op_nodes(custom_paddle_graph.node_map)

                    node_output_results = dict()
                    for k in node.output_names:
                        custom_outs = output_results[k]
                        node_outs = node.output(k)
                        assert len(custom_outs) == len(
                            node_outs
                        ), "Length of custom implementation operator's outputs is not same with the length of original operator's outputs."
                        for i in range(len(custom_outs)):
                            graph.make_node(
                                "Identity",
                                inputs=[custom_outs[i]],
                                outputs=[node_outs[i]])
            else:
                opsets = OpMapper.OPSETS[node.type]
                versions = list(opsets.keys())
                convert_version = get_max_support_version(versions,
                                                          graph.opset_version)
                mapper_func, kw = opsets[convert_version]
                mapper_func(graph, node, **kw)
        except Exception as e:
            raise Exception(
                "Error happened when mapping node ['{}'] to onnx, which op_type is '{}' with inputs: {} and outputs: {}, specific error: ".
                format(node.layer_name, node.type, node.inputs,
                       node.outputs) + str(e))

    @staticmethod
    def get_recommend_opset_version(node_map, opset_version):
        recommend_opset_version = OpMapper.check_support_status(
            node_map, opset_version, True)
        for name, node in list(node_map.items()):
            if node.type in OpMapper.REGISTER_CUSTOM_PADDLE_OP:  #如果是custom的op，获取custom的推荐op
                custom_paddle_op = OpMapper.REGISTER_CUSTOM_PADDLE_OP[
                    node.type](node)
                custom_paddle_graph, output_results = custom_paddle_op.get_paddle_graph(
                )
                custom_recommend_opset_version = OpMapper.check_support_status(
                    custom_paddle_graph.node_map, opset_version, True)
                recommend_opset_version = max(recommend_opset_version,
                                              custom_recommend_opset_version)
        if opset_version != recommend_opset_version:
            warning_info = "\n======================\n"
            warning_info += "\nFor a successful conversion, set the recommended opset version : {}\n".format(
                recommend_opset_version)
            warning_info += "\n======================\n"
            logging.warning(warning_info)
        return recommend_opset_version

    @staticmethod
    def check_support_status(node_map, opset_version, for_check=False):
        op_mapping_status = {
            OP_MAPPING_NO_REGISTER: [],
            OP_MAPPING_NO_VERSION: [],
        }
        for name, node in list(node_map.items()):
            if node.type in OpMapper.REGISTER_CUSTOM_PADDLE_OP:
                continue
            if node.type not in OpMapper.OPSETS:
                op_mapping_status[OP_MAPPING_NO_REGISTER].append(node)
            else:
                opsets = OpMapper.OPSETS[node.type]
                versions = list(opsets.keys())
                convert_version = get_max_support_version(versions,
                                                          opset_version)
                if convert_version == -1:
                    op_mapping_status[OP_MAPPING_NO_VERSION].append(node)

        if len(op_mapping_status[OP_MAPPING_NO_REGISTER]) > 0:
            unsupported_op_types = set([
                node.type for node in op_mapping_status[OP_MAPPING_NO_REGISTER]
            ])
            error_info = "\nThere's {} ops are not supported yet\n".format(
                len(unsupported_op_types))
            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)

        if len(op_mapping_status[OP_MAPPING_NO_VERSION]) > 0:
            unsupported_op_types = set([
                node.type for node in op_mapping_status[OP_MAPPING_NO_VERSION]
            ])

            recommend_opset_version = -1
            for op_type in unsupported_op_types:
                opsets = OpMapper.OPSETS[op_type]
                if min(opsets.keys()) > recommend_opset_version:
                    recommend_opset_version = min(opsets.keys())
            warning_info = "\nThere are {} ops that are not supported in opset version {}, please set opset version >= {}.\n".format(
                len(unsupported_op_types), opset_version,
                recommend_opset_version)

            for op_type in unsupported_op_types:
                warning_info += "=========== {} ===========\n".format(op_type)
            if for_check:
                logging.warning(warning_info)
                return recommend_opset_version
            raise NotImplementedError(warning_info)
        return opset_version


class CustomPaddleOp(object):
    CREATE_TIMES = {}

    def __init__(self, node):
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.inputs = self.create_place_holder(node)
        self.node = node

    def generate_scope_name(self, node):
        if node.type in CustomPaddleOp.CREATE_TIMES:
            CustomPaddleOp.CREATE_TIMES[node.type] += 1
        else:
            CustomPaddleOp.CREATE_TIMES[node.type] = 1
        scope_prefix = node.type + str(CustomPaddleOp.CREATE_TIMES[node.type] -
                                       1) + '_'
        return scope_prefix

    def create_place_holder(self, node):
        place_holders = {}
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            for arg_name, idxs in node.inputs.items():
                place_holders[arg_name] = []
                for idx in range(len(idxs)):
                    shape = node.input_shape(arg_name, idx)
                    dtype = node.input_dtype(arg_name, idx)
                    name = node.input(arg_name, idx)
                    data = paddle.static.data(
                        name=name, shape=shape, dtype=dtype)
                    place_holders[arg_name].append(data)
        return place_holders

    def input(self, name, idx=None):
        if name not in self.inputs:
            return None
        if idx is None:
            return self.inputs[name]
        if len(self.inputs[name]) <= idx:
            return None
        return self.inputs[name][idx]

    def get_paddle_graph(self):
        scope_prefix = self.generate_scope_name(self.node)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(self.main_program,
                                             self.startup_program):
                with paddle.utils.unique_name.guard(scope_prefix):
                    res = self.forward()
                    feed_var_names = [
                        var.name for vars in self.inputs.values()
                        for var in vars
                    ]
                    fetch_vars = [var for vars in res.values() for var in vars]
                    inference_program = graph_helper.get_program(
                        self.main_program, feed_var_names, fetch_vars)
                    paddle_graph = PaddleGraph.build_from_program(
                        inference_program,
                        feed_var_names,
                        fetch_vars,
                        scope=scope)

        output_results = dict()
        for arg_name, outs in res.items():
            output_results[arg_name] = [out.name for out in outs]
        return paddle_graph, output_results


def register_custom_paddle_op(paddle_op, custom_op):
    paddle_op_list = []

    if isinstance(paddle_op, six.string_types):
        paddle_op_list.append(paddle_op)
    elif isinstance(paddle_op, list):
        paddle_op_list = paddle_op
    else:
        raise ValueError("paddle_op' must be List or string, but got type {}.".
                         format(type(paddle_op)))

    if not isinstance(custom_op, six.class_types):
        raise ValueError("'custom_op' must be Class, but got type {}.".format(
            type(custom_op)))

    forward = getattr(custom_op, "forward", None)
    if not callable(forward):
        raise Exception(
            "Custom paddle operators must be implemented in function named 'forward'."
        )

    for op in paddle_op_list:
        if op not in OpMapper.REGISTER_CUSTOM_PADDLE_OP:
            OpMapper.REGISTER_CUSTOM_PADDLE_OP[op] = custom_op
