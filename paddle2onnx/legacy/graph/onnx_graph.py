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
import collections
import numpy as np
from paddle2onnx.legacy.graph import Node, Graph
from paddle2onnx.legacy.constant import NodeDomain, PRODUCER, dtypes
from paddle2onnx.legacy.op_mapper import OpMapper
from onnx import helper
from paddle2onnx.utils import check_model, logging


class ONNXNode(Node):
    def __init__(self, op_type, inputs, outputs, attrs, layer_name, domain):
        super(ONNXNode, self).__init__(op_type, inputs, outputs, attrs,
                                       layer_name, domain)
        self.domain = domain
        self.onnx_node = self.make_onnx_node()

    def make_onnx_constant_node(self):
        dtype = self.attr('dtype')
        value = self.attr('value')
        if isinstance(value, list):
            dims = (len(value), )
        elif value is None:
            dims = ()
            value = []
        else:
            dims = ()
            value = [value]

        if 'dims' in self.attrs:
            dims = self.attrs['dims']

        tensor = helper.make_tensor(
            name=self.layer_name, data_type=dtype, dims=dims, vals=value)

        onnx_node = helper.make_node(
            self.type, inputs=self.inputs, outputs=self.outputs, value=tensor)

        return onnx_node

    def make_onnx_node(self):
        if self.type in ['Constant', 'ConstantOfShape']:
            onnx_node = self.make_onnx_constant_node()
        else:
            onnx_node = helper.make_node(
                self.type,
                inputs=self.inputs,
                outputs=self.outputs,
                name=self.layer_name,
                domain=self.domain,
                **self.attrs)
        return onnx_node


class ONNXGraph(Graph):
    def __init__(self,
                 paddle_graph,
                 opset_version,
                 operator_export_type="ONNX",
                 block=None,
                 auto_update_opset=True):
        super(ONNXGraph, self).__init__()
        self.opset_version = opset_version
        self.operator_export_type = operator_export_type
        self.ctx = paddle_graph
        self.custom = []
        if auto_update_opset:
            self.update_opset_version()

    def __str__(self):
        graph_str = 'graph { \n'
        for node in self.input_nodes:
            graph_str += " input: {} \n".format(node)
        for node in self.output_nodes:
            graph_str += " output: {} \n \n".format(node)
        for name, node in self.node_map.items():
            graph_str += node.__str__()
        graph_str += ' }'
        return graph_str

    def make_node(self,
                  op_type,
                  inputs=[],
                  outputs=[],
                  attrs=None,
                  layer_name=None,
                  domain=None,
                  **kw):
        if layer_name is None:
            layer_name = self.generate_node_name(op_type)

        if domain is not None:
            if domain not in self.custom:
                self.custom.append(domain)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = []

        real_outputs = None
        if outputs is None:
            real_outputs = [layer_name]
        elif isinstance(outputs, int):
            real_outputs = []
            for i in range(outputs):
                real_outputs.append(self.generate_node_name(op_type))
        elif isinstance(outputs, list):
            real_outputs = []
            if len(outputs) == 0:
                real_outputs = [layer_name]
            else:
                for opt in outputs:
                    if isinstance(opt, Node):
                        real_outputs.append(opt.layer_name)
                    elif isinstance(opt, int):
                        real_outputs.append(self.generate_node_name(op_type))
                    else:
                        real_outputs.append(opt)
        else:
            real_outputs = outputs

        node = ONNXNode(op_type, inputs, real_outputs, attrs, layer_name,
                        domain)

        self.insert_node(node)
        if len(node.outputs) == 1:
            return node.outputs[0]
        else:
            return node.outputs

    def update_node(self,
                    node,
                    op_type=None,
                    inputs=None,
                    outputs=None,
                    attrs=None,
                    **kw):
        if op_type is None:
            op_type = node.type
        if inputs is None:
            inputs = node.inputs
        if outputs is None:
            outputs = node.outputs
        if attrs is None:
            attrs = node.attrs
        attrs.update(kw)

        node = ONNXNode(op_type, inputs, outputs, attrs, node.layer_name,
                        node.domain)
        self.insert_node(node)
        return node

    def build_parameters(self, parameters):
        # build weight nodes
        for name, param in parameters.items():
            weight = param['data']
            if weight is not np.ndarray:
                weight = np.array(weight)
            tensor = helper.make_tensor(
                name=name,
                dims=param['shape'],
                data_type=dtypes.DTYPE_PADDLE_ONNX_MAP[param['dtype']],
                vals=weight.flatten().tolist())
            node = helper.make_node(
                'Constant', inputs=[], outputs=[name], value=tensor)
            self.parameters[name] = node

    def build_input_nodes(self, input_nodes):
        # build input nodes
        for ipt in input_nodes:
            self.add_input_node(ipt.layer_name,
                                ipt.attr('shape'), ipt.attr('dtype'))

    def build_output_nodes(self, output_nodes):
        # build output nodes
        for opt in output_nodes:
            self.add_output_node(opt.layer_name,
                                 opt.attr('shape'), opt.attr('dtype'))

    def update_opset_version(self):
        node_map = self.ctx.node_map
        self.opset_version = OpMapper.get_recommend_opset_version(
            node_map, self.opset_version)

    def build_op_nodes(self, node_map):
        OpMapper.check_support_status(node_map, self.opset_version)
        # build op nodes
        for name, node in list(node_map.items()):
            OpMapper.mapping(self, node, self.operator_export_type)

    def make_value_info(self, name, shape, dtype):
        tensor_info = helper.make_tensor_value_info(
            name=name,
            shape=shape,
            elem_type=dtypes.DTYPE_PADDLE_ONNX_MAP[dtype])
        return tensor_info

    def add_input_node(self, name, shape, dtype):
        vi = self.make_value_info(name, shape, dtype)
        self.input_nodes.append(vi)

    def add_output_node(self, name, shape, dtype):
        vi = self.make_value_info(name, shape, dtype)
        self.output_nodes.append(vi)

    def find_index(self, node_inout, name):
        for i in range(len(node_inout)):
            if node_inout[i] == name:
                return i
        return -1

    def change_output_names(self, onnx_proto, output_names):
        logging.info("The output of the ONNX model is set to: {}".format(
            output_names))
        if isinstance(output_names, list):
            assert len(output_names) == len(
                onnx_proto.graph.output
            ), "The provided output names are inconsistent with the output number of the onnx model when output_names is list"
            origin_output_names = []
            for i in range(len(onnx_proto.graph.output)):
                origin_output_names.append(onnx_proto.graph.output[i].name)
                onnx_proto.graph.output[i].name = output_names[i]

            for i in range(len(onnx_proto.graph.node)):
                node = onnx_proto.graph.node[i]
                # Prevent changed names from being changed again
                output_visited_node = []
                input_visited_node = []
                for j in range(len(origin_output_names)):
                    if origin_output_names[j] in node.output:
                        index = self.find_index(node.output,
                                                origin_output_names[j])
                        if index in output_visited_node:
                            continue
                        output_visited_node.append(index)
                        onnx_proto.graph.node[i].output[index] = output_names[j]
                    if origin_output_names[j] in node.input:
                        index = self.find_index(node.input,
                                                origin_output_names[j])
                        if index in input_visited_node:
                            continue
                        input_visited_node.append(index)
                        onnx_proto.graph.node[i].input[index] = output_names[j]
        if isinstance(output_names, dict):
            for i in range(len(onnx_proto.graph.output)):
                for key, value in output_names.items():
                    if onnx_proto.graph.output[i].name == key:
                        onnx_proto.graph.output[i].name = value
                        break

            for i in range(len(onnx_proto.graph.node)):
                node = onnx_proto.graph.node[i]
                # Prevent changed names from being changed again
                output_visited_node = []
                input_visited_node = []
                for key, value in output_names.items():
                    if key in node.output:
                        index = self.find_index(node.output, key)
                        if index in output_visited_node:
                            continue
                        output_visited_node.append(index)
                        onnx_proto.graph.node[i].output[index] = value
                    if key in node.input:
                        index = self.find_index(node.input, key)
                        if index in input_visited_node:
                            continue
                        input_visited_node.append(index)
                        onnx_proto.graph.node[i].input[index] = value

        return onnx_proto

    def export_proto(self, enable_onnx_checker=False, output_names=None):

        op_nodes = [node.onnx_node for node in self.node_map.values()]
        weight_nodes = [node for node in self.parameters.values()]

        onnx_graph = helper.make_graph(
            nodes=weight_nodes + op_nodes,
            name='paddle-onnx',
            initializer=[],
            inputs=self.input_nodes,
            outputs=self.output_nodes)

        opset_imports = [helper.make_opsetid("", self.opset_version)]
        for custom_domain in self.custom:
            opset_imports.append(helper.make_opsetid(custom_domain, 1))
        onnx_proto = helper.make_model(
            onnx_graph, producer_name=PRODUCER, opset_imports=opset_imports)
        if output_names is not None:
            onnx_proto = self.change_output_names(onnx_proto, output_names)

        if enable_onnx_checker:
            check_model(onnx_proto)

        return onnx_proto

    @staticmethod
    def build(paddle_graph,
              opset_version,
              operator_export_type="ONNX",
              verbose=False,
              auto_update_opset=True):
        onnx_graph = ONNXGraph(
            paddle_graph,
            opset_version=opset_version,
            operator_export_type=operator_export_type,
            auto_update_opset=auto_update_opset)
        onnx_graph.build_parameters(paddle_graph.parameters)
        onnx_graph.build_input_nodes(paddle_graph.input_nodes)
        onnx_graph.build_output_nodes(paddle_graph.output_nodes)
        onnx_graph.build_op_nodes(paddle_graph.node_map)

        return onnx_graph
