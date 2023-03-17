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

from paddle2onnx.legacy.passes import PassManager


def get_repeated_output(inputs, outputs):
    repeated_output = {}
    for idx in range(len(outputs)):
        opt = outputs[idx]
        if opt in inputs:
            repeated_output[opt] = idx
    return repeated_output


@PassManager('inplace_node_pass')
class InplaceNodePass(object):

    name_count = dict()

    @classmethod
    def generate_new_name(cls, name):
        if name in cls.name_count:
            cls.name_count[name] += 1
        else:
            cls.name_count[name] = 1
        new_name = name + '.' + str(cls.name_count[name])
        return new_name

    @classmethod
    def run_pass(cls, onnx_graph):
        node_map = list(onnx_graph.node_map.items())
        name_mapping = {}
        for idx in range(len(node_map)):
            name, node = node_map[idx]
            inputs = node.inputs
            outputs = node.outputs
            for idx in range(len(inputs)):
                ipt = inputs[idx]
                if ipt in name_mapping:
                    inputs[idx] = name_mapping[ipt]
            repeated_output = get_repeated_output(inputs, outputs)
            if len(repeated_output) != 0:
                for opt, idx in repeated_output.items():
                    name_mapping[opt] = cls.generate_new_name(opt)
                    outputs[idx] = name_mapping[opt]
            node.set_inputs(inputs)
            node.set_outputs(outputs)
            onnx_graph.update_node(node)

        return onnx_graph
