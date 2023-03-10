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
from paddle2onnx.utils import logging


@PassManager('dumplicate_names_pass')
class DumplicateNamesPass(object):

    name_count = dict()

    @classmethod
    def generate_new_name(cls, name):
        for saved_name in cls.name_count:
            if name.startswith(saved_name):
                cls.name_count[saved_name] += 1
                new_name = saved_name + '.' + str(cls.name_count[saved_name])
                return new_name
        cls.name_count[name] = 1
        new_name = name + '.' + str(cls.name_count[name])
        return new_name

    @classmethod
    def run_pass(cls, onnx_graph):
        renamer = {}
        tensor_names = set()
        for name, node in onnx_graph.parameters.items():
            output = node.output
            for opt in output:
                assert opt not in tensor_names, "There's dumplicate names in parameters."
                tensor_names.add(opt)

        for ipt in onnx_graph.input_nodes:
            assert ipt.name not in tensor_names, "There's dumplicate names in exported parameters and inputs."
            tensor_names.add(ipt.name)

        for name, node in onnx_graph.node_map.items():
            inputs = node.inputs
            outputs = node.outputs
            update_node = False
            for idx in range(len(inputs)):
                ipt = inputs[idx]
                if ipt not in renamer:
                    continue
                updated_name = renamer[ipt]
                while updated_name in renamer:
                    updated_name = renamer[updated_name]
                inputs[idx] = updated_name
                update_node = True

            for idx in range(len(outputs)):
                opt = outputs[idx]
                if opt not in tensor_names:
                    tensor_names.add(opt)
                    continue
                renamed_tensor_name = opt
                while renamed_tensor_name in renamer:
                    renamed_tensor_name = renamer[renamed_tensor_name]
                new_name = cls.generate_new_name(renamed_tensor_name)
                logging.warning("[Renamer Pass] Will rename {}, to {}".format(
                    renamed_tensor_name, new_name))
                outputs[idx] = new_name
                update_node = True
                renamer[renamed_tensor_name] = new_name

            if update_node:
                node.set_inputs(inputs)
                node.set_outputs(outputs)
                onnx_graph.update_node(node)

        for opt in onnx_graph.output_nodes:
            if opt.name not in renamer:
                continue
            updated_name = renamer[opt.name]
            while updated_name in renamer:
                updated_name = renamer[updated_name]
            opt.name = updated_name

        return onnx_graph
