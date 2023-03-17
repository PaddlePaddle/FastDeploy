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
import six
import paddle
import numpy as np
from paddle.fluid.framework import Variable
from paddle2onnx.utils import check_model, logging
from paddle2onnx.legacy.graph import PaddleGraph, ONNXGraph
from paddle2onnx.legacy.passes import PassManager


def export_onnx(paddle_graph,
                save_file,
                opset_version=9,
                enable_onnx_checker=False,
                operator_export_type="ONNX",
                verbose=False,
                auto_update_opset=True,
                output_names=None):
    onnx_graph = ONNXGraph.build(paddle_graph, opset_version,
                                 operator_export_type, verbose,
                                 auto_update_opset)
    onnx_graph = PassManager.run_pass(
        onnx_graph, ['dumplicate_names_pass', 'inplace_node_pass'])
    onnx_proto = onnx_graph.export_proto(enable_onnx_checker, output_names)

    if save_file is None:
        return onnx_proto

    path, _ = os.path.split(save_file)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_file, 'wb') as f:
        f.write(onnx_proto.SerializeToString())
    logging.info("ONNX model saved in {}".format(save_file))


def program2onnx(program,
                 scope,
                 save_file,
                 feed_var_names=None,
                 target_vars=None,
                 opset_version=9,
                 enable_onnx_checker=False,
                 operator_export_type="ONNX",
                 auto_update_opset=True,
                 **configs):
    from paddle import fluid
    if hasattr(paddle, 'enable_static'):
        paddle.enable_static()
    if isinstance(program, paddle.fluid.framework.Program):
        if feed_var_names is not None:
            if isinstance(feed_var_names, six.string_types):
                feed_var_names = [feed_var_names]
            else:
                if not (bool(feed_var_names) and all(
                        isinstance(name, six.string_types)
                        for name in feed_var_names)):
                    raise TypeError("'feed_var_names' should be a list of str.")

        if target_vars is not None:
            if isinstance(target_vars, Variable):
                target_vars = [target_vars]
            else:
                if not (bool(target_vars) and
                        all(isinstance(var, Variable) for var in target_vars)):
                    raise TypeError(
                        "'target_vars' should be a list of variable.")

        paddle_graph = PaddleGraph.build_from_program(program, feed_var_names,
                                                      target_vars, scope)
        output_names = None
        if 'output_names' in configs:
            output_names = configs['output_names']
            if output_names is not None and not isinstance(output_names,
                                                           (list, dict)):
                raise TypeError(
                    "The output_names should be 'list' or dict, but received type is %s."
                    % type(output_names))
        return export_onnx(
            paddle_graph,
            save_file,
            opset_version,
            enable_onnx_checker,
            operator_export_type,
            auto_update_opset=auto_update_opset,
            output_names=output_names)
    else:
        raise TypeError(
            "the input 'program' should be 'Program', but received type is %s."
            % type(program))


def dygraph2onnx(layer, save_file, input_spec=None, opset_version=9, **configs):
    from paddle.nn import Layer
    from paddle.fluid import core
    from paddle.fluid.framework import Variable
    from paddle.fluid.dygraph.dygraph_to_static import program_translator
    from paddle.fluid import dygraph
    if not isinstance(layer, Layer):
        raise TypeError(
            "the input 'layer' should be 'Layer', 'TranslatedLayer', but received type is %s."
            % type(layer))

    inner_input_spec = None
    if input_spec is not None:
        if not isinstance(input_spec, list):
            raise TypeError(
                "The input input_spec should be 'list', but received type is %s."
                % type(input_spec))
        inner_input_spec = []
        for var in input_spec:
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(var, (core.VarBase, Variable)):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var))
            else:
                raise TypeError(
                    "The element in input_spec list should be 'Variable' or `paddle.static.InputSpec`, but received element's type is %s."
                    % type(var))

    output_spec = None
    if 'output_spec' in configs:
        output_spec = configs['output_spec']
        if not isinstance(output_spec, list):
            raise TypeError(
                "The output_spec should be 'list', but received type is %s." %
                type(output_spec))
        for var in output_spec:
            if not isinstance(var, (core.VarBase, Variable)):
                raise TypeError(
                    "The element in output_spec list should be 'Variable', but received element's type is %s."
                    % type(var))

    verbose = False
    if 'verbose' in configs:
        if isinstance(configs['verbose'], bool):
            verbose = configs['verbose']
        else:
            raise TypeError(
                "The verbose should be 'bool', but received type is %s." %
                type(configs['verbose']))

    enable_onnx_checker = False
    if 'enable_onnx_checker' in configs:
        if isinstance(configs['enable_onnx_checker'], bool):
            enable_onnx_checker = configs['enable_onnx_checker']
        else:
            raise TypeError(
                "The 'enable_onnx_checker' should be 'bool', but received type is %s."
                % type(configs['enable_onnx_checker']))

    operator_export_type = "ONNX"
    enable_paddle_fallback = False
    if 'enable_paddle_fallback' in configs:
        if isinstance(configs['enable_paddle_fallback'], bool):
            enable_paddle_fallback = configs['enable_paddle_fallback']
            if enable_paddle_fallback:
                operator_export_type = "PaddleFallback"
        else:
            raise TypeError(
                "The 'enable_paddle_fallback' should be 'bool', but received type is %s."
                % type(configs['enable_paddle_fallback']))

    paddle_graph = PaddleGraph.build_from_dygraph(layer, inner_input_spec,
                                                  output_spec)

    if 'get_paddle_graph' in configs:
        return paddle_graph

    auto_update_opset = True
    if 'auto_update_opset' in configs:
        if isinstance(configs['auto_update_opset'], bool):
            auto_update_opset = configs['auto_update_opset']
        else:
            raise TypeError(
                "The auto_update_opset should be 'bool', but received type is %s."
                % type(configs['auto_update_opset']))

    output_names = None
    if 'output_names' in configs:
        output_names = configs['output_names']
        if not isinstance(output_names, (list, dict)):
            raise TypeError(
                "The output_names should be 'list' or dict, but received type is %s."
                % type(output_names))

    return export_onnx(paddle_graph, save_file, opset_version,
                       enable_onnx_checker, operator_export_type, verbose,
                       auto_update_opset, output_names)
