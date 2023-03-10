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
import six
import paddle
import numpy as np
from paddle.fluid.framework import Variable

from paddle2caffe.utils import logging
from paddle2caffe.graph import build_from_program, build_from_dygraph, shape_check, PaddleGraph
from paddle2caffe.graph import build_from_graph, CaffeGraph
from paddle2caffe.graph.caffe_graph import graph_emitter
from paddle2caffe.optimizer import OptManager


def _graph2caffe(paddle_graph, save_file, use_caffe_custom=True):
    caffe_graph: CaffeGraph = build_from_graph(paddle_graph, use_caffe_custom)

    # debug & info
    logging.info('caffe graph info: ')
    caffe_graph.report_graph()

    # run opt pass
    caffe_opt_manager = OptManager()
    for OptPass in caffe_opt_manager.pick('CAFFE', 'all'):
        caffe_opt_pass = OptPass(caffe_graph)
        if caffe_opt_pass.match() is True:
            caffe_graph = caffe_opt_pass.apply()
            logging.info('caffe graph after opt info: ')
            caffe_graph.report_graph()

    # model export: graph -> code -> model export
    caffe_export = graph_emitter.CaffeEmitter(caffe_graph)
    caffe_export.export(save_file)


def program2caffe(program,
                  scope,
                  model_dir,
                  save_file,
                  feed_var_names=None,
                  target_vars=None,
                  use_caffe_custom=True):
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

        shape_check(program, feed_var_names)

        paddle_graph: PaddleGraph = \
            build_from_program(program, feed_var_names, target_vars, scope)
        paddle_graph.model_dir = model_dir

        # debug & info
        logging.info('paddle graph info: ')
        paddle_graph.report_graph()

    else:
        raise TypeError(
            "the input 'program' should be 'Program', but received type is %s."
            % type(program))

    _graph2caffe(paddle_graph, save_file, use_caffe_custom)


def dygraph2caffe(layer, save_file, input_spec=None, use_caffe_custom=True, **configs):
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
                raise TypeError("The element in input_spec list should be 'Variable' or `paddle.static.InputSpec`,"
                                " but received element's type is %s."% type(var))

    output_spec = None
    if 'output_spec' in configs:
        output_spec = configs['output_spec']
        if not isinstance(output_spec, list):
            raise TypeError(
                "The output_spec should be 'list', but received type is %s." %
                type(output_spec))
        else:
            for var in output_spec:
                if not isinstance(var, core.VarBase):
                    raise TypeError("The element in output_spec list should be 'Variable', "
                                    "but received element's type is %s." % type(var))

    verbose = False
    if 'verbose' in configs:
        if isinstance(configs['verbose'], bool):
            verbose = configs['verbose']
        else:
            raise TypeError(
                "The verbose should be 'bool', but received type is %s." %
                type(configs['verbose']))

    paddle_graph = build_from_dygraph(layer, inner_input_spec, output_spec)
    # debug & info
    logging.info('paddle graph info: ')
    paddle_graph.report_graph()

    _graph2caffe(paddle_graph, save_file, use_caffe_custom)
