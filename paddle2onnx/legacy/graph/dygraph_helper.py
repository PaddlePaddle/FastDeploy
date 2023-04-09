#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import inspect
import six
import paddle
from paddle.fluid.io import _get_valid_program
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator, StaticFunction
from paddle.fluid.layers.utils import flatten, pack_sequence_as
from collections import OrderedDict
from paddle.fluid import dygraph
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid import core
from paddle.fluid import layers
from paddle.nn import Layer
from paddle.fluid.framework import Block, ParamBase, Program, Variable, Parameter, program_guard
from paddle.fluid.dygraph.layers import Layer

from paddle2onnx.utils import logging
from paddle2onnx.legacy.graph.graph_helper import prepend_feed_ops, append_fetch_ops


def _get_input_var_names(inputs, input_spec):
    name_none_error = "The %s's name is None. " \
        "When using jit.save, please set InputSepc's name in " \
        "to_static(input_spec=[]) and jit.save(input_spec=[]) " \
        "and make sure they are consistent."
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of InputSpec or example Tensor " \
        "in input_spec is the same as the name of InputSpec in " \
        "`to_static` decorated on the Layer.forward method."
    result_list = []
    input_var_names = [
        var.name for var in flatten(inputs) if isinstance(var, Variable)
    ]
    if input_spec is None:
        # no prune
        return input_var_names
    else:
        # fileter out non-tensor type spec infos.
        input_spec = [
            spec for spec in input_spec
            if isinstance(spec, paddle.static.InputSpec)
        ]

    if len(input_spec) == len(input_var_names):
        # no prune
        result_list = input_var_names
        # if input spec name not in input_var_names, only raise warning
        for spec in input_spec:
            if spec.name is None:
                warnings.warn(name_none_error % spec)
            elif spec.name not in input_var_names:
                warnings.warn(name_no_exists_error % spec.name)
            else:
                # do nothing
                pass
    else:
        # prune
        for spec in input_spec:
            if spec.name is None:
                # name is None, the input_spec only can be InputSpec
                raise ValueError(name_none_error % spec)
            elif spec.name not in input_var_names:
                # the input_spec can be `InputSpec` or `VarBase`
                raise ValueError(name_no_exists_error % spec.name)
            else:
                result_list.append(spec.name)

    return result_list


def _get_output_vars(outputs, output_spec):
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of example Tensor " \
        "in configs.output_spec is the output tensor of " \
        "Layer.forward method."
    result_list = []
    output_vars_dict = OrderedDict()
    for var in flatten(outputs):
        if isinstance(var, Variable):
            output_vars_dict[var.name] = var
    if output_spec is None:
        result_list = output_vars_dict.values()
    elif output_spec is not None and len(output_spec) == len(output_vars_dict):
        result_list = output_vars_dict.values()
        for var in output_spec:
            if var.name not in output_vars_dict:
                warnings.warn(name_no_exists_error % var.name)
    else:
        for var in output_spec:
            if var.name not in output_vars_dict:
                raise ValueError(name_no_exists_error % var.name)
            else:
                result_list.append(output_vars_dict[var.name])
    return result_list


@dygraph.base.switch_to_static_graph
def get_program(layer, input_spec, output_spec, **configs):
    paddle.jit.set_verbosity(0)
    prog_translator = ProgramTranslator()
    if not prog_translator.enable_to_static:
        raise RuntimeError(
            "The Paddle2onnx doesn't work when setting ProgramTranslator.enable to False."
        )

    if not isinstance(layer, Layer):
        raise TypeError(
            "The input of paddle2onnx should be 'Layer', but received input type is %s."
            % type(layer))

    if isinstance(layer, paddle.DataParallel):
        inner_layer = layer._layers
    else:
        inner_layer = layer

    # avoid change user given input_spec
    inner_input_spec = None
    if input_spec is not None:
        for attr_func in dir(inner_layer):
            static_func = getattr(inner_layer, attr_func, None)
            if isinstance(static_func,
                          StaticFunction) and 'forward' != attr_func:
                raise ValueError(
                    "If there are static functions other than 'forward' that need to be saved, the input 'input_spec' should be None, but received the type of 'input_spec' is %s."
                    % type(input_spec))

        if not isinstance(input_spec, (list, tuple)):
            raise TypeError(
                "The input input_spec should be 'list', but received input_spec's type is %s."
                % type(input_spec))
        inner_input_spec = []
        for var in flatten(input_spec):
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(var, (core.VarBase, core.eager.Tensor, Variable)):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var))
            else:
                # NOTE(Aurelius84): Support non-Tensor type in `input_spec`.
                inner_input_spec.append(var)

    extra_var_info = dict()
    functions = dir(inner_layer)
    for attr_func in functions:
        static_func = getattr(inner_layer, attr_func, None)
        if isinstance(static_func, StaticFunction):
            concrete_program = static_func.concrete_program_specify_input_spec(
                inner_input_spec)
        elif 'forward' == attr_func:
            # transform in jit.save, if input_spec is incomplete, declarative will throw error
            # inner_input_spec is list[InputSpec], it should be packed with same structure
            # as original input_spec here.
            if inner_input_spec:
                inner_input_spec = pack_sequence_as(input_spec,
                                                    inner_input_spec)
            static_forward = declarative(
                inner_layer.forward, input_spec=inner_input_spec)
            concrete_program = static_forward.concrete_program
            # the input_spec has been used in declarative, which is equal to
            # @declarative with input_spec and jit.save without input_spec,
            # avoid needless warning
            inner_input_spec = None
        else:
            continue

        input_var_names = _get_input_var_names(concrete_program.inputs,
                                               inner_input_spec)

        # NOTE(chenweihang): [ Get output variables ]
        # the rule is like [ Get input variables name ]. For output var,
        # we only support VarBase spec, and actually, we only need the
        # var name of output, and we don't recommended to use output_spec
        output_vars = _get_output_vars(concrete_program.outputs, output_spec)

    feeded_var_names = input_var_names
    target_vars = output_vars
    main_program = concrete_program.main_program.clone()
    export_for_deployment = True

    if isinstance(feeded_var_names, six.string_types):
        feeded_var_names = [feeded_var_names]
    elif export_for_deployment:
        if len(feeded_var_names) > 0:
            # TODO(paddle-dev): polish these code blocks
            if not (bool(feeded_var_names) and all(
                    isinstance(name, six.string_types)
                    for name in feeded_var_names)):
                raise ValueError("'feed_var_names' should be a list of str.")

    if isinstance(target_vars, Variable):
        target_vars = [target_vars]
    elif export_for_deployment:
        if not (bool(target_vars) and
                all(isinstance(var, Variable) for var in target_vars)):
            raise ValueError("'target_vars' should be a list of Variable.")

    main_program = _get_valid_program(main_program)

    # remind user to set auc_states to zeros if the program contains auc op
    all_ops = main_program.global_block().ops
    for op in all_ops:
        # clear device of Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op._set_attr(device_attr_name, "")
        if op.type == 'auc':
            warnings.warn(
                "please ensure that you have set the auc states to zeros before saving inference model"
            )
            break

    with program_guard(main_program):
        uniq_target_vars = []
        for i, var in enumerate(target_vars):
            uniq_target_vars.append(var)
        target_vars = uniq_target_vars
    target_var_name_list = [var.name for var in target_vars]

    origin_program = main_program.clone()

    main_program = main_program.clone()
    global_block = main_program.global_block()
    need_to_remove_op_index = []
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            need_to_remove_op_index.append(i)

    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)

    main_program.desc.flush()

    main_program = main_program._prune_with_input(
        feeded_var_names=feeded_var_names, targets=target_vars)
    main_program = main_program._inference_optimize(prune_read_op=True)
    fetch_var_names = [v.name for v in target_vars]

    for target_v in target_vars:
        if not main_program.global_block().has_var(target_v.name):
            main_program.global_block().create_var(
                name=target_v.name,
                shape=target_v.shape,
                dtype=target_v.dtype,
                persistable=target_v.persistable)

    prepend_feed_ops(main_program, feeded_var_names)
    append_fetch_ops(main_program, fetch_var_names)

    main_program.desc._set_version()
    paddle.fluid.core.save_op_version_info(main_program.desc)

    main_program._copy_dist_param_info_from(origin_program)

    return main_program, feeded_var_names, target_vars
