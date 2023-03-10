#   Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np
from paddle.nn import Layer
from paddle.fluid import layers
from paddle.fluid import core
from paddle.fluid.framework import Variable, program_guard
from paddle.fluid.dygraph.dygraph_to_static import program_translator
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid import dygraph

from paddle2caffe.utils import logging


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return
    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)
    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feed_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feed_var_names "
                "if '{name}' is not involved in the fetch_vars calculation.".
                format(
                    i=i, name=name))
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def get_program(program, feed_var_names, fetch_vars):
    global_block = program.global_block()
    need_to_remove_op_index = []
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()
    program = program._prune_with_input(
        feeded_var_names=feed_var_names, targets=fetch_vars)
    program = program._inference_optimize(prune_read_op=True)
    fetch_var_names = [v.name for v in fetch_vars]
    prepend_feed_ops(program, feed_var_names)
    append_fetch_ops(program, fetch_var_names)
    return program


def is_skip_orphan_blob(op_type, port_key):
    skip_dict = {
        'batch_norm': ['MeanOut', 'SavedMean', 'SavedVariance', 'VarianceOut'],
        'dropout': ['Mask'],
        'flatten_contiguous_range': ['XShape'],
        'flatten2': ['XShape'],
        'reshape2': ['XShape'],
        'transpose2': ['XShape'],
        'norm': ['Norm'],
    }

    if skip_dict.get(op_type) is not None and port_key in skip_dict[op_type]:
        return True
    else:
        return False


def is_static_shape(shape):
    if len(shape) > 1 and shape[1:].count(-1) > 0:
        logging.warning(
            "Converting this model to Caffe need with static input shape,"
            "please fix input shape of this model, see doc Q2 in xxx."
        )
        return False
    else:
        return True


def get_inout_spec(all_vars, target_vars, return_name=False):
    result_list = []
    valid_var_dict = {}
    valid_vars = [var for var in all_vars if isinstance(var, Variable)]
    for var in valid_vars:
        valid_var_dict[var.name] = var
    if target_vars is not None:
        for i, var in enumerate(target_vars):
            # check target var whether exists
            if var.name not in valid_var_dict:
                raise RuntimeError("The variable to feed/fetch are not exist.")
            result_list.append(valid_var_dict[var.name])
    else:
        result_list = valid_vars
    if return_name:
        return result_list, [var.name for var in result_list]
    return result_list


@dygraph.base.switch_to_static_graph
def get_program(layer, input_spec, output_spec, **configs):
    paddle.jit.set_verbosity(0)
    prog_translator = program_translator.ProgramTranslator()
    if not prog_translator.enable_to_static:
        raise RuntimeError(
            "The paddle.jit.save doesn't work when setting ProgramTranslator.enable to False."
        )
    if isinstance(layer, Layer):
        if isinstance(layer.forward, program_translator.StaticFunction):
            concrete_program = layer.forward.concrete_program
        else:
            # transform in jit.save, if input_spec is incomplete, declarative will throw error
            layer = paddle.jit.to_static(layer, input_spec=input_spec)
            concrete_program = layer.forward.concrete_program
            # the input_spec has been used in declarative, which is equal to
            # @declarative with input_spec and jit.save without input_spec,
            # avoid needless warning
            input_spec = None
    else:
        raise TypeError(
            "The input Layer should be 'Layer', but received  type is %s." %
            type(layer))
    feed_var_names = paddle.fluid.dygraph.jit._get_input_var_names(
        concrete_program.inputs, input_spec)
    target_vars = paddle.fluid.dygraph.jit._get_output_vars(
        concrete_program.outputs, output_spec)
    main_program = concrete_program.main_program.clone()
    with program_guard(main_program):
        uniq_target_vars = []
        for i, var in enumerate(target_vars):
            if isinstance(var, Variable):
                var = layers.scale(
                    var, 1., name="save_infer_model/scale_{}".format(i))
            uniq_target_vars.append(var)
        target_vars = uniq_target_vars
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
        feeded_var_names=feed_var_names, targets=target_vars)
    main_program = main_program._inference_optimize(prune_read_op=True)
    fetch_var_names = [v.name for v in target_vars]
    prepend_feed_ops(main_program, feed_var_names)
    append_fetch_ops(main_program, fetch_var_names)

    return main_program, feed_var_names, target_vars