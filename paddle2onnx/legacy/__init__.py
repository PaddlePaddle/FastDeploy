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

__version__ = "0.9.6"

import paddle
from .convert import dygraph2onnx, program2onnx
from .op_mapper import register_op_mapper
from typing import TypeVar
from paddle2onnx.utils import logging
from paddle2onnx.legacy.op_mapper import OpMapper
from . import command

OP_WITHOUT_KERNEL_SET = {
    'feed', 'fetch', 'recurrent', 'go', 'rnn_memory_helper_grad',
    'conditional_block', 'while', 'send', 'recv', 'listen_and_serv',
    'fl_listen_and_serv', 'ncclInit', 'select', 'checkpoint_notify',
    'gen_bkcl_id', 'c_gen_bkcl_id', 'gen_nccl_id', 'c_gen_nccl_id',
    'c_comm_init', 'c_sync_calc_stream', 'c_sync_comm_stream',
    'queue_generator', 'dequeue', 'enqueue', 'heter_listen_and_serv',
    'c_wait_comm', 'c_wait_compute', 'c_gen_hccl_id', 'c_comm_init_hccl',
    'copy_cross_scope'
}


def process_old_ops_desc(model):
    for i in range(len(model.blocks[0].ops)):
        if model.blocks[0].ops[i].type == "matmul":
            if not model.blocks[0].ops[i].has_attr("head_number"):
                model.blocks[0].ops[i]._set_attr("head_number", 1)
        elif model.blocks[0].ops[i].type == "yolo_box":
            if not model.blocks[0].ops[i].has_attr("iou_aware"):
                model.blocks[0].ops[i]._set_attr("iou_aware", False)
            if not model.blocks[0].ops[i].has_attr("iou_aware_factor"):
                model.blocks[0].ops[i]._set_attr("iou_aware_factor", 0.5)


def get_all_registered_ops(save_file=None):
    ops = list(OpMapper.OPSETS.keys())
    logging.warning("The number of all registered OPs is: {}".format(len(ops)))
    if save_file is None:
        return
    with open(save_file, "w") as f:
        logging.warning("All registered OPs will be written to the file: {}".
                        format(save_file))
        f.write("Total OPs num: {} \n".format(len(ops)))
        for index in range(len(ops)):
            op = ops[index]
            f.write(str(index + 1) + ". " + op + "\n")
        return


def run_convert(model, input_shape_dict=None, scope=None, opset_version=9):
    paddle_version = paddle.__version__
    if isinstance(model, paddle.static.Program):
        process_old_ops_desc(model)
        if input_shape_dict is not None:
            model_version = model.desc._version()
            major_ver = model_version // 1000000
            minor_ver = (model_version - major_ver * 1000000) // 1000
            patch_ver = model_version - major_ver * 1000000 - minor_ver * 1000
            model_version = "{}.{}.{}".format(major_ver, minor_ver, patch_ver)
            if model_version != paddle_version:
                logging.warning(
                    "The model is saved by paddlepaddle v{}, but now your paddlepaddle is version of {}, this difference may cause error, it is recommend you reinstall a same version of paddlepaddle for this model".
                    format(model_version, paddle_version))
            for k, v in input_shape_dict.items():
                model.blocks[0].var(k).desc.set_shape(v)
            for i in range(len(model.blocks[0].ops)):
                if model.blocks[0].ops[i].type in OP_WITHOUT_KERNEL_SET:
                    continue
                model.blocks[0].ops[i].desc.infer_shape(model.blocks[0].desc)
        if scope is None:
            scope = paddle.static.global_scope()
        input_names = list()
        output_vars = list()
        for i in range(len(model.blocks[0].ops)):
            if model.blocks[0].ops[i].type == "feed":
                input_names.append(model.blocks[0].ops[i].output("Out")[0])
            if model.blocks[0].ops[i].type == "fetch":
                output_vars.append(model.blocks[0].var(model.blocks[0].ops[i]
                                                       .input("X")[0]))
        return program2onnx(
            model,
            scope,
            save_file=None,
            feed_var_names=input_names,
            target_vars=output_vars,
            opset_version=opset_version,
            enable_onnx_checker=True)
    elif isinstance(model, paddle.jit.TranslatedLayer):
        process_old_ops_desc(model.program())
        model_version = model.program().desc._version()
        major_ver = model_version // 1000000
        minor_ver = (model_version - major_ver * 1000000) // 1000
        patch_ver = model_version - major_ver * 1000000 - minor_ver * 1000
        model_version = "{}.{}.{}".format(major_ver, minor_ver, patch_ver)
        if model_version != paddle_version:
            logging.warning(
                "The model is saved by paddlepaddle v{}, but now your paddlepaddle is version of {}, this difference may cause error, it is recommend you reinstall a same version of paddlepaddle for this model".
                format(model_version, paddle_version))

        if input_shape_dict is not None:
            for k, v in input_shape_dict.items():
                model.program().blocks[0].var(k).desc.set_shape(v)
            for i in range(len(model.program().blocks[0].ops)):
                if model.program().blocks[0].ops[
                        i].type in OP_WITHOUT_KERNEL_SET:
                    continue
                model.program().blocks[0].ops[i].desc.infer_shape(model.program(
                ).blocks[0].desc)
        return dygraph2onnx(model, save_file=None, opset_version=opset_version)
    else:
        raise Exception(
            "Only support model loaded from paddle.static.load_inference_model() or paddle.jit.load()"
        )
