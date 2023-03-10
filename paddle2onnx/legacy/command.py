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
from six import text_type as _text_type
import argparse
import ast
import sys
import os
import paddle.fluid as fluid
from paddle2onnx.utils import logging


def str2list(v):
    if len(v) == 0:
        return None
    v = v.replace(" ", "")
    v = eval(v)
    return v


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        "-m",
        type=_text_type,
        default=None,
        help="PaddlePaddle model directory, if params stored in single file, you need define '--model_filename' and 'params_filename'."
    )
    parser.add_argument(
        "--model_filename",
        "-mf",
        type=_text_type,
        default=None,
        help="PaddlePaddle model's network file name, which under directory seted by --model_dir"
    )
    parser.add_argument(
        "--params_filename",
        "-pf",
        type=_text_type,
        default=None,
        help="PaddlePaddle model's param file name(param files combined in single file), which under directory seted by --model_dir."
    )
    parser.add_argument(
        "--save_file",
        "-s",
        type=_text_type,
        default=None,
        help="file path to save onnx model")
    parser.add_argument(
        "--opset_version",
        "-ov",
        type=int,
        default=9,
        help="set onnx opset version to export")
    parser.add_argument(
       "--input_shape_dict",
       "-isd",
       type=_text_type,
       default="None",
       help="define input shapes, e.g --input_shape_dict=\"{'image':[1, 3, 608, 608]}\" or" \
       "--input_shape_dict=\"{'image':[1, 3, 608, 608], 'im_shape': [1, 2], 'scale_factor': [1, 2]}\"")
    parser.add_argument(
        "--enable_dev_version",
        type=ast.literal_eval,
        default=False,
        help="whether to use new version of Paddle2ONNX which is under developing, default False"
    )
    parser.add_argument(
        "--enable_onnx_checker",
        type=ast.literal_eval,
        default=True,
        help="whether check onnx model validity, default True")
    parser.add_argument(
        "--enable_paddle_fallback",
        type=ast.literal_eval,
        default=False,
        help="whether use PaddleFallback for custom op, default is False")
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of paddle2onnx")
    parser.add_argument(
        "--output_names",
        "-on",
        type=str2list,
        default=None,
        help="define output names, e.g --output_names=\"[\"output1\"]\" or \
       --output_names=\"[\"output1\", \"output2\", \"output3\"]\" or \
       --output_names=\"{\"Paddleoutput\":\"Onnxoutput\"}\"")
    parser.add_argument(
        "--enable_auto_update_opset",
        type=ast.literal_eval,
        default=True,
        help="whether enable auto_update_opset, default is True")
    return parser


def c_paddle_to_onnx(model_file,
                     params_file="",
                     save_file=None,
                     opset_version=7,
                     auto_upgrade_opset=True,
                     verbose=True,
                     enable_onnx_checker=True,
                     enable_experimental_op=True,
                     enable_optimize=True):
    import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
    onnx_model_str = c_p2o.export(
        model_file, params_file, opset_version, auto_upgrade_opset, verbose,
        enable_onnx_checker, enable_experimental_op, enable_optimize)
    if save_file is not None:
        with open(save_file, "wb") as f:
            f.write(onnx_model_str)
    else:
        return onnx_model_str


def program2onnx(model_dir,
                 save_file,
                 model_filename=None,
                 params_filename=None,
                 opset_version=9,
                 enable_onnx_checker=False,
                 operator_export_type="ONNX",
                 input_shape_dict=None,
                 output_names=None,
                 auto_update_opset=True):
    try:
        import paddle
    except:
        logging.error(
            "paddlepaddle not installed, use \"pip install paddlepaddle\"")

    v0, v1, v2 = paddle.__version__.split('.')
    if v0 == '0' and v1 == '0' and v2 == '0':
        logging.warning("You are use develop version of paddlepaddle")
    elif int(v0) <= 1 and int(v1) < 8:
        raise ImportError("paddlepaddle>=1.8.0 is required")

    import paddle2onnx as p2o
    # convert model save with 'paddle.fluid.io.save_inference_model'
    if hasattr(paddle, 'enable_static'):
        paddle.enable_static()
    exe = fluid.Executor(fluid.CPUPlace())
    if model_filename is None and params_filename is None:
        [program, feed_var_names, fetch_vars] = fluid.io.load_inference_model(
            model_dir, exe)
    else:
        [program, feed_var_names, fetch_vars] = fluid.io.load_inference_model(
            model_dir,
            exe,
            model_filename=model_filename,
            params_filename=params_filename)

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
    if input_shape_dict is not None:
        import paddle2onnx
        paddle2onnx.legacy.process_old_ops_desc(program)
        paddle_version = paddle.__version__
        model_version = program.desc._version()
        major_ver = model_version // 1000000
        minor_ver = (model_version - major_ver * 1000000) // 1000
        patch_ver = model_version - major_ver * 1000000 - minor_ver * 1000
        model_version = "{}.{}.{}".format(major_ver, minor_ver, patch_ver)
        if model_version != paddle_version:
            logging.warning(
                "The model is saved by paddlepaddle v{}, but now your paddlepaddle is version of {}, this difference may cause error, it is recommend you reinstall a same version of paddlepaddle for this model".
                format(model_version, paddle_version))

        for k, v in input_shape_dict.items():
            program.blocks[0].var(k).desc.set_shape(v)
        for i in range(len(program.blocks[0].ops)):
            if program.blocks[0].ops[i].type in OP_WITHOUT_KERNEL_SET:
                continue
            program.blocks[0].ops[i].desc.infer_shape(program.blocks[0].desc)
    p2o.program2onnx(
        program,
        fluid.global_scope(),
        save_file,
        feed_var_names=feed_var_names,
        target_vars=fetch_vars,
        opset_version=opset_version,
        enable_onnx_checker=enable_onnx_checker,
        operator_export_type=operator_export_type,
        auto_update_opset=auto_update_opset,
        output_names=output_names)


def main():
    if len(sys.argv) < 2:
        logging.info("Use \"paddle2onnx -h\" to print the help information")
        logging.info(
            "For more information, please follow our github repo below:")
        logging.info("Github: https://github.com/PaddlePaddle/paddle2onnx.git")
        return

    parser = arg_parser()
    args = parser.parse_args()

    if args.version:
        import paddle2onnx
        logging.info("paddle2onnx-{} with python>=2.7, paddlepaddle>=1.8.0".
                     format(paddle2onnx.__version__))
        return

    assert args.model_dir is not None, "--model_dir should be defined while translating paddle model to onnx"
    assert args.save_file is not None, "--save_file should be defined while translating paddle model to onnx"

    input_shape_dict = eval(args.input_shape_dict)

    operator_export_type = "ONNX"
    if args.enable_paddle_fallback:
        operator_export_type = "PaddleFallback"

    if args.output_names is not None:
        if not isinstance(args.output_names, (list, dict)):
            raise TypeError(
                "The output_names should be 'list' or 'dict', but received type is %s."
                % type(args.output_names))

    if args.enable_dev_version:
        if args.enable_paddle_fallback:
            logging.warn(
                "--enable_paddle_fallback is deprecated while --enable_dev_version=True."
            )
        if args.output_names is not None:
            logging.warn(
                "--output_names is deprecated while --enable_dev_version=True.")
        if input_shape_dict is not None:
            logging.warn(
                "--input_shape_dict is deprecated while --enable_dev_version=True."
            )
        model_file = os.path.join(args.model_dir, args.model_filename)
        if args.params_filename is None:
            params_file = ""
        else:
            params_file = os.path.join(args.model_dir, args.params_filename)
        return c_paddle_to_onnx(
            model_file=model_file,
            params_file=params_file,
            save_file=args.save_file,
            opset_version=args.opset_version,
            auto_upgrade_opset=args.enable_auto_update_opset,
            verbose=True,
            enable_onnx_checker=args.enable_onnx_checker,
            enable_experimental_op=True,
            enable_optimize=True)

    program2onnx(
        args.model_dir,
        args.save_file,
        args.model_filename,
        args.params_filename,
        opset_version=args.opset_version,
        enable_onnx_checker=args.enable_onnx_checker,
        operator_export_type=operator_export_type,
        input_shape_dict=input_shape_dict,
        output_names=args.output_names,
        auto_update_opset=args.enable_auto_update_opset)


if __name__ == "__main__":
    main()
