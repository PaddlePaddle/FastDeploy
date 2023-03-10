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
        default=True,
        help="whether to use new version of Paddle2ONNX which is under developing, default True"
    )
    parser.add_argument(
        "--deploy_backend",
        "-d",
        type=_text_type,
        default="onnxruntime",
        choices=["onnxruntime", "tensorrt", "rknn", "others"],
        help="Quantize model deploy backend, default onnxruntime.")
    parser.add_argument(
        "--save_calibration_file",
        type=_text_type,
        default="calibration.cache",
        help="The calibration cache for TensorRT deploy, default calibration.cache."
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
    parser.add_argument(
        "--external_filename",
        type=_text_type,
        default=None,
        help="The filename of external_data when the model is bigger than 2G.")
    return parser


def c_paddle_to_onnx(model_file,
                     params_file="",
                     save_file=None,
                     opset_version=7,
                     auto_upgrade_opset=True,
                     verbose=True,
                     enable_onnx_checker=True,
                     enable_experimental_op=True,
                     enable_optimize=True,
                     deploy_backend="onnxruntime",
                     calibration_file="",
                     external_file=""):
    import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
    onnx_model_str = c_p2o.export(
        model_file, params_file, opset_version, auto_upgrade_opset, verbose,
        enable_onnx_checker, enable_experimental_op, enable_optimize, {},
        deploy_backend, calibration_file, external_file)
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
    logging.warning(
        "[Deprecated] `paddle2onnx.command.program2onnx` will be deprecated in the future version, the recommended usage is `paddle2onnx.export`"
    )
    from paddle2onnx.legacy.command import program2onnx
    return program2onnx(model_dir, save_file, model_filename, params_filename,
                        opset_version, enable_onnx_checker,
                        operator_export_type, input_shape_dict, output_names,
                        auto_update_opset)


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
        logging.info("paddle2onnx-{} with python>=3.6, paddlepaddle>=2.0.0".
                     format(paddle2onnx.__version__))
        return

    assert args.model_dir is not None, "--model_dir should be defined while translating paddle model to onnx"
    assert args.save_file is not None, "--save_file should be defined while translating paddle model to onnx"

    input_shape_dict = eval(args.input_shape_dict)

    operator_export_type = "ONNX"
    if args.enable_paddle_fallback:
        logging.warning(
            "[Deprecated] The flag `--enable_paddle_fallback` will be deprecated, and only works while `--enable_dev_version False` now."
        )
        operator_export_type = "PaddleFallback"

    if args.output_names is not None and args.enable_dev_version:
        logging.warning(
            "[Deprecated] The flag `--output_names` is deprecated, if you need to modify the output name, please refer to this tool https://github.com/jiangjiajun/PaddleUtils/tree/main/onnx "
        )
        if not isinstance(args.output_names, (list, dict)):
            raise TypeError(
                "The output_names should be 'list' or 'dict', but received type is %s."
                % type(args.output_names))

    if input_shape_dict is not None and args.enable_dev_version:
        logging.warning(
            "[Deprecated] The flag `--input_shape_dict` is deprecated, if you need to modify the input shape of PaddlePaddle model, please refer to this tool https://github.com/jiangjiajun/PaddleUtils/tree/main/paddle "
        )

    if args.enable_dev_version:
        model_file = os.path.join(args.model_dir, args.model_filename)
        if args.params_filename is None:
            params_file = ""
        else:
            params_file = os.path.join(args.model_dir, args.params_filename)

        if args.external_filename is None:
            args.external_filename = "external_data"

        base_path = os.path.dirname(args.save_file)
        if base_path and not os.path.exists(base_path):
            os.mkdir(base_path)
        external_file = os.path.join(base_path, args.external_filename)

        calibration_file = args.save_calibration_file
        c_paddle_to_onnx(
            model_file=model_file,
            params_file=params_file,
            save_file=args.save_file,
            opset_version=args.opset_version,
            auto_upgrade_opset=args.enable_auto_update_opset,
            verbose=True,
            enable_onnx_checker=args.enable_onnx_checker,
            enable_experimental_op=True,
            enable_optimize=True,
            deploy_backend=args.deploy_backend,
            calibration_file=calibration_file,
            external_file=external_file)
        logging.info("===============Make PaddlePaddle Better!================")
        logging.info("A little survey: https://iwenjuan.baidu.com/?code=r8hu2s")
        return

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
    logging.info("===============Make PaddlePaddle Better!================")
    logging.info("A little survey: https://iwenjuan.baidu.com/?code=r8hu2s")


if __name__ == "__main__":
    main()
