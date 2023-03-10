# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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
os.environ['GLOG_minloglevel'] = '3'
import paddle.fluid as fluid

from paddle2caffe.utils import logging


def fetch_extra_vars(block, fetch_targets, var_names_list=None):
    """
    :param block:
    :param fetch_targets:
    :param var_names_list:
    :return:
    """

    fetch_var = block.var('fetch')
    old_fetch_names = []
    for var in fetch_targets:
        old_fetch_names.append(var.name)

    new_fetch_vars = []
    # add default fetch vars
    for var_name in old_fetch_names:
        var = block.var(var_name)
        new_fetch_vars.append(var)

    i = len(new_fetch_vars)
    if var_names_list is None:
        var_names_list = block.vars.keys()

    for var_name in var_names_list:
        # if '.tmp_' in var_name and var_name not in old_fetch_names:
        if var_name not in old_fetch_names:
            var = block.var(var_name)
            new_fetch_vars.append(var)
            block.append_op(
                type='fetch',
                inputs={'X': [var_name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
            i = i + 1

    return new_fetch_vars


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
        help="file path to save caffe model")
    parser.add_argument(
        "--enable_caffe_custom",
        type=ast.literal_eval,
        default=False,
        help="whether to use custom caffe support(SSD, Upsample, etc)"
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of paddle2caffe")
    return parser


def program2caffe(model_dir, save_file,
                  model_filename=None,
                  params_filename=None,
                  enable_caffe_custom=True):
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

    import paddle2caffe as p2c
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

    # add temp tensor into output
    # extra_name_list = []
    # global_block = program.global_block()
    # fetch_vars = fetch_extra_vars(global_block, fetch_vars, extra_name_list)

    p2c.program2caffe(
        program,
        fluid.global_scope(),
        model_dir,
        save_file,
        feed_var_names=feed_var_names,
        target_vars=fetch_vars,
        use_caffe_custom=enable_caffe_custom)


def main():
    if len(sys.argv) < 2:
        logging.info("Use \"paddle2caffe -h\" to print the help information")
        logging.info(
            "For more information, please follow our github repo below:")
        logging.info("Github: https://github.com/PaddlePaddle/paddle2onnx.git")
        return

    parser = arg_parser()
    args = parser.parse_args()

    if args.version:
        import paddle2caffe
        logging.info("paddle2caffe-{}-contrib with python>=3.6, paddlepaddle>=2.1.0".
                     format(paddle2caffe.__version__))
        return

    assert args.model_dir is not None, "--model_dir should be defined while translating paddle model to caffe"
    assert args.save_file is not None, "--save_file should be defined while translating paddle model to caffe"
    program2caffe(
        args.model_dir,
        args.save_file,
        args.model_filename,
        args.params_filename,
        args.enable_caffe_custom)


if __name__ == "__main__":
    main()
