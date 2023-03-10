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

import importlib
import collections
import time
import os
import sys


def try_import(module_name):
    """Try importing a module, with an informative error message on failure."""
    install_name = module_name
    try:
        mod = importlib.import_module(module_name)
        return mod
    except ImportError:
        err_msg = (
            "Failed importing {}. This likely means that some modules "
            "requires additional dependencies that have to be "
            "manually installed (usually with `pip install {}`). ").format(
                module_name, install_name)
        raise ImportError(err_msg)


def check_model(onnx_model):
    onnx = try_import('onnx')
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        raise Exception('ONNX model is not valid.')
    finally:
        logging.info('ONNX model generated is valid.')


levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}


class logging():
    log_level = 2

    @staticmethod
    def log(level=2, message="", use_color=False):
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if logging.log_level >= level:
            if use_color:
                print("\033[1;31;40m{} [{}]\t{}\033[0m".format(
                    current_time, levels[level], message).encode("utf-8")
                      .decode("latin1"))
            else:
                print("{} [{}]\t{}".format(current_time, levels[level], message)
                      .encode("utf-8").decode("latin1"))
            sys.stdout.flush()

    @staticmethod
    def debug(message="", use_color=False):
        logging.log(level=3, message=message, use_color=use_color)

    @staticmethod
    def info(message="", use_color=False):
        logging.log(level=2, message=message, use_color=use_color)

    @staticmethod
    def warning(message="", use_color=True):
        logging.log(level=1, message=message, use_color=use_color)

    @staticmethod
    def error(message="", use_color=True, exit=True):
        logging.log(level=0, message=message, use_color=use_color)
        if exit:
            sys.exit(-1)


def compare_value(a, b, cond):
    if cond == 'equal':
        if a != b:
            return False
        return True
    if cond == 'greater_than':
        if a <= b:
            return False
        return True
    if cond == 'greater_equal':
        if a < b:
            return False
        return True
    if cond == 'less_equal':
        if a > b:
            return False
        return True
    if cond == 'less_than':
        if a >= b:
            return False
        return True


def compare_attr(actual_value, target_value, attr_name, cond='equal'):
    if not compare_value(actual_value, target_value, cond):
        raise ValueError('Support {} {} {}, actually got {}=={}.'.format(
            attr_name, cond, target_value, attr_name, actual_value))


def compare_attr_between_dims(attr, dims, attr_name, cond='equal'):
    if not compare_value(attr[dims[0]], attr[dims[1]], cond):
        expect_info = 'Support {}[{}] {} {}[{}], '.format(
            attr_name, dims[0], cond, attr_name, dims[1])
        actual_info = 'actually got {}[{}]=={}, not {} {}[{}]=={}.'.format(
            attr_name, dims[0], attr[dims[0]], cond, attr_name, dims[1],
            attr[dims[1]])
        raise ValueError(expect_info + actual_info)


def require_fixed_shape(op_name=None):
    logging.error(
        "[{}]Fixed shape is required, refer this doc for more information: https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/docs/zh/FAQ.md".
        format(op_name))
