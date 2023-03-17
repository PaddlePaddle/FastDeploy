# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import sys
from paddle2onnx.utils import logging


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_model',
        required=True,
        help='The path of input onnx model file.')
    parser.add_argument(
        '--output_model',
        required=True,
        help='The file path to write optimized onnx model file.')
    parser.add_argument(
        '--input_shape_dict',
        default="",
        help="The shape infos of inputs, e.g --input_shape_dict=\"{'image': [1, 3, 608, 608], 'scale_factor': [1, 2]}\""
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
    shape_dict = {}
    if args.input_shape_dict != "":
        shape_dict = eval(args.input_shape_dict)
    c_p2o.optimize(args.input_model, args.output_model, shape_dict)
    logging.info("Model optmized, saved in {}.".format(args.output_model))
