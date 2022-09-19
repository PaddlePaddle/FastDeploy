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

import os
import sys
import numpy as np
import time
import argparse
import paddle
from paddleslim.common import load_onnx_model
from paddleslim.quant import quant_post_static
from fdquant.dataset import FDDataset


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help="The type of input model.",
        required=True)
    parser.add_argument(
        '--model_file',
        type=str,
        default=None,
        required=True,
        help=" The path of model files.")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='quant_out',
        help=" The quantized model save path.")
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        required=True,
        help="The data path for calibration.")
    parser.add_argument(
        '--skip_tensors',
        type=str,
        default=None,
        help=" Skip these tensors when perform quantization.")
    parser.add_argument(
        '--calibration_method',
        type=str,
        default='avg',
        help="Different methods to do the calibration for getting quantization scale values."
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    return parser


def main():

    parser = argsparser()
    args = parser.parse_args()
    time_start = time.time()

    paddle.enable_static()
    paddle.set_device(args.devices)

    # 模型的输入名
    input_name = 'x2paddle_image_arrays' if args.model_type == 'YOLOv6' else 'x2paddle_images'
    # 准备数据集
    dataset = FDDataset(args.data_dir, input_name=input_name)
    data_loader = paddle.io.DataLoader(
        dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    # 初始化执行器
    place = paddle.CUDAPlace(0) if args.devices == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    # 读取onnx格式模型，并且重命名
    load_onnx_model(args.model_file)
    inference_model_path = args.model_file.rstrip().rstrip('.onnx') + '_infer'

    # 开始执行离线量化
    quant_post_static(
        executor=exe,
        model_dir=inference_model_path,
        quantize_model_path=args.save_dir,
        data_loader=data_loader,
        model_filename='model.pdmodel',
        params_filename='model.pdiparams',
        batch_size=32,
        batch_nums=10,
        algo=args.calibration_method,
        hist_percent=0.999,
        is_full_quantize=False,
        bias_correction=False,
        onnx_format=True,
        skip_tensor_list=args.skip_tensors)

    # 计时
    time_end = time.time()
    time_e = time_end - time_start
    print("Finish PTQ, the time elapsed is : ", time_e, " Seconds")


if __name__ == '__main__':
    main()
