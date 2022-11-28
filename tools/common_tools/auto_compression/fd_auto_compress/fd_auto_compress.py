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
from tqdm import tqdm
import paddle
from paddleslim.common import load_config, load_onnx_model
from paddleslim.auto_compression import AutoCompression
from paddleslim.quant import quant_post_static
from .dataset import *


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help="choose PTQ or QAT as quantization method",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    return parser


def reader_wrapper(reader, input_list):

    if isinstance(input_list, list) and len(input_list) == 1:
        input_name = input_list[0]

        def gen():
            in_dict = {}
            for i, data in enumerate(reader()):
                imgs = np.array(data[0])
                in_dict[input_name] = imgs
                yield in_dict

        return gen

    if isinstance(input_list, list) and len(input_list) > 1:

        def gen():
            for idx, data in enumerate(reader()):
                in_dict = {}
                for i in range(len(input_list)):
                    intput_name = input_list[i]
                    feed_data = np.array(data[0][i])
                    in_dict[intput_name] = feed_data

                yield in_dict

        return gen


def auto_compress(FLAGS):

    #FLAGS needs parse
    time_s = time.time()
    paddle.enable_static()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)
    global global_config

    if FLAGS.method == 'QAT':

        all_config = load_config(FLAGS.config_path)
        assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
        global_config = all_config["Global"]
        input_list = global_config['input_list']

        assert os.path.exists(global_config[
            'qat_image_path']), "image_path does not exist!"
        paddle.vision.image.set_image_backend('cv2')
        # transform could be customized.
        train_dataset = paddle.vision.datasets.ImageFolder(
            global_config['qat_image_path'],
            transform=eval(global_config['qat_preprocess']))
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=global_config['qat_batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=0)
        train_loader = reader_wrapper(train_loader, input_list=input_list)
        eval_func = None

        # ACT compression
        ac = AutoCompression(
            model_dir=global_config['model_dir'],
            model_filename=global_config['model_filename'],
            params_filename=global_config['params_filename'],
            train_dataloader=train_loader,
            save_dir=FLAGS.save_dir,
            config=all_config,
            eval_callback=eval_func)
        ac.compress()

    # PTQ compression
    if FLAGS.method == 'PTQ':

        # Read Global config and prepare dataset
        all_config = load_config(FLAGS.config_path)
        assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
        global_config = all_config["Global"]
        input_list = global_config['input_list']

        assert os.path.exists(global_config[
            'ptq_image_path']), "image_path does not exist!"

        paddle.vision.image.set_image_backend('cv2')
        # transform could be customized.
        val_dataset = paddle.vision.datasets.ImageFolder(
            global_config['ptq_image_path'],
            transform=eval(global_config['ptq_preprocess']))
        val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=0)
        val_loader = reader_wrapper(val_loader, input_list=input_list)

        # Read PTQ config
        assert "PTQ" in all_config, f"Key 'PTQ' not found in config file. \n{all_config}"
        ptq_config = all_config["PTQ"]

        # Inititalize the executor
        place = paddle.CUDAPlace(
            0) if FLAGS.devices == 'gpu' else paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        # Read ONNX or PADDLE format model
        if global_config['format'] == 'onnx':
            load_onnx_model(global_config["model_dir"])
            inference_model_path = global_config["model_dir"].rstrip().rstrip(
                '.onnx') + '_infer'
        else:
            inference_model_path = global_config["model_dir"].rstrip('/')

        quant_post_static(
            executor=exe,
            model_dir=inference_model_path,
            quantize_model_path=FLAGS.save_dir,
            data_loader=val_loader,
            model_filename=global_config["model_filename"],
            params_filename=global_config["params_filename"],
            batch_size=32,
            batch_nums=10,
            algo=ptq_config['calibration_method'],
            hist_percent=0.999,
            is_full_quantize=False,
            bias_correction=False,
            onnx_format=True,
            skip_tensor_list=ptq_config['skip_tensor_list']
            if 'skip_tensor_list' in ptq_config else None)

    time_total = time.time() - time_s
    print("Finish Compression, total time used is : ", time_total, "seconds.")
