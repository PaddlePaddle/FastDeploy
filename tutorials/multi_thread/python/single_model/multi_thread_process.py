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

from threading import Thread
import fastdeploy as fd
import cv2
import os
import psutil
from multiprocessing import Pool


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleClas model.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="The directory or path or file list of the images to be predicted."
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="Return topk results.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu' or 'ipu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    parser.add_argument("--thread_num", type=int, default=1, help="thread num")
    parser.add_argument(
        "--use_multi_process",
        type=ast.literal_eval,
        default=False,
        help="Wether to use multi process.")
    parser.add_argument(
        "--process_num", type=int, default=1, help="process num")
    return parser.parse_args()


def get_image_list(image_path):
    image_list = []
    if os.path.isfile(image_path):
        image_list.append(image_path)
    # load image in a directory
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for f in files:
                image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '{} is not found. it should be a path of image, or a directory including images.'.
            format(image_path))

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_paddle_backend()
        option.use_gpu()

    if args.device.lower() == "ipu":
        option.use_ipu()

    if args.use_trt:
        option.use_trt_backend()
    return option


def load_model(args, runtime_option):
    model_file = os.path.join(args.model, "inference.pdmodel")
    params_file = os.path.join(args.model, "inference.pdiparams")
    config_file = os.path.join(args.model, "inference_cls.yaml")
    global model
    model = fd.vision.classification.PaddleClasModel(
        model_file, params_file, config_file, runtime_option=runtime_option)
    #return model


def predict(model, img_list, topk):
    result_list = []
    # predict classification result
    for image in img_list:
        im = cv2.imread(image)
        result = model.predict(im, topk)
        result_list.append(result)
    return result_list


def process_predict(image):
    # predict classification result
    im = cv2.imread(image)
    result = model.predict(im, args.topk)
    print(result)


class WrapperThread(Thread):
    def __init__(self, func, args):
        super(WrapperThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


if __name__ == '__main__':
    args = parse_arguments()

    imgs_list = get_image_list(args.image_path)
    # configure runtime and load model
    runtime_option = build_option(args)

    if args.use_multi_process:
        process_num = args.process_num
        with Pool(
                process_num,
                initializer=load_model,
                initargs=(args, runtime_option)) as pool:
            pool.map(process_predict, imgs_list)
    else:
        load_model(args, runtime_option)
        threads = []
        thread_num = args.thread_num
        image_num_each_thread = int(len(imgs_list) / thread_num)
        # unless you want independent model in each thread, actually model.clone()
        # is the same as model when creating thead because of the existence of
        # GIL(Global Interpreter Lock) in python. In addition, model.clone() will consume
        # additional memory to store independent member variables
        for i in range(thread_num):
            if i == thread_num - 1:
                t = WrapperThread(
                    predict,
                    args=(model.clone(), imgs_list[i * image_num_each_thread:],
                          args.topk))
            else:
                t = WrapperThread(
                    predict,
                    args=(model.clone(), imgs_list[i * image_num_each_thread:(
                        i + 1) * image_num_each_thread - 1], args.topk))
            threads.append(t)
            t.start()

        for i in range(thread_num):
            threads[i].join()

        for i in range(thread_num):
            for result in threads[i].get_result():
                print('thread:', i, ', result: ', result)
