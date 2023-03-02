# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from multiprocessing import Pool


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="The directory or path or file list of the images to be predicted."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    parser.add_argument(
        "--cpu_thread_num",
        type=int,
        default=9,
        help="Number of threads while inference on CPU.")
    parser.add_argument(
        "--cls_bs",
        type=int,
        default=1,
        help="Classification model inference batch size.")
    parser.add_argument(
        "--rec_bs",
        type=int,
        default=6,
        help="Recognition model inference batch size")
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
        option.use_gpu(args.device_id)

    option.set_cpu_thread_num(args.cpu_thread_num)

    if args.device.lower() == "kunlunxin":
        option.use_kunlunxin()
        return option

    if args.backend.lower() == "trt":
        assert args.device.lower(
        ) == "gpu", "TensorRT backend require inference on device GPU."
        option.use_trt_backend()
    elif args.backend.lower() == "pptrt":
        assert args.device.lower(
        ) == "gpu", "Paddle-TensorRT backend require inference on device GPU."
        option.use_trt_backend()
        option.enable_paddle_trt_collect_shape()
        option.enable_paddle_to_trt()
    elif args.backend.lower() == "ort":
        option.use_ort_backend()
    elif args.backend.lower() == "paddle":
        option.use_paddle_infer_backend()
    elif args.backend.lower() == "openvino":
        assert args.device.lower(
        ) == "cpu", "OpenVINO backend require inference on device CPU."
        option.use_openvino_backend()
    return option


def load_model(args, runtime_option):
    # Detection模型, 检测文字框
    det_model_file = os.path.join(args.det_model, "inference.pdmodel")
    det_params_file = os.path.join(args.det_model, "inference.pdiparams")
    # Classification模型，方向分类，可选
    cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
    cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")
    # Recognition模型，文字识别模型
    rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
    rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
    rec_label_file = args.rec_label_file

    # PPOCR的cls和rec模型现在已经支持推理一个Batch的数据
    # 定义下面两个变量后, 可用于设置trt输入shape, 并在PPOCR模型初始化后, 完成Batch推理设置
    cls_batch_size = 1
    rec_batch_size = 6

    # 当使用TRT时，分别给三个模型的runtime设置动态shape,并完成模型的创建.
    # 注意: 需要在检测模型创建完成后，再设置分类模型的动态输入并创建分类模型, 识别模型同理.
    # 如果用户想要自己改动检测模型的输入shape, 我们建议用户把检测模型的长和高设置为32的倍数.
    det_option = runtime_option
    det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                   [1, 3, 960, 960])
    # 用户可以把TRT引擎文件保存至本地
    #det_option.set_trt_cache_file(args.det_model  + "/det_trt_cache.trt")
    global det_model
    det_model = fd.vision.ocr.DBDetector(
        det_model_file, det_params_file, runtime_option=det_option)

    cls_option = runtime_option
    cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                   [cls_batch_size, 3, 48, 320],
                                   [cls_batch_size, 3, 48, 1024])
    # 用户可以把TRT引擎文件保存至本地
    #cls_option.set_trt_cache_file(args.cls_model  + "/cls_trt_cache.trt")
    global cls_model
    cls_model = fd.vision.ocr.Classifier(
        cls_model_file, cls_params_file, runtime_option=cls_option)

    rec_option = runtime_option
    rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                   [rec_batch_size, 3, 48, 320],
                                   [rec_batch_size, 3, 48, 2304])
    # 用户可以把TRT引擎文件保存至本地
    #rec_option.set_trt_cache_file(args.rec_model  + "/rec_trt_cache.trt")
    global rec_model
    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file,
        rec_params_file,
        rec_label_file,
        runtime_option=rec_option)

    # 创建PP-OCR，串联3个模型，其中cls_model可选，如无需求，可设置为None
    global ppocr_v3
    ppocr_v3 = fd.vision.ocr.PPOCRv3(
        det_model=det_model, cls_model=cls_model, rec_model=rec_model)

    # 给cls和rec模型设置推理时的batch size
    # 此值能为-1, 和1到正无穷
    # 当此值为-1时, cls和rec模型的batch size将默认和det模型检测出的框的数量相同
    ppocr_v3.cls_batch_size = cls_batch_size
    ppocr_v3.rec_batch_size = rec_batch_size


def predict(model, img_list):
    result_list = []
    # predict ppocr result
    for image in img_list:
        im = cv2.imread(image)
        result = model.predict(im)
        result_list.append(result)
    return result_list


def process_predict(image):
    # predict ppocr result
    im = cv2.imread(image)
    result = ppocr_v3.predict(im)
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
    # 对于三个模型，均采用同样的部署配置
    # 用户也可根据自行需求分别配置
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
                    args=(ppocr_v3.clone(),
                          imgs_list[i * image_num_each_thread:]))
            else:
                t = WrapperThread(
                    predict,
                    args=(ppocr_v3.clone(),
                          imgs_list[i * image_num_each_thread:(i + 1) *
                                    image_num_each_thread - 1]))
            threads.append(t)
            t.start()

        for i in range(thread_num):
            threads[i].join()

        for i in range(thread_num):
            for result in threads[i].get_result():
                print('thread:', i, ', result: ', result)
