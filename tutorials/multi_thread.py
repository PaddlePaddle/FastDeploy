import numpy as np
from threading import Thread
import fastdeploy as fd
import cv2
import os
import psutil


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
        option.use_gpu()

    if args.device.lower() == "ipu":
        option.use_ipu()

    if args.use_trt:
        option.use_trt_backend()
    return option


def predict(model, img_list, topk):
    result_list = []
    # 预测图片分类结果
    for image in img_list:
        im = cv2.imread(image)
        result = model.predict(im, topk)
        result_list.append(result)
    return result_list


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

    thread_num = args.thread_num
    imgs_list = get_image_list(args.image_path)
    # 配置runtime，加载模型
    runtime_option = build_option(args)

    model_file = os.path.join(args.model, "inference.pdmodel")
    params_file = os.path.join(args.model, "inference.pdiparams")
    config_file = os.path.join(args.model, "inference_cls.yaml")
    model = fd.vision.classification.PaddleClasModel(
        model_file, params_file, config_file, runtime_option=runtime_option)
    threads = []
    image_num_each_thread = int(len(imgs_list) / thread_num)
    for i in range(thread_num):
        if i == thread_num - 1:
            t = WrapperThread(
                predict,
                args=(model, imgs_list[i * image_num_each_thread:], i))
        else:
            t = WrapperThread(
                predict,
                args=(model.clone(), imgs_list[i * image_num_each_thread:(
                    i + 1) * image_num_each_thread], i))
        threads.append(t)
        t.start()

    for i in range(thread_num):
        threads[i].join()

    for i in range(thread_num):
        for result in threads[i].get_result():
            print('thread:', i, ', result: ', result)
