import fastdeploy as fd
import cv2
import os
import numpy as np
import datetime
import json


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleClas model.")
    parser.add_argument(
        "--image", type=str, required=False, help="Path of test image file.")
    parser.add_argument(
        "--cpu_num_thread",
        type=int,
        default=12,
        help="default number of cpu thread.")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="device(gpu) id")
    parser.add_argument(
        "--iter_num",
        required=True,
        type=int,
        default=300,
        help="number of iterations for computing performace.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        nargs='+',
        type=str,
        default="ort",
        help="inference backend, ort, ov, trt, paddle.")
    args = parser.parse_args()
    return args


def build_option(args):
    option = fd.RuntimeOption()
    device = args.device
    backend = args.backend
    option.set_cpu_thread_num(args.cpu_num_thread)
    if device == "gpu":
        option.use_gpu(args.device_id)

    if backend == "trt":
        assert device == "gpu", "the trt backend need device==gpu"
        option.use_trt_backend()
    elif backend == "ov":
        assert device == "cpu", "the openvino backend need device==cpu"
        option.use_openvino_backend()
    elif backend == "paddle":
        option.use_paddle_backend()
    elif backend == "ort":
        option.use_ort_backend()
    else:
        print("%s is an unsupported backend" % backend)

    return option

args = parse_arguments()

option = build_option(args)
model_file = os.path.join(args.model, "inference.pdmodel")
params_file = os.path.join(args.model, "inference.pdiparams")
config_file = os.path.join(args.model, "inference_cls.yaml")
model = fd.vision.classification.PaddleClasModel(model_file, params_file, config_file, runtime_option=option)
model.enable_record_time_of_runtime()

import time
end2end_statis = list()
for i in range(args.iter_num):
    im = cv2.imread(args.image)
    start = time.time()
    result = model.predict(im)
    end2end_statis.append(time.time() - start)

runtime_statis = model.print_statis_info_of_runtime()

warmup_iter = args.iter_num / 5
end2end_statis = end2end_statis[warmup_iter:]

dump_result = dict()
dump_result["runtime"] = runtime_statis["avg_time"]
dump_result["end2end"] = np.mean(end2end_statis)

print(dump_result)
write_to_file(dump_result)

