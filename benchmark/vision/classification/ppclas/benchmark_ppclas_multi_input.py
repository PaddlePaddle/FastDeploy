import fastdeploy as fd
import cv2
import os
from tqdm import trange
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
        "--input_name",
        type=str,
        required=False,
        default="inputs",
        help="input name of inference file.")
    parser.add_argument(
        "--topk", type=int, default=1, help="Return topk results.")
    parser.add_argument(
        "--cpu_num_thread",
        type=int,
        default=12,
        help="default number of cpu thread.")
    parser.add_argument(
        "--size",
        nargs='+',
        type=int,
        default=[1, 3, 224, 224],
        help="size of inference array.")
    parser.add_argument(
        "--iter_num",
        required=True,
        type=int,
        default=30,
        help="number of iterations for computing performace.")
    parser.add_argument(
        "--device",
        nargs='+',
        type=str,
        default=['cpu', 'cpu', 'cpu', 'gpu', 'gpu', 'gpu'],
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        nargs='+',
        type=str,
        default=['ort', 'paddle', 'ov', 'ort', 'trt', 'paddle'],
        help="inference backend.")
    args = parser.parse_args()
    backend_list = ['ov', 'trt', 'ort', 'paddle']
    device_list = ['cpu', 'gpu']
    assert len(args.device) == len(
        args.backend), "the same number of --device and --backend is requested"
    assert args.iter_num > 10, "--iter_num has to bigger than 10"
    assert len(args.size
               ) == 4, "size should include 4 values, e.g., --size 1 3 300 300"
    for b in args.backend:
        assert b in backend_list, "%s backend is not supported" % b
    for d in args.device:
        assert d in device_list, "%s device is not supported" % d
    return args


def build_option(index, args):
    option = fd.RuntimeOption()
    device = args.device[index]
    backend = args.backend[index]
    option.set_cpu_thread_num(args.cpu_num_thread)
    if device == "gpu":
        option.use_gpu()

    if backend == "trt":
        assert device == "gpu", "the trt backend need device==gpu"
        option.use_trt_backend()
        option.set_trt_input_shape(args.input_name, args.size)
    elif backend == "ov":
        assert device == "cpu", "the openvino backend need device==cpu"
        option.use_openvino_backend()

    elif backend == "paddle":
        option.use_paddle_backend()

    elif backend == "ort":
        option.use_ort_backend()

    else:
        print("%s is an unsupported backend" % backend)

    print("============= inference using %s backend on %s device ============="
          % (args.backend[index], args.device[index]))
    return option


args = parse_arguments()

save_dict = dict()

for index, device_name in enumerate(args.device):
    if device_name not in save_dict:
        save_dict[device_name] = dict()

    # 配置runtime，加载模型
    runtime_option = build_option(index, args)

    model_file = os.path.join(args.model, "inference.pdmodel")
    params_file = os.path.join(args.model, "inference.pdiparams")
    config_file = os.path.join(args.model, "inference_cls.yaml")
    model = fd.vision.classification.PaddleClasModel(
        model_file, params_file, config_file, runtime_option=runtime_option)

    # 创建要输入的向量
    channel = args.size[1]
    height = args.size[2]
    width = args.size[3]
    input_array = np.random.randint(
        0, high=255, size=(height, width, channel), dtype=np.uint8)

    # 如果有输入图片，则使用输入的图片进行推理
    if args.image:
        input_array = cv2.imread(args.image)
    model_name = args.model.split('/')
    model_name = model_name[-1] if model_name[-1] else model_name[-2]
    print(" Model: ", model_name, " Input shape: ", input_array.shape)
    start_time = datetime.datetime.now()
    model.enable_record_time_of_runtime()
    warmup_iter = args.iter_num // 5
    warmup_end2end_time = 0
    if "iter_num" not in save_dict:
        save_dict["iter_num"] = args.iter_num
    if "warmup_iter" not in save_dict:
        save_dict["warmup_iter"] = warmup_iter
    if "cpu_num_thread" not in save_dict:
        save_dict["cpu_num_thread"] = args.cpu_num_thread
    for i in trange(args.iter_num, desc="Inference Progress"):
        if i == warmup_iter:
            # 计算warmup端到端总时间(s)
            warmup_time = datetime.datetime.now()
            warmup_end2end_time = warmup_time - start_time
            warmup_end2end_time = (
                warmup_end2end_time.days * 24 * 60 * 60 +
                warmup_end2end_time.seconds
            ) * 1000 + warmup_end2end_time.microseconds / 1000
        result = model.predict(input_array, args.topk)
    end_time = datetime.datetime.now()
    # 计算端到端（前处理，推理，后处理）的总时间
    statis_info_of_runtime_dict = model.print_statis_info_of_runtime()
    end2end_time = end_time - start_time
    end2end_time = (end2end_time.days * 24 * 60 * 60 + end2end_time.seconds
                    ) * 1000 + end2end_time.microseconds / 1000
    remain_end2end_time = end2end_time - warmup_end2end_time
    pre_post_process = end2end_time - statis_info_of_runtime_dict[
        "total_time"] * 1000
    end2end = remain_end2end_time / (args.iter_num - warmup_iter)
    runtime = statis_info_of_runtime_dict["avg_time"] * 1000
    print("Total time of end2end: %s ms" % str(end2end_time))
    print("Average time of end2end exclude warmup step: %s ms" % str(end2end))
    print("Total time of preprocess and postprocess in warmup step: %s ms" %
          str(warmup_end2end_time - statis_info_of_runtime_dict["warmup_time"]
              * 1000))
    print(
        "Average time of preprocess and postprocess exclude warmup step: %s ms"
        % str((remain_end2end_time - statis_info_of_runtime_dict["remain_time"]
               * 1000) / (args.iter_num - warmup_iter)))
    # 结构化输出
    backend_name = args.backend[index]
    save_dict[device_name][backend_name] = {
        "end2end": end2end,
        "runtime": runtime
    }
    json_str = json.dumps(save_dict)
    with open("%s.json" % model_name, 'w', encoding='utf-8') as fw:
        json.dump(json_str, fw, indent=4, ensure_ascii=False)
