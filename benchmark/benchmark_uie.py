import numpy as np
import os
import time
import distutils.util
import sys
import json

from paddlenlp.utils.log import logger
import fastdeploy as fd
from fastdeploy.text import UIEModel, SchemaLanguage
import pynvml
import psutil
import GPUtil
import multiprocessing


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="The directory of model and tokenizer.")
    parser.add_argument(
        "--data_path", required=True, help="The path of uie data.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['gpu', 'cpu'],
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default='pp',
        choices=['ort', 'pp', 'trt', 'pp-trt', 'openvino'],
        help="The inference runtime backend.")
    parser.add_argument(
        "--device_id", type=int, default=0, help="device(gpu) id")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="The max length of sequence.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="The interval of logging.")
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=1,
        help="The number of threads when inferring on cpu.")
    parser.add_argument(
        "--use_fp16",
        type=distutils.util.strtobool,
        default=False,
        help="Use FP16 mode")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    if args.device == 'cpu':
        option.use_cpu()
        option.set_cpu_thread_num(args.cpu_num_threads)
    else:
        option.use_gpu(args.device_id)
    if args.backend == 'pp':
        option.use_paddle_backend()
    elif args.backend == 'ort':
        option.use_ort_backend()
    elif args.backend == 'openvino':
        option.use_openvino_backend()
    else:
        option.use_trt_backend()
        if args.backend == 'pp-trt':
            option.enable_paddle_to_trt()
            option.enable_paddle_trt_collect_shape()
        trt_file = os.path.join(args.model_dir, "infer.trt")
        option.set_trt_input_shape(
            'input_ids',
            min_shape=[1, args.max_length],
            opt_shape=[args.batch_size, args.max_length],
            max_shape=[args.batch_size, args.max_length])
        option.set_trt_input_shape(
            'token_type_ids',
            min_shape=[1, args.max_length],
            opt_shape=[args.batch_size, args.max_length],
            max_shape=[args.batch_size, args.max_length])
        option.set_trt_input_shape(
            'pos_ids',
            min_shape=[1, args.max_length],
            opt_shape=[args.batch_size, args.max_length],
            max_shape=[args.batch_size, args.max_length])
        option.set_trt_input_shape(
            'att_mask',
            min_shape=[1, args.max_length],
            opt_shape=[args.batch_size, args.max_length],
            max_shape=[args.batch_size, args.max_length])
        if args.use_fp16:
            option.enable_trt_fp16()
            trt_file = trt_file + ".fp16"
        option.set_trt_cache_file(trt_file)
    return option


def get_current_memory_mb(gpu_id=None):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    if gpu_id is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return cpu_mem, gpu_mem


def get_current_gputil(gpu_id):
    GPUs = GPUtil.getGPUs()
    gpu_load = GPUs[gpu_id].load
    return gpu_load


def sample_gpuutil(gpu_id, gpu_utilization=[]):
    while True:
        gpu_utilization.append(get_current_gputil(gpu_id))
        time.sleep(0.01)


def get_dataset(data_path, max_seq_len=512):
    json_lines = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line['content'].strip()
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            json_lines.append(json_line)

    return json_lines


def run_inference(ds, uie):
    for i, sample in enumerate(ds):
        uie.set_schema([sample['prompt']])
        result = uie.predict([sample['content']])
        if (i + 1) % args.log_interval == 0:
            runtime_statis = uie.print_statis_info_of_runtime()
            print(f"Step {i + 1}:")
            print(runtime_statis)
            print()

    runtime_statis = uie.print_statis_info_of_runtime()
    print(f"Final:")
    print(runtime_statis)
    print()


if __name__ == '__main__':
    args = parse_arguments()
    runtime_option = build_option(args)
    model_path = os.path.join(args.model_dir, "inference.pdmodel")
    param_path = os.path.join(args.model_dir, "inference.pdiparams")
    vocab_path = os.path.join(args.model_dir, "vocab.txt")

    ds = get_dataset(args.data_path)
    schema = ["时间"]
    uie = UIEModel(
        model_path,
        param_path,
        vocab_path,
        position_prob=0.5,
        max_length=args.max_length,
        schema=schema,
        runtime_option=runtime_option,
        schema_language=SchemaLanguage.ZH)

    uie.enable_record_time_of_runtime()
    run_inference(ds, uie)
