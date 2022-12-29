import numpy as np
import os
import time
import distutils.util
import sys
import json

import fastdeploy as fd
from fastdeploy.text import UIEModel, SchemaLanguage


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
        default='paddle',
        choices=['ort', 'paddle', 'trt', 'paddle_trt', 'ov'],
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
        "--cpu_num_threads",
        type=int,
        default=8,
        help="The number of threads when inferring on cpu.")
    parser.add_argument(
        "--enable_trt_fp16",
        type=distutils.util.strtobool,
        default=False,
        help="whether enable fp16 in trt backend")
    parser.add_argument(
        "--epoch", type=int, default=1, help="The epoch of test")
    parser.add_argument(
        "--enable_collect_memory_info",
        type=ast.literal_eval,
        default=False,
        help="whether enable collect memory info")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    if args.device == 'cpu':
        option.use_cpu()
        option.set_cpu_thread_num(args.cpu_num_threads)
    else:
        option.use_gpu(args.device_id)
    if args.backend == 'paddle':
        option.use_paddle_backend()
    elif args.backend == 'ort':
        option.use_ort_backend()
    elif args.backend == 'ov':
        option.use_openvino_backend()
    else:
        option.use_trt_backend()
        if args.backend == 'paddle_trt':
            option.enable_paddle_to_trt()
            option.enable_paddle_trt_collect_shape()
        trt_file = os.path.join(args.model_dir, "infer.trt")
        option.set_trt_input_shape(
            'input_ids',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        option.set_trt_input_shape(
            'token_type_ids',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        option.set_trt_input_shape(
            'pos_ids',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        option.set_trt_input_shape(
            'att_mask',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        if args.enable_trt_fp16:
            option.enable_trt_fp16()
            trt_file = trt_file + ".fp16"
        option.set_trt_cache_file(trt_file)
    return option


class StatBase(object):
    """StatBase"""
    nvidia_smi_path = "nvidia-smi"
    gpu_keys = ('index', 'uuid', 'name', 'timestamp', 'memory.total',
                'memory.free', 'memory.used', 'utilization.gpu',
                'utilization.memory')
    nu_opt = ',nounits'
    cpu_keys = ('cpu.util', 'memory.util', 'memory.used')


class Monitor(StatBase):
    """Monitor"""

    def __init__(self, use_gpu=False, gpu_id=0, interval=0.1):
        self.result = {}
        self.gpu_id = gpu_id
        self.use_gpu = use_gpu
        self.interval = interval
        self.cpu_stat_q = multiprocessing.Queue()

    def start(self):
        cmd = '%s --id=%s --query-gpu=%s --format=csv,noheader%s -lms 50' % (
            StatBase.nvidia_smi_path, self.gpu_id, ','.join(StatBase.gpu_keys),
            StatBase.nu_opt)
        if self.use_gpu:
            self.gpu_stat_worker = subprocess.Popen(
                cmd,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                close_fds=True,
                preexec_fn=os.setsid)
        # cpu stat
        pid = os.getpid()
        self.cpu_stat_worker = multiprocessing.Process(
            target=self.cpu_stat_func,
            args=(self.cpu_stat_q, pid, self.interval))
        self.cpu_stat_worker.start()

    def stop(self):
        try:
            if self.use_gpu:
                os.killpg(self.gpu_stat_worker.pid, signal.SIGUSR1)
            # os.killpg(p.pid, signal.SIGTERM)
            self.cpu_stat_worker.terminate()
            self.cpu_stat_worker.join(timeout=0.01)
        except Exception as e:
            print(e)
            return

        # gpu
        if self.use_gpu:
            lines = self.gpu_stat_worker.stdout.readlines()
            lines = [
                line.strip().decode("utf-8") for line in lines
                if line.strip() != ''
            ]
            gpu_info_list = [{
                k: v
                for k, v in zip(StatBase.gpu_keys, line.split(', '))
            } for line in lines]
            if len(gpu_info_list) == 0:
                return
            result = gpu_info_list[0]
            for item in gpu_info_list:
                for k in item.keys():
                    if k not in ["name", "uuid", "timestamp"]:
                        result[k] = max(int(result[k]), int(item[k]))
                    else:
                        result[k] = max(result[k], item[k])
            self.result['gpu'] = result

        # cpu
        cpu_result = {}
        if self.cpu_stat_q.qsize() > 0:
            cpu_result = {
                k: v
                for k, v in zip(StatBase.cpu_keys, self.cpu_stat_q.get())
            }
        while not self.cpu_stat_q.empty():
            item = {
                k: v
                for k, v in zip(StatBase.cpu_keys, self.cpu_stat_q.get())
            }
            for k in StatBase.cpu_keys:
                cpu_result[k] = max(cpu_result[k], item[k])
        cpu_result['name'] = cpuinfo.get_cpu_info()['brand_raw']
        self.result['cpu'] = cpu_result

    def output(self):
        return self.result

    def cpu_stat_func(self, q, pid, interval=0.0):
        """cpu stat function"""
        stat_info = psutil.Process(pid)
        while True:
            # pid = os.getpid()
            cpu_util, mem_util, mem_use = stat_info.cpu_percent(
            ), stat_info.memory_percent(), round(stat_info.memory_info().rss /
                                                 1024.0 / 1024.0, 4)
            q.put([cpu_util, mem_util, mem_use])
            time.sleep(interval)
        return


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


if __name__ == '__main__':
    args = parse_arguments()
    runtime_option = build_option(args)
    model_path = os.path.join(args.model_dir, "inference.pdmodel")
    param_path = os.path.join(args.model_dir, "inference.pdiparams")
    vocab_path = os.path.join(args.model_dir, "vocab.txt")

    gpu_id = args.device_id
    enable_collect_memory_info = args.enable_collect_memory_info
    dump_result = dict()
    end2end_statis = list()
    cpu_mem = list()
    gpu_mem = list()
    gpu_util = list()
    if args.device == "cpu":
        file_path = args.model_dir + "_model_" + args.backend + "_" + \
            args.device + "_" + str(args.cpu_num_threads) + ".txt"
    else:
        if args.enable_trt_fp16:
            file_path = args.model_dir + "_model_" + \
                args.backend + "_fp16_" + args.device + ".txt"
        else:
            file_path = args.model_dir + "_model_" + args.backend + "_" + args.device + ".txt"
    f = open(file_path, "w")
    f.writelines("===={}====: \n".format(os.path.split(file_path)[-1][:-4]))

    ds = get_dataset(args.data_path)
    schema = ["时间"]
    uie = UIEModel(
        model_path,
        param_path,
        vocab_path,
        position_prob=0.5,
        max_length=args.max_length,
        batch_size=args.batch_size,
        schema=schema,
        runtime_option=runtime_option,
        schema_language=SchemaLanguage.ZH)

    try:
        if enable_collect_memory_info:
            import multiprocessing
            import subprocess
            import psutil
            import signal
            import cpuinfo
            enable_gpu = args.device == "gpu"
            monitor = Monitor(enable_gpu, gpu_id)
            monitor.start()
        uie.enable_record_time_of_runtime()

        for ep in range(args.epoch):
            for i, sample in enumerate(ds):
                curr_start = time.time()
                uie.set_schema([sample['prompt']])
                result = uie.predict([sample['content']])
                end2end_statis.append(time.time() - curr_start)
        runtime_statis = uie.print_statis_info_of_runtime()

        warmup_iter = args.epoch * len(ds) // 5

        end2end_statis_repeat = end2end_statis[warmup_iter:]
        if enable_collect_memory_info:
            monitor.stop()
            mem_info = monitor.output()
            dump_result["cpu_rss_mb"] = mem_info['cpu'][
                'memory.used'] if 'cpu' in mem_info else 0
            dump_result["gpu_rss_mb"] = mem_info['gpu'][
                'memory.used'] if 'gpu' in mem_info else 0
            dump_result["gpu_util"] = mem_info['gpu'][
                'utilization.gpu'] if 'gpu' in mem_info else 0

        dump_result["runtime"] = runtime_statis["avg_time"] * 1000
        dump_result["end2end"] = np.mean(end2end_statis_repeat) * 1000

        time_cost_str = f"Runtime(ms): {dump_result['runtime']}\n" \
                        f"End2End(ms): {dump_result['end2end']}\n"
        f.writelines(time_cost_str)
        print(time_cost_str)

        if enable_collect_memory_info:
            mem_info_str = f"cpu_rss_mb: {dump_result['cpu_rss_mb']}\n" \
                           f"gpu_rss_mb: {dump_result['gpu_rss_mb']}\n" \
                           f"gpu_util: {dump_result['gpu_util']}\n"
            f.writelines(mem_info_str)
            print(mem_info_str)
    except:
        f.writelines("!!!!!Infer Failed\n")

    f.close()
