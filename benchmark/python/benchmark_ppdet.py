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

import fastdeploy as fd
import cv2
import os
import numpy as np
import time
from sympy import EX
from tqdm import tqdm


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
        default=8,
        help="default number of cpu thread.")
    parser.add_argument(
        "--device_id", type=int, default=0, help="device(gpu) id")
    parser.add_argument(
        "--profile_mode",
        type=str,
        default="runtime",
        help="runtime or end2end.")
    parser.add_argument(
        "--repeat",
        required=True,
        type=int,
        default=1000,
        help="number of repeats for profiling.")
    parser.add_argument(
        "--warmup",
        required=True,
        type=int,
        default=50,
        help="number of warmup for profiling.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="inference backend, default, ort, ov, trt, paddle, paddle_trt.")
    parser.add_argument(
        "--enable_trt_fp16",
        type=ast.literal_eval,
        default=False,
        help="whether enable fp16 in trt backend")
    parser.add_argument(
        "--enable_lite_fp16",
        type=ast.literal_eval,
        default=False,
        help="whether enable fp16 in Paddle Lite backend")
    parser.add_argument(
        "--enable_collect_memory_info",
        type=ast.literal_eval,
        default=False,
        help="whether enable collect memory info")
    parser.add_argument(
        "--include_h2d_d2h",
        type=ast.literal_eval,
        default=False,
        help="whether run profiling with h2d and d2h")
    args = parser.parse_args()
    return args


def build_option(args):
    option = fd.RuntimeOption()
    device = args.device
    backend = args.backend
    enable_trt_fp16 = args.enable_trt_fp16
    enable_lite_fp16 = args.enable_lite_fp16
    if args.profile_mode == "runtime":
        option.enable_profiling(args.include_h2d_d2h, args.repeat, args.warmup)
    option.set_cpu_thread_num(args.cpu_num_thread)
    if device == "gpu":
        option.use_gpu()
        if backend == "ort":
            option.use_ort_backend()
        elif backend == "paddle":
            option.use_paddle_backend()
        elif backend == "ov":
            option.use_openvino_backend()
            # Using GPU and CPU heterogeneous execution mode
            option.set_openvino_device("HETERO:GPU,CPU")
            # change name and shape for models
            option.set_openvino_shape_info({
                "image": [1, 3, 320, 320],
                "scale_factor": [1, 2]
            })
            # Set CPU up operator
            option.set_openvino_cpu_operators(["MulticlassNms"])
        elif backend in ["trt", "paddle_trt"]:
            option.use_trt_backend()
            if backend == "paddle_trt":
                option.enable_paddle_to_trt()
            if enable_trt_fp16:
                option.enable_trt_fp16()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with GPU, only support default/ort/paddle/trt/paddle_trt now, {} is not supported.".
                format(backend))
    elif device == "cpu":
        if backend == "ort":
            option.use_ort_backend()
        elif backend == "ov":
            option.use_openvino_backend()
        elif backend == "paddle":
            option.use_paddle_backend()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with CPU, only support default/ort/ov/paddle now, {} is not supported.".
                format(backend))
    elif device == "kunlunxin":
        option.use_kunlunxin()
        if backend == "lite":
            option.use_lite_backend()
        elif backend == "ort":
            option.use_ort_backend()
        elif backend == "paddle":
            option.use_paddle_backend()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with CPU, only support default/ort/lite/paddle now, {} is not supported.".
                format(backend))
    elif device == "ascend":
        option.use_ascend()
        if backend == "lite":
            option.use_lite_backend()
            if enable_lite_fp16:
                option.enable_lite_fp16()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with CPU, only support default/lite now, {} is not supported.".
                format(backend))
    else:
        raise Exception(
            "Only support device CPU/GPU/Kunlunxin/Ascend now, {} is not supported.".
            format(device))

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


if __name__ == '__main__':

    args = parse_arguments()
    option = build_option(args)
    model_file = os.path.join(args.model, "model.pdmodel")
    params_file = os.path.join(args.model, "model.pdiparams")
    config_file = os.path.join(args.model, "infer_cfg.yml")

    gpu_id = args.device_id
    enable_collect_memory_info = args.enable_collect_memory_info
    dump_result = dict()
    cpu_mem = list()
    gpu_mem = list()
    gpu_util = list()
    if args.device == "cpu":
        file_path = args.model + "_model_" + args.backend + "_" + \
            args.device + "_" + str(args.cpu_num_thread) + ".txt"
    else:
        if args.enable_trt_fp16:
            file_path = args.model + "_model_" + args.backend + "_fp16_" + args.device + ".txt"
        else:
            file_path = args.model + "_model_" + args.backend + "_" + args.device + ".txt"
    f = open(file_path, "w")
    f.writelines("===={}====: \n".format(os.path.split(file_path)[-1][:-4]))

    try:
        if "ppyoloe" in args.model:
            model = fd.vision.detection.PPYOLOE(
                model_file, params_file, config_file, runtime_option=option)
        elif "picodet" in args.model:
            model = fd.vision.detection.PicoDet(
                model_file, params_file, config_file, runtime_option=option)
        elif "yolox" in args.model:
            model = fd.vision.detection.PaddleYOLOX(
                model_file, params_file, config_file, runtime_option=option)
        elif "yolov3" in args.model:
            model = fd.vision.detection.YOLOv3(
                model_file, params_file, config_file, runtime_option=option)
        elif "yolov8" in args.model:
            model = fd.vision.detection.PaddleYOLOv8(
                model_file, params_file, config_file, runtime_option=option)
        elif "ppyolo_r50vd_dcn_1x_coco" in args.model or "ppyolov2_r101vd_dcn_365e_coco" in args.model:
            model = fd.vision.detection.PPYOLO(
                model_file, params_file, config_file, runtime_option=option)
        elif "faster_rcnn" in args.model:
            model = fd.vision.detection.FasterRCNN(
                model_file, params_file, config_file, runtime_option=option)
        else:
            raise Exception("model {} not support now in ppdet series".format(
                args.model))
        if enable_collect_memory_info:
            import multiprocessing
            import subprocess
            import psutil
            import signal
            import cpuinfo
            enable_gpu = args.device == "gpu"
            monitor = Monitor(enable_gpu, gpu_id)
            monitor.start()

        im_ori = cv2.imread(args.image)
        if args.profile_mode == "runtime":
            result = model.predict(im_ori)
            profile_time = model.get_profile_time()
            dump_result["runtime"] = profile_time * 1000
            f.writelines("Runtime(ms): {} \n".format(
                str(dump_result["runtime"])))
            print("Runtime(ms): {} \n".format(str(dump_result["runtime"])))
        else:
            # end2end
            for i in range(args.warmup):
                result = model.predict(im_ori)

            start = time.time()
            for i in tqdm(range(args.repeat)):
                result = model.predict(im_ori)
            end = time.time()
            dump_result["end2end"] = ((end - start) / args.repeat) * 1000.0
            f.writelines("End2End(ms): {} \n".format(
                str(dump_result["end2end"])))
            print("End2End(ms): {} \n".format(str(dump_result["end2end"])))

        if enable_collect_memory_info:
            monitor.stop()
            mem_info = monitor.output()
            dump_result["cpu_rss_mb"] = mem_info['cpu'][
                'memory.used'] if 'cpu' in mem_info else 0
            dump_result["gpu_rss_mb"] = mem_info['gpu'][
                'memory.used'] if 'gpu' in mem_info else 0
            dump_result["gpu_util"] = mem_info['gpu'][
                'utilization.gpu'] if 'gpu' in mem_info else 0

        if enable_collect_memory_info:
            f.writelines("cpu_rss_mb: {} \n".format(
                str(dump_result["cpu_rss_mb"])))
            f.writelines("gpu_rss_mb: {} \n".format(
                str(dump_result["gpu_rss_mb"])))
            f.writelines("gpu_util: {} \n".format(
                str(dump_result["gpu_util"])))
            print("cpu_rss_mb: {} \n".format(str(dump_result["cpu_rss_mb"])))
            print("gpu_rss_mb: {} \n".format(str(dump_result["gpu_rss_mb"])))
            print("gpu_util: {} \n".format(str(dump_result["gpu_util"])))
    except Exception as e:
        f.writelines("!!!!!Infer Failed\n")
        raise e

    f.close()
