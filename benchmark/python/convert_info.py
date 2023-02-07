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
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--txt_path', type=str, default='result.txt')
parser.add_argument('--domain', type=str, default='ppcls')
parser.add_argument(
    "--enable_collect_memory_info",
    type=bool,
    default=False,
    help="whether enable collect memory info")
args = parser.parse_args()
txt_path = args.txt_path
domain = args.domain
enable_collect_memory_info = args.enable_collect_memory_info

f1 = open(txt_path, "r")
lines = f1.readlines()
line_nums = len(lines)
ort_cpu_thread1 = dict()
ort_cpu_thread8 = dict()
ort_gpu = dict()
ov_cpu_thread1 = dict()
ov_cpu_thread8 = dict()
paddle_cpu_thread1 = dict()
paddle_cpu_thread8 = dict()
paddle_gpu = dict()
paddle_trt_gpu = dict()
paddle_trt_gpu_fp16 = dict()
trt_gpu = dict()
trt_gpu_fp16 = dict()
model_name_set = set()

for i in range(line_nums):
    if "====" in lines[i]:
        model_name = lines[i].strip().split("_model")[0][4:]
        model_name_set.add(model_name)
        runtime = "-"
        end2end = "-"
        cpu_rss_mb = "-"
        gpu_rss_mb = "-"
        if "Runtime(ms)" in lines[i + 1]:
            runtime_ori = lines[i + 1].split(": ")[1]
            # two decimal places
            runtime_list = runtime_ori.split(".")
            runtime = runtime_list[0] + "." + runtime_list[1][:2]
        if "End2End(ms)" in lines[i + 2]:
            end2end_ori = lines[i + 2].split(": ")[1]
            # two decimal places
            end2end_list = end2end_ori.split(".")
            end2end = end2end_list[0] + "." + end2end_list[1][:2]
        if enable_collect_memory_info:
            if "cpu_rss_mb" in lines[i + 3]:
                cpu_rss_mb_ori = lines[i + 3].split(": ")[1]
                # two decimal places
                cpu_rss_mb_list = cpu_rss_mb_ori.split(".")
                cpu_rss_mb = cpu_rss_mb_list[0] + "." + cpu_rss_mb_list[1][:2]
            if "gpu_rss_mb" in lines[i + 4]:
                gpu_rss_mb_ori = lines[i + 4].split(": ")[1].strip()
                gpu_rss_mb = str(gpu_rss_mb_ori) + ".0"
        if "ort_cpu_1" in lines[i]:
            ort_cpu_thread1[
                model_name] = runtime + "\t" + end2end + "\t" + cpu_rss_mb
        elif "ort_cpu_8" in lines[i]:
            ort_cpu_thread8[
                model_name] = runtime + "\t" + end2end + "\t" + cpu_rss_mb
        elif "ort_gpu" in lines[i]:
            ort_gpu[model_name] = runtime + "\t" + end2end + "\t" + gpu_rss_mb
        elif "ov_cpu_1" in lines[i]:
            ov_cpu_thread1[
                model_name] = runtime + "\t" + end2end + "\t" + cpu_rss_mb
        elif "ov_cpu_8" in lines[i]:
            ov_cpu_thread8[
                model_name] = runtime + "\t" + end2end + "\t" + cpu_rss_mb
        elif "paddle_cpu_1" in lines[i]:
            paddle_cpu_thread1[
                model_name] = runtime + "\t" + end2end + "\t" + cpu_rss_mb
        elif "paddle_cpu_8" in lines[i]:
            paddle_cpu_thread8[
                model_name] = runtime + "\t" + end2end + "\t" + cpu_rss_mb
        elif "paddle_gpu" in lines[i]:
            paddle_gpu[
                model_name] = runtime + "\t" + end2end + "\t" + gpu_rss_mb
        elif "paddle_trt_gpu" in lines[i]:
            paddle_trt_gpu[
                model_name] = runtime + "\t" + end2end + "\t" + gpu_rss_mb
        elif "paddle_trt_fp16_gpu" in lines[i]:
            paddle_trt_gpu_fp16[
                model_name] = runtime + "\t" + end2end + "\t" + gpu_rss_mb
        elif "trt_gpu" in lines[i]:
            trt_gpu[model_name] = runtime + "\t" + end2end + "\t" + gpu_rss_mb
        elif "trt_fp16_gpu" in lines[i]:
            trt_gpu_fp16[
                model_name] = runtime + "\t" + end2end + "\t" + gpu_rss_mb

f2 = open("struct_cpu_" + domain + ".txt", "w")
f2.writelines(
    "model_name\tthread_nums\tort_run\tort_end2end\tcpu_mem\tov_run\tov_end2end\tcpu_mem\tpaddle_run\tpaddle_end2end\tcpu_mem\n"
)
for model_name in model_name_set:
    lines1 = model_name + '\t1\t'
    lines2 = model_name + '\t8\t'
    if model_name in ort_cpu_thread1 and ort_cpu_thread1[model_name] != "":
        lines1 += ort_cpu_thread1[model_name] + '\t'
    else:
        lines1 += "-\t-\t-\t"
    if model_name in ov_cpu_thread1 and ov_cpu_thread1[model_name] != "":
        lines1 += ov_cpu_thread1[model_name] + '\t'
    else:
        lines1 += "-\t-\t-\t"
    if model_name in paddle_cpu_thread1 and paddle_cpu_thread1[
            model_name] != "":
        lines1 += paddle_cpu_thread1[model_name] + '\n'
    else:
        lines1 += "-\t-\t-\n"
    f2.writelines(lines1)
    if model_name in ort_cpu_thread8 and ort_cpu_thread8[model_name] != "":
        lines2 += ort_cpu_thread8[model_name] + '\t'
    else:
        lines2 += "-\t-\t-\t"
    if model_name in ov_cpu_thread8 and ov_cpu_thread8[model_name] != "":
        lines2 += ov_cpu_thread8[model_name] + '\t'
    else:
        lines2 += "-\t-\t-\t"
    if model_name in paddle_cpu_thread8 and paddle_cpu_thread8[
            model_name] != "":
        lines2 += paddle_cpu_thread8[model_name] + '\n'
    else:
        lines2 += "-\t-\t-\n"
    f2.writelines(lines2)
f2.close()

f3 = open("struct_gpu_" + domain + ".txt", "w")
f3.writelines(
    "model_name\tort_run\tort_end2end\tgpu_mem\tpaddle_run\tpaddle_end2end\tgpu_mem\tpaddle_trt_run\tpaddle_trt_end2end\tgpu_mem\tpaddle_trt_fp16_run\tpaddle_trt_fp16_end2end\tgpu_mem\ttrt_run\ttrt_end2end\tgpu_mem\ttrt_fp16_run\ttrt_fp16_end2end\tgpu_mem\n"
)
for model_name in model_name_set:
    lines1 = model_name + '\t'
    if model_name in ort_gpu and ort_gpu[model_name] != "":
        lines1 += ort_gpu[model_name] + '\t'
    else:
        lines1 += "-\t-\t-\t"
    if model_name in paddle_gpu and paddle_gpu[model_name] != "":
        lines1 += paddle_gpu[model_name] + '\t'
    else:
        lines1 += "-\t-\t-\t"
    if model_name in paddle_trt_gpu and paddle_trt_gpu[model_name] != "":
        lines1 += paddle_trt_gpu[model_name] + '\t'
    else:
        lines1 += "-\t-\t-\t"
    if model_name in paddle_trt_gpu_fp16 and paddle_trt_gpu_fp16[
            model_name] != "":
        lines1 += paddle_trt_gpu_fp16[model_name] + '\t'
    else:
        lines1 += "-\t-\t-\t"
    if model_name in trt_gpu and trt_gpu[model_name] != "":
        lines1 += trt_gpu[model_name] + '\t'
    else:
        lines1 += "-\t-\t-\t"
    if model_name in trt_gpu_fp16 and trt_gpu_fp16[model_name] != "":
        lines1 += trt_gpu_fp16[model_name] + '\n'
    else:
        lines1 += "-\t-\t-\n"
    f3.writelines(lines1)
f3.close()
