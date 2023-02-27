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
parser.add_argument('--log_path', type=str, default='result.log')
parser.add_argument('--domain', type=str, default='ppcls')

args = parser.parse_args()
log_path = args.log_path
domain = args.domain

f1 = open(log_path, "r")
lines = f1.readlines()
line_nums = len(lines)
# cpu
paddle_cpu_runtime = dict()
paddle_cpu_runtime_all = dict()
paddle_cpu_e2e = dict()
ort_cpu_runtime = dict()
ort_cpu_runtime_all = dict()
ort_cpu_e2e = dict()
ov_cpu_runtime = dict()
ov_cpu_runtime_all = dict()
ov_cpu_e2e = dict()
paddle_cpu_mem = dict()
ort_cpu_mem = dict()
ov_cpu_mem = dict()
# gpu
paddle_gpu_runtime = dict()
paddle_gpu_runtime_all = dict()
paddle_gpu_e2e = dict()
ort_gpu_runtime = dict()
ort_gpu_runtime_all = dict()
ort_gpu_e2e = dict()
paddle_trt_gpu_runtime = dict()
paddle_trt_gpu_runtime_all = dict()
paddle_trt_gpu_e2e = dict()
paddle_trt_gpu_fp16_runtime = dict()
paddle_trt_gpu_fp16_runtime_all = dict()
paddle_trt_gpu_fp16_e2e = dict()
trt_gpu_runtime = dict()
trt_gpu_runtime_all = dict()
trt_gpu_e2e = dict()
trt_gpu_fp16_runtime = dict()
trt_gpu_fp16_runtime_all = dict()
trt_gpu_fp16_e2e = dict()
paddle_gpu_mem = dict()
ort_gpu_mem = dict()
paddle_trt_gpu_mem = dict()
paddle_trt_gpu_fp16_mem = dict()
trt_gpu_mem = dict()
trt_gpu_fp16_mem = dict()
model_name_set = set()

for i in range(line_nums):
    # init
    model_name = ""
    device = ""
    profile_mode = "runtime"
    include_h2d_d2h = 0
    use_fp16 = 0
    collect_memory_info = 0
    runtime = "-"
    end2end = "-"
    cpu_rss_mb = "-"
    gpu_rss_mb = "-"
    gpu_util = "-"
    if "======= Model Info =======" in lines[i]:
        model_name = lines[i + 1].strip().split(": ")[1]
        model_name_set.add(model_name)
        profile_mode = lines[i + 2].strip().split(": ")[1]
        if profile_mode == "runtime":
            include_h2d_d2h = int(lines[i + 3].strip().split(": ")[1])
        if profile_mode == "runtime":
            backend_info_start = i + 5
        else:
            backend_info_start = i + 4
        device = lines[backend_info_start + 3].strip().split(": ")[1]
        start = backend_info_start + 3
        # find next model info
        while backend_info_start < line_nums:
            if "======= Model Info =======" in lines[backend_info_start]:
                end = backend_info_start - 1
                break
            backend_info_start += 1
        for j in range(start, end):
            if "backend: " in lines[j]:
                backend = lines[j].strip().split(": ")[1]
            if "use_fp16: " in lines[j]:
                use_fp16 = int(lines[j].strip().split(": ")[1])
            if "collect_memory_info: " in lines[j]:
                collect_memory_info = int(lines[j].strip().split(": ")[1])
            if "Runtime(ms)" in lines[j]:
                runtime_ori = lines[j].split(": ")[1]
                # two decimal places
                runtime_list = runtime_ori.split(".")
                runtime = runtime_list[0] + "." + runtime_list[1][:2]
            if "End2End(ms)" in lines[j]:
                end2end_ori = lines[j].split(": ")[1]
                # two decimal places
                end2end_list = end2end_ori.split(".")
                end2end = end2end_list[0] + "." + end2end_list[1][:2]
            if "cpu_rss_mb: " in lines[j]:
                cpu_rss_mb_ori = lines[j].strip().split(": ")[1]
                # two decimal places
                cpu_rss_mb_list = cpu_rss_mb_ori.split(".")
                cpu_rss_mb = cpu_rss_mb_list[0] + "." + cpu_rss_mb_list[1][:2]
            if "gpu_rss_mb: " in lines[j]:
                gpu_rss_mb_ori = lines[j].strip().split(": ")[1][:-3]
                gpu_rss_mb = str(gpu_rss_mb_ori) + ".0"
            if "gpu_util: " in lines[j]:
                gpu_util = lines[j].strip().split(": ")[1]
    if device == "cpu":
        if backend == "paddle":
            if profile_mode == "runtime":
                if include_h2d_d2h == 0:
                    paddle_cpu_runtime[model_name] = runtime
                else:
                    paddle_cpu_runtime_all[model_name] = runtime
            elif profile_mode == "end2end":
                if collect_memory_info:
                    paddle_cpu_mem[model_name] = cpu_rss_mb
                else:
                    paddle_cpu_e2e[model_name] = end2end
        elif backend == "ort":
            if profile_mode == "runtime":
                if include_h2d_d2h == 0:
                    ort_cpu_runtime[model_name] = runtime
                else:
                    ort_cpu_runtime_all[model_name] = runtime
            elif profile_mode == "end2end":
                if collect_memory_info:
                    ort_cpu_mem[model_name] = cpu_rss_mb
                else:
                    ort_cpu_e2e[model_name] = end2end
        elif backend == "ov":
            if profile_mode == "runtime":
                if include_h2d_d2h == 0:
                    ov_cpu_runtime[model_name] = runtime
                else:
                    ov_cpu_runtime_all[model_name] = runtime
            elif profile_mode == "end2end":
                if collect_memory_info:
                    ov_cpu_mem[model_name] = cpu_rss_mb
                else:
                    ov_cpu_e2e[model_name] = end2end
    elif device == "gpu":
        if backend == "paddle":
            if profile_mode == "runtime":
                if include_h2d_d2h == 0:
                    paddle_gpu_runtime[model_name] = runtime
                else:
                    paddle_gpu_runtime_all[model_name] = runtime
            elif profile_mode == "end2end":
                if collect_memory_info:
                    paddle_gpu_mem[model_name] = gpu_rss_mb
                else:
                    paddle_gpu_e2e[model_name] = end2end
        elif backend == "ort":
            if profile_mode == "runtime":
                if include_h2d_d2h == 0:
                    ort_gpu_runtime[model_name] = runtime
                else:
                    ort_gpu_runtime_all[model_name] = runtime
            elif profile_mode == "end2end":
                if collect_memory_info:
                    ort_gpu_mem[model_name] = gpu_rss_mb
                else:
                    ort_gpu_e2e[model_name] = end2end
        elif backend == "paddle_trt":
            if profile_mode == "runtime":
                if include_h2d_d2h == 0:
                    if use_fp16:
                        paddle_trt_gpu_fp16_runtime[model_name] = runtime
                    else:
                        paddle_trt_gpu_runtime[model_name] = runtime
                else:
                    if use_fp16:
                        paddle_trt_gpu_fp16_runtime_all[model_name] = runtime
                    else:
                        paddle_trt_gpu_runtime_all[model_name] = runtime
            elif profile_mode == "end2end":
                if collect_memory_info:
                    if use_fp16:
                        paddle_trt_gpu_fp16_mem[model_name] = gpu_rss_mb
                    else:
                        paddle_trt_gpu_mem[model_name] = gpu_rss_mb
                else:
                    if use_fp16:
                        paddle_trt_gpu_fp16_e2e[model_name] = end2end
                    else:
                        paddle_trt_gpu_e2e[model_name] = end2end
        elif backend == "trt":
            if profile_mode == "runtime":
                if include_h2d_d2h == 0:
                    if use_fp16:
                        trt_gpu_fp16_runtime[model_name] = runtime
                    else:
                        trt_gpu_runtime[model_name] = runtime
                else:
                    if use_fp16:
                        trt_gpu_fp16_runtime_all[model_name] = runtime
                    else:
                        trt_gpu_runtime_all[model_name] = runtime
            elif profile_mode == "end2end":
                if collect_memory_info:
                    if use_fp16:
                        trt_gpu_fp16_mem[model_name] = gpu_rss_mb
                    else:
                        trt_gpu_mem[model_name] = gpu_rss_mb
                else:
                    if use_fp16:
                        trt_gpu_fp16_e2e[model_name] = end2end
                    else:
                        trt_gpu_e2e[model_name] = end2end
    else:
        continue

f2 = open("struct_cpu_" + domain + ".txt", "w")
f2.writelines(
    "model_name\tthread_nums\tPaddle_Inference_runtime\tORT_runtime\tOpenVINO_runtime\tPaddle_Inference_h2d_runtime_d2h\tORT_h2d_runtime_d2h\tOpenVINO_h2d_runtime_d2h\tPaddle_Inference_e2e\tORT_e2e\tOpenVINO_e2e\tPaddle_Inference_cpu_mem\tORT_cpu_mem\tOpenVINO_cpu_mem\n"
)
for model_name in model_name_set:
    lines1 = model_name + '\t8\t'
    if model_name in paddle_cpu_runtime and paddle_cpu_runtime[
            model_name] != "":
        lines1 += paddle_cpu_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_cpu_runtime and ort_cpu_runtime[model_name] != "":
        lines1 += ort_cpu_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ov_cpu_runtime and ov_cpu_runtime[model_name] != "":
        lines1 += ov_cpu_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_cpu_runtime_all and paddle_cpu_runtime_all[
            model_name] != "":
        lines1 += paddle_cpu_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_cpu_runtime_all and ort_cpu_runtime_all[
            model_name] != "":
        lines1 += ort_cpu_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ov_cpu_runtime_all and ov_cpu_runtime_all[
            model_name] != "":
        lines1 += ov_cpu_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_cpu_e2e and paddle_cpu_e2e[model_name] != "":
        lines1 += paddle_cpu_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_cpu_e2e and ort_cpu_e2e[model_name] != "":
        lines1 += ort_cpu_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ov_cpu_e2e and ov_cpu_e2e[model_name] != "":
        lines1 += ov_cpu_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_cpu_mem and paddle_cpu_mem[model_name] != "":
        lines1 += paddle_cpu_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_cpu_mem and ort_cpu_mem[model_name] != "":
        lines1 += ort_cpu_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ov_cpu_mem and ov_cpu_mem[model_name] != "":
        lines1 += ov_cpu_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    f2.writelines(lines1 + "\n")
f2.close()
print("cpu info saved in {}".format("struct_cpu_" + domain + ".txt"))

f3 = open("struct_gpu_" + domain + ".txt", "w")
f3.writelines(
    "model_name\tpaddle_run\tort_run\tpaddle_run_all\tort_run_all\tpaddle_end2end\tort_end2end\tpaddle_gpu_mem\tort_gpu_mem\tpaddle_trt_run\ttrt_run\tpaddle_trt_run_all\ttrt_run_all\tpaddle_trt_end2end\ttrt_end2end\tpaddle_trt_gpu_mem\ttrt_gpu_mem\tpaddle_trt_fp16_run\ttrt_fp16_run\tpaddle_trt_fp16_run_all\ttrt_fp16_run_all\tpaddle_trt_fp16_end2end\ttrt_fp16_end2end\tpaddle_trt_fp16_gpu_mem\ttrt_fp16_gpu_mem\n"
)
for model_name in model_name_set:
    lines1 = model_name + '\t'
    if model_name in paddle_gpu_runtime and paddle_gpu_runtime[
            model_name] != "":
        lines1 += paddle_gpu_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_gpu_runtime and ort_gpu_runtime[model_name] != "":
        lines1 += ort_gpu_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_gpu_runtime_all and paddle_gpu_runtime_all[
            model_name] != "":
        lines1 += paddle_gpu_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_gpu_runtime_all and ort_gpu_runtime_all[
            model_name] != "":
        lines1 += ort_gpu_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_gpu_e2e and paddle_gpu_e2e[model_name] != "":
        lines1 += paddle_gpu_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_gpu_e2e and ort_gpu_e2e[model_name] != "":
        lines1 += ort_gpu_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_gpu_mem and paddle_gpu_mem[model_name] != "":
        lines1 += paddle_gpu_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in ort_gpu_mem and ort_gpu_mem[model_name] != "":
        lines1 += ort_gpu_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_runtime and paddle_trt_gpu_runtime[
            model_name] != "":
        lines1 += paddle_trt_gpu_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_runtime and trt_gpu_runtime[model_name] != "":
        lines1 += trt_gpu_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_runtime_all and paddle_trt_gpu_runtime_all[
            model_name] != "":
        lines1 += paddle_trt_gpu_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_runtime_all and trt_gpu_runtime_all[
            model_name] != "":
        lines1 += trt_gpu_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_e2e and paddle_trt_gpu_e2e[
            model_name] != "":
        lines1 += paddle_trt_gpu_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_e2e and trt_gpu_e2e[model_name] != "":
        lines1 += trt_gpu_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_mem and paddle_trt_gpu_mem[
            model_name] != "":
        lines1 += paddle_trt_gpu_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_mem and trt_gpu_mem[model_name] != "":
        lines1 += trt_gpu_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_fp16_runtime and paddle_trt_gpu_fp16_runtime[
            model_name] != "":
        lines1 += paddle_trt_gpu_fp16_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_fp16_runtime and trt_gpu_fp16_runtime[
            model_name] != "":
        lines1 += trt_gpu_fp16_runtime[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_fp16_runtime_all and paddle_trt_gpu_fp16_runtime_all[
            model_name] != "":
        lines1 += paddle_trt_gpu_fp16_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_fp16_runtime_all and trt_gpu_fp16_runtime_all[
            model_name] != "":
        lines1 += trt_gpu_fp16_runtime_all[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_fp16_e2e and paddle_trt_gpu_fp16_e2e[
            model_name] != "":
        lines1 += paddle_trt_gpu_fp16_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_fp16_e2e and trt_gpu_fp16_e2e[model_name] != "":
        lines1 += trt_gpu_fp16_e2e[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in paddle_trt_gpu_fp16_mem and paddle_trt_gpu_fp16_mem[
            model_name] != "":
        lines1 += paddle_trt_gpu_fp16_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    if model_name in trt_gpu_fp16_mem and trt_gpu_fp16_mem[model_name] != "":
        lines1 += trt_gpu_fp16_mem[model_name] + '\t'
    else:
        lines1 += "-\t"
    f3.writelines(lines1 + "\n")
f3.close()

print("gpu info saved in {}".format("struct_gpu_" + domain + ".txt"))
