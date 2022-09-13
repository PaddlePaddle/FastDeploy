import os
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--txt_path', type=str, default='result.txt')
parser.add_argument('--domain', type=str, default='ppcls')
args = parser.parse_args()
txt_path = args.txt_path
domain = args.domain

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
trt_gpu = dict()
trt_gpu_fp16 = dict()
model_name_set = set()

for i in range(line_nums):
    if "====" in lines[i]:
        model_name = lines[i].strip().split("_model")[0][4:]
        model_name_set.add(model_name)
        runtime = "-"
        end2end = "-"
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
        if "ort_cpu_1" in lines[i]:
            ort_cpu_thread1[model_name] = runtime + "\t" + end2end
        elif "ort_cpu_8" in lines[i]:
            ort_cpu_thread8[model_name] = runtime + "\t" + end2end
        elif "ort_gpu" in lines[i]:
            ort_gpu[model_name] = runtime + "\t" + end2end
        elif "ov_cpu_1" in lines[i]:
            ov_cpu_thread1[model_name] = runtime + "\t" + end2end
        elif "ov_cpu_8" in lines[i]:
            ov_cpu_thread8[model_name] = runtime + "\t" + end2end
        elif "paddle_cpu_1" in lines[i]:
            paddle_cpu_thread1[model_name] = runtime + "\t" + end2end
        elif "paddle_cpu_8" in lines[i]:
            paddle_cpu_thread8[model_name] = runtime + "\t" + end2end
        elif "paddle_gpu" in lines[i]:
            paddle_gpu[model_name] = runtime + "\t" + end2end
        elif "trt_gpu" in lines[i]:
            trt_gpu[model_name] = runtime + "\t" + end2end
        elif "trt_fp16_gpu" in lines[i]:
            trt_gpu_fp16[model_name] = runtime + "\t" + end2end

f2 = open("struct_cpu_" + domain + ".txt", "w")
f2.writelines(
    "model_name\tthread_nums\tort_run\tort_end2end\tov_run\tov_end2end\tpaddle_run\tpaddle_end2end\n"
)
for model_name in model_name_set:
    lines1 = model_name + '\t1\t'
    lines2 = model_name + '\t8\t'
    if model_name in ort_cpu_thread1 and ort_cpu_thread1[model_name] != "":
        lines1 += ort_cpu_thread1[model_name] + '\t'
    else:
        lines1 += "-\t-\t"
    if model_name in ov_cpu_thread1 and ov_cpu_thread1[model_name] != "":
        lines1 += ov_cpu_thread1[model_name] + '\t'
    else:
        lines1 += "-\t-\t"
    if model_name in paddle_cpu_thread1 and paddle_cpu_thread1[
            model_name] != "":
        lines1 += paddle_cpu_thread1[model_name] + '\n'
    else:
        lines1 += "-\t-\n"
    f2.writelines(lines1)
    if model_name in ort_cpu_thread8 and ort_cpu_thread8[model_name] != "":
        lines2 += ort_cpu_thread8[model_name] + '\t'
    else:
        lines2 += "-\t-\t"
    if model_name in ov_cpu_thread8 and ov_cpu_thread8[model_name] != "":
        lines2 += ov_cpu_thread8[model_name] + '\t'
    else:
        lines2 += "-\t-\t"
    if model_name in paddle_cpu_thread8 and paddle_cpu_thread8[
            model_name] != "":
        lines2 += paddle_cpu_thread8[model_name] + '\n'
    else:
        lines2 += "-\t-\n"
    f2.writelines(lines2)
f2.close()

f3 = open("struct_gpu_" + domain + ".txt", "w")
f3.writelines(
    "model_name\tort_run\tort_end2end\tpaddle_run\tpaddle_end2end\ttrt_run\ttrt_end2end\ttrt_fp16_run\ttrt_fp16_end2end\n"
)
for model_name in model_name_set:
    lines1 = model_name + '\t'
    if model_name in ort_gpu and ort_gpu[model_name] != "":
        lines1 += ort_gpu[model_name] + '\t'
    else:
        lines1 += "-\t-\t"
    if model_name in paddle_gpu and paddle_gpu[model_name] != "":
        lines1 += paddle_gpu[model_name] + '\t'
    else:
        lines1 += "-\t-\t"
    if model_name in trt_gpu and trt_gpu[model_name] != "":
        lines1 += trt_gpu[model_name] + '\t'
    else:
        lines1 += "-\t-\t"
    if model_name in trt_gpu_fp16 and trt_gpu_fp16[model_name] != "":
        lines1 += trt_gpu_fp16[model_name] + '\n'
    else:
        lines1 += "-\t-\n"
    f3.writelines(lines1)
f3.close()
