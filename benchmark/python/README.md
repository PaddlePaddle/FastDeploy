# FastDeploy Benchmarks

在跑benchmark前，需确认以下两个步骤

* 1. 软硬件环境满足要求，参考[FastDeploy环境要求](..//docs/cn/build_and_install/download_prebuilt_libraries.md)
* 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../docs/cn/build_and_install/download_prebuilt_libraries.md)

FastDeploy 目前支持多种推理后端，下面以 PaddleClas MobileNetV1 为例，跑出多后端在 CPU/GPU 对应 benchmark 数据

```bash
# 下载 MobileNetV1 模型
wget https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_x0_25_infer.tgz
tar -xvf MobileNetV1_x0_25_infer.tgz

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# CPU
# Paddle Inference
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend paddle

# ONNX Runtime
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ort

# OpenVINO
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ov

# GPU
# Paddle Inference
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle

# Paddle Inference + TensorRT
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle_trt

# Paddle Inference + TensorRT fp16
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle_trt --enable_trt_fp16 True

# ONNX Runtime
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend ort

# TensorRT
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend trt

# TensorRT fp16
python benchmark_ppcls.py --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend trt --enable_trt_fp16 True

```

**具体参数说明**

| 参数                 | 作用                                        |
| -------------------- | ------------------------------------------ |
| --model              | 模型路径                                     |
| --image              | 图片路径    |
| --device             | 选择 CPU 还是 GPU，默认 CPU  |
| --cpu_num_thread     | CPU 线程数      |
| --device_id          | GPU 卡号                             |
| --iter_num           | 跑 benchmark 的迭代次数 |
| --backend            | 指定后端类型，有ort, ov, trt, paddle, paddle_trt 五个选项  |
| --enable_trt_fp16    | 当后端为trt或paddle_trt时，是否开启fp16  |
| --enable_collect_memory_info    | 是否记录 cpu/gpu memory信息，默认 False  |

**最终txt结果**

将当前目录的所有txt汇总并结构化，执行下列命令

```bash
# 汇总
cat *.txt >> ./result_ppcls.txt

# 结构化信息
python convert_info.py --txt_path result_ppcls.txt --domain ppcls --enable_collect_memory_info True
```

得到 CPU 结果```struct_cpu_ppcls.txt```以及 GPU 结果```struct_gpu_ppcls.txt```如下所示

```bash
# struct_cpu_ppcls.txt
model_name	thread_nums	ort_run	ort_end2end	cpu_rss_mb	ov_run	ov_end2end	cpu_rss_mb	paddle_run	paddle_end2end	cpu_rss_mb
MobileNetV1_x0_25	8	1.18	3.27	270.43	0.87	1.98	272.26	3.13	5.29	899.57

# struct_gpu_ppcls.txt
model_name	ort_run	ort_end2end	gpu_rss_mb	paddle_run	paddle_end2end	gpu_rss_mb	trt_run	trt_end2end	gpu_rss_mb	trt_fp16_run	trt_fp16_end2end	gpu_rss_mb
MobileNetV1_x0_25	1.25	3.24	677.06	2.00	3.77	945.06	0.67	2.66	851.06	0.53    2.46	839.06
```

**结果说明**

* ```_run```后缀代表一次infer耗时，包括H2D以及D2H；```_end2end```后缀代表包含前后处理耗时
* ```cpu_rss_mb```代表内存占用；```gpu_rss_mb```代表显存占用

若有多个PaddleClas模型，在当前目录新建ppcls_model目录，将所有模型放入该目录即可，运行下列命令

```bash
sh run_benchmark_ppcls.sh
```

一键得到所有模型在 CPU 以及 GPU 的 benchmark 数据


**添加新设备**

如果添加了一种新设备，想进行 benchmark 测试，以```ipu```为例

在对应 benchmark 脚本```--device```中加入```ipu```选项，并通过```option.use_ipu()```进行开启

输入下列命令，进行 benchmark 测试

```shell
python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --iter_num 2000 --backend paddle --device ipu
```
