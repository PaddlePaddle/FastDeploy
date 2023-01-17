[English](README.md) | 简体中文
# PaddleClas模型 Python多线程/进程部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`multi_thread_process.py`快速完成ResNet50_vd在CPU/GPU，以及GPU上通过TensorRT加速部署的多线程/进程示例。执行如下脚本即可完成


```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/tutorials/multi_thread/python

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU多线程推理
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device cpu --topk 1 --thread_num 1
# CPU多进程推理
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device cpu --topk 1 --use_multi_process True --process_num 1

# GPU多线程推理
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --topk 1 --thread_num 1
# GPU多进程推理
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --topk 1 --use_multi_process True --process_num 1

# GPU上使用TensorRT多线程推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True --topk 1 --thread_num 1
# GPU上使用TensorRT多进程推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True --topk 1 --use_multi_process True --process_num 1

# IPU多线程推理（注意：IPU推理首次运行会有序列化模型的操作，有一定耗时，需要耐心等待）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device ipu --topk 1 --thread_num 1
# IPU多进程推理（注意：IPU推理首次运行会有序列化模型的操作，有一定耗时，需要耐心等待）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device ipu --topk 1 --use_multi_process True --process_num 1
```
>> **注意**: `--image_path` 可以输入图片文件夹的路径

运行完成后返回结果如下所示
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```
