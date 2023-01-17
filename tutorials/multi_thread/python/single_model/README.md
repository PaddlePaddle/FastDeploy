English | [简体中文](README_CN.md)
# Example of PaddleClas models Python multi-thread/multi-process Deployment

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install the FastDeploy Python whl package. Please refer to [FastDeploy Python Installation](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides example file `multi_thread_process.py` to fast deploy multi-thread/multi-process ResNet50_vd on CPU/GPU and GPU accelerated by TensorRT. The script is as follows


```bash
# Download deployment example code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/tutorials/multi_thread/python

# Download the ResNet50_vd model file and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU multi-thread inference
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device cpu --topk 1 --thread_num 1
# CPU multi-process inference
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device cpu --topk 1 --use_multi_process True --process_num 1

# GPU multi-thread inference
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --topk 1 --thread_num 1
# GPU multi-process inference
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --topk 1 --use_multi_process True --process_num 1

# Use TensorRT multi-thread inference on GPU （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True --topk 1 --thread_num 1
# Use TensorRT multi-process inference on GPU （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True --topk 1 --use_multi_process True --process_num 1

# IPU multi-thread inference（Attention: It is somewhat time-consuming for the operation of model serialization when running IPU inference for the first time. Please be patient.）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device ipu --topk 1 --thread_num 1
# IPU multi-process inference（Attention: It is somewhat time-consuming for the operation of model serialization when running IPU inference for the first time. Please be patient.）
python multi_thread_process.py --model ResNet50_vd_infer --image_path ILSVRC2012_val_00000010.jpeg --device ipu --topk 1 --use_multi_process True --process_num 1
```
>> **Notice**: `--image_path` can be the path of the pictures folder

The result returned after running is as follows
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```
