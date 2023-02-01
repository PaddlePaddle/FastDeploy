[English](README.md) | 中文

# PaddleClas Python Example

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

```bash
# Get FastDeploy codes
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/tutorials/intel_gpu/python

# Download model and test image
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Inference with CPU
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --topk 1

# Inference with Intel GPU
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device intel_gpu --topk 1


# Download PaddleDetection/PP-YOLOE model and test image
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# Inference with CPU
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device cpu

# Inference with Intel GPU
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device intel_gpu
```
