English | [中文](README_CN.md)

# PaddleClas Python Example

Before deployment, confirm the following two steps

- 1. The software and hardware environment meet the requirements. Refer to [FastDeploy Environment Requirements](../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. Install FastDeploy Python wheel package. Refer to [Install FastDeploy](../../../docs/en/build_and_install/download_prebuilt_libraries.md)

```bash
# Get FastDeploy codes
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/tutorials/intel_gpu/python

# Download PaddleClas model and test image
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
tar -xvf ResNet50_vd_infer.tgz

# Inference with CPU
python infer_resnet50.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --topk 1

# Inference with Intel GPU
python infer_resnet50.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device intel_gpu --topk 1



# Download PaddleDetection/PP-YOLOE model and test image
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# Inference with CPU
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device cpu

# Inference with Intel GPU
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device intel_gpu
```
