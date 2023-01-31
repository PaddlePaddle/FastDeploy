English | [中文](README_CN.md)

# PaddleClas Python Example

Before deployment, confirm the following two steps

- 1. The software and hardware environment meet the requirements. Refer to [FastDeploy Environment Requirements](../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. Install FastDeploy Python wheel package. Refer to [Install FastDeploy](../../../docs/en/build_and_install/download_prebuilt_libraries.md)

**Notice** This doc require FastDeploy version >= 1.0.2, or just use nightly built version.

```bash
# Get FastDeploy codes
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/tutorials/intel_gpu/cpu

mkdir build && cd build

# Please the preparation step to get the download link
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz

# Download PaddleClas model and test image
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
tar -xvf ResNet50_vd_infer.tgz

cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Inference with CPU
./infer_resnet50 -model ResNet50_vd_infer -image ILSVRC2012_val_00000010.jpeg -device cpu -topk 3

# Inference with Intel GPU
./infer_resnet50 -model ResNet50_vd_infer -image ILSVRC2012_val_00000010.jpeg -device intel_gpu -topk 3


# Download PaddleDetection/PP-YOLOE model and test image
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# Inference with CPU
./infer_ppyoloe -model ppyoloe_crn_l_300e_coco -image 000000014439.jpg -device cpu

# Inference with Intel GPU
./infer_ppyoloe -model ppyoloe_crn_l_300e_coco -image 000000014439.jpg -device intel_gpu
```

This documents only shows how to compile on Linux/Mac, if you are using Windows, please refer the following documents

- [How to use FastDeploy C++ SDK on Windows](../../../docs/en/faq/use_sdk_on_windows.md)
