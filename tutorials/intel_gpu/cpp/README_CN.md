[English](README.md) | 中文

# PaddleClas Python Example

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

**注意** 本文档依赖FastDeploy>=1.0.2版本，或nightly built版本。

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

这篇文档展示的是如何在Linux/Mac上编译和运行，如果你是使用Windows系统，请参考下面的文档进行使用

- [Windows上使用FastDeploy C++ SDK](../../../docs/cn/faq/use_sdk_on_windows.md)
