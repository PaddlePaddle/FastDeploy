# PaddleDetection C++部署示例

本目录下提供`infer_xxx.cc`快速完成PaddleDetection模型包括PPYOLOE/PicoDet/YOLOX/YOLOv3/PPYOLO/FasterRCNN在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上推理为例，在本目录执行如下命令即可完成编译测试

```bash
以ppyoloe为例进行推理部署

#下载SDK，编译模型examples代码（SDK中包含了examples代码）
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.1.tgz
tar xvf fastdeploy-linux-x64-gpu-0.2.1.tgz
cd fastdeploy-linux-x64-gpu-0.2.1/examples/vision/detection/paddledetection/cpp
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../../fastdeploy-linux-x64-gpu-0.2.1
make -j

# 下载PPYOLOE模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz


# CPU推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 0
# GPU推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 1
# GPU上TensorRT推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 2
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)

## PaddleDetection C++接口

### 模型类

PaddleDetection目前支持6种模型系列，类名分别为`PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，所有类名的构造函数和预测函数在参数上完全一致，本文档以PPYOLOE为例讲解API
```c++
fastdeploy::vision::detection::PPYOLOE(
        const string& model_file,
        const string& params_file,
        const string& config_file
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE)
```

PaddleDetection PPYOLOE模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 配置文件路径，即PaddleDetection导出的部署yaml文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为PADDLE格式

#### Predict函数

> ```c++
> PPYOLOE::Predict(cv::Mat* im, DetectionResult* result)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, DetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../how_to_change_backend.md)
