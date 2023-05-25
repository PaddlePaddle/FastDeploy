[English](README.md) | 简体中文

# PaddleDetection C++部署示例

本目录下提供`infer_ppyoloe_demo.cc`快速完成PPDetection模型使用TVM加速部署的示例。

## 转换模型并运行

```bash
# build example
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=/path/to/fastdeploy-sdk
make -j
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
./infer_ppyoloe_demo ../tvm_save 000000014439.jpg
 ```


## PaddleDetection C++接口

### 模型类

PaddleDetection目前支持6种模型系列，类名分别为`PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，`SSD`,`PaddleYOLOv5`,`PaddleYOLOv6`,`PaddleYOLOv7`,`RTMDet`,`CascadeRCNN`,`PSSDet`,`RetinaNet`,`PPYOLOESOD`,`FCOS`,`TTFNet`,`TOOD`,`GFL`所有类名的构造函数和预测函数在参数上完全一致，本文档以PPYOLOE为例讲解API
```c++
fastdeploy::vision::detection::PPYOLOE(
        const string& model_file,
        const string& params_file,
        const string& config_file
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PaddleDetection PPYOLOE模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 配置文件路径，即PaddleDetection导出的部署yaml文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为PADDLE格式

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
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
