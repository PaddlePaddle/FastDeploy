# PaddleSeg C++部署示例

本目录下提供`infer.cc`快速完成Unet在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```bash
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.1.tgz
tar xvf fastdeploy-linux-x64-gpu-0.2.1.tgz
cd fastdeploy-linux-x64-gpu-0.2.1/examples/vision/segmentation/paddleseg/cpp/
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../../fastdeploy-linux-x64-gpu-0.2.1
make -j

# 下载Unet模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz
tar -xvf Unet_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png


# CPU推理
./infer_demo Unet_cityscapes_without_argmax_infer Unet_cityscapes_without_argmax_infer cityscapes_demo.png 0
# GPU推理
./infer_demo Unet_cityscapes_without_argmax_infer Unet_cityscapes_without_argmax_infer cityscapes_demo.png 1
# GPU上TensorRT推理
./infer_demo Unet_cityscapes_without_argmax_infer Unet_cityscapes_without_argmax_infer cityscapes_demo.png 2
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/184588768-45ee673b-ef1f-40f4-9fbd-6b1a9ce17c59.png", width=512px, height=256px />
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)

## PaddleSeg C++接口

### PaddleSeg类

```c++
fastdeploy::vision::segmentation::PaddleSegModel(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE)
```

PaddleSegModel模型加载和初始化，其中model_file为导出的Paddle模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为Paddle格式

#### Predict函数

> ```c++
> PaddleSegModel::Predict(cv::Mat* im, DetectionResult* result)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 分割结果，包括分割预测的标签以及标签对应的概率值, SegmentationResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **is_vertical_screen**(bool): PP-HumanSeg系列模型通过设置此参数为`True`表明输入图片是竖屏，即height大于width的图片

#### 后处理参数
> > * **with_softmax**(bool): 当模型导出时，并未指定`with_softmax`参数，可通过此设置此参数为`True`，将预测的输出分割标签（label_map）对应的概率结果(score_map)做softmax归一化处理

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../how_to_change_backend.md)
