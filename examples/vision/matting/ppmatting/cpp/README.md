# PPMatting C++部署示例

本目录下提供`infer.cc`快速完成PPMatting在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/the%20software%20and%20hardware%20requirements.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上 PPMatting 推理为例，在本目录执行如下命令即可完成编译测试（如若只需在CPU上部署，可在[Fastdeploy C++预编译库](../../../../../docs/quick_start/CPP_prebuilt_libraries.md)下载CPU推理库）

```bash
#下载SDK，编译模型examples代码（SDK中包含了examples代码）
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.0.tgz
tar xvf fastdeploy-linux-x64-gpu-0.2.0.tgz
cd fastdeploy-linux-x64-gpu-0.2.0/examples/vision/matting/ppmatting/cpp/
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../../fastdeploy-linux-x64-gpu-0.2.0
make -j

# 下载PPMatting模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz
tar -xvf PP-Matting-512.tgz
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_matting_input.jpg
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_matting_bgr.jpg


# CPU推理
./infer_demo PP-Matting-512 test_lite_matting_input.jpg test_lite_matting_bgr.jpg 0
# GPU推理 (TODO: ORT-GPU 推理会报错)
./infer_demo PP-Matting-512 test_lite_matting_input.jpg test_lite_matting_bgr.jpg 1
# GPU上TensorRT推理
./infer_demo PP-Matting-512 test_lite_matting_input.jpg test_lite_matting_bgr.jpg 2
```

运行完成可视化结果如下图所示
<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184301892-457f7014-2dc0-4ad1-b688-43b41fac299a.jpg">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184301871-c234dfdf-3b3d-46e4-8886-e1ac156c9e4a.jpg">
<!-- <img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG"> -->
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)

## PPMatting C++接口

### PPMatting类

```c++
fastdeploy::vision::matting::PPMatting(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE)
```

PPMatting模型加载和初始化，其中model_file为导出的Paddle模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为Paddle格式

#### Predict函数

> ```c++
> PPMatting::Predict(cv::Mat* im, MattingResult* result)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 分割结果，包括分割预测的标签以及标签对应的概率值, MattingResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果


- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
