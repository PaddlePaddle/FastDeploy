# PP-Tracking C++部署示例

本目录下提供`infer.cc`快速完成PP-Tracking在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上 PP-Tracking 推理为例，在本目录执行如下命令即可完成编译测试（如若只需在CPU上部署，可在[Fastdeploy C++预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md/CPP_prebuilt_libraries.md)下载CPU推理库）

```bash
#下载SDK，编译模型examples代码（SDK中包含了examples代码）
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.3.0.tgz
tar xvf fastdeploy-linux-x64-gpu-0.3.0.tgz
cd fastdeploy-linux-x64-gpu-0.3.0/examples/vision/tracking/pptracking/cpp/
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../../fastdeploy-linux-x64-gpu-0.3.0
make -j

# 下载PP-Tracking模型文件和测试视频
wget https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
tar -xvf fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4


# CPU推理
./infer_demo fairmot_hrnetv2_w18_dlafpn_30e_576x320 person.mp4 0
# GPU推理
./infer_demo fairmot_hrnetv2_w18_dlafpn_30e_576x320 person.mp4 1
# GPU上TensorRT推理
./infer_demo fairmot_hrnetv2_w18_dlafpn_30e_576x320 person.mp4 2
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## PP-Tracking C++接口

### PPTracking类

```c++
fastdeploy::vision::tracking::PPTracking(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PP-Tracking模型加载和初始化，其中model_file为导出的Paddle模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

#### Predict函数

> ```c++
> PPTracking::Predict(cv::Mat* im, MOTResult* result)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，跟踪id，各个框的置信度，对象类别id，MOTResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
