# PP-TinyPose C++部署示例

本目录下提供`pptinypose_infer.cc`快速完成PP-TinyPose在CPU/GPU，以及GPU上通过TensorRT加速部署的`单图单人关键点检测`示例
>> **注意**: PP-Tinypose单模型目前只支持单图单人关键点检测，因此输入的图片应只包含一个人或者进行过裁剪的图像。多人关键点检测请参考[PP-TinyPose Pipeline](../../det_keypoint_unite/cpp/README.md)

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)


以Linux上推理为例，在本目录执行如下命令即可完成编译测试

```bash
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.3.0.tgz
tar xvf fastdeploy-linux-x64-gpu-0.3.0.tgz
cd fastdeploy-linux-x64-gpu-0.3.0/examples/vision/keypointdetection/tiny_pose/cpp/
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../../fastdeploy-linux-x64-gpu-0.3.0
make -j

# 下载Unet模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/hrnet_demo.jpg


# CPU推理
./infer_tinypose_demo PP_TinyPose_256x192_infer hrnet_demo.jpg 0
# GPU推理
./infer_tinypose_demo PP_TinyPose_256x192_infer hrnet_demo.jpg 1
# GPU上TensorRT推理
./infer_tinypose_demo PP_TinyPose_256x192_infer hrnet_demo.jpg 2
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196386764-dd51ad56-c410-4c54-9580-643f282f5a83.jpeg", width=359px, height=423px />
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## PP-TinyPose C++接口

### PP-TinyPose类

```c++
fastdeploy::vision::keypointdetection::PPTinyPose(
        const string& model_file,
        const string& params_file = "",
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PPTinyPose模型加载和初始化，其中model_file为导出的Paddle模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

#### Predict函数

> ```c++
> PPTinyPose::Predict(cv::Mat* im, KeyPointDetectionResult* result)
> ```
>
> 模型预测接口，输入图像直接输出关键点检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 关键点检测结果，包括关键点的坐标以及关键点对应的概率值, KeyPointDetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性
#### 后处理参数
> > * **use_dark**(bool): 是否使用DARK进行后处理[参考论文](https://arxiv.org/abs/1910.06278)

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
