# PP-PicoDet + PP-TinyPose (Pipeline) C++部署示例

本目录下提供`det_keypoint_unite_infer.cc`快速完成多人模型配置 PP-PicoDet + PP-TinyPose 在CPU/GPU，以及GPU上通过TensorRT加速部署的`单图多人关键点检测`示例。执行如下脚本即可完成
>> **注意**: PP-TinyPose单模型独立部署，请参考[PP-TinyPose 单模型](../../tiny_pose/cpp/README.md)

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

# 下载PP-TinyPose和PP-PicoDet模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
tar -xvf PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/000000018491.jpg

# CPU推理
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 0
# GPU推理
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 1
# GPU上TensorRT推理
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 2
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196393343-eeb6b68f-0bc6-4927-871f-5ac610da7293.jpeg", width=359px, height=423px />
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## PP-TinyPose C++接口

### PP-TinyPose类

```c++
fastdeploy::pipeline::PPTinyPose(
        fastdeploy::vision::detection::PPYOLOE* det_model,
        fastdeploy::vision::keypointdetection::PPTinyPose* pptinypose_model)
```

PPTinyPose Pipeline模型加载和初始化。

**参数**

> * **model_det_modelfile**(fastdeploy::vision::detection): 初始化后的检测模型，参考[PP-TinyPose](../../tiny_pose/README.md)
> * **pptinypose_model**(fastdeploy::vision::keypointdetection): 初始化后的检测模型[Detection](../../../detection/paddledetection/README.md)，暂时只提供PaddleDetection系列

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
> > * **detection_model_score_threshold**(bool):
输入PP-TinyPose模型前，Detectin模型过滤检测框的分数阈值

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
