# InsightFace C++部署示例
本目录下提供infer_xxx.cc快速完成InsighFace模型包括ArcFace\CosFace\VPL\Partial_FC在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。
以ArcFace为例提供`infer_arcface.cc`快速完成ArcFace在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/quick_start/requirements.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/compile/prebuild_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```
mkdir build
cd build
wget https://xxx.tgz
tar xvf fastdeploy-linux-x64-0.2.0.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.2.0
make -j

#下载官方转换好的ArcFace模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx
wget https://bj.bcebos.com/paddlehub/test_samples/test_lite_focal_arcface_0.JPG
wget https://bj.bcebos.com/paddlehub/test_samples/test_lite_focal_arcface_1.JPG
wget https://bj.bcebos.com/paddlehub/test_samples/test_lite_focal_arcface_2.JPG


# CPU推理
./infer_arcface_demo ms1mv3_arcface_r100.onnx test_lite_focal_arcface_0.JPG test_lite_focal_arcface_1.JPG test_lite_focal_arcface_2.JPG 0
# GPU推理
./infer_arcface_demo ms1mv3_arcface_r100.onnx test_lite_focal_arcface_0.JPG test_lite_focal_arcface_1.JPG test_lite_focal_arcface_2.JPG 1
# GPU上TensorRT推理
./infer_arcface_demo ms1mv3_arcface_r100.onnx test_lite_focal_arcface_0.JPG test_lite_focal_arcface_1.JPG test_lite_focal_arcface_2.JPG 2
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/182562483-2719648c-8fe2-48af-a8e0-82e4ebe15133.jpg">

## ArcFace C++接口

### ArcFace类

```
fastdeploy::vision::detection::ArcFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX)
```

ArcFace模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为ONNX格式

#### Predict函数

> ```
> ArcFace::Predict(cv::Mat* im, DetectionResult* result,
>                 float conf_threshold = 0.25,
>                 float nms_iou_threshold = 0.5)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, DetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)
> > * **conf_threshold**: 检测框置信度过滤阈值
> > * **nms_iou_threshold**: NMS处理过程中iou阈值

### 类成员变量


> > * **size**(vector&lt;int&gt;): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[112, 112]
> > * **alpha**(vector&lt;float&gt;): 预处理归一化的alpha值，计算公式为`x'=x*alpha+beta`，alpha默认为[1. / 127.5, 1.f / 127.5, 1. / 127.5]
> > * **beta**(vector&lt;float&gt;): 预处理归一化的beta值，计算公式为`x'=x*alpha+beta`，beta默认为[-1.f, -1.f, -1.f]
> > * **swap_rb**(bool): 预处理是否将BGR转换成RGB，默认true
> > * **l2_normalize**(bool): 输出人脸向量之前是否执行l2归一化，默认false

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
