# AdaFace C++部署示例
本目录下提供infer_xxx.py快速完成AdaFace模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

以AdaFace为例提供`infer_adaface.cc`快速完成AdaFace在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```bash
# “如果预编译库不包含本模型，请从最新代码编译SDK”
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.2.1.tgz
tar xvf fastdeploy-linux-x64-0.2.1.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.2.1
make -j

#下载官方转换好的AdaFace模型文件和测试图片
wget https://bj.bcebos.com/fastdeploy/models/onnx/mobile_face_net_ada_face_112x112.onnx
wget https://bj.bcebos.com/paddlehub/test_samples/test_lite_focal_arcface_0.JPG
wget https://bj.bcebos.com/paddlehub/test_samples/test_lite_focal_arcface_1.JPG
wget https://bj.bcebos.com/paddlehub/test_samples/test_lite_focal_arcface_2.JPG


# CPU推理
./infer_adaface_demo mobile_face_net_ada_face_112x112.onnx test_lite_focal_arcface_0.JPG test_lite_focal_arcface_1.JPG test_lite_focal_arcface_2.JPG 0
# GPU推理
./infer_adaface_demo mobile_face_net_ada_face_112x112.onnx test_lite_focal_arcface_0.JPG test_lite_focal_arcface_1.JPG test_lite_focal_arcface_2.JPG 1
# GPU上TensorRT推理
./infer_adaface_demo mobile_face_net_ada_face_112x112.onnx test_lite_focal_arcface_0.JPG test_lite_focal_arcface_1.JPG test_lite_focal_arcface_2.JPG 2
```

运行完成可视化结果如下图所示

<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321537-860bf857-0101-4e92-a74c-48e8658d838c.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184322004-a551e6e4-6f47-454e-95d6-f8ba2f47b516.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG">
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)

## AdaFace C++接口

### AdaFace类

```c++
fastdeploy::vision::faceid::AdaFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

AdaFace模型加载和初始化，其中model_file为导出的ONNX模型格式。

#### Predict函数

> ```c++
> AdaFace::Predict(cv::Mat* im, FaceRecognitionResult* result)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, FaceRecognitionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员变量
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果


> > * **size**(vector&lt;int&gt;): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[112, 112]
> > * **alpha**(vector&lt;float&gt;): 预处理归一化的alpha值，计算公式为`x'=x*alpha+beta`，alpha默认为[1. / 127.5, 1.f / 127.5, 1. / 127.5]
> > * **beta**(vector&lt;float&gt;): 预处理归一化的beta值，计算公式为`x'=x*alpha+beta`，beta默认为[-1.f, -1.f, -1.f]
> > * **swap_rb**(bool): 预处理是否将BGR转换成RGB，默认true
> > * **l2_normalize**(bool): 输出人脸向量之前是否执行l2归一化，默认false

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
