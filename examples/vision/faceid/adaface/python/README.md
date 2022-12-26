# AdaFace Python部署示例
本目录下提供infer_xxx.py快速完成AdaFace模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以AdaFace为例子, 提供`infer.py`快速完成AdaFace在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/faceid/adaface/python/

#下载AdaFace模型文件和测试图片
#下载测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/face_demo.zip
unzip face_demo.zip

# 如果为Paddle模型，运行以下代码
wget https://bj.bcebos.com/paddlehub/fastdeploy/mobilefacenet_adaface.tgz
tar zxvf mobilefacenet_adaface.tgz -C ./

# CPU推理
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face face_0.jpg \
                --face_positive face_1.jpg \
                --face_negative face_2.jpg \
                --device cpu
# GPU推理
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face face_0.jpg \
                --face_positive face_1.jpg \
                --face_negative face_2.jpg \
                --device gpu
# GPU上使用TensorRT推理
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face face_0.jpg \
                --face_positive face_1.jpg \
                --face_negative face_2.jpg \
                --device gpu \
                --use_trt True

# 昆仑芯XPU推理
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face test_lite_focal_arcface_0.JPG \
                --face_positive test_lite_focal_arcface_1.JPG \
                --face_negative test_lite_focal_arcface_2.JPG \
                 --device kunlunxin
```

运行完成可视化结果如下图所示

<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321537-860bf857-0101-4e92-a74c-48e8658d838c.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184322004-a551e6e4-6f47-454e-95d6-f8ba2f47b516.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG">
</div>

```bash
FaceRecognitionResult: [Dim(512), Min(-0.133213), Max(0.148838), Mean(0.000293)]
FaceRecognitionResult: [Dim(512), Min(-0.102777), Max(0.120130), Mean(0.000615)]
FaceRecognitionResult: [Dim(512), Min(-0.116685), Max(0.142919), Mean(0.001595)]
Cosine 01:  0.7483505506964364
Cosine 02:  -0.09605773855893639
```

## AdaFace Python接口

```python
fastdeploy.vision.faceid.AdaFace(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.PADDLE)
```

AdaFace模型加载和初始化，其中model_file为导出的ONNX模型格式或PADDLE静态图格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为PADDLE

### predict函数

> ```python
> AdaFace.predict(image_data)
> ```
>
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.FaceRecognitionResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

#### AdaFacePreprocessor的成员变量
以下变量为AdaFacePreprocessor的成员变量
> > * **size**(list[int]): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[112, 112]
> > * **alpha**(list[float]): 预处理归一化的alpha值，计算公式为`x'=x*alpha+beta`，alpha默认为[1. / 127.5, 1.f / 127.5, 1. / 127.5]
> > * **beta**(list[float]): 预处理归一化的beta值，计算公式为`x'=x*alpha+beta`，beta默认为[-1.f, -1.f, -1.f]
> > * **swap_rb**(bool): 预处理是否将BGR转换成RGB，默认True

#### AdaFacePostprocessor的成员变量
以下变量为AdaFacePostprocessor的成员变量
> > * **l2_normalize**(bool): 输出人脸向量之前是否执行l2归一化，默认False


## 其它文档

- [AdaFace 模型介绍](..)
- [AdaFace C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
