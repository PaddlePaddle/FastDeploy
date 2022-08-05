# ArcFace API说明

## 0. 特别说明  
fastdeploy支持 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition) 的人脸识别模块recognition中大部分模型的部署，包括ArcFace、CosFace、Partial FC、VPL等，由于用法类似，这里仅用ArcFace来说明参数设置。

## 1. Python API

### 1.1 ArcFace 类

#### 1.1.1 类初始化说明
```python
fastdeploy.vision.deepinsight.ArcFace(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
```
ArcFace模型加载和初始化，当model_format为`fd.Frontend.ONNX`时，只需提供model_file，如`xxx.onnx`；当model_format为`fd.Frontend.PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### 1.1.2 predict函数
> ```python
> ArcFace.predict(image_data)
> ```
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

示例代码参考[arcface.py](./arcface.py)

### 1.2 其他支持的类
```python
fastdeploy.vision.deepinsight.ArcFace(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
fastdeploy.vision.deepinsight.CosFace(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
fastdeploy.vision.deepinsight.PartialFC(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
fastdeploy.vision.deepinsight.VPL(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
fastdeploy.vision.deepinsight.InsightFaceRecognitionModel(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
```
Tips: 如果 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition) 人脸识别的推理逻辑没有随它自身的版本发生太大变化，则可以都统一使用 InsightFaceRecognitionModel 进行推理。



## 2. C++ API

### 2.1 ArcFace 类
#### 2.1.1 类初始化说明
```C++
fastdeploy::vision::deepinsight::ArcFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX)
```
ArcFace模型加载和初始化，当model_format为`Frontend::ONNX`时，只需提供model_file，如`xxx.onnx`；当model_format为`Frontend::PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### 2.1.2 Predict函数
> ```C++
> ArcFace::Predict(cv::Mat* im, FaceRecognitionResult* result)
> ```
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，result的成员embedding包含人脸向量

示例代码参考[cpp/arcface.cc](cpp/arcface.cc)

### 2.2 其他支持的类
```C++
fastdeploy::vision::deepinsight::ArcFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX);
fastdeploy::vision::deepinsight::CosFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX);
fastdeploy::vision::deepinsight::PartialFC(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX);
fastdeploy::vision::deepinsight::VPL(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX);
fastdeploy::vision::deepinsight::InsightFaceRecognitionModel(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX);  
```
Tips: 如果 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition) 人脸识别的推理逻辑没有随它自身的版本发生太大变化，则可以都统一使用 InsightFaceRecognitionModel 进行推理。


## 3. 其它API使用

- [模型部署RuntimeOption配置](../../../docs/api/runtime_option.md)
