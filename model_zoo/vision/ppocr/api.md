# DBDetector API说明

## Python API

### DBDetector类
```
fastdeploy.vision.ppocr.DBDetector(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.PADDLE)
```
DBDetector模型加载和初始化，当model_format为`fd.Frontend.ONNX`时，只需提供model_file，如`DBDetector.onnx`；当model_format为`fd.Frontend.PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### predict函数
> ```
> DBDetector.predict(image_data)
> ```
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

示例代码参考[dbdetector.py](./dbdetector.py)


## C++ API

### DBDetector类
```
fastdeploy::vision::ppocr::DBDetector(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE)
```
DBDetector模型加载和初始化，当model_format为`Frontend::ONNX`时，只需提供model_file，如`DBDetector.onnx`；当model_format为`Frontend::PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### Predict函数
> ```
> DBDetector::Predict(cv::Mat* im, OCRPredictResult* result)
> ```
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度

示例代码参考[cpp/dbdetector.cc](cpp/dbdetector.cc)

## 其它API使用

- [模型部署RuntimeOption配置](../../../docs/api/runtime_option.md)
