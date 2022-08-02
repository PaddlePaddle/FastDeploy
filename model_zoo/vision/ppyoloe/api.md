# PPYOLOE API说明

## Python API

### PPYOLOE类
```
fastdeploy.vision.ppdet.PPYOLOE(model_file, params_file, config_file, runtime_option=None, model_format=fd.Frontend.PADDLE)
```
PPYOLOE模型加载和初始化，需同时提供model_file和params_file, 当前仅支持model_format为Paddle格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 模型推理配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### predict函数
> ```
> PPYOLOE.predict(image_data)
> ```
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

示例代码参考[ppyoloe.py](./ppyoloe.py)


## C++ API

### PPYOLOE类
```
fastdeploy::vision::ppdet::PPYOLOE(
        const string& model_file,
        const string& params_file,
        const string& config_file,
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE)
```
PPYOLOE模型加载和初始化，需同时提供model_file和params_file, 当前仅支持model_format为Paddle格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 模型推理配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### Predict函数
> ```
> PPYOLOE::Predict(cv::Mat* im, DetectionResult* result)
> ```
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度

示例代码参考[cpp/ppyoloe.cc](cpp/ppyoloe.cc)

## 其它API使用

- [模型部署RuntimeOption配置](../../../docs/api/runtime_option.md)
