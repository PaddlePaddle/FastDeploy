# MODNet API说明

## 1. Python API

### 1.1 MODNet 类

#### 1.1.1 类初始化说明
```python
fastdeploy.vision.zhkkke.MODNet(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
```
MODNet模型加载和初始化，当model_format为`fd.Frontend.ONNX`时，只需提供model_file，如`xxx.onnx`；当model_format为`fd.Frontend.PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### 1.1.2 predict函数
> ```python
> MODNet.predict(image_data)
> ```
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

示例代码参考[modnet.py](./modnet.py)


## 2. C++ API

### 2.1 MODNet 类
#### 2.1.1 类初始化说明
```C++
fastdeploy::vision::zhkkke::MODNet(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX)
```
MODNet模型加载和初始化，当model_format为`Frontend::ONNX`时，只需提供model_file，如`xxx.onnx`；当model_format为`Frontend::PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### 2.1.2 Predict函数
> ```C++
> MODNet::Predict(cv::Mat* im, MattingResult* result)
> ```
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包含的成员如下
> >     * alpha: std::vector\<float\> 包含透明度
> >     * contain_foreground: bool 表示输出是否包含预测的前景
> >     * foreground: std::vector\<float\> 如果模型包含前景预测，则此项为预测的前景
> >     * shape: std::vector\<int\> 包含输出alpha的维度(h,w), 如果包含前景，则shape为(h,w,c) c表示前景的通道数，一般为c=3

示例代码参考[cpp/modnet.cc](cpp/modnet.cc)

## 3. 其它API使用

- [模型部署RuntimeOption配置](../../../docs/api/runtime_option.md)
