# RuntimeOption

`RuntimeOption`用于配置模型在不同后端、硬件上的推理参数。

## Python 类

```
class RuntimeOption()
```

### 成员函数

```
set_model_path(model_file, params_file="", model_format="paddle")
```
设定加载的模型路径

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当为onnx模型格式时，无需指定
> * **model_format**(str): 模型格式，支持paddle, onnx, 默认paddle

```
use_gpu(device_id=0)
```
设定使用GPU推理

**参数**

> * **device_id**(int): 环境中存在多个GPU卡时，此参数指定推理的卡，默认为0

```
use_cpu()
```
设定使用CPU推理


```
set_cpu_thread_num(thread_num=-1)
```
设置CPU上推理时线程数量

**参数**

> * **thread_num**(int): 线程数量，当小于或等于0时为后端自动分配，默认-1

```
use_paddle_backend()
```
使用Paddle Inference后端进行推理，支持CPU/GPU，支持Paddle模型格式

```
use_ort_backend()
```
使用ONNX Runtime后端进行推理，支持CPU/GPU，支持Paddle/ONNX模型格式

```
use_trt_backend()
```
使用TensorRT后端进行推理，支持GPU，支持Paddle/ONNX模型格式

```
use_openvino_backend()
```
使用OpenVINO后端进行推理，支持CPU, 支持Paddle/ONNX模型格式

```
enable_paddle_mkldnn()
disable_paddle_mkldnn()
```
当使用Paddle Inference后端时，通过此开关开启或关闭CPU上MKLDNN推理加速，后端默认为开启

```
enable_paddle_log_info()
disable_paddle_log_info()
```
当使用Paddle Inference后端时，通过此开关开启或关闭模型加载时的优化日志，后端默认为关闭

```
set_paddle_mkldnn_cache_size(cache_size)
```
当使用Paddle Inference后端时，通过此接口控制MKLDNN加速时的Shape缓存大小

**参数**
> * **cache_size**(int): 缓存大小

```
set_trt_input_shape(tensor_name, min_shape, opt_shape=None, max_shape=None)
```
当使用TensorRT后端时，通过此接口设置模型各个输入的Shape范围，当只设置min_shape时，会自动将opt_shape和max_shape设定为与min_shape一致。

此接口用户也可以无需自行调用，FastDeploy在推理过程中，会根据推理真实数据自动更新Shape范围，但每次遇到新的shape更新范围后，会重新构造后端引擎，带来一定的耗时。可能过此接口提前配置，来避免推理过程中的引擎重新构建。

**参数**
> * **tensor_name**(str): 需要设定输入范围的tensor名
> * **min_shape(list of int): 对应tensor的最小shape，例如[1, 3, 224, 224]
> * **opt_shape(list of int): 对应tensor的最常用shape，例如[2, 3, 224, 224], 当为None时，即保持与min_shape一致，默认为None
> * **max_shape(list of int): 对应tensor的最大shape，例如[8, 3, 224, 224], 当为None时，即保持与min_shape一致，默认为None

```
set_trt_cache_file(cache_file_path)
```
当使用TensorRT后端时，通过此接口将构建好的TensorRT模型引擎缓存到指定路径，或跳过构造引擎步骤，直接加载本地缓存的TensorRT模型
- 当调用此接口，且`cache_file_path`不存在时，FastDeploy将构建TensorRT模型，并将构建好的模型保持至`cache_file_path`
- 当调用此接口，且`cache_file_path`存在时，FastDeploy将直接加载`cache_file_path`存储的已构建好的TensorRT模型，从而大大减少模型加载初始化的耗时

通过此接口，可以在第二次运行代码时，加速模型加载初始化的时间，但因此也需注意，如需您修改了模型加载配置，例如TensorRT的max_workspace_size，或重新设置了`set_trt_input_shape`，以及更换了原始的paddle或onnx模型，需先删除已缓存在本地的`cache_file_path`文件，避免重新加载旧的缓存，影响程序正确性。

**参数**
> * **cache_file_path**(str): 缓存文件路径，例如`/Downloads/resnet50.trt`

```
enable_trt_fp16()
disable_trt_fp16()
```
当使用TensorRT后端时，通过此接口开启或关闭半精度推理加速，会带来明显的性能提升，但并非所有GPU都支持半精度推理。 在不支持半精度推理的GPU上，将会回退到FP32推理，并给出提示`Detected FP16 is not supported in the current GPU, will use FP32 instead.`

## C++ 结构体

```
struct RuntimeOption
```

### 成员函数

```
void SetModelPath(const string& model_file, const string& params_file = "", const string& model_format = "paddle")
```
设定加载的模型路径

**参数**

> * **model_file**: 模型文件路径
> * **params_file**: 参数文件路径，当为onnx模型格式时，指定为""即可
> * **model_format**: 模型格式，支持"paddle", "onnx", 默认"paddle"

```
void UseGpu(int device_id = 0)
```
设定使用GPU推理

**参数**

> * **device_id**: 环境中存在多个GPU卡时，此参数指定推理的卡，默认为0

```
void UseCpu()
```
设定使用CPU推理


```
void SetCpuThreadNum(int thread_num=-1)
```
设置CPU上推理时线程数量

**参数**

> * **thread_num**: 线程数量，当小于或等于0时为后端自动分配，默认-1

```
void UsePaddleBackend()
```
使用Paddle Inference后端进行推理，支持CPU/GPU，支持Paddle模型格式

```
void UseOrtBackend()
```
使用ONNX Runtime后端进行推理，支持CPU/GPU，支持Paddle/ONNX模型格式

```
void UseTrtBackend()
```
使用TensorRT后端进行推理，支持GPU，支持Paddle/ONNX模型格式

```
void UseOpenVINOBackend()
```
使用OpenVINO后端进行推理，支持CPU, 支持Paddle/ONNX模型格式

```
void SetPaddleMKLDNN(bool pd_mkldnn = true)
void DisablePaddleMKLDNN()
```
当使用Paddle Inference后端时，通过此开关开启或关闭CPU上MKLDNN推理加速，后端默认为开启

```
void EnablePaddleLogInfo()
void DisablePaddleLogInfo()
```
当使用Paddle Inference后端时，通过此开关开启或关闭模型加载时的优化日志，后端默认为关闭

```
void SetPaddleMKLDNNCacheSize(int cache_size)
```
当使用Paddle Inference后端时，通过此接口控制MKLDNN加速时的Shape缓存大小

**参数**
> * **cache_size**: 缓存大小

```
void SetTrtInputShape(const string& tensor_name, const vector<int32_t>& min_shape,
                      const vector<int32_t>& opt_shape = vector<int32_t>(),
                      const vector<int32_t>& opt_shape = vector<int32_t>())
```
当使用TensorRT后端时，通过此接口设置模型各个输入的Shape范围，当只设置min_shape时，会自动将opt_shape和max_shape设定为与min_shape一致。

此接口用户也可以无需自行调用，FastDeploy在推理过程中，会根据推理真实数据自动更新Shape范围，但每次遇到新的shape更新范围后，会重新构造后端引擎，带来一定的耗时。可能过此接口提前配置，来避免推理过程中的引擎重新构建。

**参数**
> * **tensor_name**: 需要设定输入范围的tensor名
> * **min_shape: 对应tensor的最小shape，例如[1, 3, 224, 224]
> * **opt_shape: 对应tensor的最常用shape，例如[2, 3, 224, 224], 当为默认参数即空vector时，则视为保持与min_shape一致，默认为空vector
> * **max_shape: 对应tensor的最大shape，例如[8, 3, 224, 224], 当为默认参数即空vector时，则视为保持与min_shape一致，默认为空vector

```
void SetTrtCacheFile(const string& cache_file_path)
```
当使用TensorRT后端时，通过此接口将构建好的TensorRT模型引擎缓存到指定路径，或跳过构造引擎步骤，直接加载本地缓存的TensorRT模型
- 当调用此接口，且`cache_file_path`不存在时，FastDeploy将构建TensorRT模型，并将构建好的模型保持至`cache_file_path`
- 当调用此接口，且`cache_file_path`存在时，FastDeploy将直接加载`cache_file_path`存储的已构建好的TensorRT模型，从而大大减少模型加载初始化的耗时

通过此接口，可以在第二次运行代码时，加速模型加载初始化的时间，但因此也需注意，如需您修改了模型加载配置，例如TensorRT的max_workspace_size，或重新设置了`SetTrtInputShape`，以及更换了原始的paddle或onnx模型，需先删除已缓存在本地的`cache_file_path`文件，避免重新加载旧的缓存，影响程序正确性。

**参数**
> * **cache_file_path**: 缓存文件路径，例如`/Downloads/resnet50.trt`

```
void EnableTrtFp16()
void DisableTrtFp16()
```
当使用TensorRT后端时，通过此接口开启或关闭半精度推理加速，会带来明显的性能提升，但并非所有GPU都支持半精度推理。 在不支持半精度推理的GPU上，将会回退到FP32推理，并给出提示`Detected FP16 is not supported in the current GPU, will use FP32 instead.`
