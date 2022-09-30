# RuntimeOption

`RuntimeOption` is used to configure the inference parameters of the model on different backends and hardware.

## Python  Class

```
class RuntimeOption()
```

### Member function

```
set_model_path(model_file, params_file="", model_format="paddle")
```

Set the model path for loading

**Parameters**

> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path. This parameter is not required for onnx model format
> * **model_format**(str): Model format. The model supports paddle, onnx format (Paddle by default).

```
use_gpu(device_id=0)
```

Inference on GPU

**Parameters**

> * **device_id**(int): When there are multiple GPU cards in the environment, this parameter specifies the card for inference. The default is 0.

```
use_cpu()
```

Inference on CPU

```
set_cpu_thread_num(thread_num=-1)
```

Set the number of threads on the CPU for inference

**Parameters**

> * **thread_num**(int): Number of threads, automatically allocated for the backend when the number is smaller than or equal to 0. The default is -1

```
use_paddle_backend()
```

Inference with Paddle Inference backend (CPU/GPU supported, Paddle model format supported).

```
use_ort_backend()
```

Inference with ONNX Runtime backend (CPU/GPU supported, Paddle and ONNX model format supported).

```
use_trt_backend()
```

Inference with TensorRT backend (GPU supported, Paddle/ONNX model format supported)

```
use_openvino_backend()
```

Inference with OpenVINO backend (CPU supported, Paddle/ONNX model format supported)

```
enable_paddle_mkldnn()
disable_paddle_mkldnn()
```

When using the Paddle Inference backend, this parameter determines whether the MKLDNN inference acceleration on the CPU is on or off. It is on by default.

```
enable_paddle_log_info()
disable_paddle_log_info()
```

When using the Paddle Inference backend, this parameter determines whether the optimization log on model loading is on or off. It is off by default.

```
set_paddle_mkldnn_cache_size(cache_size)
```

When using the Paddle Inference backend, this interface controls the shape cache size of MKLDNN acceleration

**Parameters**

> * **cache_size**(int): Cache size

```
set_trt_input_shape(tensor_name, min_shape, opt_shape=None, max_shape=None)
```

When using the TensorRT backend, this interface is used to set the Shape range of each input to the model. If only min_shape is set, the opt_shape and max_shape are automatically set to match min_shape.

FastDeploy will automatically update the shape range during the inference process according to the real-time data. But it will lead to a rebuilding of the back-end engine when it encounters a new shape range, costing more time. It is advisable to configure this interface in advance to avoid engine rebuilding during the inference process.

**Parameters**

> * **tensor_name**(str): tensor name of the range
> * **min_shape(list of int): Minimum shape of the corresponding tensor, e.g. [1, 3, 224, 224]
> * **opt_shape(list of int): The most common shape for the corresponding tensor, e.g. [2, 3, 224, 224]. When it is None, i.e. it remains the same as min_shape. The default is None.
> * **max_shape(list of int): The maximum shape for the corresponding tensor, e.g. [8, 3, 224, 224]. When it is None, i.e. it remains the same as min_shape. The default is None.

```
set_trt_cache_file(cache_file_path)
```

When using the TensorRT backend, developers can use this interface to cache the built TensorRT model engine to the designated path, or skip the building engine step and load the locally cached TensorRT model directly

- When this interface is called and `cache_file_path` does not exist, FastDeploy will build the TensorRT model and save the built model to `cache_file_path`
- When this interface is called and `cache_file_path` exists, FastDeploy will directly load the built TensorRT model stored in `cache_file_path`, thus greatly reducing the time spent on model load initialization.

This interface allows developers to speed up the initialisation of the model loading for later use. However, if developers change the model loading configuration, for example the max_workspace_size of TensorRT, or reset `set_trt_input_shape`, as well as replace the original paddle or onnx model, it is better to delete the `cache_file_path` file that has been cached locally first to avoid reloading the old cache, which could affect the program working.

**Parameters**

> * **cache_file_path**(str): cache file path. e.g.`/Downloads/resnet50.trt`

```
enable_trt_fp16()
disable_trt_fp16()
```

When using the TensorRT backend, turning half-precision inference acceleration on or off via this interface brings a significant performance boost. However, half-precision inference is not supported on all GPUs. On GPUs that do not support half-precision inference, it will fall back to FP32 inference and give the prompt `Detected FP16 is not supported in the current GPU, will use FP32 instead.`

## C++ Struct

```
struct RuntimeOption
```

### Member function

```
void SetModelPath(const string& model_file, const string& params_file = "", const string& model_format = "paddle")
```

Set the model path for loading

**Parameters**

> * **model_file**: Model file path
> * **params_file**: Parameter file path. This parameter could be optimized as "" for onnx model format
> * **model_format**: Model format. The model supports paddle, onnx format (Paddle by default).

```
void UseGpu(int device_id = 0)
```

Set to inference on GPU

**Parameters**

> * **device_id**: 0When there are multiple GPU cards in the environment, this parameter specifies the card for inference. The default is 0.

```
void UseCpu()
```

Set to inference on CPU

```
void SetCpuThreadNum(int thread_num=-1)
```

Set the number of threads on the CPU for inference

**Parameters**

> * **thread_num**: Number of threads, automatically allocated for the backend when the number is smaller than or equal to 0. The default is -1

```
void UsePaddleBackend()
```

Inference with Paddle Inference backend (CPU/GPU supported, Paddle model format supported).

```
void UseOrtBackend()
```

Inference with ONNX Runtime backend (CPU/GPU supported, Paddle and ONNX model format supported).

```
void UseTrtBackend()
```

Inference with TensorRT backend (GPU supported, Paddle/ONNX model format supported)

```
void UseOpenVINOBackend()
```

Inference with OpenVINO backend (CPU supported, Paddle/ONNX model format supported)

```
void SetPaddleMKLDNN(bool pd_mkldnn = true)
void DisablePaddleMKLDNN()
```

When using the Paddle Inference backend, this parameter determines whether the MKLDNN inference acceleration on the CPU is on or off. It is on by default.

```
void EnablePaddleLogInfo()
void DisablePaddleLogInfo()
```

When using the Paddle Inference backend, this parameter determines whether the optimization log on model loading is on or off. It is off by default.

```
void SetPaddleMKLDNNCacheSize(int cache_size)
```

When using the Paddle Inference backend, this interface controls the shape cache size of MKLDNN acceleration

**Parameters**

> * **cache_size**: Cache size

```
void SetTrtInputShape(const string& tensor_name, const vector<int32_t>& min_shape,
                      const vector<int32_t>& opt_shape = vector<int32_t>(),
                      const vector<int32_t>& opt_shape = vector<int32_t>())
```

When using the TensorRT backend, this interface sets the Shape range of each input to the model. If only min_shape is set, the opt_shape and max_shape are automatically set to match min_shape.

FastDeploy will automatically update the shape range during the inference process according to the real-time data. But it will lead to a rebuilding of the back-end engine when it encounters a new shape range, costing more time. It is advisable to configure this interface in advance to avoid engine rebuilding during the inference process.

**Parameters**

> - **tensor_name**(str): tensor name of the range
> - **min_shape(list of int): Minimum shape of the corresponding tensor, e.g. [1, 3, 224, 224]
> - **opt_shape(list of int): The most common shape for the corresponding tensor, e.g. [2, 3, 224, 224]. When it is empty vector, i.e. it remains the same as min_shape. The default is empty vector.
> - **max_shape(list of int): The maximum shape for the corresponding tensor, e.g. [8, 3, 224, 224]. When it is empty vector, i.e. it remains the same as min_shape. The default is empty vector.

```
void SetTrtCacheFile(const string& cache_file_path)
```

When using the TensorRT backend, developers can use this interface to cache the built TensorRT model engine to the designated path, or skip the building engine step and load the locally cached TensorRT model directly

- When this interface is called and `cache_file_path` does not exist, FastDeploy will build the TensorRT model and save the built model to `cache_file_path`
- When this interface is called and `cache_file_path` exists, FastDeploy will directly load the built TensorRT model stored in `cache_file_path`, thus greatly reducing the time spent on model load initialization.

This interface allows developers to speed up the initialisation of the model loading for later use. However, if developers change the model loading configuration, for example the max_workspace_size of TensorRT, or reset `SetTrtInputShape`, as well as replace the original paddle or onnx model, it is better to delete the `cache_file_path` file that has been cached locally first to avoid reloading the old cache, which could affect the program working.

**Parameters**

> * **cache_file_path**: cache file path, such as `/Downloads/resnet50.trt`

```
void EnableTrtFp16()
void DisableTrtFp16()
```

When using the TensorRT backend, turning half-precision inference acceleration on or off via this interface brings a significant performance boost. However, half-precision inference is not supported on all GPUs. On GPUs that do not support half-precision inference, it will fall back to FP32 inference and give the prompt `Detected FP16 is not supported in the current GPU, will use FP32 instead.`
