# RuntimeOption Inference Backend Deployment

The Runtime in the FastDeploy product contains multiple inference backends:

| Model Format\Inference Backend | ONNXRuntime                    | Paddle Inference                           | TensorRT                       | OpenVINO |
|:------------------------------ |:------------------------------ |:------------------------------------------ |:------------------------------ |:-------- |
| Paddle                         | Support (built-in Paddle2ONNX) | Support                                    | Support (built-in Paddle2ONNX) | Support  |
| ONNX                           | Support                        | Support (requires conversion via X2Paddle) | Support                        | Support  |

The hardware supported by Runtime is as follows

| Hardware/Inference Backend | ONNXRuntime | Paddle Inference | TensorRT    | OpenVINO |
|:-------------------------- |:----------- |:---------------- |:----------- |:-------- |
| CPU                        | Support     | Support          | Not Support | Support  |
| GPU                        | Support     | Support          | Support     | Support  |

Each model uses `RuntimeOption` to configure the inference backend and parameters, e.g. in python, the inference configuration can be printed after loading the model with the following code

```python
model = fastdeploy.vision.detection.YOLOv5("yolov5s.onnx")
print(model.runtime_option)
```

See below:

```python
RuntimeOption(
  backend : Backend.ORT                # Inference Backend ONNXRuntime
  cpu_thread_num : 8                   # Number of CPU threads (valid only when using CPU)
  device : Device.CPU                  # Inference hardware is CPU
  device_id : 0                        # Inference hardware id (for GPU)
  model_file : yolov5s.onnx            # Path to the model file
  params_file :                        # Parameter file path
  model_format : ModelFormat.ONNX         # odel format
  ort_execution_mode : -1              # The prefix ort indicates ONNXRuntime backend parameters
  ort_graph_opt_level : -1
  ort_inter_op_num_threads : -1
  trt_enable_fp16 : False              # The prefix of trt indicates a TensorRT backend  parameter
  trt_enable_int8 : False
  trt_max_workspace_size : 1073741824
  trt_serialize_file :
  trt_fixed_shape : {}
  trt_min_shape : {}
  trt_opt_shape : {}
  trt_max_shape : {}
  trt_max_batch_size : 32
)
```

## Python

### RuntimeOption Class

`fastdeploy.RuntimeOption()`Configuration

#### Configuration options

> * **backend**(fd.Backend): `fd.Backend.ORT`/`fd.Backend.TRT`/`fd.Backend.PDINFER`/`fd.Backend.OPENVINO`
> * **cpu_thread_num**(int): Number of CPU inference threads, valid only on CPU inference
> * **device**(fd.Device): `fd.Device.CPU`/`fd.Device.GPU`
> * **device_id**(int): Device id, used on GPU
> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path
> * **model_format**(ModelFormat): Model format, `fd.ModelFormat.PADDLE`/`fd.ModelFormat.ONNX`
> * **ort_execution_mode**(int): ORT back-end execution mode, 0 for sequential execution of all operators, 1 for parallel execution of operators, default is -1, i.e. execution in the ORT default configuration
> * **ort_graph_opt_level**(int): ORT back-end image optimisation level; 0: disable image optimisation; 1: basic optimisation 2: additional expanded optimisation; 99: all optimisation; default is -1, i.e. executed in the ORT default configuration
> * **ort_inter_op_num_threads**(int): When `ort_execution_mode` is 1, this parameter sets the number of threads in parallel between operators
> * **trt_enable_fp16**(bool): TensorRT turns on FP16 inference
> * **trt_enable_int8**(bool):TensorRT turns on INT8 inference
> * **trt_max_workspace_size**(int):  `max_workspace_size` parameter configured on TensorRT
> * **trt_fixed_shape**(dict[str : list[int]]):When the model is a dynamic shape, but the input shape remains constant for the actual inference, the input fixed shape is configured with this parameter
> * **trt_min_shape**(dict[str : list[int]]): When the model is a dynamic shape and the input shape changes during the actual inference, the minimum shape of the input is configured with this parameter
> * **trt_opt_shape**(dict[str : list[int]]): When the model is a dynamic shape and the input shape changes during the actual inference, the optimal shape of the input is configured with this parameter
> * **trt_max_shape**(dict[str : list[int]]): When the model is a dynamic shape and the input shape changes during the actual inference, the maximum shape of the input is configured with this parameter
> * **trt_max_batch_size**(int): Maximum number of batches for TensorRT inference

```python
import fastdeploy as fd

option = fd.RuntimeOption()
option.backend = fd.Backend.TRT
# When using a TRT backend with a dynamic input shape
# Configure input shape information
option.trt_min_shape = {"x": [1, 3, 224, 224]}
option.trt_opt_shape = {"x": [4, 3, 224, 224]}
option.trt_max_shape = {"x": [8, 3, 224, 224]}

model = fd.vision.classification.PaddleClasModel(
    "resnet50/inference.pdmodel",
    "resnet50/inference.pdiparams",
    "resnet50/inference_cls.yaml",
    runtime_option=option)
```

## C++

### RuntimeOption  Struct

`fastdeploy::RuntimeOption()`Configuration options

#### Configuration options

> * **backend**(fastdeploy::Backend): `Backend::ORT`/`Backend::TRT`/`Backend::PDINFER`/`Backend::OPENVINO`
> * **cpu_thread_num**(int): 、Number of CPU inference threads, valid only on CPU inference
> * **device**(fastdeploy::Device): `Device::CPU`/`Device::GPU`
> * **device_id**(int): Device id, used on GPU
> * **model_file**(string): Model file path
> * **params_file**(string): Parameter file path
> * **model_format**(fastdeploy::ModelFormat): Model format,`ModelFormat::PADDLE`/`ModelFormat::ONNX`
> * **ort_execution_mode**(int): ORT back-end execution mode, 0 for sequential execution of all operators, 1 for parallel execution of operators, default is -1, i.e. execution in the ORT default configuration
> * **ort_graph_opt_level**(int): ORT back-end image optimisation level; 0: disable image optimisation; 1: basic optimisation 2: additional expanded optimisation; 99: all optimisation; default is -1, i.e. executed in the ORT default configuration
> * **ort_inter_op_num_threads**(int): When `ort_execution_mode` is 1, this parameter sets the number of threads in parallel between operators
> * **trt_enable_fp16**(bool): TensorRT turns on FP16 inference
> * **trt_enable_int8**(bool): TensorRT turns on INT8 inference
> * **trt_max_workspace_size**(int): `max_workspace_size` parameter configured on TensorRT
> * **trt_fixed_shape**(map<string, vector<int>>): When the model is a dynamic shape, but the input shape remains constant for the actual inference, the input fixed shape is configured with this parameter
> * **trt_min_shape**(map<string, vector<int>>): When the model is a dynamic shape and the input shape changes during the actual inference, the minimum shape of the input is configured with this parameter
> * **trt_opt_shape**(map<string, vector<int>>): When the model is a dynamic shape and the input shape changes during the actual inference, the optimal shape of the input is configured with this parameter
> * **trt_max_shape**(map<string, vector<int>>): When the model is a dynamic shape and the input shape changes during the actual inference, the maximum shape of the input is configured with this parameter
> * **trt_max_batch_size**(int): Maximum number of batches for TensorRT inference

```c++
#include "fastdeploy/vision.h"

int main() {
  auto option = fastdeploy::RuntimeOption();
  option.trt_min_shape["x"] = {1, 3, 224, 224};
  option.trt_opt_shape["x"] = {4, 3, 224, 224};
  option.trt_max_shape["x"] = {8, 3, 224, 224};

  auto model = fastdeploy::vision::classification::PaddleClasModel(
                           "resnet50/inference.pdmodel",
                           "resnet50/inference.pdiparams",
                           "resnet50/inference_cls.yaml",
                            option);
  return 0;
}
```
