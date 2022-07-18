# RuntimeOption 推理后端配置

FastDeploy产品中的Runtime包含多个推理后端，其各关系如下所示

| 模型格式\推理后端 | ONNXRuntime | Paddle Inference | TensorRT | OpenVINO |
| :---------------  | :---------- | :--------------- | :------- | :------- |
|     Paddle        | 支持(内置Paddle2ONNX) | 支持 | 支持(内置Paddle2ONNX) | 支持 |
|     ONNX          | 支持        | 支持(需通过X2Paddle转换) | 支持 | 支持 |

各Runtime支持的硬件情况如下

| 硬件/推理后端 | ONNXRuntime | Paddle Inference | TensorRT | OpenVINO |
| :---------------  | :---------- | :--------------- | :------- | :------- |
|   CPU        |  支持       | 支持        | 不支持 |   支持 |
|   GPU       |   支持       | 支持       | 支持    | 支持   |

在各模型的，均通过`RuntimeOption`来配置推理的后端，以及推理时的参数，例如在python中，加载模型后可通过如下代码打印推理配置
```
model = fastdeploy.vision.ultralytics.YOLOv5("yolov5s.onnx")
print(model.runtime_option)
```
可看下如下输出

```
RuntimeOption(
  backend : Backend.ORT                # 推理后端ONNXRuntime
  cpu_thread_num : 8                   # CPU线程数（仅当使用CPU推理时有效）
  device : Device.CPU                  # 推理硬件为CPU
  device_id : 0                        # 推理硬件id（针对GPU）
  model_file : yolov5s.onnx            # 模型文件路径
  params_file :                        # 参数文件路径
  model_format : Frontend.ONNX         # 模型格式
  ort_execution_mode : -1              # 前辍为ort的表示为ONNXRuntime后端专用参数
  ort_graph_opt_level : -1
  ort_inter_op_num_threads : -1
  trt_enable_fp16 : False              # 前辍为trt的表示为TensorRT后端专用参数
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

## Python 使用

### RuntimeOption类
`fastdeploy.RuntimeOption()`配置选项

#### 配置选项
> * **backend**(fd.Backend): `fd.Backend.ORT`/`fd.Backend.TRT`/`fd.Backend.PDINFER`/`fd.Backend.OPENVINO`等
> * **cpu_thread_num**(int): CPU推理线程数，仅当CPU推理时有效
> * **device**(fd.Device): `fd.Device.CPU`/`fd.Device.GPU`等
> * **device_id**(int): 设备id，在GPU下使用
> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **model_format**(Frontend): 模型格式, `fd.Frontend.PADDLE`/`fd.Frontend.ONNX`
> * **ort_execution_mode**(int): ORT后端执行方式，0表示按顺序执行所有算子，1表示并行执行算子，默认为-1，即按ORT默认配置方式执行
> * **ort_graph_opt_level**(int): ORT后端图优化等级；0：禁用图优化；1：基础优化 2：额外拓展优化；99：全部优化； 默认为-1，即按ORT默认配置方式执行
> * **ort_inter_op_num_threads**(int): 当`ort_execution_mode`为1时，此参数设置算子间并行的线程数
> * **trt_enable_fp16**(bool): TensorRT开启FP16推理
> * **trt_enable_int8**(bool): TensorRT开启INT8推理
> * **trt_max_workspace_size**(int): TensorRT配置的`max_workspace_size`参数
> * **trt_fixed_shape**(dict[str : list[int]]): 当模型为动态shape，但实际推理时输入shape保持不变，则通过此参数配置输入的固定shape
> * **trt_min_shape**(dict[str : list[int]]): 当模型为动态shape，且实际推理时输入shape也会变化，通过此参数配置输入的最小shape
> * **trt_opt_shape**(dict[str : list[int]]): 当模型为动态shape, 且实际推理时输入shape也会变化，通过此参数配置输入的最优shape
> * **trt_max_shape**(dict[str : list[int]]): 当模型为动态shape，且实际推理时输入shape也会变化，通过此参数配置输入的最大shape
> * **trt_max_batch_size**(int): TensorRT推理时的最大batch数

```
import fastdeploy as fd

option = fd.RuntimeOption()
option.backend = fd.Backend.TRT
# 当使用TRT后端，且为动态输入shape时
# 需配置输入shape信息
option.trt_min_shape = {"x": [1, 3, 224, 224]}
option.trt_opt_shape = {"x": [4, 3, 224, 224]}
option.trt_max_shape = {"x": [8, 3, 224, 224]}

model = fd.vision.ppcls.Model("resnet50/inference.pdmodel",
                              "resnet50/inference.pdiparams",
                              "resnet50/inference_cls.yaml",
                              runtime_option=option)
```

## C++ 使用

### RuntimeOption 结构体
`fastdeploy::RuntimeOption()`配置选项

#### 配置选项
> * **backend**(fastdeploy::Backend): `Backend::ORT`/`Backend::TRT`/`Backend::PDINFER`/`Backend::OPENVINO`等
> * **cpu_thread_num**(int): CPU推理线程数，仅当CPU推理时有效
> * **device**(fastdeploy::Device): `Device::CPU`/`Device::GPU`等
> * **device_id**(int): 设备id，在GPU下使用
> * **model_file**(string): 模型文件路径
> * **params_file**(string): 参数文件路径
> * **model_format**(fastdeploy::Frontend): 模型格式, `Frontend::PADDLE`/`Frontend::ONNX`
> * **ort_execution_mode**(int): ORT后端执行方式，0表示按顺序执行所有算子，1表示并行执行算子，默认为-1，即按ORT默认配置方式执行
> * **ort_graph_opt_level**(int): ORT后端图优化等级；0：禁用图优化；1：基础优化 2：额外拓展优化；99：全部优化； 默认为-1，即按ORT默认配置方式执行
> * **ort_inter_op_num_threads**(int): 当`ort_execution_mode`为1时，此参数设置算子间并行的线程数
> * **trt_enable_fp16**(bool): TensorRT开启FP16推理
> * **trt_enable_int8**(bool): TensorRT开启INT8推理
> * **trt_max_workspace_size**(int): TensorRT配置的`max_workspace_size`参数
> * **trt_fixed_shape**(map<string, vector<int>>): 当模型为动态shape，但实际推理时输入shape保持不变，则通过此参数配置输入的固定shape
> * **trt_min_shape**(map<string, vector<int>>): 当模型为动态shape，且实际推理时输入shape也会变化，通过此参数配置输入的最小shape
> * **trt_opt_shape**(map<string, vector<int>>): 当模型为动态shape, 且实际推理时输入shape也会变化，通过此参数配置输入的最优shape
> * **trt_max_shape**(map<string, vector<int>>): 当模型为动态shape，且实际推理时输入shape也会变化，通过此参数配置输入的最大shape
> * **trt_max_batch_size**(int): TensorRT推理时的最大batch数

```
#include "fastdeploy/vision.h"

int main() {
  auto option = fastdeploy::RuntimeOption();
  option.trt_min_shape["x"] = {1, 3, 224, 224};
  option.trt_opt_shape["x"] = {4, 3, 224, 224};
  option.trt_max_shape["x"] = {8, 3, 224, 224};

  auto model = fastdeploy::vision::ppcls.Model(
                           "resnet50/inference.pdmodel",
                           "resnet50/inference.pdiparams",
                           "resnet50/inference_cls.yaml",
                           option);
  return 0;
}
```
