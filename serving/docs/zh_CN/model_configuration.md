中文 ｜ [English](../EN/model_configuration-en.md)
# 模型配置
模型存储库中的每个模型都必须包含一个模型配置，该配置提供了关于模型的必要和可选信息。这些配置信息一般写在 *config.pbtxt* 文件中，[ModelConfig protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)格式。

## 模型通用最小配置
详细的模型通用配置请看官网文档: [model_configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md).Triton的最小模型配置必须包括: *platform* 或 *backend* 属性、*max_batch_size* 属性和模型的输入输出.

例如一个Paddle模型，有两个输入*input0* 和 *input1*，一个输出*output0*，输入输出都是float32类型的tensor，最大batch为8.则最小的配置如下:

```
  backend: "fastdeploy"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```

## CPU、GPU和实例个数配置

通过*instance_group*属性可以配置服务使用哪种硬件资源，分别部署多少个模型推理实例。

CPU部署例子：
```
  instance_group [
    {
      # 创建两个CPU实例
      count: 2
      # 使用CPU部署  
      kind: KIND_CPU
    }
  ]
```

在*GPU 0*上部署2个实例，在*GPU1*和*GPU*上分别部署1个实例

```
  instance_group [
    {
      # 创建两个GPU实例
      count: 2
      # 使用GPU推理
      kind: KIND_GPU
      # 部署在GPU卡0上
      gpus: [ 0 ]
    },
    {
      count: 1
      kind: KIND_GPU
      # 在GPU卡1、2都部署
      gpus: [ 1, 2 ]
    }
  ]
```

### Name, Platform and Backend
模型配置中 *name* 属性是可选的。如果模型没有在配置中指定，则使用模型的目录名；如果指定了该属性，它必须要跟模型的目录名一致。

使用 *fastdeploy backend*，没有*platform*属性可以配置，必须配置*backend*属性为*fastdeploy*。

```
backend: "fastdeploy"
```

### FastDeploy Backend配置

FastDeploy后端目前支持*cpu*和*gpu*推理，*cpu*上支持*paddle*、*onnxruntime*和*openvino*三个推理引擎，*gpu*上支持*paddle*、*onnxruntime*和*tensorrt*三个引擎。


#### 配置使用Paddle引擎
除去配置 *Instance Groups*，决定模型运行在CPU还是GPU上。Paddle引擎中，还可以进行如下配置,具体例子可参照[PP-OCRv3例子中Runtime配置](../../../examples/vision/ocr/PP-OCRv3/serving/models/cls_runtime/config.pbtxt):

```
optimization {
  execution_accelerators {
    # CPU推理配置， 配合KIND_CPU使用
    cpu_execution_accelerator : [
      {
        name : "paddle"
        # 设置推理并行计算线程数为4
        parameters { key: "cpu_threads" value: "4" }
        # 开启mkldnn加速，设置为0关闭mkldnn
        parameters { key: "use_mkldnn" value: "1" }
      }
    ],
    # GPU推理配置， 配合KIND_GPU使用
    gpu_execution_accelerator : [
      {
        name : "paddle"
        # 设置推理并行计算线程数为4
        parameters { key: "cpu_threads" value: "4" }
        # 开启mkldnn加速，设置为0关闭mkldnn
        parameters { key: "use_mkldnn" value: "1" }
      }
    ]
  }
}
```

### 配置使用ONNXRuntime引擎
除去配置 *Instance Groups*，决定模型运行在CPU还是GPU上。ONNXRuntime引擎中，还可以进行如下配置，具体例子可参照[YOLOv5的Runtime配置](../../../examples/vision/detection/yolov5/serving/models/runtime/config.pbtxt):

```
optimization {
  execution_accelerators {
    cpu_execution_accelerator : [
      {
        name : "onnxruntime"
        # 设置推理并行计算线程数为4
        parameters { key: "cpu_threads" value: "4" }
      }
    ],
    gpu_execution_accelerator : [
      {
        name : "onnxruntime"
      }
    ]
  }
}
```

### 配置使用OpenVINO引擎
OpenVINO引擎只支持CPU推理，配置如下:

```
optimization {
  execution_accelerators {
    cpu_execution_accelerator : [
      {
        name : "openvino"
        # 设置推理并行计算线程数为4（所有实例总共线程数）
        parameters { key: "cpu_threads" value: "4" }
        # 设置OpenVINO的num_streams（一般设置为跟实例数一致）
        parameters { key: "num_streams" value: "1" }
      }
    ]
  }
}
```

### 配置使用TensorRT引擎
TensorRT引擎只支持GPU推理，配置如下:

```
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "tensorrt"
        # 使用TensorRT的FP16推理,其他可选项为: trt_fp32
        # 如果加载的是量化模型，此精度设置无效，会默认使用int8进行推理
        parameters { key: "precision" value: "trt_fp16" }
      }
    ]
  }
}
```

配置TensorRT动态shape的格式如下，可参照[PaddleCls例子中Runtime配置](../../../examples/vision/classification/paddleclas/serving/models/runtime/config.pbtxt):
```
optimization {
  execution_accelerators {
  gpu_execution_accelerator : [ {
    # use TRT engine
    name: "tensorrt",
    # use fp16 on TRT engine
    parameters { key: "precision" value: "trt_fp16" }
  },
  {
    # Configure the minimum shape of dynamic shape
    name: "min_shape"
    # All input name and minimum shape
    parameters { key: "input1" value: "1 3 224 224" }
    parameters { key: "input2" value: "1 10" }
  },
  {
    # Configure the optimal shape of dynamic shape
    name: "opt_shape"
    # All input name and optimal shape
    parameters { key: "input1" value: "2 3 224 224" }
    parameters { key: "input2" value: "2 20" }
  },
  {
    # Configure the maximum shape of dynamic shape
    name: "max_shape"
    # All input name and maximum shape
    parameters { key: "input1" value: "8 3 224 224" }
    parameters { key: "input2" value: "8 30" }
  }
  ]
}}
```
