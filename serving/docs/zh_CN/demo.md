# 服务化部署示例
我们以最简单的yolov5模型为例，讲述如何进行服务化部署，详细的代码和操作步骤见[yolov5服务化部署](../../../examples/vision/detection/yolov5/serving)，阅读本文之前建议您先阅读以下文档：
- [服务化模型目录说明](model_repository.md) (说明如何准备模型目录)
- [服务化部署配置说明](model_configuration.md)  (说明runtime的配置选项)

## 目录结构与配置的原理介绍
像常见的深度学习模型一样，yolov5完整的运行过程包含前处理+模型预测+后处理三个阶段。

在Triton中，将前处理、模型预测、后处理均视为1个**Triton-Model**，每个Triton-Model的**config.pbtxt**配置文件中均描述了其输入数据格式、输出数据格式、Triton-Model的类型（即config.pbtxt中的**backend**或**platform**字段）、以及其他的一些配置选项。

前处理和后处理一般是运行一段Python代码，为了方便后续描述，我们称之为**Python-Triton-Model**，其config.pbtxt配置文件中的`backend: "python"`。

模型预测阶段是深度学习模型预测引擎（如ONNXRuntime、Paddle、TRT、FastDeploy）加载用户提供的深度学习模型文件来运行模型预测，我们称之为**Runtime-Triton-Model**，其config.pbtxt配置文件中的`backend: "fastdeploy"`。

根据用户提供的模型类型的不同，可以在**optimization**字段中设置使用CPU、GPU、TRT、ONNX等配置，配置方法参考[服务化部署配置说明](model_configuration.md)。

除此之外，还需要一个**Ensemble-Triton-Model**来将前处理、模型预测、后处理3个**Triton-Model**组合为1个整体，并描述3个Triton-Model之间的关联关系，例如，前处理的输出与模型预测的输入之间的对应关系，3个Triton-Model的调用顺序、串并联关系等，**Ensemble-Triton-Model**的config.pbtxt配置文件中的`platform: "ensemble"`。

在本文的yolov5服务化示例中，**Ensemble-Triton-Model**将前处理、模型预测、后处理3个**Triton-Model**串联组合为1个整体，整体的结构如下图所示。


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
除去配置 *Instance Groups*，决定模型运行在CPU还是GPU上。Paddle引擎中，还可以进行如下配置:

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
除去配置 *Instance Groups*，决定模型运行在CPU还是GPU上。ONNXRuntime引擎中，还可以进行如下配置:

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
        # 使用TensorRT的FP16推理,其他可选项为: trt_fp32、trt_int8
        parameters { key: "precision" value: "trt_fp16" }
      }
    ]
  }
}
```
