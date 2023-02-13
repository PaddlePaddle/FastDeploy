English | [中文](../zh_CN/model_configuration.md)
# Model Configuration
Each model in the model repository must contain a configuration that provides required and optional information about the model. The configuration information is generally written in [ModelConfig protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto) format in file *config.pbtxt*.

## Minimum Model General Configuration
Please see the official website for detailed general configuration: [model_configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md). Minimum model configuration of Triton must include: attribute *platform* or *backend*, attribute *max_batch_size* and input and output of the model.

For example, the minimum configuration of a Paddle model should be (with two inputs *input0* and *input1*, one output *output0*, both inputs and outputs are tensors of type float32, and the maximum batch is 8):


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

## Configuration of CPU, GPU and Instances number

The attribute *instance_group* allows you to configure hardware resource and model inference instances number.

Here's an example of CPU deployment:
```
  instance_group [
    {
      # Create two CPU instances
      count: 2
      # Use CPU for deployment 
      kind: KIND_CPU
    }
  ]
```
Another example of deploying two instances on *GPU 0*, and one instance each on *GPU1* and *GPU*:

```
  instance_group [
    {
      # Create tow GPU instances
      count: 2
      # Use GPU for inference
      kind: KIND_GPU
      # Deploy on GPU 0
      gpus: [ 0 ]
    },
    {
      count: 1
      kind: KIND_GPU
      # Deploy on GPU 1,2
      gpus: [ 1, 2 ]
    }
  ]
```

### Name, Platform and Backend
The attribute *name* is optional. If the model is not specified in the configuration,  then the name is the model's directory name. When the name is specified, it should match the directory name.

Set *fastdeploy backend*. You should not configure attribute *platform*, but please instead configure attribute *backend* to *fastdeploy*.

```
backend: "fastdeploy"
```

### FastDeploy Backend Configuration

Currently FastDeploy backend supports *cpu* and *gpu* inference, with *paddle*, *onnxruntime* and *openvino* inference engines supported on *cpu*, and *paddle*, *onnxruntime* and *tensorrt* engines supported on *gpu*.


#### Paddle Engine Configuration
In addition to configuring *Instance Groups*, deciding whether the model runs on CPU or GPU, the Paddle engine can be configured as follows. You can see more specific examples in [A PP-OCRv3 example for Runtime configuration](../../../examples/vision/ocr/PP-OCRv3/serving/models/cls_runtime/config.pbtxt).

```
optimization {
  execution_accelerators {
    # CPU inference configuration, used with KIND_CPU.
    cpu_execution_accelerator : [
      {
        name : "paddle"
        # Set parallel inference computing threads number to 4.
        parameters { key: "cpu_threads" value: "4" }
        # Set mkldnn acceleration on, or off when set to 0.
        parameters { key: "use_mkldnn" value: "1" }
      }
    ],
    # GPU inference configuration, used with KIND_GPU.
    gpu_execution_accelerator : [
      {
        name : "paddle"
        # Set parallel inference computing threads number to 4.
        parameters { key: "cpu_threads" value: "4" }
        # Set mkldnn acceleration on, or off when set to 0.
        parameters { key: "use_mkldnn" value: "1" }
      }
    ]
  }
}
```

#### ONNXRuntime Engine Configuration
In addition to configuring *Instance Groups*, deciding whether the model runs on CPU or GPU, the ONNXRuntime engine can be configured as follows. You can see more specific examples in [A YOLOv5 example for Runtime configuration](../../../examples/vision/detection/yolov5/serving/models/runtime/config.pbtxt).

```
optimization {
  execution_accelerators {
    cpu_execution_accelerator : [
      {
        name : "onnxruntime"
        # Set parallel inference computing threads number to 4.
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

### OpenVINO Engine Configuration
The OpenVINO engine only supports inferring on CPU, which can be configured as:

```
optimization {
  execution_accelerators {
    cpu_execution_accelerator : [
      {
        name : "openvino"
        # Set parallel inference computing threads number to 4 (total number of threads for all instances).
        parameters { key: "cpu_threads" value: "4" }
        # Set num_streams in OpenVINO (usually the same as instances number)
        parameters { key: "num_streams" value: "1" }
      }
    ]
  }
}
```

### TensorRT Engine Configuration
The TensorRT engine only supports inferring on GPU, which can be configured as:

```
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "tensorrt"
        # Use FP16 inference in TensorRT. You can also choose: trt_fp32
        # If the loaded model is a quantized model, this precision will be int8 automatically
        parameters { key: "precision" value: "trt_fp16" }
      }
    ]
  }
}
```

You can configure the TensorRT dynamic shape in the following format, and refer to [A PaddleCls example for Runtime configuration](../../../examples/vision/classification/paddleclas/serving/models/runtime/config.pbtxt):
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
