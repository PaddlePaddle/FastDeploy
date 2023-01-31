中文 ｜ [English](../EN/demo-en.md)
# 服务化部署示例
我们以最简单的yolov5模型为例，讲述如何进行服务化部署，详细的代码和操作步骤见[yolov5服务化部署](../../../examples/vision/detection/yolov5/serving)，阅读本文之前建议您先阅读以下文档：
- [服务化模型目录说明](model_repository.md) (说明如何准备模型目录)
- [服务化部署配置说明](model_configuration.md)  (说明runtime的配置选项)

## 基本原理介绍
像常见的深度学习模型一样，yolov5完整的运行过程包含前处理+模型预测+后处理三个阶段。

在FastDeployServer中，将前处理、模型预测、后处理均视为1个**模型服务**，每个模型服务的**config.pbtxt**配置文件中均描述了其输入数据格式、输出数据格式、模型服务的类型（即config.pbtxt中的**backend**或**platform**字段）、以及其他的一些配置选项。

前处理和后处理一般是运行一段Python代码，为了方便后续描述，我们称之为**Python模型服务**，其config.pbtxt配置文件中的`backend: "python"`。

模型预测阶段是深度学习模型预测引擎加载用户提供的深度学习模型文件来运行模型预测，我们称之为**Runtime模型服务**，其config.pbtxt配置文件中的`backend: "fastdeploy"`。

根据用户提供的模型类型的不同，可以在**optimization**字段中设置使用CPU、GPU、TRT、ONNX等配置，配置方法参考[服务化部署配置说明](model_configuration.md)。

除此之外，还需要一个**Ensemble模型服务**来将前处理、模型预测、后处理3个**模型服务**组合为1个整体，并描述3个模型服务之间的关联关系。例如，前处理的输出与模型预测的输入之间的对应关系，多个模型服务的调用顺序、串并联关系等，**Ensemble模型服务**的config.pbtxt配置文件中的`platform: "ensemble"`。

在本文的yolov5服务化示例中，**Ensemble模型服务**将前处理、模型预测、后处理3个**模型服务**串联组合为1个整体，整体的结构如下图所示。
<p align="center">
    <br>
<img src='https://user-images.githubusercontent.com/35565423/204268774-7b2f6b4a-50b1-4962-ade9-cd10cf3897ab.png'>
    <br>
</p>
  
对于像[OCR这样多个深度学习模型的组合模型](../../../examples/vision/ocr/PP-OCRv3/serving)，或者[流式输入输出的深度学习模型](../../../examples/audio/pp-tts/serving)，其**Ensemble模型服务**的配置会更加复杂。
  
  
## Python模型服务简介
我们以[yolov5前处理](../../../examples/vision/detection/yolov5/serving/models/preprocess/1/model.py)为例，简单介绍一下编写Python模型服务中的注意事项。

Python模型服务的代码model.py的整体结构框架如下所示。Python代码的核心是1个`class TritonPythonModel`类，类中包含3个成员函数`initialize`、`execute`、`finalize`，类名、成员函数名、函数输入变量都不允许更改。在此基础上，用户可以自行编写代码。

```
import json
import numpy as np
import time
import fastdeploy as fd

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        #你的初始化操作代码，initialize函数只在模型加载的时候调用1次
        
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        #你的前处理代码，每次预测调用均会调用execute函数
        #FastDeploy提供了部分模型的python前后处理函数，无须用户编写
        #调用方式为fd.vision.detection.YOLOv5.preprocess(data)
        #用户也可以自行编写需要的处理逻辑
        
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        #你的析构代码，finalize只在模型卸载的时候被调用1次
```

`initialize`中一般放置初始化的一些操作，该函数只在Python模型服务被加载的时候执行1次。

`finalize`中一般放置一些析构释放的操作，该函数只在Python模型服务被卸载的时候执行1次。

`execute`中放置用户需要的前后处理的逻辑，该函数在每次服务端收到客户端请求的时候被执行1次。

`execute`函数的输入参数requests为InferenceRequest的集合，在没有开启[动态合并Batch功能](#动态合并Batch功能)时候，requests的长度是1，即只有1个InferenceRequest。

`execute`函数的返回参数responses必须是InferenceResponse的集合，通常情况下长度与requests的长度一致，即有N个InferenceRequest就必须返回N个InferenceResponse。

用户可以在`execute`函数中自行编写Python代码进行数据前处理或后处理，为了方便用户，FastDeploy提供了部分模型的python前后处理函数，使用方式如下：

```
import fastdeploy as fd
fd.vision.detection.YOLOv5.preprocess(data)
```

## 动态合并Batch功能
动态合并Batch的原理如下图所示。当用户请求request并发量较大，但GPU利用率较小时，可以通过将不同用户的request组合为1个大的Batch进行模型预测，从而提高服务的吞吐性能。
<p align="center">
    <br>
<img src='https://user-images.githubusercontent.com/35565423/204285444-1f9aaf24-05c2-4aae-bbd5-47dc3582dc01.png'>
    <br>
</p>

开启动态合并Batch功能非常简单，仅需在config.pbtxt结尾处，增加`dynamic_batching{}`字段即可，但最大的合并Batch不会超过`max_batch_size`。

**注意**：`ensemble_scheduling`字段与`dynamic_batching`字段不可共存，即对于**Ensemble模型服务**不存在动态合并Batch功能，这也可以理解，因为**Ensemble模型服务**本身仅仅是多个模型服务的组合。

## 多模型实例
多模型实例的原理如下图所示。当前后处理（通常不支持Batch）成为整个服务的性能瓶颈时，能通过增加多个前后处理的**Python模型服务**实例，来提高整个服务的时延性能。

当然也可以开启多个**Runtime模型服务**实例，来提升GPU利用率。
<p align="center">
    <br>
<img src='https://user-images.githubusercontent.com/35565423/204268809-6ea95a9f-e014-468a-8597-98b67ebc7381.png'>
    <br>
</p>

设置多模式实例的方法很简单，如下所示。
```
instance_group [
  {
      count: 3
      kind: KIND_CPU
  }
]
```
