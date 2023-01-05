English | [中文](../zh_CN/demo.md)
# Service-oriented Deployment Demo
We take the YOLOv5 model as an simple example, and introduce how to execute a service-oriented deployment. For the detailed code, please refer to [Service-oriented Deployment of YOLOv5](../../../examples/vision/detection/yolov5/serving). It is recommend that you read the following documents before reading this document.
- [Service-oriented Model Catalog Description](model_repository-en.md) (how to prepare the model catalog)
- [Service-oriented Deployment Configuration Description](model_configuration-en.md) (the configuration option for runtime)

## Fundamental Introduction
Similar to common deep learning models, the process of YOLOv5 consists of three stages: pre-processing, model prediction and post-processing.

Pre-processing, model prediction, and post-processing are all considered as one **model service** in FastDeployServer. The **config.pbtxt** configuration file of each model service describes its input data format, output data format, the type of model service (i.e. **backend** or **platform** in config.pbtxt), and some other options.

In pre-processing and post-processing stage, we generally run a piece of Python code. So let us simply call it **Python model service**, and the corresponding config.pbtxt configure `backend: "python"`.

The model prediction stage is when the deep learning model prediction engine loads the deep learning model files user supplied to run the model prediction, which we call **Runtime model service**, and the corresponding config.pbtxt configure `backend: "fastdeploy"`.

Depending on different type of model provided, the configuration of using CPU, GPU, TRT, ONNX, etc. can be set in **optimization**. Please refer to [Service-based Deployment Configuration Introduction](model_configuration-en.md) for configuring methods.

In addition, **Ensemble model service** is required to combine the 3 **model services** stages of pre-processing, model prediction, and post-processing into one whole, and is used to describe the correlation between the 3 model services. For example, the correspondence between the output of pre-processing and the input of model prediction, the calling order of multiple model services, the series-parallel relationship, etc. Its corresponding config.pbtxt configure `platform: "ensemble"`.

In this YOLOv5  example, **Ensemble model service** combines 3 **model services** stages of pre-processing, model prediction, and post-processing as a whole, and the overall structure is shown in the figure below.
<p align="center">
    <br>
<img src='https://user-images.githubusercontent.com/35565423/204268774-7b2f6b4a-50b1-4962-ade9-cd10cf3897ab.png'>
    <br>
</p>

For [a combinition model of multiple deep learning models like OCR](../../../examples/vision/ocr/PP-OCRv3/serving), or [a deep learning model with streaming input and output](../../../examples/audio/pp-tts/serving), the **Ensemble model service** configuration is more complex.
  
  
## Introduction to Python Model Service
Let us take [Pre-processing in YOLOv5](../../../examples/vision/detection/yolov5/serving/models/preprocess/1/model.py) as an example to briefly introduce the notes in programming a Python model service.

The overall structure framework of the Python model service code model.py is shown below. The core is the class `TritonPythonModel`, which contains three member functions `initialize`, `execute`, and `finalize`. Name of classes, member functions, and input variables are not allowed to be changed. On this basis, you can write your own code.

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
        #The initialize function is only called when loading the model.
        
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
        #Pre-processing code that calls the execute function for each prediction.
        #FastDeploy provides pre and post processing python functions for some models, so you don't need to program them.
        #Please use fd.vision.detection.YOLOv5.preprocess(data) for calling.
        #You can write your own processing logic
        
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        #Destructor code, it is only called when the model is being unloaded
```

Initialization operations are generally in function `initialize` , and this function is only executed once when the Python model service is being loaded.

Destructive release operations are generally in function `finalize`, and this function is only executed once when the Python model service is being unloaded.

The pre and post processing logic is generally designed in function `execute`, and this function is executed once each time the server receives a client request.

The input parameter `requests` of the function `execute`, is a collection of InferenceRequest. When [Dynamic Batching](#Dynamic-Batching) is not enabled, the length of requests is 1, i.e. there is only one InferenceRequest.

The return parameter `responses` of the function `execute` must be a collection of InferenceResponse, with the length usually the same as the length of `requests`, that is, N InferenceRequest must return N InferenceResponse.

You can write your own code in the `execute` function for data pre-processing or post-processing. For convenience, FastDeploy provides pre and post processing python functions for some models. You can write:

```
import fastdeploy as fd
fd.vision.detection.YOLOv5.preprocess(data)
```

## Dynamic Batching
The principle of dynamic batching is shown in the figure. When the user request concurrency is high while the GPU utilization is low, the throughput performance can be improved by merging different user requests into a large Batch for model prediction.
<p align="center">
    <br>
<img src='https://user-images.githubusercontent.com/35565423/204285444-1f9aaf24-05c2-4aae-bbd5-47dc3582dc01.png'>
    <br>
</p>

Enabling dynamic batching is as simple as adding the lines `dynamic_batching{}` to the end of config.pbtxt. Please note that the maximum batch size should not exceed `max_batch_size`.

**Note**: The field `ensemble_scheduling` and the field `dynamic_batching` should not coexist. That is, dynamic batching is not available for **Ensemble Model Service**, since **Ensemble Model Service** itself is just a combination of multiple model services.

## Multi-Model Instance
The principle of multi-model instance is shown in the figure below. When pre and post processing (which usually does not support Batch) becomes the performance bottleneck of the whole service, it is possible to improve the latency performance by adding **Python Model Service** instances for pre and post processing.

Of course, you can also turn on multiple **Runtime Model Service** instances to improve GPU utilization.
<p align="center">
    <br>
<img src='https://user-images.githubusercontent.com/35565423/204268809-6ea95a9f-e014-468a-8597-98b67ebc7381.png'>
    <br>
</p>

It is simple to set a multi-model instance, just write:
```
instance_group [
  {
      count: 3
      kind: KIND_CPU
  }
]
```