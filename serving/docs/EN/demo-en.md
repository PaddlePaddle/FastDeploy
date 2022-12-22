English | [中文](../zh_CN/demo.md)
# Service-oriented Deployment Demo
We take the YOLOv5 model as an simple example, and introduce how to execute a service-oriented deployment. For the detailed code, please refer to [Service-oriented Deployment of YOLOv5](../../../examples/vision/detection/yolov5/serving). It is recommend that you read the following documents before reading this article.
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
        #FastDeploy provides python pre and post processing functions for some models, so you don't need to program them.
        #Please use fd.vision.detection.YOLOv5.preprocess(data) for calling.
        #用户也可以自行编写需要的处理逻辑
        
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        #你的析构代码，finalize只在模型卸载的时候被调用1次
```