# Paddle2ONNX

[简体中文](README_zh.md) | English

## Introduction

Paddle2ONNX enables users to convert models from PaddlePaddle to ONNX.

- Supported model format. Paddle2ONNX supports both dynamic and static computational graph of PaddlePaddle. For static computational graph, Paddle2ONNX converts PaddlePaddle models saved by API [save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/save_inference_model_cn.html#save-inference-model), for example [IPthon example](examples/tutorial.ipynb). For dynamic computational graph, it is now under experiment and more details will be released after the release of PaddlePaddle 2.0.
- Supported operators. Paddle2ONNX can stably export models to ONNX Opset 9~11, and partialy support lower version opset. More details please refer to [Operator list](docs/en/op_list.md).
- Supported models. You can find officially verified models by Paddle2ONNX in [model zoo](docs/en/model_zoo.md).

## AIStudio Tutorials

- [Export and inference ONNX model in PaddlePaddle 2.0](https://aistudio.baidu.com/aistudio/projectdetail/1461212)
- [How to deploy PP-OCR model using ONNX RunTime](https://aistudio.baidu.com/aistudio/projectdetail/1479970)

## What we can do with Paddle2ONNX
- Deploy PaddlePaddle model by ADLIK, [more details](https://github.com/Adlik/Adlik/tree/master/examples/paddle_model)
- Deploy PaddlePaddle model by OpenVINO, [more details](https://paddlex.readthedocs.io/zh_CN/develop/deploy/openvino/index.html)
- Deploy PaddlePaddle model by OpenCV, [more details](https://github.com/opencv/opencv/tree/master/samples/dnn/dnn_model_runner/dnn_conversion/paddlepaddle)
- Deploy PaddlePaddle model by Triton, [more details](https://github.com/PaddlePaddle/PaddleX/blob/develop/deploy/cpp/docs/compile/triton/docker.md)

## Environment Dependencies

### Configuration
     python >= 2.7  
     static computational graph: paddlepaddle >= 1.8.0
     dynamic computational graph: paddlepaddle >= 2.0.0
     onnx == 1.7.0 | Optional
## Installation

### Via Pip

     pip install paddle2onnx


### From Source

     git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
     cd Paddle2ONNX
     python setup.py install

## Usage
### Static Computational Graph
#### Via Command Line Tool
Uncombined PaddlePaddle model(parameters saved in different files)

    paddle2onnx --model_dir paddle_model  --save_file onnx_file --opset_version 10 --enable_onnx_checker True

Combined PaddlePaddle model(parameters saved in one binary file)

    paddle2onnx --model_dir paddle_model  --model_filename model_filename --params_filename params_filename --save_file onnx_file --opset_version 10 --enable_onnx_checker True

If you need to configure the input shape, use the following command:

    paddle2onnx --model_dir paddle_model  --model_filename model_filename --params_filename params_filename --save_file onnx_file --opset_version 10 --enable_onnx_checker True  --input_shape_dict "{'x': [1, 3, 224, 224]}"

#### Parameters
| Parameters | Description |
|----------|--------------|
|--model_dir | The directory path of the paddlepaddle model saved by `paddle.fluid.io.save_inference_model`|
|--model_filename |**[Optional]** The model file name under the directory designated by`--model_dir`. Only needed when all the model parameters saved in one binary file. Default value None|
|--params_filename |**[Optional]** the parameter file name under the directory designated by`--model_dir`. Only needed when all the model parameters saved in one binary file. Default value None|
|--save_file | the directory path for the exported ONNX model|
|--opset_version | **[Optional]** To configure the ONNX Opset version. Opset 9-11 are stably supported. Default value is 9.|
|--enable_dev_version | **[Optional]** Whether to use new version of Paddle2ONNX while is under developing. Default value is False.|
|--enable_onnx_checker| **[Optional]**  To check the validity of the exported ONNX model. It is suggested to turn on the switch. If set to True, onnx>=1.7.0 is required. Default value is False|
|--enable_paddle_fallback| **[Optional]**  Whether custom op is exported using paddle_fallback mode. Default value is False|
|--enable_auto_update_opset| **[Optional]**  Whether enable auto_update_opset. Default value is True|
|--input_shape_dict| **[Optional]**  Configure the input shape, the default is empty|
|--version |**[Optional]** check the version of paddle2onnx |
|--output_names| **[Optional]**  Set the output name of the model, the default is empty, support configuration in list form，for example：--output_names "['my_output1','my_output2']"，or in dict form，for example："{'paddle_output1':'my_output1', 'paddle_output2':'my_output2'}"|

- Two types of PaddlePaddle models
   - Combined model, parameters saved in one binary file. --model_filename and --params_filename represents the file name and parameter name under the directory designated by --model_dir. --model_filename and --params_filename are valid only with parameter --model_dir.
   - Uncombined model, parameters saved in different files. Only --model_dir is needed，which contains '\_\_model\_\_' file and the seperated parameter files.
- Use onnxruntime to verify the Converted model
    - When using onnxruntime to verify the converted onnx model, please note that the onnxruntime and onnx versions need to match. [Onnxruntime and onnx version requirements. ](https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md)
- If there is a prompt that OP does not support during model conversion, users are welcome to develop by themselves, please refer to the document [OP Development Guide](docs/zh/Paddle2ONNX_Development_Guide.md)


#### IPython tutorials

- [Convert to ONNX from static computational graph](examples/tutorial.ipynb)

### Dynamic Computational Graph

```
import paddle
from paddle import nn
from paddle.static import InputSpec
import paddle2onnx as p2o

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(784, 10)

    def forward(self, x):
        return self._linear(x)

layer = LinearNet()

# configure model inputs
x_spec = InputSpec([None, 784], 'float32', 'x')

# convert model to inference mode
layer.eval()

save_path = 'onnx.save/linear_net'
p2o.dygraph2onnx(layer, save_path + '.onnx', input_spec=[x_spec])

# when paddlepaddle>2.0.0, you can try:
# paddle.onnx.export(layer, save_path, input_spec=[x_spec])

```

#### IPython tutorials

- [Convert to ONNX from dynamic computational graph](examples/tutorial_dygraph2onnx.ipynb)

## Documents

- [model zoo](docs/en/model_zoo.md)
- [op list](docs/en/op_list.md)
- [update notes](docs/en/change_log.md)

## License
[Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
