# Deploy PaddleDetection by OpenVINO

[中文版本](./openvino_ppdet_cn.md)

In this document, we will show how to convert PaddleDetection's model to ONNX, and convert ONNX model to OpenVINO IR.

## Some issues

- 1. OpenVINO requires that all the shapes of model node should be fixed
- 2. The NMS operator of object detection, will lead to a dynamic shape of result

## How to fix the problem

we provide a NMS convertor plugin in this directory, make sure that the output of NMS keeps a fixed shape

- The output of NMS(shape is N*6), will include some invalid result which label_id is negative. We should filter this invalid result by ourself, you can refer to the `openvino_ppdet/yolov3_infer.py:postprocess()` for details

## Introduce to PaddleDetection models

Currently, this document only supports serials of YOLOv3 in PaddleDetection, e.g YOLOv3-DarkNet, YOLOv3-ResNet34. These models contains 3 inputs and 2 outputs. The inputs are ["image", "im_shape", "scale_factor"], means preprocessed image data(N*3*H*W), the shape of origin image before preprocessing(N*2), the scale factor while resize origin image(N*2).

In order to convert model to openvino, we need to fix all the shapes of input tensors, e.g {"image": [1, 3, 608, 608], "im_shape": [1, 2], "scale_factor": [1, 2]}, 1 means batch size, and 608 means resized height/width(this has to be times of 32, like 320, 608, 640)

## Start to Convert

### 1. Export PaddleDetection model

```
cd PaddleDetection
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
                             -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams \
                             --output_dir inference_model
# Assume the model save in /User/XXX/PaddleDetection/inference_model/yolov3_darknet53_270e_coco
```

### 2. Export to ONNX format
We need to install Paddle2ONNX from its source code
```
# if paddle2onnx is installed, uninstall it
# pip uninstall paddle2onnx
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git checkout release/0.9
python setup.py install

cd experimental
```

Currenly, you are in the directory of `Paddle2ONNX/experimental`, and you will see a sub directory `openvino_ppdet` there. Execute the follwoing python code here
```
import paddle2onnx
import paddle
from openvino_ppdet import nms_mapper

model_prefix = "/User/XXX/PaddleDetection/inference_model/yolov3_darknet53_270e_coco/model"
# load model by paddle
model = paddle.jit.load(model_prefix)
input_shape_dict = {
    "image": [1, 3, 608, 608],
    "scale_factor": [1, 2],
    "im_shape": [1, 2]
    }
onnx_model = paddle2onnx.run_convert(model, input_shape_dict=input_shape_dict, opset_version=11)

with open("./yolov3.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

Make sure the following logs are printed in your screen, means the nms convertor plugin are enabled
```
===============================


You are using a nms convertor for OpenVINO!


===============================
```

## 3. Convert to OpenVINO IR
If OpenVINO is installed already, and you have initialized the environment, using the folloing command to convert ONNX model to OpenVINO IR
```
mo.py --framework onnx --input_model yolov3.onnx --output_dir ov_model
```

## 4. Inference by OpenVINO

Currently, you need in the directory of `Paddle2ONNX/experimental`, and there's a sub directory of `openvino_ppdet`. We implement a simple code `yolov3_infer.py` which include image preprocess, model inference, post porcess and visualize. Execute the following python code to get a visualized result.
```
from openvino_ppdet.yolov3_infer import YOLOv3

xml_file = "ov_model/yolov3.xml"
bin_file = "ov_model/yolov3.bin"
model = YOLOv3(xml_file=xml_file,
               bin_file=bin_file,
               model_input_shape=[608, 608])
boxes = model.predict("./test.jpg", visualize_out="./result.jpg", threshold=0.5)
```
- model_input_shape should be same with the ONNX model, it will used to preprocess the image file
- visualized_out defines the path to save the visualized result, set it to None if you don't need visualize
- threshold will used to filter the objects which confidence is lower than it.
