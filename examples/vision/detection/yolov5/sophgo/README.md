English | [简体中文](README_CN.md)
# YOLOv5 SOPHGO Deployment Example

## Supporting Model List

For YOLOv5 v6.0 model deployment, please refer to [YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.0) and [Pretrained model based on COCO](https://github.com/ultralytics/yolov5/releases/tag/v6.0).

## Preparing YOLOv5 Model Deployment and Conversion

Before deploying SOPHGO-TPU model, you need to first convert Paddle model to bmodel. Specific steps are as follows:
- Download the pre-trained ONNX model. Please refer to [YOLOv5 Ready-to-deploy Model](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/yolov5).
- Convert ONNX model to bmodel. Please refer to [TPU-MLIR](https://github.com/sophgo/tpu-mlir).

## Model conversion example

Here we take YOLOv5s as an example to show you how to convert ONNX model to SOPHGO-TPU model.

## Download YOLOv5s Model

### Download ONNX YOLOv5s Static Map Model
```shell
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx

```
### Export bmodel Model

Here we take BM1684x bmodel as an example. You need to download [TPU-MLIR](https://github.com/sophgo/tpu-mlir) project. For the installing process, please refer to [TPU-MLIR Document](https://github.com/sophgo/tpu-mlir/blob/master/README.md).
### 1.	Installation
``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can customize your own name.
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

source ./envsetup.sh
./build.sh
```

### 2.	Convert ONNX model to bmodel
``` shell
mkdir YOLOv5s && cd YOLOv5s

# Put the test image in this file, and put the yolov5s.onnx into this folder.
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
# Put in the onnx model file yolov5s.onnx

mkdir workspace && cd workspace

# Convert ONNX model to mlir model, the parameter --output_names can be viewed via NETRON.
model_transform.py \
    --model_name yolov5s \
    --model_def ../yolov5s.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names output,350,498,646 \
    --test_input ../image/dog.jpg \
    --test_result yolov5s_top_outputs.npz \
    --mlir yolov5s.mlir

# Convert mlir model to BM1684x F32 bmodel.
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --model yolov5s_1684x_f32.bmodel
```
The final bmodel, yolov5s_1684x_f32.bmodel, can run on BM1684x. If you want to further accelerate the model, you can convert ONNX model to INT8 bmodel. For details, please refer to [TPU-MLIR Document](https://github.com/sophgo/tpu-mlir/blob/master/README.md).

## Other Documents
- [Cpp Deployment](./cpp)
