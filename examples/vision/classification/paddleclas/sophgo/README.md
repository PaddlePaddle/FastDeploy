English | [简体中文](README_CN.md)
# PaddleDetection SOPHGO Deployment Example

## Supporting Model List

Currently FastDeploy supports the following model deployment: [ResNet series model](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/models/ResNet_and_vd_en.md).

## Preparing ResNet Model Deployment and Conversion

Before deploying SOPHGO-TPU model, you need to first convert Paddle model to bmodel. Specific steps are as follows:
- Convert Paddle dynamic map model to ONNX model, please refer to [Paddle2ONNX model conversion](https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/model_zoo/classification).
- For the process of converting ONNX model to bmodel, please refer to [TPU-MLIR](https://github.com/sophgo/tpu-mlir).

## Model Converting Example

Here we take [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz) as an example to show you how to convert Paddle model to SOPHGO-TPU model.

## Export ONNX Model

### Download and Unzip Paddle ResNet50_vd Static Map Model
```shell
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar xvf ResNet50_vd_infer.tgz
```

### Convert Static Map Model to ONNX Model, note that the save_file here aligns with the zip name
```shell
paddle2onnx --model_dir ResNet50_vd_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ResNet50_vd_infer.onnx \
            --enable_dev_version True
```
### Export bmodel

Take converting BM1684x model to bmodel as an example. You need to download [TPU-MLIR](https://github.com/sophgo/tpu-mlir) project. For the process of installation, please refer to [TPU-MLIR Document](https://github.com/sophgo/tpu-mlir/blob/master/README.md).
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
mkdir ResNet50_vd_infer && cd ResNet50_vd_infer

# Put the test image in this file, and put the ResNet50_vd_infer.onnx into this folder.
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
# Put in the onnx model file ResNet50_vd_infer.onnx.

mkdir workspace && cd workspace

# Convert ONNX model to mlir model, the parameter --output_names can be viewed via NETRON.
model_transform.py \
    --model_name ResNet50_vd_infer \
    --model_def ../ResNet50_vd_infer.onnx \
    --input_shapes [[1,3,224,224]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names save_infer_model/scale_0.tmp_1 \
    --test_input ../image/dog.jpg \
    --test_result ResNet50_vd_infer_top_outputs.npz \
    --mlir ResNet50_vd_infer.mlir

# Convert mlir model to BM1684x F32 bmodel.
model_deploy.py \
  --mlir ResNet50_vd_infer.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input ResNet50_vd_infer_in_f32.npz \
  --test_reference ResNet50_vd_infer_top_outputs.npz \
  --model ResNet50_vd_infer_1684x_f32.bmodel
```
The final bmodel, ResNet50_vd_infer_1684x_f32.bmodel, can run on BM1684x. If you want to further accelerate the model, you can convert ONNX model to INT8 bmodel. For details, please refer to [TPU-MLIR Document](https://github.com/sophgo/tpu-mlir/blob/master/README.md).

## Other Documents
- [Cpp Deployment](./cpp)
