English | [简体中文](README_CN.md)
# PaddleSeg C++ Deployment Example

## Supporting Model List

- PP-LiteSeg deployment models are from [PaddleSeg PP-LiteSeg series model](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/pp_liteseg/README.md).

## PP-LiteSeg Model Deployment and Conversion Preparations

Befor SOPHGO-TPU model deployment, you should first convert Paddle model to bmodel model. Specific steps are as follows:
- Download Paddle model: [PP-LiteSeg-B(STDC2)-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz).
- Convert Paddle model to ONNX model. Please refer to [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX).
- For the process of converting ONNX model to bmodel, please refer to [TPU-MLIR](https://github.com/sophgo/tpu-mlir).

## Model Converting Example

Here we take [PP-LiteSeg-B(STDC2)-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz) as an example to show you how to convert Paddle model to SOPHGO-TPU model.

### Download PP-LiteSeg-B(STDC2)-cityscapes-without-argmax, and convert it to ONNX
```shell
https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz

# Modify the input shape of PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer model from dynamic input to constant input.
python paddle_infer_shape.py --model_dir PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir pp_liteseg_fix \
                             --input_shape_dict="{'x':[1,3,512,512]}"

# Convert constant input Paddle model to ONNX model.
paddle2onnx --model_dir pp_liteseg_fix \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file pp_liteseg.onnx \
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
mkdir pp_liteseg && cd pp_liteseg

# Put the test image in this file, and put the converted pp_liteseg.onnx into this folder.
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
# Put in the onnx model file pp_liteseg.onnx.

mkdir workspace && cd workspace

# Convert ONNX model to mlir model, the parameter --output_names can be viewed via NETRON.
model_transform.py \
    --model_name pp_liteseg \
    --model_def ../pp_liteseg.onnx \
    --input_shapes [[1,3,512,512]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names bilinear_interp_v2_6.tmp_0 \
    --test_input ../image/dog.jpg \
    --test_result pp_liteseg_top_outputs.npz \
    --mlir pp_liteseg.mlir

# Convert mlir model to BM1684x F32 bmodel.
model_deploy.py \
  --mlir pp_liteseg.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input pp_liteseg_in_f32.npz \
  --test_reference pp_liteseg_top_outputs.npz \
  --model pp_liteseg_1684x_f32.bmodel
```
The final bmodel, pp_liteseg_1684x_f32.bmodel, can run on BM1684x. If you want to further accelerate the model, you can convert ONNX model to INT8 bmodel. For details, please refer to [TPU-MLIR Document](https://github.com/sophgo/tpu-mlir/blob/master/README.md).

## Other Documents
- [Cpp Deployment](./cpp)
