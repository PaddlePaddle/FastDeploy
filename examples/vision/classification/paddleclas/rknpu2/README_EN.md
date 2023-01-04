English | [简体中文](README.md)
# PaddleClas Model RKNPU2 Deployment

## Convert the model
Taking ResNet50_vd as an example, this document demonstrates how to convert classification model to RKNN model.

### Export the ONNX model
```bash
# Install paddle2onnx
pip install paddle2onnx

# Download ResNet50_vd model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz

# From static map to ONNX model. Attention: Align the save_file with the zip file name
paddle2onnx --model_dir ResNet50_vd_infer  \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams  \
            --save_file ResNet50_vd_infer/ResNet50_vd_infer.onnx  \
            --enable_dev_version True  \
            --opset_version 10  \
            --enable_onnx_checker True

# Fix shape. Attention: the inputs here should correspond to the name of the inputs shown in netron.app, which may be image or x
python -m paddle2onnx.optimize --input_model ResNet50_vd_infer/ResNet50_vd_infer.onnx \
                               --output_model ResNet50_vd_infer/ResNet50_vd_infer.onnx \
                               --input_shape_dict "{'inputs':[1,3,224,224]}"
```  

### Write the model export configuration file
Taking the example of RKNN model from RK3588, we need to edit tools/rknpu2/config/ResNet50_vd_infer_rknn.yaml to convert ONNX model to RKNN model.

If you need to perform the normalize operation on NPU, configure the normalize parameters based on your model. For example:
```yaml
model_path: ./ResNet50_vd_infer/ResNet50_vd_infer.onnx
output_folder: ./ResNet50_vd_infer
mean:
  -
    - 123.675
    - 116.28
    - 103.53
std:
  -
    - 58.395
    - 57.12
    - 57.375
outputs_nodes:
do_quantization: False
dataset: "./ResNet50_vd_infer/dataset.txt"
```

To **normalize on CPU**, refer to the following yaml：
```yaml
model_path: ./ResNet50_vd_infer/ResNet50_vd_infer.onnx
output_folder: ./ResNet50_vd_infer
mean:
  -
    - 0
    - 0
    - 0
std:
  -
    - 1
    - 1
    - 1
outputs_nodes:
do_quantization: False
dataset: "./ResNet50_vd_infer/dataset.txt"
```
Here we perform the normalize operation on NPU.


### From ONNX model to RKNN model
```shell
python tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/ResNet50_vd_infer_rknn.yaml \
        --target_platform rk3588
```

## Other Links
- [Cpp Deployment](./cpp)
- [Python Deployment](./python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
