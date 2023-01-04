English | [简体中文](README.md)
# PaddleDetection RKNPU2 Deployment Example

## List of Supported Models

Now FastDeploy supports the deployment of the following models
- [PicoDet models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)

## Prepare PaddleDetection deployment models and convert models
Before RKNPU deployment, you need to transform Paddle model to RKNN model:
* From Paddle dynamic map to ONNX model, refer to [PaddleDetection Model Export](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md)
  , and set **export.nms=True** during transformation.
* From ONNX model to RKNN model, refer to [Transformation Document](../../../../../docs/cn/faq/rknpu2/export.md).


## Model Transformation Example
The following steps are conducted on Ubuntu computers, refer to the configuration document to prepare the environment. Taking Picodet-s as an example, this document demonstrates how to transform PaddleDetection model to RKNN model.
### Export the ONNX model
```bash
# Download Paddle static map model and unzip it
wget https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar
tar xvf picodet_s_416_coco_lcnet.tar

# From static map to ONNX model. Attention: Align the save_file with the zip file name
paddle2onnx --model_dir picodet_s_416_coco_lcnet \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
            --enable_dev_version True

# Fix shape
python -m paddle2onnx.optimize --input_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --output_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --input_shape_dict "{'image':[1,3,416,416]}"
```

### Write the model export configuration file
Taking the example of RKNN model from RK3588, we need to edit tools/rknpu2/config/RK3568/picodet_s_416_coco_lcnet.yaml to convert ONNX model to RKNN model.

**Modify normalize parameter**

If you need to perform the normalize operation on NPU, configure the normalize parameters based on your model. For example:
```yaml
model_path: ./picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx
output_folder: ./picodet_s_416_coco_lcnet
target_platform: RK3568
normalize:
  mean: [[0.485,0.456,0.406]]
  std: [[0.229,0.224,0.225]]
outputs: ['tmp_17','p2o.Concat.9']
```

**Modify outputs parameter**
The output node names of the transformation model are various based on different versions of Paddle2ONNX. Please use [Netron](https://netron.app) and find the NonMaxSuppression node marked by the blue box below, and the node name in the red box is the target name.

For example, we can obtain the following image after visualization with Netron:
![](https://user-images.githubusercontent.com/58363586/202728663-4af0b843-d012-4aeb-8a66-626b7b87ca69.png)

Find the NonMaxSuppression node marked by the blue box,and we can see the names of the two nodes marked by the red box: tmp_17 and p2o.Concat.9. So we need to modify the outputs parameter:
```yaml
model_path: ./picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx
output_folder: ./picodet_s_416_coco_lcnet
target_platform: RK3568
normalize: None
outputs: ['tmp_17','p2o.Concat.9']
```

### model transformation
```bash

# Transform ONNX modle to RKNN model
# The transformed model is in the picodet_s_320_coco_lcnet_non_postprocess directory
python tools/rknpu2/export.py --config_path tools/rknpu2/config/picodet_s_416_coco_lcnet.yaml \
                              --target_platform rk3588
```

### Modify the configuration file during runtime

n the config file, we need to modify **Normalize** and **Permute** under **Preprocess**.

**Remove Permute**

The Permute operation needs removing considering that RKNPU only supports the input format of NHWC. After removal, the Precess is as follows:
```yaml
Preprocess:
- interp: 2
  keep_ratio: false
  target_size:
  - 416
  - 416
  type: Resize
- is_scale: true
  mean:
  - 0.485
  - 0.456
  - 0.406
  std:
  - 0.229
  - 0.224
  - 0.225
  type: NormalizeImage
```

**Decide whether to remove Normalize based on model transformation file**

RKNPU supports Normalize on NPU. Remove **Normalize** if you configured the Normalize parameter when exporting the model. After removal, the Precess is as follows:
```yaml
Preprocess:
- interp: 2
  keep_ratio: false
  target_size:
  - 416
  - 416
  type: Resize
```

## Other Links
- [Cpp Deployment](./cpp)
- [Python Deployment](./python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
