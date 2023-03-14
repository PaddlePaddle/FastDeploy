English | [简体中文](README_CN.md)
# PaddleDetection RKNPU2 Deployment Example

## List of Supported Models

Now FastDeploy supports the deployment of the following models
The PaddleDetection models that have been tested on RKNPU2 are as follows:

- Picodet
- PPYOLOE(int8)
- YOLOV8

If detailed speed information is required, you can check out the [RKNPU2 Model Speed Table](../../../../../docs/cn/faq/rknpu2/rknpu2.md)

## Prepare PaddleDetection deployment models and convert models
Before RKNPU deployment, you need to transform Paddle model to RKNN model:
* From Paddle dynamic map to ONNX model, refer to [PaddleDetection Model Export](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md), and set **export.nms=True** during transformation.
* From ONNX model to RKNN model, refer to [Transformation Document](../../../../../docs/en/faq/rknpu2/export.md).


## Model Transformation Example
### Notes

When deploying PPDetection models on RKNPU2, you may pay attention to the following points:

* Include Decode for model export
* Since RKNPU2 does not support NMS, the output nodes must be clipped before NMS 
* Limited by the RKNPU2 Div operator, the output nodes need to be clipped before the Div operator 

### From Paddle model to ONNX model

Because the rknN-toolkit2 provided by Rockchip does not support the direct support from Paddle models to RKNN models, you need to export the Paddle model to the ONNX model first, and then convert the ONNX model to the RKNN model. 
```bash
# Taking Picodet as an example
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
                                --input_shape_dict "{'image':[1,3,416,416], 'scale_factor':[1,2]}"
```

### Write the yaml file

**Modify normalize parameter**

If you need to perform the normalize operation on NPU, configure the normalize parameters based on your model. For example:
```yaml
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
```

**Modify outputs parameter**
The output node names of the transformation model are various based on different versions of Paddle2ONNX. Please use [Netron](https://netron.app) and find the NonMaxSuppression node marked by the blue box below, and the node name in the red box is the target name.

For example, we can obtain the following image after visualization with Netron:
![](https://ai-studio-static-online.cdn.bcebos.com/8bce6b904a6b479e8b30da9f7c719fad57517ffb2f234aeca3b8ace0761754d5)

Find the NonMaxSuppression node marked by the blue box,and we can see the names of the two nodes marked by the red box: tmp_17 and p2o.Concat.9. So we need to modify the outputs parameter:
```yaml
outputs_nodes:
  - 'p2o.Mul.179'
  - 'p2o.Concat.9'
```

### From ONNX model to RKNN model

For your convenience, we provide python scripts that will enable you to quickly convert ONNX models to RKNN models through our pre-configured config file 
### model transformation
```bash
python tools/rknpu2/export.py --config_path 
tools/rknpu2/config/picodet_s_416_coco_lcnet_unquantized.yaml \
                              --target_platform rk3588
```

## List of RKNN models 

For people’s testing, we provide two models, picodet and ppyoloe, which can be used after unzipping:

| Model Name                    | Download Address                                                    |
| --------------------------- | ------------------------------------------------------------ |
| picodet_s_416_coco_lcnet    | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/picodet_s_416_coco_lcnet.zip |
| ppyoloe_plus_crn_s_80e_coco | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/ppyoloe_plus_crn_s_80e_coco.zip |

## Other Links
- [Cpp Deployment](./cpp)
- [Python Deployment](./python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
