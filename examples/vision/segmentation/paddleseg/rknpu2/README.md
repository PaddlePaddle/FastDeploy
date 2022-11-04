# PaddleSeg 模型部署

## 模型版本说明

- [PaddleSeg develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

目前FastDeploy使用RKNPU2推理PPSeg支持如下模型的部署:

| 模型                                                                                                                                           | 参数文件大小 | 输入Shape  | mIoU   | mIoU (flip) | mIoU (ms+flip) |
|:---------------------------------------------------------------------------------------------------------------------------------------------|:-------|:---------|:-------|:------------|:---------------|
| [Unet-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz)                                       | 52MB   | 1024x512 | 65.00% | 66.02%      | 66.89%         |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz)          | 31MB   | 1024x512 | 77.04% | 77.73%      | 77.46%         |
| [PP-HumanSegV1-Lite(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz)                                      | 543KB  | 192x192  | 86.2%  | -           | -              |
| [PP-HumanSegV2-Lite(通用人像分割模型)](https://bj.bcebos.com/paddle2onnx/libs/PP_HumanSegV2_Lite_192x192_infer.tgz)                                  | 12MB   | 192x192  | 92.52% | -           | -              |
| [PP-HumanSegV2-Mobile(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz)                          | 29MB   | 192x192  | 93.13% | -           | -              |
| [PP-HumanSegV1-Server(通用人像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz)                                  | 103MB  | 512x512  | 96.47% | -           | -              |
| [Portait-PP-HumanSegV2_Lite(肖像分割模型)](https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz)               | 3.6M   | 256x144  | 96.63% | -           | -              |
| [FCN-HRNet-W18-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz)                     | 37MB   | 1024x512 | 78.97% | 79.49%      | 79.74%         |
| [Deeplabv3-ResNet101-OS8-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz) | 150MB  | 1024x512 | 79.90% | 80.22%      | 80.47%         |

## 准备PaddleSeg部署模型以及转换模型
RKNPU部署模型前需要将Paddle模型转换成RKNN模型，具体步骤如下:
* Paddle动态图模型转换为ONNX模型，请参考[PaddleSeg模型导出说明](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg)
* ONNX模型转换RKNN模型的过程，请参考[转换文档](../../../../../docs/cn/faq/rknpu2/export.md)进行转换。

## 模型转换example

下面以Portait-PP-HumanSegV2_Lite(肖像分割模型)为例子，教大家如何转换PPSeg模型到RKNN模型。
```bash
# 下载Paddle2ONNX仓库
git clone https://github.com/PaddlePaddle/Paddle2ONNX

# 下载Paddle静态图模型并为Paddle静态图模型固定输入shape
## 进入为Paddle静态图模型固定输入shape的目录
cd Paddle2ONNX/tools/paddle
## 下载Paddle静态图模型并解压
wget https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz
tar xvf Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz
python paddle_infer_shape.py --model_dir Portrait_PP_HumanSegV2_Lite_256x144_infer/ \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir Portrait_PP_HumanSegV2_Lite_256x144_infer \
                             --input_shape_dict="{'x':[1,3,144,256]}"

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir Portrait_PP_HumanSegV2_Lite_256x144_infer \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file Portrait_PP_HumanSegV2_Lite_256x144_infer/Portrait_PP_HumanSegV2_Lite_256x144_infer.onnx \
            --enable_dev_version True

# ONNX模型转RKNN模型
# 将ONNX模型目录拷贝到Fastdeploy根目录
cp -r ./Portrait_PP_HumanSegV2_Lite_256x144_infer /path/to/Fastdeploy
# 转换模型,模型将生成在Portrait_PP_HumanSegV2_Lite_256x144_infer目录下
python tools/rknpu2/export.py --config_path tools/rknpu2/config/RK3588/Portrait_PP_HumanSegV2_Lite_256x144_infer.yaml
```

## 修改yaml配置文件

在**模型转换example**中，我们对模型的shape进行了固定，因此对应的yaml文件也要进行修改，如下:

**原yaml文件**
```yaml
Deploy:
  input_shape:
  - -1
  - 3
  - -1
  - -1
  model: model.pdmodel
  output_dtype: float32
  output_op: none
  params: model.pdiparams
  transforms:
  - target_size:
    - 256
    - 144
    type: Resize
  - type: Normalize
```

**修改后的yaml文件**
```yaml
Deploy:
  input_shape:
  - 1
  - 3
  - 144
  - 256
  model: model.pdmodel
  output_dtype: float32
  output_op: none
  params: model.pdiparams
  transforms:
  - target_size:
    - 256
    - 144
    type: Resize
  - type: Normalize
```

## 详细部署文档
- [RKNN总体部署教程](../../../../../docs/cn/faq/rknpu2/rknpu2.md)
- [C++部署](cpp)
- [Python部署](python)
