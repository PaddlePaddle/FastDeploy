# PaddleSeg 模型部署

## 模型版本说明

- [PaddleSeg develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

目前FastDeploy支持如下模型的部署

- [U-Net系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/unet/README.md)
- [PP-LiteSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/pp_liteseg/README.md)
- [PP-HumanSeg系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README.md)
- [FCN系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/fcn/README.md)
- [DeepLabV3系列模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/deeplabv3/README.md)

【注意】如你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../matting)

## 准备PaddleSeg部署模型

PaddleSeg模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export_cn.md)

## 下载预训练模型

为了方便开发者的测试，下面提供了PaddleSeg导出的部分模型（导出方式为：**指定**`--input_shape`，**指定**`--output_op none`，**指定**`--without_argmax`，**指定**`--with_softmax`），开发者可直接下载使用。

| 任务场景             | 模型                | 模型版本(表示已经测试的版本)                                                                                                                            | 大小  | ONNX/RKNN是否支持 | ONNX/RKNN速度(ms) |
|------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----|---------------|-----------------|
| Segmentation     | PP-LiteSeg        | [PP_LiteSeg_T_STDC1_cityscapes](https://bj.bcebos.com/fastdeploy/models/rknn2/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_3588.tgz) | -   | True/True     | 6634/5598       |
| Segmentation     | PP-HumanSegV2Lite | [portrait](https://bj.bcebos.com/fastdeploy/models/rknn2/portrait_pp_humansegv2_lite_256x144_inference_model_without_softmax_3588.tgz)     | -   | True/True     | 456/266         |
| Segmentation     | PP-HumanSegV2Lite | [human](https://bj.bcebos.com/fastdeploy/models/rknn2/human_pp_humansegv2_lite_192x192_pretrained_3588.tgz)                                | -   | True/True     | 496/256         |

## 详细部署文档

- [C++部署](cpp)
