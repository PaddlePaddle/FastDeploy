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

## 准备PaddleSeg部署模型以及转换模型

RKNPU部署模型前需要将模型转换成RKNN模型，其过程一般可以简化为如下步骤:
*   Paddle动态图模型 -> Paddle静态图模型 -> ONNX模型 -> RKNN模型。
    *   对于Paddle动态图模型 -> Paddle静态图模型 -> ONNX模型的过程，请参考其文档说明([模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export_cn.md))。
        !!!!!注意!!!!!导出模型时请不要导出softmax部分。详细配置可以参考AIStudio项目，大家Fork项目后可一键运行体验。
        ```text
        我发现了一篇高质量的实训项目，使用免费算力即可一键运行，还能额外获取8小时免费GPU运行时长，快来Fork一下体验吧。
        模型集市——Paddle系列模型ONNX合集：https://aistudio.baidu.com/aistudio/projectdetail/4618218?contributionType=1&sUid=790375&shared=1&ts=1667027805784
        ```
    *   对于ONNX模型 -> RKNN模型的过程，我将其集成在[LuFeng仓库](https://github.com/Zheng-Bicheng/LuFeng)中，请参考[转换文档](https://github.com/Zheng-Bicheng/LuFeng/blob/main/docs/export.md)进行转换。
        以PPHumanSeg为例，在获取到ONNX模型后，其转换步骤如下:
        * 下载LuFeng仓库
        ```bash
        git clone https://github.com/Zheng-Bicheng/LuFeng.git
        ```
        * 编写config.yaml文件
        ```yaml
        model_path: ./portrait_pp_humansegv2_lite_256x144_pretrained.onnx
        output_folder: ./
        target_platform: RK3588
        normalize:
        mean: [0.5,0.5,0.5]
        std: [0.5,0.5,0.5]
        outputs: None
        ```
        * 执行转换代码
        ```bash
        python tools/export.py  --config_path=./config.yaml
        ```
## 下载预训练模型

为了方便开发者的测试，下面提供了PaddleSeg导出的部分模型（导出方式为：**指定**`--input_shape`，**指定**`--output_op none`，**指定**`--without_argmax`，**指定**`--with_softmax`），开发者可直接下载使用。

| 任务场景             | 模型                | 模型版本(表示已经测试的版本)                                                                                                                            | 大小  | ONNX/RKNN是否支持 | ONNX/RKNN速度(ms) |
|------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----|---------------|-----------------|
| Segmentation     | PP-LiteSeg        | [PP_LiteSeg_T_STDC1_cityscapes](https://bj.bcebos.com/fastdeploy/models/rknn2/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_3588.tgz) | -   | True/True     | 6634/5598       |
| Segmentation     | PP-HumanSegV2Lite | [portrait](https://bj.bcebos.com/fastdeploy/models/rknn2/portrait_pp_humansegv2_lite_256x144_inference_model_without_softmax_3588.tgz)     | -   | True/True     | 456/266         |
| Segmentation     | PP-HumanSegV2Lite | [human](https://bj.bcebos.com/fastdeploy/models/rknn2/human_pp_humansegv2_lite_192x192_pretrained_3588.tgz)                                | -   | True/True     | 496/256         |

## 详细部署文档
- [RKNN总体部署教程](../../../../../docs/cn/faq/rknpu2.md)
- [C++部署](cpp)
- [Python部署](python)