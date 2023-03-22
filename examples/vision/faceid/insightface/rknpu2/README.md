English | [简体中文](README_CN.md)
# InsightFace RKNPU Deployment Example

This document provides the deployment of the InsightFace model in the RKNPU2 environment. For details, please refer to [Model Introduction Document].本教程提供InsightFace模型在RKNPU2环境下的部署，模型的详细介绍已经ONNX模型的下载请查看[模型介绍文档](../README.md)。

## 支持模型列表
目前FastDeploy支持如下模型的部署
- ArcFace
- CosFace
- PartialFC
- VPL

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了InsightFace导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）其中精度指标来源于InsightFace中对各模型的介绍，详情各参考InsightFace中的说明

| 模型                                                                                         | 大小    | 精度 (AgeDB_30) |
|:-------------------------------------------------------------------------------------------|:------|:--------------|
| [CosFace-r18](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r18.onnx)       | 92MB  | 97.7          |
| [CosFace-r34](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r34.onnx)       | 131MB | 98.3          |
| [CosFace-r50](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r50.onnx)       | 167MB | 98.3          |
| [CosFace-r100](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r100.onnx)     | 249MB | 98.4          |
| [ArcFace-r18](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r18.onnx)          | 92MB  | 97.7          |
| [ArcFace-r34](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r34.onnx)          | 131MB | 98.1          |
| [ArcFace-r50](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r50.onnx)          | 167MB | -             |
| [ArcFace-r100](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx)        | 249MB | 98.4          |
| [ArcFace-r100_lr0.1](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_r100_lr01.onnx)     | 249MB | 98.4          |
| [PartialFC-r34](https://bj.bcebos.com/paddlehub/fastdeploy/partial_fc_glint360k_r50.onnx)  | 167MB | -             |
| [PartialFC-r50](https://bj.bcebos.com/paddlehub/fastdeploy/partial_fc_glint360k_r100.onnx) | 249MB | -             |


## 转换为RKNPU模型

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r18.onnx

python -m paddle2onnx.optimize --input_model ./ms1mv3_arcface_r18/ms1mv3_arcface_r18.onnx \
                               --output_model ./ms1mv3_arcface_r18/ms1mv3_arcface_r18.onnx \
                               --input_shape_dict "{'data':[1,3,112,112]}"

python  /Path/To/FastDeploy/tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/arcface_unquantized.yaml \
        --target_platform rk3588
```

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[InsightFace CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5) 编写
