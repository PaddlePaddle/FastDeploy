[English](README.md) | 简体中文
# SCRFD RKNPU2部署模型

本教程提供SCRFD模型在RKNPU2环境下的部署，模型的详细介绍已经ONNX模型的下载请查看[模型介绍文档](../README.md)。

## ONNX模型转换RKNN模型

下面以scrfd_500m_bnkps_shape640x640为例子，快速的转换SCRFD ONNX模型为RKNN量化模型。 以下命令在Ubuntu18.04下执行:
```bash
wget  https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/scrfd_500m_bnkps_shape640x640.zip
unzip scrfd_500m_bnkps_shape640x640.zip
python  /Path/To/FastDeploy/tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/scrfd_quantized.yaml \
        --target_platform rk3588
```



## 详细部署文档

- [Python部署](python/README.md)
- [C++部署](cpp/README.md)


## 版本说明

- 本版本文档和代码基于[SCRFD CommitID:17cdeab](https://github.com/deepinsight/insightface/tree/17cdeab12a35efcebc2660453a8cbeae96e20950) 编写
