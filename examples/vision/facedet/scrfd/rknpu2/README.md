English | [简体中文](README_CN.md)
# SCRFD RKNPU2 Deployment Models

This tutorial demonstrates the deployment of SCRFD models in RKNPU2. For model description and download of ONNX models, refer to [Model Description](../README.md)。

## From ONNX model to RKNN model

Taking scrfd_500m_bnkps_shape640x640 as an example, the following commands in Ubuntu18.0 demonstrate how to fast convert SCRFD ONNX models to RKNN quantification models:
```bash
wget  https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/scrfd_500m_bnkps_shape640x640.zip
unzip scrfd_500m_bnkps_shape640x640.zip
python  /Path/To/FastDeploy/tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/scrfd.yaml \
        --target_platform rk3588
```



## Detailed Deployment Tutorials

- [Python Deployment](python/README.md)
- [C++ Deployment](cpp/README.md)


## Release Note

- This document and code are written based on [SCRFD CommitID:17cdeab](https://github.com/deepinsight/insightface/tree/17cdeab12a35efcebc2660453a8cbeae96e20950)
