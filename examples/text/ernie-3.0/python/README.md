# ERNIE-3.0 模型Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`seq_cls_infer.py` 快速完成ERNIE-3.0模型在CPU/GPU，以及CPU上通过OpenVINO加速CPU端文本分类任务的部署示例。执行如下脚本即可完成。

## 快速开始
```bash

#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
