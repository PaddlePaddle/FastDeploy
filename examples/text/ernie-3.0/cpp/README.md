# ERNIE-3.0 模型Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`seq_cls_infer.cc`快速完成在CPU/GPU的文本分类任务的示例。

## 快速开始
以Linux上ERNIE-3.0-me模型推理为例，在本目录执行如下命令即可完成编译测试。

```
#下载SDK，编译模型examples代码（SDK中包含了examples代码）
