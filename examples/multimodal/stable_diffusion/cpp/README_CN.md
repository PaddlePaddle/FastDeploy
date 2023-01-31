简体中文 ｜ [English](README.md)
# StableDiffusion C++部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`*_infer.cc`快速完成StableDiffusion各任务的C++部署示例。

## Inpaint任务

StableDiffusion Inpaint任务是一个根据提示文本补全图片的任务，具体而言就是用户给定提示文本，原始图片以及原始图片的mask图片，该任务输出补全后的图片。
