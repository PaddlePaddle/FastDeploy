---
name: 报告issue
about: Describe this issue template's purpose here.
title: ''
labels: ''
assignees: ''

---

## 环境

- FastDeploy版本： 说明具体的版本，如fastdeploy-linux-gpu-0.8.0或自行编译的develop代码（附上自行编译的方式，及cmake时print的编译选项截图）
- 系统平台: Linux x64(Ubuntu 18.04) / Windows x64(Windows10) / Mac OSX arm(12.0) / Mac OSX intel(12.0)
- 硬件： 说明具体硬件型号，如 Nvidia GPU 3080TI， CUDA 11.2 CUDNN 8.3
- 编译语言： C++ / Python(3.7或3.8等）

## 问题描述
- 附上详细的问题日志有助于工程师快速定位分析
- 性能问题，描述清楚对比的方式
- - 注意性能测试，循环跑N次，取后80%的用时平均（模型启动时，刚开始受限于资源分配，速度会较慢）
- - FastDeploy的Predict包含模型本身之外的数据前后处理用时
