# FastDeploy

FastDeploy分为`Runtime`和`应用`模块。

## Runtime
`Runtime`对应于不同硬件上的不同后端，大部分情况下，一种硬件对应于一种后端，但对于CPU、GPU, 存在多种后端，用户可根据自己的需求进行选择。

| Runtime | 后端 |
| :------ | :---- |
| CPU(x86_64) | `fastdeploy::Backend::ORT` |
| GPU(Nvidia) | `fastdeploy::Backend::ORT` / `fastdeploy::Backend::TRT` |

具体文档参考 [Runtime文档](runtime.md)


## 应用

应用是基于`Runtime`提供的上层模型推理，集成了模型端到端的推理功能

- Vision
- Text
- Audio

具体文档参考 [Vision文档](vision.md)
