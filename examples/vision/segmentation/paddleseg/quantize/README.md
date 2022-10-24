# PaddleSeg 量化模型部署
FastDeploy已支持部署量化模型,并提供一键模型量化的工具.
用户可以使用一键模型量化工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## FastDeploy一键模型量化工具
FastDeploy 提供了一键量化工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型量化工具](../../../../../tools/quantization/)
注意: 推理量化后的分类模型仍然需要FP32模型文件夹下的deploy.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。

## 下载量化完成的PaddleSeg模型
用户也可以直接下载下表中的量化模型进行部署.
| 模型                 |推理后端            |部署硬件    | FP32推理时延    | INT8推理时延  | 加速比    | FP32 MIoU | INT8 MIoU |量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ.tar)            | Paddle Inference        |    CPU    |    1217.46      |   651.79    |     1.87       | 77.37  | 70.52 | 离线量化 |


上表中的数据, 为模型量化前后，在FastDeploy部署的Runtime推理性能.
- 测试数据为Cityscapes验证集中的所有图片.
- 推理时延为在不同Runtime上推理的时延, 单位是毫秒.
- CPU为Intel(R) Xeon(R) Gold 6271C, GPU为Tesla T4, TensorRT版本8.4.15, 所有测试中固定CPU线程数为1
- PP-LiteSeg量化模型目前暂只支持在Paddle Inference上推理

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
