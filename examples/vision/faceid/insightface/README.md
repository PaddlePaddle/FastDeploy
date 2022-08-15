# InsightFace准备部署模型

- [InsightFace](https://github.com/deepinsight/insightface/commit/babb9a5)
  - （1）[官方库](https://github.com/deepinsight/insightface/)中提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的InsightFace模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。


## 支持模型列表
目前FastDeploy支持如下模型的部署
- ArcFace
- CosFace
- PartialFC
- VPL


## 导出ONNX模型
以ArcFace为例:
访问[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)官方github库，按照指引下载安装，下载pt模型文件，利用 `torch2onnx.py` 得到`onnx`格式文件。

* 下载ArcFace模型文件
  ```
  Link: https://pan.baidu.com/share/init?surl=CL-l4zWqsI1oDuEEYVhj-g code: e8pw  
  ```

* 导出onnx格式文件
  ```bash
  PYTHONPATH=. python ./torch2onnx.py ms1mv3_arcface_r100_fp16/backbone.pth --output ms1mv3_arcface_r100.onnx --network r100 --simplify 1
  ```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了InsightFace导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库的.pt文件测试结果）其中精度指标来源于InsightFace中对各模型的介绍，详情各参考InsightFace中的说明

| 模型                                                               | 大小    | 精度 (AgeDB_30)   |
|:---------------------------------------------------------------- |:----- |:----- |
| [CosFace-r18](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r18.onnx) | 92MB | 97.7 |
| [CosFace-r34](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r34.onnx) | 131MB | 98.3|
| [CosFace-r50](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r50.onnx) | 167MB | 98.3 |
| [CosFace-r100](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r100.onnx) | 249MB | 98.4 |
| [ArcFace-r18](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r18.onnx) | 92MB | 97.7 |
| [ArcFace-r34](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r34.onnx) | 131MB | 98.1|
| [ArcFace-r50](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r50.onnx) | 167MB | - |
| [ArcFace-r100](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx) | 249MB | 98.4 |
| [ArcFace-r100_lr0.1](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_r100_lr01.onnx) | 249MB | 98.4 |
| [PartialFC-r34](https://bj.bcebos.com/paddlehub/fastdeploy/partial_fc_glint360k_r50.onnx) | 167MB | -|
| [PartialFC-r50](https://bj.bcebos.com/paddlehub/fastdeploy/partial_fc_glint360k_r100.onnx) | 249MB | - |




## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[InsightFace CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5) 编写
