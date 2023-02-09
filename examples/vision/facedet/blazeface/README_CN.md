# BlazeFace准备部署模型

- BlazeFace部署模型实现来自[BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection),和[基于WiderFace的预训练模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection)
  - （1）[官方库](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/tools)中提供的*.params,通过[export_model.py](#导出PADDLE模型)操作后，可进行部署；
  - （2）开发者基于自己数据训练的BlazeFace模型，可按照[export_model.py](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/tools/export_model.py)导出模型后，完成部署。

## 导出PADDLE模型

访问[BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection)github库，按照指引下载安装，下载`.yml`和`.params` 模型参数，利用 `export_model.py` 得到`paddle`模型文件`.yml, .pdiparams, .pdmodel`。

* 下载BlazeFace模型参数文件

| 网络结构 | 输入尺寸 | 图片个数/GPU | 学习率策略 | Easy/Medium/Hard Set  | 预测时延（SD855）| 模型大小(MB) | 下载 | 配置文件 |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|:----------:|:---------:|:--------:|
| BlazeFace  | 640  |    8    | 1000e     | 0.885 / 0.855 / 0.731 | - | 0.472 |[下载链接](https://paddledet.bj.bcebos.com/models/blazeface_1000e.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection/blazeface_1000e.yml) |
| BlazeFace-FPN-SSH  | 640  |    8    | 1000e     | 0.907 / 0.883 / 0.793 | - | 0.479 |[下载链接](https://paddledet.bj.bcebos.com/models/blazeface_fpn_ssh_1000e.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection/blazeface_fpn_ssh_1000e.yml) |

* 导出paddle格式文件
  ```bash
  python tools/export_model.py -c configs/face_detection/blazeface_1000e.yml -o weights=blazeface_1000e.pdparams --export_serving_model=True
  ```

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection) 编写
