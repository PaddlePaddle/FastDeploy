[English](README.md) | 简体中文
# RKYOLO Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/rknpu2.md)

本目录下提供`infer.py`快速完成Picodet在RKNPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/rkyolo/python

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# copy model
cp -r ./model /path/to/FastDeploy/examples/vision/detection/rkyolo/python

# 推理
python3 infer.py --model_file /path/to/model  --image /path/to/000000014439.jpg
```

## 常见问题

如果你使用自己训练的YOLOv5模型，你可能会碰到运行FastDeploy的demo后出现`segmentation fault`的问题，很大概率是label数目不一致，你可以使用以下方案来解决:

```python
model.postprocessor.class_num = 3
```

## 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，需要先调用DisablePermute(C++) `disable_permute(Python)`，在预处理阶段禁用归一化以及数据格式的转换。

## 其它文档

- [PaddleDetection 模型介绍](..)
- [PaddleDetection C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [转换PaddleDetection RKNN模型文档](../README.md)
