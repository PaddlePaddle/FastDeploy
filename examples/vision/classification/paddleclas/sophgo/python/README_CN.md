[English](README.md) | 简体中文
# PaddleClas Python部署示例

在部署前，需确认以下步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/sophgo.md)

本目录下提供`infer.py`快速完成 ResNet50_vd 在SOPHGO TPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/classification/paddleclas/sophgo/python

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 手动设置推理使用的模型、配置文件和图片路径
python3 infer.py --auto False --model_file ./bmodel/resnet50_1684x_f32.bmodel  --config_file ResNet50_vd_infer/inference_cls.yaml  --image ILSVRC2012_val_00000010.jpeg

# 自动完成下载数据-模型编译-推理，不需要设置模型、配置文件和图片路径
python3 infer.py --auto True --model '' --config_file '' --image ''


# 运行完成后返回结果如下所示
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)
```

## 其它文档
- [ResNet50_vd C++部署](../cpp)
- [转换ResNet50_vd SOPHGO模型文档](../README.md)
