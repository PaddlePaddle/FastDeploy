# PaddleSeg Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/sophgo.md)

本目录下提供`infer.py`快速完成 pp_liteseg 在SOPHGO TPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/sophgo/python

# 下载图片
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 推理
python3 infer.py --model_file ./bmodel/pp_liteseg_1684x_f32.bmodel --config_file ./bmodel/deploy.yaml --image cityscapes_demo.png

# 运行完成后返回结果如下所示
运行结果保存在sophgo_img.png中
```

## 其它文档
- [pp_liteseg C++部署](../cpp)
- [转换 pp_liteseg SOPHGO模型文档](../README.md)
