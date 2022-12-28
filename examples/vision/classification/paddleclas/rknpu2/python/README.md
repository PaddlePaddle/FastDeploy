# PPClas Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/rknpu2.md)

本目录下提供`infer.py`快速完成 ResNet50_vd 在RKNPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/classification/paddleclas/rknpu2/python

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 推理
python3 infer.py --model_file ./ResNet50_vd_infer/ResNet50_vd_infer_rk3588.rknn  --config_file ResNet50_vd_infer/inference_cls.yaml  --image ILSVRC2012_val_00000010.jpeg

# 运行完成后返回结果如下所示
ClassifyResult(
label_ids: 153, 
scores: 0.684570, 
)
```


## 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，
DisablePermute(C++)或`disable_permute(Python)，在预处理阶段禁用数据格式的转换。

## 其它文档
- [ResNet50_vd C++部署](../cpp)
- [模型预测结果说明](../../../../../../docs/api/vision_results/)
- [转换ResNet50_vd RKNN模型文档](../README.md)