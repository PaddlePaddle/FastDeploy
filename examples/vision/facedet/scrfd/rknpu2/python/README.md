# Scrfd Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/rknpu2.md)


本目录下提供`infer.py`快速完成Scrfd在RKNPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/facedet/scrfd/rknpu2/python

# 下载图片
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# 推理
python3 infer.py --model_file ./scrfd_500m_bnkps_shape640x640_rk3588.rknn \
                 --image test_lite_face_detector_3.jpg
```


## 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，
需要先调用DisableNormalizePermute(C++)或`disable_normalize_permute(Python)，在预处理阶段禁用归一化以及数据格式的转换。
## 其它文档

- [Scrfd 模型介绍](..)
- [Scrfd C++部署](../cpp)
- [模型预测结果说明](../../../../../../docs/api/vision_results/)
- [转换Scrfd RKNN模型文档](../README.md)
