[English](../../../en/quick_start/models/python.md) | 中文

# PPYOLOE Python部署

确认开发环境已安装FastDeploy，参考[FastDeploy安装](../../build_and_install/)安装预编译的FastDeploy，或根据自己需求进行编译安装。

本文档以PaddleDetection目标检测模型PPYOLOE为例展示CPU上的推理示例

## 1. 获取模型和测试图像

``` python
import fastdeploy as fd

model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz"
image_url = "https://bj.bcebos.com/fastdeploy/tests/test_det.jpg"
fd.download_and_decompress(model_url, path=".")
fd.download(image_url, path=".")
```

## 2. 加载模型

- 更多模型的示例可参考[FastDeploy/examples](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples)

``` python
model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel"
params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams"
infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml"

# 模型推理的配置信息
option = fd.RuntimeOption()
model = fd.vision.detection.PPYOLOE(model_file, params_file, infer_cfg_file, option)
```
加载模型完后，会输出提示如下，说明模型初始化的后端，以及运行的硬件设备
```
[INFO] fastdeploy/fastdeploy_runtime.cc(283)::Init	Runtime initialized with Backend::OPENVINO in device Device::CPU.
```

## 3. 预测图片检测结果

``` python
import cv2
im = cv2.imread("test_det.jpg")

result = model.predict(im)
print(result)
```
预测完，输出预测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
415.047180,89.311569, 506.009613, 283.863098, 0.950423, 0
163.665710,81.914932, 198.585342, 166.760895, 0.896433, 0
581.788635,113.027618, 612.623474, 198.521713, 0.842596, 0
267.217224,89.777306, 298.796051, 169.361526, 0.837951, 0
104.465584,45.482422, 127.688850, 93.533867, 0.773348, 0
...
...
```

## 4. 可视化图片预测结果

``` python
vis_im = fd.vision.visualize.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```
可视化执行完，打开`vis_image.jpg`可视化效果如下
<div  align="center">
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

## 其它文档

- [切换模型推理的硬件和后端](../../faq/how_to_change_backend.md)
