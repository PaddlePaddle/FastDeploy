# ⚡️FastDeploy

</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/FastDeploy?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af"></a>
    <a href="https://pypi.org/project/FastDeploy-python/"><img src="https://img.shields.io/pypi/dm/FastDeploy-python?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf"></a>
</p>

**⚡️FastDeploy**是一款**简单易用**的推理部署工具箱。覆盖业界主流**优质预训练模型**并提供**开箱即用**的开发体验，包括图像分类、目标检测、图像分割、人脸检测、人体关键点识别、文字识别等多任务，满足开发者**多场景**，**多硬件**、**多平台**的快速部署需求。

## 支持模型

| 任务场景 | 模型                                                         | X64 CPU | Nvidia-GPU | Nvidia-GPU TensorRT |
| -------- | ------------------------------------------------------------ | ------- | ---------- | ------------------- |
| 图像分类 | [PaddleClas/ResNet50](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/PPLCNet](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/GhostNet](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas) | √       | √          | √                   |
|          | [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas) | √       | √          | √                   |
| 目标检测 | [PaddleDetection/PPYOLOE](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection) | √       | √          | √                   |
|          | [PaddleDetection/PPYOLO](./examples/vision/detection/paddledetection) | √       | √          | -                   |
|          | [PaddleDetection/PPYOLOv2](./examples/vision/detection/paddledetection) | √       | √          | -                   |
|          | [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | √       | √          | -                   |

#### 快速开始

#### 安装FastDeploy Python

用户根据开发环境选择安装版本，更多安装环境参考[安装文档](docs/quick_start/install.md).

```
pip install https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl
```

准备目标检测模型和测试图片
```
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

加载模型预测
```
import fastdeploy.vision as vis
import cv2

model = vis.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                              "ppyoloe_crn_l_300e_coco/model.pdiparams",
                              "ppyoloe_crn_l_300e_coco/infer_cfg.yml")

im = cv2.imread("000000014439.jpg")
result = model.predict(im.copy())
print(result)

vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```

预测完成，可视化结果保存至`vis_image.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
415.047363,89.311523, 506.009613, 283.863129, 0.950423, 0
163.665710,81.914894, 198.585342, 166.760880, 0.896433, 0
581.788635,113.027596, 612.623474, 198.521713, 0.842597, 0
267.217224,89.777321, 298.796051, 169.361496, 0.837951, 0
104.465599,45.482410, 127.688835, 93.533875, 0.773348, 0
...
```

## 更多部署示例

FastDeploy提供了大量部署示例供开发者参考，支持模型在CPU、GPU以及TensorRT的部署

- [PaddleDetection模型部署](examples/vision/detection/paddledetection)
- [PaddleClas模型部署](examples/vision/classification/paddleclas)
- [PaddleSeg模型部署](examples/vision/segmentation/paddleseg)
- [YOLOv7部署](examples/vision/detection/yolov7)
- [YOLOv6部署](examples/vision/detection/yolov6)
- [YOLOv5部署](examples/vision/detection/yolov5)
- [人脸检测模型部署](examples/vision/facedet)
- [更多视觉模型部署示例...](examples/vision)

## Acknowledge

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，再次表示感谢。

## License

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
