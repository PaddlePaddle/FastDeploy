# Python部署

确认开发环境已安装FastDeploy，参考[FastDeploy安装](../../build_and_install/)安装预编译的FastDeploy，或根据自己需求进行编译安装。

本文档以PaddleDetection目标检测模型PPYOLOE为例展示CPU上的推理示例

## 1. 获取模型和测试图像

``` python
import fastdeploy as fd

model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz"
image_url - "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
fd.download_and_decompress(model_url, path=".")
fd.download(image_url, path=".")
```

## 2. 加载模型

- 更多模型的示例可参考[FastDeploy/examples](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples)
- 模型API说明见API文档[FastDeploy模型Python API文档](../../apis/models/python/)

``` python
model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel"
params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams"
infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml"
model = fd.vision.detection.PPYOLOE(model_file, params_file, infer_cfg_file)
```

## 3. 预测图片检测结果

``` python
import cv2
im = cv2.imread("000000014439.jpg")

result = model.predict(im)
print(result)
```

## 4. 可视化图片预测结果

``` python
vis_im = fd.vision.visualize.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```
