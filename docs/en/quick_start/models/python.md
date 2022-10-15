# Python Deployment

Make sure that FastDeploy is installed in the development environment. Refer to [FastDeploy Installation](../../build_and_install/) to install the pre-built FastDeploy, or build and install according to your own needs.

This document uses the PaddleDetection target detection model PPYOLOE as an example to show an inference example on the CPU.

## 1. Get the Model and Test Image

``` python
import fastdeploy as fd

model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz"
image_url - "https://bj.bcebos.com/fastdeploy/tests/test_det.jpg"
fd.download_and_decompress(model_url, path=".")
fd.download(image_url, path=".")
```

## 2. Load Model

- More examples of models can be found in[FastDeploy/examples](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples)

``` python
model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel"
params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams"
infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml"
model = fd.vision.detection.PPYOLOE(model_file, params_file, infer_cfg_file)
```

## 3. Get Prediction for Image Object Detection 

``` python
import cv2
im = cv2.imread("000000014439.jpg")

result = model.predict(im)
print(result)
```

## 4. Visualize image prediction results

``` python
vis_im = fd.vision.visualize.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```

After the visualization is executed, open `vis_image.jpg` and the visualization effect is as follows：

<div  align="center">
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

## other documents

- [Switch model inference hardware and backend](../../faq/how_to_change_backend.md)
