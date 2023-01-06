English | [简体中文](README_CN.md)
# PP-OCRv3 Wechat Mini-program Deployment Example

This document introduces the deployment of PP-OCRv3 model from PaddleOCR in Wechat mini-program, and the js interface in the @paddle-js-models/ocr npm package.


## Deploy PP-OCRv3 models in Wechat Mini-program

For the deployment of PP-OCRv3 models in Wechat mini-program, refer to [**reference document**](../../../../application/js/mini_program)


## PP-OCRv3 js interface

```
import * as ocr from "@paddle-js-models/ocr";
await ocr.init(detConfig, recConfig);
const res = await ocr.recognize(img, option, postConfig);
```
ocr model loading and initialization, where the model is in Paddle.js model format. For the conversion of js models, refer to [document](../../../../application/js/web_demo/README.md)

**init function parameter**

> * **detConfig**(dict): The configuration parameter for text detection model. Default {modelPath: 'https://js-models.bj.bcebos.com/PaddleOCR/PP-OCRv3/ch_PP-OCRv3_det_infer_js_960/model.json', fill: '#fff', mean: [0.485, 0.456, 0.406],std: [0.229, 0.224, 0.225]}; Among them, modelPath  is the path of the text detection model; fill is the padding value in the image pre-processing; mean and std are the mean and standard deviation in the pre-processing
> * **recConfig**(dict)): The configuration parameter for text recognition model. Default {modelPath: 'https://js-models.bj.bcebos.com/PaddleOCR/PP-OCRv3/ch_PP-OCRv3_rec_infer_js/model.json', fill: '#000', mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}; Among them, modelPath is the path of the text detection model, fill is the padding value in the image pre-processing, and mean/std are the mean and standard deviation in the pre-processing


**recognize function parameter**

> * **img**(HTMLImageElement): Enter an image parameter in HTMLImageElement. 
> * **option**(dict): The canvas parameter of the visual text detection box. No need to set.
> * **postConfig**(dict): Text detection post-processing parameter. Default: {shape: 960, thresh: 0.3, box_thresh: 0.6, unclip_ratio:1.5}; thresh is the binarization threshold of the output prediction image; box_thresh is the threshold of the output box, below which the prediction box will be discarded; unclip_ratio is the expansion ratio of the output box.


## Other Documents

- [PP-OCR Model Description](../../)
- [PP-OCRv3 C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
- [Web demo document of PP-OCRv3 models](../../../../application/js/web_demo/README.md)
