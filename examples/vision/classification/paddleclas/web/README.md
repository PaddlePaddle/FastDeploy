English | [简体中文](README_CN.md)
# MobileNet Front-end Deployment Example

This document introduces the deployment of PaddleClas's mobilenet models for image classification to run in the browser, and the js interface in the @paddle-js-models/mobilenet npm package.


## Front-end Deployment of Image Classification Model

To use the web demo of image classification models, refer to [**Reference Document**](../../../../application/js/web_demo/)


## MobileNet js Interface

```
import * as mobilenet from "@paddle-js-models/mobilenet";
# mobilenet model loading and initialization
await mobilenet.load()
# mobilenet model performs the prediction and obtains the classification result
const res = await mobilenet.classify(img);
console.log(res);
```

**load() function parameter**

> * **Config**(dict): The configuration parameter for the image classification model. Default {Path: 'https://paddlejs.bj.bcebos.com/models/fuse/mobilenet/mobileNetV2_fuse_activation/model.json', fill: '#fff', mean: [0.485, 0.456, 0.406],std: [0.229, 0.224, 0.225]}; Among them, modelPath is the path of the js model, fill is the padding value in the image pre-processing, and mean/std are the mean and standard deviation in the pre-processing

**classify() function parameter**
> * **img**(HTMLImageElement): Enter an image parameter in HTMLImageElement. 



## Other Documents

- [PaddleClas model python deployment](../../paddleclas/python/)
- [PaddleClas model C++ deployment](../cpp/)
