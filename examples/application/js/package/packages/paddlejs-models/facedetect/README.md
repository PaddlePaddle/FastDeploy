[中文版](./README_cn.md)

# Facedetect

Facedetect is used for face detection in image. It provides a simple interface. At the same time, you can use your own model.

<img src="https://img.shields.io/npm/v/@paddle-js-models/facedetect?color=success" alt="version"> <img src="https://img.shields.io/bundlephobia/min/@paddle-js-models/facedetect" alt="size"> <img src="https://img.shields.io/npm/dm/@paddle-js-models/facedetect?color=orange" alt="downloads"> <img src="https://img.shields.io/npm/dt/@paddle-js-models/facedetect" alt="downloads">

# Usage

```js
import { FaceDetector } from '@paddle-js-models/facedetect';

const faceDetector = new FaceDetector();
await faceDetector.init();
// Required parameter：imgEle(HTMLImageElement)
// Optional parameter: shrink, threshold
// Result is face area information. It includes left, top, width, height, confidence
const res = await faceDetector.detect(
    imgEle,
    { shrink: 0.4, threshold: 0.6 }
);
```

# Performance
+ **multi small-sized face**  
  <img width="500"  src="https://mms-voice-fe.cdn.bcebos.com/pdmodel/face/detection/pic/small.png"/>

+ **single big-sized face**  
  <img width="500"  src="https://mms-voice-fe.cdn.bcebos.com/pdmodel/face/detection/pic/big.png"/>

# Postprocess
This model has a better recognition effect for small-sized faces, and the image will be shrink before prediction, so it is necessary to transform the prediction output data.  
<img width="500"  src="https://mms-voice-fe.cdn.bcebos.com/pdmodel/face/detection/pic/example.png"/>  
The red line indicates the predicted output result, and the green line indicates the converted result. dx dy fw fh are known parameters.

# Reference
[original model link](https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.2/modules/image/face_detection/pyramidbox_lite_mobile)
