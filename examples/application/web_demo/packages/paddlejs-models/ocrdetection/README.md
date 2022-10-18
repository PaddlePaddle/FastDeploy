[中文版](./README_cn.md)

# ocr_detect

ocr_detect model is used to detect the text area in the image.

<img src="https://img.shields.io/npm/v/@paddle-js-models/ocrdet?color=success" alt="version"> <img src="https://img.shields.io/bundlephobia/min/@paddle-js-models/ocrdet" alt="size"> <img src="https://img.shields.io/npm/dm/@paddle-js-models/ocrdet?color=orange" alt="downloads"> <img src="https://img.shields.io/npm/dt/@paddle-js-models/ocrdet" alt="downloads">

# Usage

```js
import * as ocr from '@paddle-js-models/ocrdet';
// Load ocr_detect model
await ocr.load();
// Get text area points
const res = await ocr.detect(img);
```

# Online experience

https://paddlejs.baidu.com/ocrdet

# Performance
<img alt="image" src="https://user-images.githubusercontent.com/43414102/156394295-5650b6c5-65c4-42a7-bccc-3ed459577b9d.png">
