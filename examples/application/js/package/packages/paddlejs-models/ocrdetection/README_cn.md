[English](./README.md)

# ocr_detection

ocr_detection模型用于检测图像中文字区域。

<img src="https://img.shields.io/npm/v/@paddle-js-models/ocrdet?color=success" alt="version"> <img src="https://img.shields.io/bundlephobia/min/@paddle-js-models/ocrdet" alt="size"> <img src="https://img.shields.io/npm/dm/@paddle-js-models/ocrdet?color=orange" alt="downloads"> <img src="https://img.shields.io/npm/dt/@paddle-js-models/ocrdet" alt="downloads">


ocr_detection模型是[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)发布[PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/PP-OCRv3_introduction.md)模型的压缩版本，压缩后的模型仅0.47M，在损失一小部分精度的情况下，大幅提升在js上的运行速度。

# 使用

```js
import * as ocr from '@paddle-js-models/ocrdet';
// ocr_detect模型加载
await ocr.load();
// 获取文字区域坐标
const res = await ocr.detect(img);
```

# 效果
<img alt="image" src="https://user-images.githubusercontent.com/43414102/156394295-5650b6c5-65c4-42a7-bccc-3ed459577b9d.png">

