[中文版](./README_cn.md)

# OCR

OCR is a text recognition module, which includes two models: ocr_detection and ocr_recognition。 ocr_detection model detects the region of the text in the picture, ocr_recognition model can recognize the characters (Chinese / English / numbers) in each text area.

<img src="https://img.shields.io/npm/v/@paddle-js-models/ocr?color=success" alt="version"> <img src="https://img.shields.io/bundlephobia/min/@paddle-js-models/ocr" alt="size"> <img src="https://img.shields.io/npm/dm/@paddle-js-models/ocr?color=orange" alt="downloads"> <img src="https://img.shields.io/npm/dt/@paddle-js-models/ocr" alt="downloads">

The module provides a simple and easy-to-use interface. Users only need to upload pictures to obtain text recognition results.

The ocr_detection and ocr_recognition models are compressed from [PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/PP-OCRv3_introduction_en.md) which released by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), which greatly improves the running speed on js at a small loss of accuracy.

The input shape of the ocr_recognition model is [1, 3, 48, 320], and the selected area of the picture text box will be processed before the model reasoning: the width height ratio of the selected area of the picture text box is < = 10, and the whole selected area will be transferred into the recognition model; If the width height ratio of the frame selected area is > 10, the frame selected area will be cropped according to the width, the cropped area will be introduced into the recognition model, and finally the recognition results of each part of the cropped area will be spliced.


# Usage

```js
import * as ocr from '@paddle-js-models/ocr';
// Model initialization
await ocr.init();
// Get the text recognition result API, img is the user's upload picture, and option is an optional parameter
// option.canvas as HTMLElementCanvas：if the user needs to draw the selected area of the text box, pass in the canvas element
// option.style as object：if the user needs to configure the canvas style, pass in the style object
// option.style.strokeStyle as string：select a color for the text box
// option.style.lineWidth as number：width of selected line segment in text box
// option.style.fillStyle as string：select the fill color for the text box
const res = await ocr.recognize(img, option?);
// character recognition results
console.log(res.text);
// text area points
console.log(res.points);
```

# Performance
<img alt="ocr" src="https://user-images.githubusercontent.com/43414102/156380942-2ee5ad8d-d023-4cd3-872c-b18ebdcbb3f3.gif">