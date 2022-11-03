English | [简体中文](README.md)

# Front-end AI application

The development of artificial intelligence technology has led to industrial upgrading in the fields of computer vision(CV) and natural language processing(NLP). In addition, the deployment of AI models in browsers to achieve front-end intelligence has already provided good basic conditions with the steady increase in computing power on PCs and mobile devices, iterative updates of model compression technologies, and the continuous emergence of various innovative needs.
In response to the difficulty of deploying AI deep learning models on the front-end, Baidu has open-sourced the Paddle.js front-end deep learning model deployment framework, which can easily deploy deep learning models into front-end projects.

# Introduction of Paddle.js

[Paddle.js](https://github.com/PaddlePaddle/Paddle.js) is a web sub-project of Baidu `PaddlePaddle`, an open source deep learning framework running in the browser. `Paddle.js` can load the deep learning model trained by `PaddlePaddle`, and convert it into a browser-friendly model through the model conversion tool `paddlejs-converter` of `Paddle.js`, which is easy to use for online reasoning and prediction. `Paddle.js` supports running in browsers of `WebGL/WebGPU/WebAssembly`, and can also run in the environment of Baidu applet and WeChat applet.

Finally, we can launch AI functions in front-end application scenarios such as browsers and mini-program using `Paddle.js`, including but not limited to AI capabilities such as object detection, image segmentation, OCR, and item classification.

## Web Demo

Refer to this [document](./WebDemo_en.md) for steps to run computer vision demo in the browser.

|demo|web demo directory|visualization|
|-|-|-|
|object detection|[ScrewDetection、FaceDetection](./web_demo/src/pages/cv/detection/)| <img src="https://user-images.githubusercontent.com/26592129/196874536-b7fa2c0a-d71f-4271-8c40-f9088bfad3c9.png" height="200px">|
|human segmentation|[HumanSeg](./web_demo/src/pages/cv/segmentation/HumanSeg)|<img src="https://user-images.githubusercontent.com/26592129/196874452-4ef2e770-fbb3-4a35-954b-f871716d6669.png" height="200px">|
|classification|[GestureRecognition、ItemIdentification](./web_demo/src/pages/cv/recognition/)|<img src="https://user-images.githubusercontent.com/26592129/196874416-454e6bb0-4ebd-4b51-a88a-8c40614290ae.png" height="200px">|
|OCR|[TextDetection、TextRecognition](./web_demo/src/pages/cv/ocr/)|<img src="https://user-images.githubusercontent.com/26592129/196874354-1b5eecb0-f273-403c-aa6c-4463bf6d78db.png" height="200px">|


## Wechat Mini-program

Run the official demo reference in the WeChat mini-program [document](./mini_program/README.md)

|Name|Directory|
|-|-|
|OCR Text Detection| [ocrdetecXcx](./mini_program/ocrdetectXcx/) |
|OCR Text Recognition| [ocrXcx](./mini_program/ocrXcx/) |
|object detection| coming soon |
|Image segmentation | coming soon |
|Item Category| coming soon |

## Contributor

Thanks to Paddle Paddle Developer Expert (PPDE) Chen Qianhe (github: [chenqianhe](https://github.com/chenqianhe)) for the Web demo, mini-program.