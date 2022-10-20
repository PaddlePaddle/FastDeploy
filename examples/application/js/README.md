
# 前端AI应用

人工智能技术的快速发展带动了计算机视觉、自然语言处理领域的产业升级。另外，随着PC和移动设备上算力的稳步增强、模型压缩技术迭代更新以及各种创新需求的不断催生，在浏览器中部署AI模型实现前端智能已经具备了良好的基础条件。
针对前端部署AI深度学习模型困难的问题，百度开源了Paddle.js前端深度学习模型部署框架，可以很容易的将深度学习模型部署到前端项目中。

## Paddle.js简介

[Paddle.js](https://github.com/PaddlePaddle/Paddle.js)是百度`PaddlePaddle`的web方向子项目，是一个运行在浏览器中的开源深度学习框架。`Paddle.js`可以加载`PaddlePaddle`动转静的模型，经过`Paddle.js`的模型转换工具`paddlejs-converter`转换成浏览器友好的模型，易于在线推理预测使用。`Paddle.js`支持`WebGL/WebGPU/WebAssembly`的浏览器中运行，也可以在百度小程序和微信小程序环境下运行。

简言之，利用Paddle.js，我们可以在浏览器、小程序等前端应用场景上线AI功能，包括但不限于目标检测，图像分割，OCR，物品分类等AI能力。

## Paddle.js Web Demo使用

在浏览器中直接运行官方demo参考[文档](./web_demo/README.md)

|demo名称|web demo目录|
|-|-|
|人脸检测|[FaceDetection](./web_demo/demo/src/pages/cv/detection/FaceDetection/)| 
|螺丝钉检测|[ScrewDetection](./web_demo//demo/src/pages/cv/detection/ScrewDetection)| 
|人像分割背景替换|[HumanSeg](./web_demo//demo/src/pages/cv/segmentation/HumanSeg)|
|手势识别AI猜丁壳|[GestureRecognition](./web_demo//demo/src/pages/cv/recognition/GestureRecognition)|
|1000种物品识别|[ItemIdentification](./web_demo//demo/src/pages/cv/recognition/ItemIdentification)|
|文本检测|[TextDetection](./web_demo//demo/src/pages/cv/ocr/TextDetection)|
|文本识别|[TextRecognition](./web_demo//demo/src/pages/cv/ocr/TextRecognition)|


## Paddle.js 小程序Demo使用

在微信小程序运行官方demo参考[文档](./mini_program/README.md)

|名称|目录|
|-|-|
|OCR文本检测| [ocrdetecXcx](./mini_program/ocrdetectXcx/) |
|OCR文本识别| [ocrXcx](./mini_program/ocrXcx/) |
|目标检测| coming soon |
|图像分割| coming soon | 
|物品分类| coming soon | 
