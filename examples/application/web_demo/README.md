
# 简介

本项目基于 Paddle.js 实现在浏览器中实现目标检测，人像分割，OCR，物品分类等计算机视觉任务。Paddle.js 是百度 PaddlePaddle 的一个 web 项目，它是一个运行在浏览器中的开源深度学习框架，所有支持 WebGL/WebGPU/WebAssembly 的浏览器中都可以使用 Paddle.js 部署模型。


|Demo名称|WebDemo目录|源码目录|npm包|
|-|-|-|-|
|人脸检测|./src/pages/cv/detection/FaceDetection| ./packages/paddlejs-models/detect|[@paddle-js-models/detect](https://www.npmjs.com/package/@paddle-js-models/detect)|
|螺丝检测|src/pages/cv/detection/ScrewDetection| ./packages/paddlejs-models/facedetect|[@paddle-js-models/facedetect](https://www.npmjs.com/package/@paddle-js-models/facedetect)|
|人像分割背景替换|./src/pages/cv/segmentation/HumanSeg|./packages/paddlejs-models//humanseg|[@paddle-js-models/humanse](https://www.npmjs.com/package/@paddle-js-models/humanseg)|
|手势识别AI猜丁壳|./src/pages/cv/recognition/GestureRecognition|./packages/paddlejs-models/gesture|[@paddle-js-models/gesture](https://www.npmjs.com/package/@paddle-js-models/gesture)|
|1000种物品识别|./src/pages/cv/recognition/ItemIdentification|./packages/paddlejs-models/mobilenet|[@paddle-js-models/mobilenet](https://www.npmjs.com/package/@paddle-js-models/mobilenet)|
|文本检测|./src/pages/cv/ocr/TextDetection|./packages/paddlejs-models/ocrdetection|[@paddle-js-models/ocrdet](https://www.npmjs.com/package/@paddle-js-models/ocrdet)|
|文本识别|./src/pages/cv/ocr/TextRecognition|./packages/paddlejs-models/ocr|[@paddle-js-models/ocr](https://www.npmjs.com/package/@paddle-js-models/ocr)|



# 1. 快速开始

本节介绍如何在浏览器中运行计算机视觉任务。

**安装Node.js**

从 Node.js官网https://nodejs.org/en/download/ 下载适合自己平台的 Node.js 安装包并安装。

在 ./web_mode/demo 目录下执行如下指令：

```
npm install
npm run dev
```

在浏览器中打开网址 http://localhost:5173/main/index.html 即可快速体验在浏览器中运行计算机视觉任务。

![02f81ab34d6007b54daef9a451240a5c](https://user-images.githubusercontent.com/26592129/196321732-1f089e4a-d053-4d9a-9685-e2eb467e51fb.png)


# 2. npm包使用方式


本节介绍npm包的使用方式，每个demo均提供简单易用的接口，用户只需初始化上传图片即可获得结果，使用步骤如下：
1. 调用模块
2. 初始化模型
3. 传入输入，执行预测

以 OCR 为例，在前端项目中，ocr包的使用方式如下：

```
// 1. 调用ocr模块
import * as ocr from '@paddle-js-models/ocr';

// 2. 初始化ocr模型
await ocr.init();

// 3. 传入HTMLImageElement类型的图像作为输入并获得结果
const res = await ocr.recognize(img);

// 打印OCR模型得到的文本坐标以及文本内容
console.log(res.text);
console.log(res.points);
```

**更换模型**
ocr.init()函数中，包含默认初始化的模型链接，如果要替换模型参考下述步骤。

步骤1：将模型转成js格式：
```
# 安装paddlejsconverter
pip3 install paddlejsconverter
# 转换模型格式，输入模型为inference模型
paddlejsconverter --modelPath=./inference.pdmodel --paramPath=./inference.pdiparams --outputDir=./  --useGPUOpt=True
# 注意：useGPUOpt 选项默认不开启，如果模型用在 gpu backend（webgl/webgpu），则开启 useGPUOpt，如果模型运行在（wasm/plain js）则不要开启。
```

导出成功后，本地目录下会出现 model.json chunk_1.dat文件，分别是js模型的网络结构和模型参数二进制文件。

步骤2：将导出的js模型上传到支持跨域访问的服务器，服务器的CORS配置参考下图：
![image](https://user-images.githubusercontent.com/26592129/196356384-6bd25caf-6cc0-4509-92af-0370149825d8.png)


步骤3：修改代码替换默认的模型。以OCR demo为例，修改OCR web demo中[模型初始化代码](https://github.com/DataVizU/Paddle.js/blob/4b2796c15bcc22f5a99a52fd9a2d9bbf667ee73d/demo/src/pages/cv/ocr/TextRecognition/TextRecognition.vue#L64)，即
```
await ocr.init();
修改为：
await ocr.init("https://js-models.bj.bcebos.com/PaddleOCR/PP-OCRv3/ch_PP-OCRv3_det_infer_js_960/model.json");   # 第一个参数传入新的文本检测模型链接
```

重新在demo目录下执行即可体验新的模型效果。
```
npm run dev
```

注：
1. OCR文本识别demo模型部署的源代码链接：
https://github.com/DataVizU/Paddle.js/blob/main/packages/paddlejs-models/ocr/src/index.ts
2. ocr.init()函数有两个参数，分别为检测模型参数的替换和识别模型参数，参考[链接](https://github.com/DataVizU/Paddle.js/blob/4b2796c15bcc22f5a99a52fd9a2d9bbf667ee73d/packages/paddlejs-models/ocr/src/index.ts#L52)


**自定义前后处理参数**

以OCR文本检测 demo为例，修改文本检测后处理的参数实现扩大文本检测框的效果，修改OCR web demo中执行[模型预测代码](https://github.com/DataVizU/Paddle.js/blob/4b2796c15bcc22f5a99a52fd9a2d9bbf667ee73d/demo/src/pages/cv/ocr/TextRecognition/TextRecognition.vue#L99)，即：

```
const res = await ocr.recognize(img, { canvas: canvas.value });
修改为：
// 定义超参数，将unclip_ratio参数从1.5 增大为3.5
const detConfig = {shape: 960, thresh: 0.3, box_thresh: 0.6, unclip_ratio:3.5};
const res = await ocr.recognize(img, { canvas: canvas.value }, detConfig);
```

注：OCR文本检测demo模型部署的源代码链接：
https://github.com/DataVizU/Paddle.js/blob/main/packages/paddlejs-models/ocrdetection/src/index.ts

