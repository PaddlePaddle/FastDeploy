# Paddle.js Web Demo

- [简介](#0)
- [1. 快速开始](#1)
- [2. npm包调用](#2)
- [3. 模型替换](#3)
- [4. 自定义前后处理参数](#4)
- [5. 其他](#5)

<a name="0"></a>
## 简介

本项目基于[Paddle.js](https://github.com/PaddlePaddle/Paddle.js)在浏览器中实现目标检测，人像分割，OCR，物品分类等计算机视觉任务。


|demo名称|web demo组件|源码目录|npm包|
|-|-|-|-|
|人脸检测|[FaceDetection](./demo/src/pages/cv/detection/FaceDetection/)| [facedetect](./packages/paddlejs-models/facedetect)|[@paddle-js-models/facedetect](https://www.npmjs.com/package/@paddle-js-models/facedetect)|
|螺丝钉检测|[ScrewDetection](./demo/src/pages/cv/detection/ScrewDetection)| [detect](./packages/paddlejs-models/detect)|[@paddle-js-models/detect](https://www.npmjs.com/package/@paddle-js-models/detect)|
|人像分割背景替换|[HumanSeg](./demo/src/pages/cv/segmentation/HumanSeg)|[humanseg](./packages/paddlejs-models/humanseg)|[@paddle-js-models/humanseg](https://www.npmjs.com/package/@paddle-js-models/humanseg)|
|手势识别AI猜丁壳|[GestureRecognition](./demo/src/pages/cv/recognition/GestureRecognition)|[gesture](./packages/paddlejs-models/gesture)|[@paddle-js-models/gesture](https://www.npmjs.com/package/@paddle-js-models/gesture)|
|1000种物品识别|[ItemIdentification](./demo/src/pages/cv/recognition/ItemIdentification)|[mobilenet](./packages/paddlejs-models/mobilenet)|[@paddle-js-models/mobilenet](https://www.npmjs.com/package/@paddle-js-models/mobilenet)|
|文本检测|[TextDetection](./demo/src/pages/cv/ocr/TextDetection)|[ocrdetection](./packages/paddlejs-models/ocrdetection)|[@paddle-js-models/ocrdet](https://www.npmjs.com/package/@paddle-js-models/ocrdet)|
|文本识别|[TextRecognition](./demo/src/pages/cv/ocr/TextRecognition)|[ocr](./packages/paddlejs-models/ocr)|[@paddle-js-models/ocr](https://www.npmjs.com/package/@paddle-js-models/ocr)|


<a name="1"></a>
## 1. 快速开始

本节介绍如何在浏览器中直接运行官方demo。

**安装Node.js**

从`Node.js`官网https://nodejs.org/en/download/ 下载适合自己平台的`Node.js`安装包并安装。

**安装demo依赖并运行**
在`./web_mode/demo`目录下执行如下指令：

```
# 安装依赖
npm install
# 启动demo
npm run dev
```

在浏览器中打开网址 `http://localhost:5173/main/index.html` 即可快速体验在浏览器中运行计算机视觉任务。

![22416f4a3e7d63f950b838be3cd11e80](https://user-images.githubusercontent.com/26592129/196685868-93ab53bd-cb2e-44ff-a56b-50c1781b8679.jpg)

<a name="2"></a>
## 2. npm包调用

本节介绍npm包的使用方式，每个demo均提供简单易用的接口，用户只需初始化上传图片即可获得结果，使用步骤如下：
1. 调用模块
2. 初始化模型
3. 传入输入，执行预测

以 OCR 为例，在前端项目中，`@paddle-js-models/ocr`包的使用方式如下：

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

<a name="3"></a>
## 3. 模型替换

由于前端环境和计算资源限制，在前端部署深度学习模型时，我们对模型的性能有着更严格的要求，简单来说，模型需要足够轻量化。理论上模型的输入shape越小、模型大小越小，则对应的模型的flops越小，在前端运行也能更流畅。经验总结，使用`Paddle.js`部署的模型存储尽量不超过*5M*，实际情况根据硬件和计算资源情况决定。

在实际应用中，常常根据垂类的场景定制化模型，官方的demo支持修改传入参数替换模型。

以OCR demo为例，[ocr.init()函数](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/web_demo/packages/paddlejs-models/ocr/src/index.ts#L52)中，包含默认初始化的模型链接，如果要替换模型参考下述步骤。

步骤1：将模型转成js格式：
```
# 安装paddlejsconverter
pip3 install paddlejsconverter
# 转换模型格式，输入模型为inference模型
paddlejsconverter --modelPath=./inference.pdmodel --paramPath=./inference.pdiparams --outputDir=./  --useGPUOpt=True
# 注意：useGPUOpt 选项默认不开启，如果模型用在 gpu backend（webgl/webgpu），则开启 useGPUOpt，如果模型运行在（wasm/plain js）则不要开启。
```

导出成功后，本地目录下会出现 `model.json chunk_1.dat`等文件，分别是对应js模型的网络结构、模型参数二进制文件。

步骤2：将导出的js模型上传到支持跨域访问的服务器，服务器的CORS配置参考下图：
![image](https://user-images.githubusercontent.com/26592129/196612669-5233137a-969c-49eb-b8c7-71bef5088686.png)


步骤3：修改代码替换默认的模型。以OCR demo为例，修改OCR web demo中[模型初始化代码](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/web_demo/demo/src/pages/cv/ocr/TextRecognition/TextRecognition.vue#L64)，即

```
await ocr.init();
修改为：
await ocr.init({modelPath: "https://js-models.bj.bcebos.com/PaddleOCR/PP-OCRv3/ch_PP-OCRv3_det_infer_js_960/model.json"});   # 第一个参数传入新的文本检测字典类型参数
```

重新在demo目录下执行下述命令，即可体验新的模型效果。
```
npm run dev
```

<a name="4"></a>
## 4. 自定义前后处理参数

**自定义前处理参数**

在不同计算机视觉任务中，不同的模型可能有不同的预处理参数，比如mean,std,keep_ratio等参数，替换模型后也需要对预处理参数进行修改。paddle.js发布的npm包中提供了自定义预处理参数的简单方案。只需要在调用模型初始化函数时，传入自定义的参数即可。

```
# 默认参数初始化
await model.init();

自定义参数初始化
const Config = {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], keepratio: false};
await model.init(Config);
```

以OCR文本检测demo为例，修改模型前处理的mean和std参数，只需要在模型初始化时传入自定义的mean和std参数。
```
await ocr.init();
修改为：
const detConfig = {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]};
await ocr.init(detConfig);  # 第一个参数传入新的文本检测模型链接
```

**自定义后处理参数**

同理，paddle.js发布的npm包也提供了后处理参数的自定义方案。

```
# 默认参数运行
await model.predict();

# 自定义后处理参数
const postConfig = {thresh: 0.5};
await model.predict(Config);
```

以OCR文本检测 demo为例，修改文本检测后处理的参数实现扩大文本检测框的效果，修改OCR web demo中执行[模型预测代码](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/web_demo/demo/src/pages/cv/ocr/TextRecognition/TextRecognition.vue#L99)，即：

```
const res = await ocr.recognize(img, { canvas: canvas.value });
修改为：
// 定义超参数，将unclip_ratio参数从1.5 增大为3.5
const detConfig = {shape: 960, thresh: 0.3, box_thresh: 0.6, unclip_ratio:3.5};
const res = await ocr.recognize(img, { canvas: canvas.value }, detConfig);
```

注：不同的任务有不同的后处理参数，详细参数参考npm包中的API。

<a name="5"></a>
## 5. 其他

`Paddle.js`转换后的模型不仅支持浏览器中使用，也可以在百度小程序和微信小程序环境下运行。

|名称|目录|
|-|-|
|OCR文本检测| [ocrdetecXcx](../mini_program/ocrdetectXcx/) |
|OCR文本识别| [ocrXcx](../mini_program/ocrXcx/) |
|目标检测| coming soon |
|图像分割| coming soon | 
|物品分类| coming soon | 
