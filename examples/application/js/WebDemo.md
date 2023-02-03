English | [简体中文](WebDemo_CN.md)

# Introduction to Web Demo

- [Introduction](#0)
- [1. Quick Start](#1)
- [2. npm package call](#2)
- [3. Model Replacement](#3)
- [4. custom hyperparameters](#4)
- [5. Other](#5)

<a name="0"></a>
## Introduction

Based on [Paddle.js](https://github.com/PaddlePaddle/Paddle.js), this project implements computer vision tasks such as target detection, portrait segmentation, OCR, and item classification in the browser.


|demo name|web demo component|source directory|npm package|
|-|-|-|-|
|Face Detection|[FaceDetection](./web_demo/src/pages/cv/detection/FaceDetection/)| [facedetect](./package/packages/paddlejs-models/facedetect)|[@paddle-js-models/ facedetect](https://www.npmjs.com/package/@paddle-js-models/facedetect)|
|Screw Detection|[ScrewDetection](./web_demo/src/pages/cv/detection/ScrewDetection)| [detect](./package/packages/paddlejs-models/detect)|[@paddle-js-models/detect](https://www.npmjs.com/package/@paddle-js-models/detect)|
|Portrait segmentation background replacement|[HumanSeg](./web_demo/src/pages/cv/segmentation/HumanSeg)|[humanseg](./package/packages/paddlejs-models/humanseg)|[@paddle-js-models/ humanseg](https://www.npmjs.com/package/@paddle-js-models/humanseg)|
|Gesture Recognition AI Guessing Shell|[GestureRecognition](./web_demo/src/pages/cv/recognition/GestureRecognition)|[gesture](./package/packages/paddlejs-models/gesture)|[@paddle-js- models/gesture](https://www.npmjs.com/package/@paddle-js-models/gesture)|
|1000 Item Identification|[ItemIdentification](./web_demo/src/pages/cv/recognition/ItemIdentification)|[mobilenet](./package/packages/paddlejs-models/mobilenet)|[@paddle-js-models/ mobilenet](https://www.npmjs.com/package/@paddle-js-models/mobilenet)|
|Text Detection|[TextDetection](./web_demo/src/pages/cv/ocr/TextDetection)|[ocrdetection](./package/packages/paddlejs-models/ocrdetection)|[@paddle-js-models/ocrdet](https://www.npmjs.com/package/@paddle-js-models/ocrdet)|
|Text Recognition|[TextRecognition](./web_demo/src/pages/cv/ocr/TextRecognition)|[ocr](./package/packages/paddlejs-models/ocr)|[@paddle-js-models/ocr](https://www.npmjs.com/package/@paddle-js-models/ocr)|


<a name="1"></a>
## 1. Quick Start

This section describes how to run the official demo directly in the browser.

**1. Install Node.js**

Download the `Node.js` installation package suitable for your platform from the `Node.js` official website https://nodejs.org/en/download/ and install it.

**2. Install demo dependencies and start**
Execute the following command in the `./web_demo` directory:

````
# install dependencies
npm install
# start demo
npm run dev
````

Open the URL `http://localhost:5173/main/index.html` in the browser to quickly experience running computer vision tasks in the browser.

![22416f4a3e7d63f950b838be3cd11e80](https://user-images.githubusercontent.com/26592129/196685868-93ab53bd-cb2e-44ff-a56b-50c1781b8679.jpg)


<a name="2"></a>
## 2. npm package call

This section introduces how to use npm packages. Each demo provides an easy-to-use interface. Users only need to initialize and upload images to get the results. The steps are as follows:
1. Call the module
2. Initialize the model
3. Pass in input, perform prediction

Taking OCR as an example, in a front-end project, the `@paddle-js-models/ocr` package is used as follows:

````
// 1. Call the ocr module
import * as ocr from '@paddle-js-models/ocr';

// 2. Initialize the ocr model
await ocr.init();

// 3. Pass in an image of type HTMLImageElement as input and get the result
const res = await ocr.recognize(img);

// Print the text coordinates and text content obtained by the OCR model
console.log(res.text);
console.log(res.points);
````

<a name="3"></a>
## 3. Model replacement

Due to the limitations of the front-end environment and computing resources, when deploying deep learning models on the front-end, we have stricter requirements on the performance of the models. In short, the models need to be lightweight enough. In theory, the smaller the input shape of the model and the smaller the model size, the smaller the flops of the corresponding model, and the smoother the front-end operation. Based on experience, the model storage deployed with `Paddle.js` should not exceed *5M* as much as possible, and the actual situation depends on the hardware and computing resources.

In practical applications, models are often customized according to vertical scenarios, and the official demo supports modifying incoming parameters to replace models.

Take the OCR demo as an example, [ocr.init()function](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/package/packages/paddlejs-models/ocr/src/index.ts#L52), contains the default initialization model link, if you want to replace the model, please refer to the following steps.

Step 1: Convert the model to js format:
````
# Install paddlejsconverter
pip3 install paddlejsconverter
# Convert the model format, the input model is the inference model
paddlejsconverter --modelPath=./inference.pdmodel --paramPath=./inference.pdiparams --outputDir=./ --useGPUOpt=True
# Note: The useGPUOpt option is not enabled by default. If the model is used on the gpu backend (webgl/webgpu), enable useGPUOpt. If the model is running on (wasm/plain js), do not enable it.
````

After the export is successful, files such as `model.json chunk_1.dat` will appear in the local directory, which are the network structure and model parameter binary files corresponding to the js model.

Step 2: Upload the exported js model to a server that supports cross-domain access. For the CORS configuration of the server, refer to the following image:
![image](https://user-images.githubusercontent.com/26592129/196612669-5233137a-969c-49eb-b8c7-71bef5088686.png)


Step 3: Modify the code to replace the default model. Take the OCR demo as an example, modify the [model initialization code](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/web_demo/src/pages/cv/ocr/TextRecognition/TextRecognition.vue#L64) in the OCR web demo , i.e.

````
await ocr.init();
change into:
await ocr.init({modelPath: "https://js-models.bj.bcebos.com/PaddleOCR/PP-OCRv3/ch_PP-OCRv3_det_infer_js_960/model.json"}); # The first parameter passes in the new text Check dictionary type parameter
````

Re-execute the following command in the demo directory to experience the new model effect.
````
npm run dev
````

<a name="4"></a>
## 4. custom hyperparameters

**Custom preprocessing parameters**

In different computer vision tasks, different models may have different preprocessing parameters, such as mean, std, keep_ratio and other parameters. After replacing the model, the preprocessing parameters also need to be modified. A simple solution for customizing preprocessing parameters is provided in the npm package published by paddle.js. You only need to pass in custom parameters when calling the model initialization function.

````
# Default parameter initialization
await model.init();

Custom parameter initialization
const Config = {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], keepratio: false};
await model.init(Config);
````

Taking the OCR text detection demo as an example, to modify the mean and std parameters of the model preprocessing, you only need to pass in the custom mean and std parameters when the model is initialized.
````
await ocr.init();
change into:
const detConfig = {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]};
await ocr.init(detConfig); # The first parameter passes in the new text detection model link
````

**Custom postprocessing parameters**

Similarly, the npm package published by paddle.js also provides a custom solution for post-processing parameters.

````
# run with default parameters
await model.predict();

# custom post-processing parameters
const postConfig = {thresh: 0.5};
await model.predict(Config);
````

Take the OCR text detection demo as an example, modify the parameters of the text detection post-processing to achieve the effect of expanding the text detection frame, and modify the OCR web demo to execute the [model prediction code](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/web_demo/src/pages/cv/ocr/TextRecognition/TextRecognition.vue#L99), ie:

````
const res = await ocr.recognize(img, { canvas: canvas.value });
change into:
// Define hyperparameters, increase the unclip_ratio parameter from 1.5 to 3.5
const detConfig = {shape: 960, thresh: 0.3, box_thresh: 0.6, unclip_ratio:3.5};
const res = await ocr.recognize(img, { canvas: canvas.value }, detConfig);
````

Note: Different tasks have different post-processing parameters. For detailed parameters, please refer to the API in the npm package.

<a name="5"></a>
## 5. Others

The converted model of `Paddle.js` can not only be used in the browser, but also can be run in the Baidu mini-program and WeChat mini-program environment.

|Name|Directory|
|-|-|
|OCR Text Detection| [ocrdetecXcx](./mini_program/ocrdetectXcx/) |
|OCR Text Recognition| [ocrXcx](./mini_program/ocrXcx/) |
|target detection| coming soon |
| Image segmentation | coming soon |
|Item Category| coming soon |

