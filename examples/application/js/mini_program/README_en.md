English | [中文](README.md)

# Paddle.js WeChat mini-program Demo

- [1. Introduction](#1)
- [2. Project Start](#2)
  * [2.1 Preparations](#21)
  * [2.2 Startup steps](#22)
  * [2.3 visualization](#23)
- [3. Model inference pipeline](#3)
- [4. FAQ](#4)

<a name="1"></a>
## 1 Introduction


This directory contains the text detection, text recognition mini-program demo, by using [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) and [Paddle.js WeChat mini-program plugin](https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx7138a7bb793608c3&token=956931339&lang=zh_CN) to complete the text detection frame selection effect on the mini-program using the computing power of the user terminal.

<a name="2"></a>
## 2. Project start

<a name="21"></a>
### 2.1 Preparations
* [Apply for a WeChat mini-program account](https://mp.weixin.qq.com/)
* [WeChat Mini Program Developer Tools](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
* Front-end development environment preparation: node, npm
* Configure the server domain name in the mini-program management background, or open the developer tool [do not verify the legal domain name]

For details, please refer to [document.](https://mp.weixin.qq.com/wxamp/devprofile/get_profile?token=1132303404&lang=zh_CN)

<a name="22"></a>
### 2.2 Startup steps

#### **1. Clone the demo code**
````sh
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy/examples/application/js/mini_program
````

#### **2. Enter the mini-program directory and install dependencies**

````sh
# Run the text recognition demo and enter the ocrXcx directory
cd ./ocrXcx && npm install
# Run the text detection demo and enter the ocrdetectXcx directory
# cd ./ocrdetectXcx && npm install
````

#### **3. WeChat mini-program import code**
Open WeChat Developer Tools --> Import --> Select a directory and enter relevant information

#### **4. Add Paddle.js WeChat mini-program plugin**
Mini Program Management Interface --> Settings --> Third Party Settings --> Plugin Management --> Add Plugins --> Search for `wx7138a7bb793608c3` and add
[Reference document](https://developers.weixin.qq.com/miniprogram/dev/framework/plugin/using.html)

#### **5. Build dependencies**
Click on the menu bar in the developer tools: Tools --> Build npm

Reason: The node_modules directory will not be involved in compiling, uploading and packaging. If a small program wants to use npm packages, it must go through the process of "building npm". After the construction is completed, a miniprogram_npm directory will be generated, which will store the built and packaged npm packages. It is the npm package that the mini-program actually uses. *
[Reference Documentation](https://developers.weixin.qq.com/miniprogram/dev/devtools/npm.html)

<a name="23"></a>
### 2.3 visualization

<img src="https://user-images.githubusercontent.com/43414102/157648579-cdbbee61-9866-4364-9edd-a97ac0eda0c1.png" width="300px">

<a name="3"></a>
## 3. Model inference pipeline

```typescript
// Introduce paddlejs and paddlejs-plugin, register the mini-program environment variables and the appropriate backend
import * as paddlejs from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';
const plugin = requirePlugin('paddlejs-plugin');
plugin.register(paddlejs, wx);

// Initialize the inference engine
const runner = new paddlejs.Runner({modelPath, feedShape, mean, std});
await runner.init();

// get image information
wx.canvasGetImageData({
    canvasId: canvasId,
    x: 0,
    y: 0,
    width: canvas.width,
    height: canvas.height,
    success(res) {
        // inference prediction
        runner.predict({
            data: res.data,
            width: canvas.width,
            height: canvas.height,
        }, function (data) {
            // get the inference result
            console.log(data)
        });
    }
});
````

<a name="4"></a>
## 4. FAQ

- 4.1 An error occurs `Invalid context type [webgl2] for Canvas#getContext`

You can leave it alone, it will not affect the normal code operation and demo function

- 4.2 Preview can't see the result

It is recommended to try real machine debugging

- 4.3 A black screen appears in the WeChat developer tool, and then there are too many errors

Restart WeChat Developer Tools

- 4.4 The debugging results of the simulation and the real machine are inconsistent; the simulation cannot detect the text, etc.

The real machine can prevail;

If the simulation cannot detect the text, etc., you can try to change the code at will (add, delete, newline, etc.) and then click to compile


- 4.5 Prompts such as no response for a long time appear when the phone is debugged or running

Please continue to wait, model inference will take some time