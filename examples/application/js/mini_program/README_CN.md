[English](README.md) | 简体中文

# Paddle.js微信小程序Demo

- [1.简介](#1)
- [2. 项目启动](#2)
  * [2.1 准备工作](#21)
  * [2.2 启动步骤](#22)
  * [2.3 效果展示](#23)
- [3. 模型推理pipeline](#3)
- [4. 常见问题](#4)

<a name="1"></a>
## 1.简介


本目录下包含文本检测、文本识别小程序demo，通过使用 [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) 以及 [Paddle.js微信小程序插件](https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx7138a7bb793608c3&token=956931339&lang=zh_CN) 完成在小程序上利用用户终端算力实现文本检测框选效果。

<a name="2"></a>
## 2. 项目启动

<a name="21"></a>
### 2.1 准备工作
* [申请微信小程序账号](https://mp.weixin.qq.com/)
* [微信小程序开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
* 前端开发环境准备：node、npm
* 小程序管理后台配置服务器域名，或打开开发者工具【不校验合法域名】 

详情参考:https://mp.weixin.qq.com/wxamp/devprofile/get_profile?token=1132303404&lang=zh_CN)

<a name="22"></a>
### 2.2 启动步骤

#### **1. 克隆Demo代码**
```sh
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy/examples/application/js/mini_program
```

#### **2. 进入小程序目录，安装依赖**

```sh
# 运行文本识别demo，进入到ocrXcx目录
cd ./ocrXcx && npm install
# 运行文本检测demo，进入到ocrdetectXcx目录
# cd ./ocrdetectXcx && npm install
```

#### **3. 微信小程序导入代码**
打开微信开发者工具 --> 导入 --> 选定目录，输入相关信息

#### **4. 添加 Paddle.js微信小程序插件**
小程序管理界面 --> 设置 --> 第三方设置 --> 插件管理 --> 添加插件 --> 搜索 `wx7138a7bb793608c3` 并添加  
[参考文档](https://developers.weixin.qq.com/miniprogram/dev/framework/plugin/using.html)

#### **5. 构建依赖**
点击开发者工具中的菜单栏：工具 --> 构建 npm

原因：node_modules 目录不会参与编译、上传和打包中，小程序想要使用 npm 包必须走一遍“构建 npm”的过程，构建完成会生成一个 miniprogram_npm 目录，里面会存放构建打包后的 npm 包，也就是小程序真正使用的 npm 包。*  
[参考文档](https://developers.weixin.qq.com/miniprogram/dev/devtools/npm.html)

<a name="23"></a>
### 2.3 效果展示

<img src="https://user-images.githubusercontent.com/43414102/157648579-cdbbee61-9866-4364-9edd-a97ac0eda0c1.png" width="300px">

<a name="3"></a>
## 3. 模型推理pipeline

```typescript
// 引入 paddlejs 和 paddlejs-plugin，注册小程序环境变量和合适的 backend
import * as paddlejs from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';
const plugin = requirePlugin('paddlejs-plugin');
plugin.register(paddlejs, wx);

// 初始化推理引擎
const runner = new paddlejs.Runner({modelPath, feedShape, mean, std}); 
await runner.init();

// 获取图像信息
wx.canvasGetImageData({
    canvasId: canvasId,
    x: 0,
    y: 0,
    width: canvas.width,
    height: canvas.height,
    success(res) {
        // 推理预测
        runner.predict({
            data: res.data,
            width: canvas.width,
            height: canvas.height,
        }, function (data) {
            // 获取推理结果
            console.log(data)
        });
    }
});
```

<a name="4"></a>
## 4. 常见问题

- 4.1 出现报错 `Invalid context type [webgl2] for Canvas#getContext`

    **答:** 可以不管，不影响正常代码运行和demo功能

- 4.2 预览看不到结果

    **答:** 建议尝试真机调试

- 4.3 微信开发者工具出现黑屏，然后出现超多报错

    **答:** 重启微信开发者工具

- 4.4 模拟和真机调试结果不一致；模拟检测不到文本等

    **答:** 可以以真机为准；模拟检测不到文本等可以尝试随意改动下代码（增删换行等）再点击编译


- 4.5 手机调试或运行时出现 长时间无反应等提示

    **答:** 请继续等待，模型推理需要一定时间


