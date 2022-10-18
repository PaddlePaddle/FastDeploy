# ocr detection 微信小程序代码示例

## 1.介绍
本目录为文本检测小程序代码，通过使用 [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) 以及 [Paddle.js微信小程序插件](https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx7138a7bb793608c3&token=956931339&lang=zh_CN) 完成在小程序上利用用户终端算力实现文本检测框选效果。

文本检测详见 [@paddlejs-models/ocrdet](https://github.com/PaddlePaddle/Paddle.js/tree/release/v2.2.3/packages/paddlejs-models/ocrdetection) 。

开发者可在demo基础上增加文本识别能力，可参考 [@paddlejs-models/ocr](https://github.com/PaddlePaddle/Paddle.js/blob/release/v2.2.3/packages/paddlejs-models/ocr/src/index.ts) 实现。

## 2. 项目启动

### 2.1 准备工作
* [申请微信小程序账号](https://mp.weixin.qq.com/)
* [微信小程序开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
* 前端开发环境准备：node、npm
* 小程序管理后台配置服务器域名，或打开开发者工具【不校验合法域名】 

详情参考:https://mp.weixin.qq.com/wxamp/devprofile/get_profile?token=1132303404&lang=zh_CN)

### 2.2 启动步骤
#### **1. clone Paddle.js**
```sh
git clone https://github.com/PaddlePaddle/Paddle.js.git
```

#### **2. 进入 xxx 目录，安装依赖**
```sh
cd Paddle.js/packages/paddlejs-examples/ocrdetectXcx && npm install
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

### 2.3 效果展示
![image](https://user-images.githubusercontent.com/43414102/157648579-cdbbee61-9866-4364-9edd-a97ac0eda0c1.png)

## 3. Paddle.js 框架推理
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

## 4. 更多
* [详细文档](https://mp.weixin.qq.com/s/KBjXawSfBreUCsIYbUgF-w)
