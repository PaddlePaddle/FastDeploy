# Paddle.js-demo

## Demo 目录

| 分类 | 名称             | 目录                                                     |
|:----:| :--------------- | -------------------------------------------------------- |
|  CV  | 人像扣图         | /src/pages/cv/segmentation/HumanSeg               |
|  CV  | 人像分割背景替换 | /src/pages/cv/segmentation/HumanSeg |
|  CV  | 手势识别AI猜丁壳 | /src/pages/cv/recognition/GestureRecognition             |
|  CV  | 1000种物品识别   | /src/pages/cv/recognition/ItemIdentification             |
|  CV  | 酒瓶识别         | /src/pages/cv/recognition/WineBottleIdentification       |
|  CV  | 文本检测         | /src/pages/cv/ocr/TextDetection                          |
|  CV  | 文本识别         | /src/pages/cv/ocr/TextRecognition                        |

## 开发简介

### 安装依赖

```sh
npm install
```

### 开发

```sh
npm run dev
```

### 查看页面

访问 `http://localhost:5173/main/index.html` 进入主页

### 构建

```sh
npm run build
```

### [ESLint](https://eslint.org/) 格式化

```sh
npm run lint
```

### 工程风格

1. 项目使用TypeScript
2. 推荐使用 Vue 的组合式 API，可以根据 'src/pages/ExampleFile.vue' 模板创建新的组件
3. CSS 使用 Less
4. eslint 使用的是 Vue 推荐的，一般情况请尽量符合对应的要求
5. store 使用的是 [Pinia](https://pinia.web3doc.top/)
6. router 使用的是 [vue-router](https://router.vuejs.org/zh/)

### src 目录简介

```text
├─assets 资源文件
├─components 全局组件
├─router 路由
├─stores 存储库
└─pages 
    └─cv cv相关demo
        ├─ocr ocr相关demo
        │  ├─TextDetection
        │  └─TextRecognition
        ├─...
        ├─recognition 识别相关demo
        │  ├─GestureRecognition
        │  ├─ItemIdentification
        │  ├─...
        │  └─WineBottleIdentification
        └─segmentation 分割相关demo
            ├─PortraitBackgroundReplacement
            ├─...
            └─PortraitMatting

```
新增组件在对应类别下新增即可，可以参考模板 'src/pages/ExampleFile.vue'
