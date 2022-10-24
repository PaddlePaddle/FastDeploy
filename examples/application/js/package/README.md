# Paddle.js Model Module介绍

该部分是基于 Paddle.js 进行开发的模型库，主要提供 Web 端可直接引入使用模型的能力。

| demo名称         | 源码目录                                               | npm包                                                        |
| ---------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| 人脸检测         | [facedetect](./packages/paddlejs-models/facedetect) | [@paddle-js-models/facedetect](https://www.npmjs.com/package/@paddle-js-models/facedetect) |
| 螺丝钉检测       | [detect](./packages/paddlejs-models/detect)      | [@paddle-js-models/detect](https://www.npmjs.com/package/@paddle-js-models/detect) |
| 人像分割背景替换 | [humanseg](./packages/paddlejs-models/humanseg)  | [@paddle-js-models/humanseg](https://www.npmjs.com/package/@paddle-js-models/humanseg) |
| 手势识别AI猜丁壳 | [gesture](./packages/paddlejs-models/gesture)    | [@paddle-js-models/gesture](https://www.npmjs.com/package/@paddle-js-models/gesture) |
| 1000种物品识别   | [mobilenet](./packages/paddlejs-models/mobilenet) | [@paddle-js-models/mobilenet](https://www.npmjs.com/package/@paddle-js-models/mobilenet) |
| 文本检测         | [ocrdetection](./packages/paddlejs-models/ocrdetection) | [@paddle-js-models/ocrdet](https://www.npmjs.com/package/@paddle-js-models/ocrdet) |
| 文本识别         | [ocr](./packages/paddlejs-models/ocr)           | [@paddle-js-models/ocr](https://www.npmjs.com/package/@paddle-js-models/ocr) |

## 开发使用

该部分是使用 `pnpm` 搭建的 Menorepo

### 安装依赖

```sh
pnpm i
```

### 开发
参考 Package.json 使用 `yalc` 进行开发测试。

```sh
pnpm run dev:xxx
```

### 整体简介

1. 使用 rollup 一次性打包生成 commonjs 和 es 规范的代码；同时具有可扩展性；目前由于依赖的cv库有些问题；就没有配置umd打包。
2. 打包时基于 api-extractor 实现 d.ts 文件生成，实现支持 ts 引入生成我们的包
3. 基于 jest 支持测试并显示测试相关覆盖率等
4. 基于 ts 和 eslint 维护代码风格，保证代码更好开发
5. 基于 conventional-changelog-cli 实现自定义关键词生成对应生成changelog
6. 基于 yalc 实现本地打包开发测试


