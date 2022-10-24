[English](./README.md)

# detect

detect模型用于检测图像中label框选位置。

<img src="https://img.shields.io/npm/v/@paddle-js-models/detect?color=success" alt="version"> <img src="https://img.shields.io/bundlephobia/min/@paddle-js-models/detect" alt="size"> <img src="https://img.shields.io/npm/dm/@paddle-js-models/detect?color=orange" alt="downloads"> <img src="https://img.shields.io/npm/dt/@paddle-js-models/detect" alt="downloads">

# 使用

```js
import * as det from '@paddle-js-models/detect';

// 模型加载
await det.load();

// 获取label对应索引、置信度、检测框选坐标
const res = await det.detect(img);

res.forEach(item => {
    // 获取label对应索引
    console.log(item[0]);
    // 获取label置信度
    console.log(item[1]);
    // 获取检测框选left顶点
    console.log(item[2]);
    // 获取检测框选top顶点
    console.log(item[3]);
    // 获取检测框选right顶点
    console.log(item[4]);
    // 获取检测框选bottom顶点
    console.log(item[5]);
});
```

# 效果
![img.png](https://user-images.githubusercontent.com/43414102/153805288-80f289bf-ca92-4788-b1dd-44854681a03f.png)

