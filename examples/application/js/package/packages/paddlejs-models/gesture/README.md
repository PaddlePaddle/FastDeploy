[中文版](./README_cn.md)

# gesture

gesture is a gesture recognition module, including two models: gesture_detect and gesture_rec. gesture_detect model is used to identify the palm area of the person in the image. gesture_rec model is used to recognize human gesture. The interface provided by the module is simple, users only need to pass in gesture images to get the results.

<img src="https://img.shields.io/npm/v/@paddle-js-models/gesture?color=success" alt="version"> <img src="https://img.shields.io/bundlephobia/min/@paddle-js-models/gesture" alt="size"> <img src="https://img.shields.io/npm/dm/@paddle-js-models/gesture?color=orange" alt="downloads"> <img src="https://img.shields.io/npm/dt/@paddle-js-models/gesture" alt="downloads">

# Usage

```js

import * as gesture from '@paddle-js-models/gesture';

// Load gesture_detect model and gesture_rec model
await gesture.load();

// Get the image recognition results. The results include: palm frame coordinates and recognition results
const res = await gesture.classify(img);

```

# Online experience

https://paddlejs.baidu.com/gesture

# Performance
<img alt="gesture" src="https://user-images.githubusercontent.com/43414102/156379706-065a4f57-cc75-4457-857a-18619589492f.gif">
