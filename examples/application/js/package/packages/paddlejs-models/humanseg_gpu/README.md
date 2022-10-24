[中文版](./README_cn.md)

# humanseg

A real-time human-segmentation model. You can use it to change background. The output of the model is gray value. Model supplies simple api for users.

Api drawHumanSeg can draw human segmentation with a specified background.
Api blurBackground can draw human segmentation with a blurred origin background.
Api drawMask can draw the background without human.


<img src="https://img.shields.io/npm/v/@paddle-js-models/humanseg?color=success" alt="version"> <img src="https://img.shields.io/bundlephobia/min/@paddle-js-models/humanseg" alt="size"> <img src="https://img.shields.io/npm/dm/@paddle-js-models/humanseg?color=orange" alt="downloads"> <img src="https://img.shields.io/npm/dt/@paddle-js-models/humanseg" alt="downloads">

# Usage

```js

import * as humanseg from '@paddle-js-models/humanseg';

// load humanseg model, use 398x224 shape model, and preheat
await humanseg.load();

// use 288x160 shape model, preheat and predict faster with a little loss of precision
// await humanseg.load(true, true);

// get the gray value [2, 398, 224] or [2, 288, 160];
const { data } = await humanseg.getGrayValue(img);

// background canvas
const back_canvas = document.getElementById('background') as HTMLCanvasElement;

// draw human segmentation
const canvas1 = document.getElementById('back') as HTMLCanvasElement;
humanseg.drawHumanSeg(data, canvas1, back_canvas) ;

// blur background
const canvas2 = document.getElementById('blur') as HTMLCanvasElement;
humanseg.blurBackground(data, canvas2) ;

// draw the mask with background
const canvas3 = document.getElementById('mask') as HTMLCanvasElement;
humanseg.drawMask(data, canvas3, back_canvas);

```

## gpu pipeline

```js

// 引入 humanseg sdk
import * as humanseg from '@paddle-js-models/humanseg/lib/index_gpu';

// load humanseg model, use 398x224 shape model, and preheat
await humanseg.load();

// use 288x160 shape model, preheat and predict faster with a little loss of precision
// await humanseg.load(true, true);


// background canvas
const back_canvas = document.getElementById('background') as HTMLCanvasElement;

// draw human segmentation
const canvas1 = document.getElementById('back') as HTMLCanvasElement;
await humanseg.drawHumanSeg(input, canvas1, back_canvas) ;

// blur background
const canvas2 = document.getElementById('blur') as HTMLCanvasElement;
await humanseg.blurBackground(input, canvas2) ;

// draw the mask with background
const canvas3 = document.getElementById('mask') as HTMLCanvasElement;
await humanseg.drawMask(input, canvas3, back_canvas);

```

# Online experience

image human segmentation：https://paddlejs.baidu.com/humanseg

video-streaming human segmentation：https://paddlejs.baidu.com/humanStream

# Performance

  <img width="800"  src="https://user-images.githubusercontent.com/10822846/126873788-1e2d4984-274f-45be-8716-2a87ddda8c75.png"/>
  <img width="800"  src="https://user-images.githubusercontent.com/10822846/126873838-e5b68c9b-279f-4cb4-ae90-6aaaecd06aa4.png"/>


# Used in Video Meeting
  <p>
  <img width="400"  src="https://user-images.githubusercontent.com/10822846/126872499-c3fd680e-a01b-4daa-b0cb-acd3290862bd.gif"/>
  <img width="400"  src="https://user-images.githubusercontent.com/10822846/126872930-4f4c5c5d-5c51-44fe-b2d6-3f83c4e124bc.png"/>
  </p>
