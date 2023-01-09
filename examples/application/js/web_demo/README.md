English | [简体中文](README_CN.md)
# Paddle.js-demo

## Demo Directory

| Classification | Name             | Directory                                                     |
|:----:| :--------------- | -------------------------------------------------------- |
|  CV  | Portrait matting         | /src/pages/cv/segmentation/HumanSeg               |
|  CV  | Portrait segmentation background replacement | /src/pages/cv/segmentation/HumanSeg |
|  CV  | Gesture recognition AI 'Rock Paper Scissors' | /src/pages/cv/recognition/GestureRecognition             |
|  CV  | Identify 1000 items   | /src/pages/cv/recognition/ItemIdentification             |
|  CV  | Wine bottle recognition         | /src/pages/cv/recognition/WineBottleIdentification       |
|  CV  | Text detection         | /src/pages/cv/ocr/TextDetection                          |
|  CV  | Text Recognition        | /src/pages/cv/ocr/TextRecognition                        |

## Introduction to Development

### Install dependencies

```sh
npm install
```

### Development

```sh
npm run dev
```

### Page View

Visit `http://localhost:5173/main/index.html` and enter homepage

### Construction

```sh
npm run build
```

### [ESLint](https://eslint.org/) Formatting

```sh
npm run lint
```

### Project style

1. Use TypeScript
2. Vue's compositional API is recommended. Creating new components according to the 'src/pages/ExampleFile.vue' template
3. use Less for CSS
4. Use what Vue recommends for eslint. Try to meet the requirements.
5. Use [Pinia](https://pinia.web3doc.top/) for store
6. Use [vue-router](https://router.vuejs.org/zh/) for router

### Brief introduction to src

```text
├─assets  
├─components  
├─router 
├─stores 
└─pages 
    └─cv  demo of cv
        ├─ocr  demo of ocr
        │  ├─TextDetection
        │  └─TextRecognition
        ├─...
        ├─recognition  demo of recognition
        │  ├─GestureRecognition
        │  ├─ItemIdentification
        │  ├─...
        │  └─WineBottleIdentification
        └─segmentation  demo of segmentation
            ├─PortraitBackgroundReplacement
            ├─...
            └─PortraitMatting

```
Add new components under corresponding categories. Refer to 'src/pages/ExampleFile.vue' for its template
