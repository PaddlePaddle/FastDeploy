English | [简体中文](README_CN.md)

# Introduction to Paddle.js Demo Module

This part is a model library developed based on Paddle.js, which mainly provides the ability to directly introduce and use models on the web side.

| demo name | source directory | npm package |
| - | - | - |
| face detection | [facedetect](./packages/paddlejs-models/facedetect) | [@paddle-js-models/facedetect](https://www.npmjs.com/package/@paddle-js-models/facedetect) |
| Screw detection | [detect](./packages/paddlejs-models/detect) | [@paddle-js-models/detect](https://www.npmjs.com/package/@paddle-js-models/detect ) |
| Portrait segmentation background replacement | [humanseg](./packages/paddlejs-models/humanseg) | [@paddle-js-models/humanseg](https://www.npmjs.com/package/@paddle-js-models/humanseg) |
| Gesture Recognition AI Guessing Shell | [gesture](./packages/paddlejs-models/gesture) | [@paddle-js-models/gesture](https://www.npmjs.com/package/@paddle-js-models/gesture) |
| 1000 Item Recognition | [mobilenet](./packages/paddlejs-models/mobilenet) | [@paddle-js-models/mobilenet](https://www.npmjs.com/package/@paddle-js-models/mobilenet) |
| Text Detection | [ocrdetection](./packages/paddlejs-models/ocrdetection) | [@paddle-js-models/ocrdet](https://www.npmjs.com/package/@paddle-js-models/ocrdet ) |
| Text Recognition | [ocr](./packages/paddlejs-models/ocr) | [@paddle-js-models/ocr](https://www.npmjs.com/package/@paddle-js-models/ocr) |

## Usage

This part is Menorepo built with `pnpm`

### Install dependencies

````sh
pnpm i
````

### Development
See Package.json for development testing with `yalc`.

````sh
pnpm run dev:xxx
````

### Overall Introduction

1. Use rollup to package the code of commonjs and es specifications at one time; at the same time, it is extensible; at present, there are some problems with the dependent cv library; there is no configuration for umd packaging.
2. The d.ts file is generated based on api-extractor during packaging, and the introduction of ts is supported to generate our package
3. Support testing based on jest and display test related coverage, etc.
4. Maintain code style based on ts and eslint to ensure better code development
5. Generate custom keywords based on conventional-changelog-cli and generate changelog accordingly
6. Implement local packaging development and testing based on yalc
