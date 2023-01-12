English | [简体中文](README_CN.md)

# Silero VAD - pre-trained enterprise-grade Voice Activity Detector

The deployment model comes from [silero-vad](https://github.com/snakers4/silero-vad)

![](https://user-images.githubusercontent.com/36505480/198026365-8da383e0-5398-4a12-b7f8-22c2c0059512.png)

## Key Features

* Stellar accuracy

Silero VAD has excellent results on speech detection tasks.

* Fast

One audio chunk (30+ ms) takes less than 1ms to be processed on a single CPU thread. Using batching or GPU can also improve performance considerably.

* General

Silero VAD was trained on huge corpora that include over 100 languages and it performs well on audios from different domains with various background noise and quality levels.

* Flexible sampling rate

Silero VAD supports 8000 Hz and 16000 Hz sampling rates.

## Download Pre-trained ONNX Model

For developers' testing, model exported by VAD are provided below. Developers can download them directly.

| 模型                                                         | 大小  | 备注                                                         |
| :----------------------------------------------------------- | :---- | :----------------------------------------------------------- |
| [silero-vad](https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad.tgz) | 1.8MB | This model file is sourced from [snakers4/silero-vad](https://github.com/snakers4/silero-vad)，MIT License |

## Detailed Deployment Documents

- [C++ deployment](cpp)

## Source

[https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)
