简体中文 ｜ [English](README.md)

# Silero VAD 预训练的企业级语音活动检测器

该部署模型来自于 [silero-vad](https://github.com/snakers4/silero-vad)

![](https://user-images.githubusercontent.com/36505480/198026365-8da383e0-5398-4a12-b7f8-22c2c0059512.png)

## 主要特征

* 高准确率

Silero VAD在语音检测任务上有着优异的成绩。

* 快速推理

一个音频块（30+ 毫秒）在单个 CPU 线程上处理时间不到 1毫秒。

* 通用性

Silero VAD 在包含100多种语言的庞大语料库上进行了训练，它在来自不同领域、具有不同背景噪音和质量水平的音频上表现良好。

* 灵活采样率

Silero VAD支持 8000 Hz和16000 Hz 采样率。

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了 VAD 导出模型，开发者可直接下载使用。
| 模型                                                         | 大小  | 备注                                                         |
| :----------------------------------------------------------- | :---- | :----------------------------------------------------------- |
| [silero-vad](https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad.tgz) | 1.8MB | 此模型文件来源于[snakers4/silero-vad](https://github.com/snakers4/silero-vad)，MIT License |

## 详细部署文档

- [C++ 部署](cpp)

## 模型来源

[https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)
