English | [简体中文](README_CN.md)

# Silero VAD - pre-trained enterprise-grade Voice Activity Detector

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

## Detailed Deployment Documents

- [C++ deployment](cpp)

## Source

[https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)