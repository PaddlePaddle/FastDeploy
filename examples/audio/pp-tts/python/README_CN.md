简体中文 | [English](README.md)

# PP-TTS流式语音合成Python示例

## 介绍
本文介绍了使用FastDeploy运行流式语音合成的示例.

## 使用
### 1. 安装
```bash
apt-get install libsndfile1 wget zip
对于Centos系统,使用yum install libsndfile-devel wget zip
python3 -m pip install --upgrade pip
pip3 install -U fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip3 install -U paddlespeech paddlepaddle soundfile matplotlib
```

### 2. 运行示例
```bash
python3 stream_play_tts.py
```

### 3. 运行效果
完整的语音合成音频被保存为`demo_stream.wav`文件.

用户可以在自己的终端上安装pyaudio, 对语音合成的结果进行实时播放, 相关代码在stream_play_tts.py处于注释状态, 用户可自行调试运行.
