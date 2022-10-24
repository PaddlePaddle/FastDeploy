([简体中文](./README_cn.md)|English)

# PP-TTS Streaming Text-to-Speech Python Example

## Introduction
This demo is an implementation of starting the streaming speech synthesis.

## Usage

### 1. Installation
```bash
apt-get install libsndfile1 wget zip
**For Centos,use the command `yum install libsndfile-devel wget zip`**
python3 -m pip install --upgrade pip
pip3 install -U fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip3 install -U paddlespeech paddlepaddle soundfile matplotlib
```

### 2. Run the example
```bash
python3 stream_play_tts.py
```

### 3. Result
The complete voice synthesis audio is saved as `demo_stream.wav`.

Users can install `pyaudio` on their own terminals to play the results of speech synthesis in real time. The relevant code is in `stream_play_tts.py` and you can debug and run it yourself.
