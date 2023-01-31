English | [简体中文](README_CN.md)
# PaddleJsConverter

## Installation

System Requirements:

* paddlepaddle >= 2.0.0
* paddlejslite >= 0.0.2
* Python3： 3.5.1+ / 3.6 / 3.7
* Python2： 2.7.15+

#### Install PaddleJsConverter

<img src="https://img.shields.io/pypi/v/paddlejsconverter" alt="version">

```shell
pip install paddlejsconverter

# or
pip3 install paddlejsconverter
```


## Usage

```shell
paddlejsconverter --modelPath=user_model_path --paramPath=user_model_params_path --outputDir=model_saved_path --useGPUOpt=True
```
Note: The option useGPUOpt is not turned on by default. Turn on useGPUOpt if the model is used on gpu backend (webgl/webgpu), don't turn on if is running on (wasm/plain js).
