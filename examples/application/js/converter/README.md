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
注意：useGPUOpt 选项默认不开启，如果模型用在 gpu backend（webgl/webgpu），则开启 useGPUOpt，如果模型运行在（wasm/plain js）则不要开启。
