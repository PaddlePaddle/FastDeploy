[中文版](./DEVELOPMENT_cn.md)
# paddlejs-converter

paddlejs-converter is a model transformation tool for Paddle.js. Its role is to convert PaddlePaddle models (also known as fluid models) into a browser-friendly format that Paddle.js can use to load and predict usage in browsers as well as other environments. In addition, paddlejs-converter provides powerful model optimization capabilities to help developers optimize the model structure and improve runtime performance.

## 1. Tutorial

### 1.1. Environment Construction
#### Python Version
Confirm whether the python environment and version of the running platform meet the requirements. If Python 3 is used, you may need to change the `python` in subsequent commands to `python3`:
- Python3： 3.5.1+ / 3.6 / 3.7
- Python2： 2.7.15+

#### Install Virtual Environment
*Since the development environment may have multiple versions of Python installed, there may be different versions of dependent packages. In order to avoid conflicts, it is strongly recommended to use Python virtual environment to execute the commands required by the conversion tool to avoid various problems. If you are not using a virtual environment or if you have a virtual environment installed, you can skip this step.*

Take Anaconda as an example：
Go to [Anaconda](https://www.anaconda.com/) main page，Select the corresponding platform and python version of anaconda and install it according to the official prompts；

After installation, execute the following command on the command line to create a python virtual environment:
``` bash
conda create --name <your_env_name>
```

Execute the following command to switch to the virtual environment
``` bash
# Linux or macOS
source activate <your_env_name>

# Windows
activate <your_env_name>
```

#### Installation Dependency
- If you don't need to optimize model, execute the command：
``` bash
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```
- Otherwise，execute the command：
``` bash
python -m pip install paddlepaddle paddlelite==2.6.0 -i https://mirror.baidu.com/pypi/simple
```

### 1.2. Get Start
- If the weight file of fluid model to be converted is merged format which means one model corresponds to one weight file, then execute:
``` bash
python convertToPaddleJSModel.py --modelPath=<fluid_model_file_path> --paramPath=<fluid_param_file_path> --outputDir=<paddlejs_model_directory>
```
- Otherwise，execute：
``` bash
# Note that in this way, you need to ensure that the model file name '__ model__ ' in the inputDir
python convertToPaddleJSModel.py --inputDir=<fluid_model_directory> --outputDir=<paddlejs_model_directory>
````
The model converter generates the following two types of files for Paddle.js:

- model.json (Contains the model structure and parameter list)
- chunk_\*.dat (The collection of binary weight files)

## 2. Detailed Documentation

Parameter | description
:-: | :-:
--inputDir | The fluid model directory, If and only if weight files are not merged format, `modelPath` and `paramPath` below will be ignored，and the model file name should be `__model__`.
--modelPath | The model file path, used when the weight file is merged.
--paramPath | The weight file path，used when the weight file is merged.
--outputDir | `Necessary`, the output model directory generated after converting.
--disableOptimize | Whether to disable optimize model, `1`is to disable, `0`is use optimize(need to install PaddleLite), default 0.
--logModelInfo | Whether to print model structure information， `0` means not to print, `1` means to print, default 0.
--sliceDataSize | Shard size (in KB) of each weight file. Default size is 4096.
--useGPUOpt | Whether to use gpu opt, default is False.

## 3. Other information
If the model to be converted is in `tensorflow / Cafe / onnx` format, there is [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) tool in PaddlePaddle program for converting other models with different formats to fluid model, and then you can use paddlejs-converter to get a Paddle.js model.
