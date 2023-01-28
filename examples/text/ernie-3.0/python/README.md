English | [简体中文](README_CN.md)

# Example of ERNIE 3.0 Models Python Deployment

Before deployment, two steps require confirmation.

- 1. Environment of software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).
- 2. FastDeploy Python whl package should be installed. Please refer to [FastDeploy Python Installation](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

This directory provides deployment examples that seq_cls_inferve.py fast finish text classification tasks on CPU/GPU.

## Dependency Installation

The Python Predictor in this project uses AutoTokenizer provided by PaddleNLP to conduct word segmentation and fast_tokenizer to speed up word segmentation. Run the following command to install it.

```bash
pip install -r requirements.txt
```


## Text Classification Tasks

### A Quick Start

The following example shows how to employ FastDeploy library to complete Python predictive deployment of ERNIE 3.0 Medium model on [AFQMC Dataset](https://bj.bcebos.com/paddlenlp/datasets/afqmc_public.zip)of CLUE Benchmark for text classification tasks.

```bash

# Download the deployment example code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/text/ernie-3.0/python

# Download fine-tuned ERNIE 3.0 models for AFQMC dataset
wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc.tgz
tar xvfz ernie-3.0-medium-zh-afqmc.tgz

# CPU Inference
python seq_cls_infer.py --device cpu --model_dir ernie-3.0-medium-zh-afqmc

# GPU Inference
python seq_cls_infer.py --device gpu --model_dir ernie-3.0-medium-zh-afqmc

# KunlunXin XPU Inference
python seq_cls_infer.py --device kunlunxin --model_dir ernie-3.0-medium-zh-afqmc

```
The result returned after running is as follows:


```bash
[INFO] fastdeploy/runtime.cc(469)::Init	Runtime initialized with Backend::ORT in Device::CPU.
Batch id:0, example id:0, sentence1:花呗收款额度限制, sentence2:收钱码，对花呗支付的金额有限制吗, label:1, similarity:0.5819
Batch id:1, example id:0, sentence1:花呗支持高铁票支付吗, sentence2:为什么友付宝不支持花呗付款, label:0, similarity:0.9979
```

### Parameter Description

`seq_cls_infer.py`supports more command-line arguments in addition to the preceding example. The following is a description of every command-line argument

| Parameter | Parameter Description |
|----------|--------------|
|--model_dir | Specify the directory where the model is to be deployed |
|--batch_size |Maximum measurable batch size, default 1|
|--max_length |Maximum sequence length, default 128|
|--device | Running devices, optional range: ['cpu', 'gpu'], default 'cpu' |
|--backend | Supported inference backend, optional range: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，default 'onnx_runtime' |
|--use_fp16 | Whether to use FP16 mode for inference.Enabled when using tensorrt and paddle tensorrt backend, and default is False |
|--use_fast| Whether to use FastTokenizer to speed up the tokenization, and default is True|

## Related Documents

[ERNIE 3.0 Model detailed introduction](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0)

[ERNIE 3.0 Model Export Method](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0)

[ERNIE 3.0 Model C++ Deployment Method](../cpp/README.md)
