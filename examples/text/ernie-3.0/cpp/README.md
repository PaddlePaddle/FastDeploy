English | [简体中文](README_CN.md)
# Example of ERNIE 3.0 models C++ Deployment

Before deployment, two steps require confirmation.

- 1. Environment of software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).
- 2. Based on the develop environment, download the precompiled deployment library and samples code. Please refer to [FastDeploy Precompiled Library](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

This directory provides deployment examples that seq_cls_inferve.py fast finish text classification tasks on CPU/GPU.


##  Text Classification Tasks

### A Quick Start

The following example shows how to employ FastDeploy library to complete C++ predictive deployment of ERNIE 3.0 Medium model on [AFQMC dataset](https://bj.bcebos.com/paddlenlp/datasets/afqmc_public.zip) of CLUE Benchmark for text classification tasks.FastDeploy version 0.7.0 or above is required to support this model(x.x.x>=0.7.0) )


```bash
mkdir build
cd build
# Download FastDeploy precompiled library. Uses can choose proper version in the `FastDeploy Precompiled Library`mentioned above.
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the fine-tuned ERNIE 3.0 models for the AFQMC dataset and the word lists.
wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc.tgz
tar xvfz ernie-3.0-medium-zh-afqmc.tgz

# CPU Inference
./seq_cls_infer_demo --device cpu --model_dir ernie-3.0-medium-zh-afqmc

# GPU Inference
./seq_cls_infer_demo --device gpu --model_dir ernie-3.0-medium-zh-afqmc

# KunlunXin XPU Inference
./seq_cls_infer_demo --device kunlunxin --model_dir ernie-3.0-medium-zh-afqmc
```
The result returned after running is as follows：
```bash
[INFO] /paddle/FastDeploy/examples/text/ernie-3.0/cpp/seq_cls_infer.cc(93)::CreateRuntimeOption	model_path = ernie-3.0-medium-zh-afqmc/infer.pdmodel, param_path = ernie-3.0-medium-zh-afqmc/infer.pdiparams
[INFO] fastdeploy/runtime.cc(469)::Init	Runtime initialized with Backend::ORT in Device::CPU.
Batch id: 0, example id: 0, sentence 1: 花呗收款额度限制, sentence 2: 收钱码，对花呗支付的金额有限制吗, label: 1, confidence:  0.581852
Batch id: 1, example id: 0, sentence 1: 花呗支持高铁票支付吗, sentence 2: 为什么友付宝不支持花呗付款, label: 0, confidence:  0.997921
```



### Parameter Description

`seq_cls_infer_demo` supports more command-line arguments in addition to the preceding example. The following is a description of every command-line argument:

| Parameter | Parameter Description |
|----------|--------------|
|--model_dir | Specify the directory where the model is to be deployed |
|--batch_size |Maximum measurable batch size, default 1|
|--max_length |Maximum sequence length, default 128|
|--device | Running devices, optional range: ['cpu', 'gpu'], default 'cpu' |
|--backend | Supported inference backend, optional range: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，default'onnx_runtime' |
|--use_fp16 | Whether to use FP16 mode for inference. Enabled when using tensorrt and paddle_tensorrt backend, and default is False |

## Related Documents

[ERNIE 3.0 Model Detailed Instruction](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0)

[ERNIE 3.0 Model Export Method](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0)

[ERNIE 3.0 Model Python Deployment Method](../python/README.md)
