[English](README_EN.md) | 简体中文 
# ERNIE 3.0 Model Deployment

## Model Description
- [PaddleNLP ERNIE 3.0 model description](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0)

## A List of Supported Models

| Model |  Structure  | Language |
| :---: | :--------: | :--------: |
| `ERNIE 3.0-Base`| 12-layers, 768-hidden, 12-heads | Chinese |
| `ERNIE 3.0-Medium`| 6-layers, 768-hidden, 12-heads | Chinese |
| `ERNIE 3.0-Mini`| 6-layers, 384-hidden, 12-heads | Chinese |
| `ERNIE 3.0-Micro`| 4-layers, 384-hidden, 12-heads | Chinese |
| `ERNIE 3.0-Nano `| 4-layers, 312-hidden, 12-heads | Chinese |

## A List of Supported NLP Tasks

| Task |  Yes or No  |
| :--------------- | ------- |
| text classification | ✅ |
|  sequence labeling  | ❌ |

## Export Deployment Models

Before deployment, it is required to export trained ERNIE models into deployment models. The export steps can be found in the document [Export Model](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-3.0).

## Download Fine-tuning Models

### Classification Task

For developers' testing, the ERNIE 3.0-Medium Model fine-tuned on the text classification [AFQMC dataset](https://bj.bcebos.com/paddlenlp/datasets/afqmc_public.zip) is provided below. Developers can download it directly.

- [ERNIE 3.0 Medium AFQMC](https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc.tgz)

## Detailed Deployment Documents 

- [Python Deployment](python)
- [C++ Deployment](cpp)
- [Serving Deployment](serving)
