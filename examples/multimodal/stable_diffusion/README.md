# Diffusion模型高性能部署

本部署示例基于Huggingface团队的[Diffusers](https://github.com/huggingface/diffusers)项目设计的`Diffusion Pipeline`基础上，使用FastDeploy完成Diffusion模型的高性能部署。本项目支持两种部署方式：[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)模型部署以及[Diffusers](https://github.com/huggingface/diffusers)模型部署。

## PPDiffusers 模型部署

[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)是一款支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱，其借鉴了🤗 Huggingface团队的[Diffusers](https://github.com/huggingface/diffusers)的优秀设计，并且依托[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)框架和[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)自然语言处理库。下面介绍如何使用FastDeploy将PPDiffusers提供的Diffusion模型进行高性能部署。

### 依赖安装

模型导出需要依赖`paddlepaddle`, `paddlenlp`以及`ppdiffusers`，可使用`pip`执行下面的命令进行快速安装。

```shell
pip install -r requirements_paddle.txt
```

### 模型导出

可直接执行执行以下命令行完成模型导出。

```shell
python export_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path paddle_diffusion_model
```

输出的模型目录结构如下：
```shell
paddle_diffusion_model/
├── text_encoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── unet
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── vae_decoder
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

#### 参数说明

`export_model.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--pretrained_model_name_or_path | ppdiffuers提供的diffusion预训练模型。默认为："CompVis/stable-diffusion-v1-4	"。更多diffusion预训练模型可参考[ppdiffuser模型列表](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/textual_inversion)。|
|--output_path | 导出的模型目录。 |

### 预测运行


## Diffusers 模型部署

[Diffusers](https://github.com/huggingface/diffusers)是一款由HuggingFace打造的支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱。其底层的模型代码提供PyTorch实现的版本以及Flax实现的版本两种版本。本示例将介绍如何使用FastDeploy将PyTorch实现的Diffusion模型进行高性能部署。

### 依赖安装

模型导出需要依赖`onnx`, `torch`, `diffusers`以及`transformers`，可使用`pip`执行下面的命令进行快速安装。

```shell
pip install -r requirements_torch.txt
```

### 模型导出

可直接执行执行以下命令行完成模型导出。

```shell
python export_torch_to_onnx_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path paddle_diffusion_model
```

输出的模型目录结构如下：

```shell
torch_diffusion_model/
├── text_encoder
│   └── inference.onnx
├── unet
│   └── inference.onnx
└── vae_decoder
    └── inference.onnx
```

#### 参数说明

`export_torch_to_onnx_model.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--pretrained_model_name_or_path | ppdiffuers提供的diffusion预训练模型。默认为："CompVis/stable-diffusion-v1-4	"。更多diffusion预训练模型可参考[HuggingFace模型列表说明](https://huggingface.co/CompVis/stable-diffusion-v1-4)。|
|--output_path | 导出的模型目录。 |

### 预测运行
