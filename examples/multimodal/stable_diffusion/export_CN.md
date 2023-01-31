简体中文｜[English](export.md)
# Diffusion模型导出教程

本项目支持两种模型导出方式：[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)模型导出以及[Diffusers](https://github.com/huggingface/diffusers)模型导出。下面分别介绍这两种模型导出方式。

## PPDiffusers 模型导出

[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)是一款支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱，其借鉴了🤗 Huggingface团队的[Diffusers](https://github.com/huggingface/diffusers)的优秀设计，并且依托[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)框架和[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)自然语言处理库。下面介绍如何使用FastDeploy将PPDiffusers提供的Diffusion模型进行高性能部署。

### 依赖安装

模型导出需要依赖`paddlepaddle`, `paddlenlp`以及`ppdiffusers`，可使用`pip`执行下面的命令进行快速安装。

```shell
pip install -r requirements_paddle.txt
```

### 模型导出

___注意：模型导出过程中，需要下载StableDiffusion模型。为了使用该模型与权重，你必须接受该模型所要求的License，请访问HuggingFace的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的License，然后签署该协议。___

___Tips: Stable Diffusion是基于以下的License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

可执行以下命令行完成模型导出。

```shell
python export_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path stable-diffusion-v1-4
```

输出的模型目录结构如下：
```shell
stable-diffusion-v1-4/
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
|<div style="width: 230pt">--pretrained_model_name_or_path </div> | ppdiffuers提供的diffusion预训练模型。默认为："CompVis/stable-diffusion-v1-4	"。更多diffusion预训练模型可参考[ppdiffuser模型列表](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/textual_inversion)。|
|--output_path | 导出的模型目录。 |


## Diffusers 模型导出

[Diffusers](https://github.com/huggingface/diffusers)是一款由HuggingFace打造的支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱。其底层的模型代码提供PyTorch实现的版本以及Flax实现的版本两种版本。本示例将介绍如何使用FastDeploy将PyTorch实现的Diffusion模型进行高性能部署。

### 依赖安装

模型导出需要依赖`onnx`, `torch`, `diffusers`以及`transformers`，可使用`pip`执行下面的命令进行快速安装。

```shell
pip install -r requirements_torch.txt
```

### 模型导出

___注意：模型导出过程中，需要下载StableDiffusion模型。为了使用该模型与权重，你必须接受该模型所要求的License，并且获取HF Hub授予的Token。请访问HuggingFace的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的License，然后签署该协议。___

___Tips: Stable Diffusion是基于以下的License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

若第一次导出模型，需要先登录HuggingFace客户端。执行以下命令进行登录：

```shell
huggingface-cli login
```

完成登录后，执行以下命令行完成模型导出。

```shell
python export_torch_to_onnx_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path torch_diffusion_model
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
|<div style="width: 230pt">--pretrained_model_name_or_path </div> | ppdiffuers提供的diffusion预训练模型。默认为："CompVis/stable-diffusion-v1-4	"。更多diffusion预训练模型可参考[HuggingFace模型列表说明](https://huggingface.co/CompVis/stable-diffusion-v1-4)。|
|--output_path | 导出的模型目录。 |
