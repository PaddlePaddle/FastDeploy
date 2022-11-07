# FastDeploy Diffusion模型高性能部署

本部署示例使用⚡️`FastDeploy`在Huggingface团队的[Diffusers](https://github.com/huggingface/diffusers)项目设计的`Diffusion Pipeline`基础上，完成Diffusion模型的高性能部署。本项目支持两种部署方式：[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)模型部署以及[Diffusers](https://github.com/huggingface/diffusers)模型部署。

## PPDiffusers 模型部署

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
|<div style="width: 230pt">--pretrained_model_name_or_path </div> | ppdiffuers提供的diffusion预训练模型。默认为："CompVis/stable-diffusion-v1-4	"。更多diffusion预训练模型可参考[ppdiffuser模型列表](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/textual_inversion)。|
|--output_path | 导出的模型目录。 |

### 预测运行

经过上述模型导出步骤，将模型导出到`paddle_diffusion_model`目录。下面将指定模型目录以及推理引擎后端，运行`infer.py`脚本，完成推理。

```
python infer.py --model_dir paddle_diffusion_model/ --backend paddle
```

得到的图像文件为fd_astronaut_rides_horse.png。生成的图片示例：

![fd_astronaut_rides_horse.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

#### 参数说明

`infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --backend | 推理引擎后端。默认为`paddle`, 可选列表：`['onnx_runtime', 'tensorrt', 'paddle', 'paddle-tensorrt']`。 |
| --model_format | 模型格式。默认为`'paddle'`, 可选列表：`['paddle', 'onnx']`。 |
| --unet_model_prefix | UNet模型前缀。默认为`unet`。 |
| --vae_model_prefix | VAE模型前缀。默认为`vae_decoder`。 |
| --text_encoder_model_prefix | TextEncoder模型前缀。默认为`text_encoder`。 |
| --inference_steps | UNet模型运行的次数，默认为100。 |
| --image_path | 生成图片的路径。默认为`fd_astronaut_rides_horse.png`。  |
| --device_id | gpu设备的id。若`device_id`为-1，视为使用cpu推理。 |
| --use_fp16 | 是否使用fp16精度。默认为`False`。使用tensorrt或者paddle-tensorrt后端时可以设为`True`开启。 |

## Diffusers 模型部署

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

### 预测运行

经过上述模型导出步骤，将模型导出到`torch_diffusion_model`目录。下面将指定模型目录、推理引擎后端以及模型格式，运行`infer.py`脚本，完成推理。

```
python infer.py --model_dir torch_diffusion_model/ --backend onnx_runtime --model_format onnx
```

由于导出的模型为ONNX格式模型，所以部署Diffusers模型时仅能指定后端为ONNX Runtime或者TensorRT，并且需要将`model_format`指定为onnx。推理后得到的图像文件为fd_astronaut_rides_horse.png。生成的图片示例：

![fd_astronaut_rides_horse.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

#### 参数说明

`infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --backend | 推理引擎后端。默认为`paddle`, 可选列表：`['onnx_runtime', 'tensorrt']`。 |
| --model_format | 模型格式。默认为`'paddle'`, 可选列表：`['paddle', 'onnx']`。 |
| --unet_model_prefix | UNet模型前缀。默认为`unet`。 |
| --vae_model_prefix | VAE模型前缀。默认为`vae_decoder`。 |
| --text_encoder_model_prefix | TextEncoder模型前缀。默认为`text_encoder`。 |
| --inference_steps | UNet模型运行的次数，默认为100。 |
| --image_path | 生成图片的路径。默认为`fd_astronaut_rides_horse.png`。  |
| --device_id | gpu设备的id。若`device_id`为-1，视为使用cpu推理。 |
| --use_fp16 | 是否使用fp16精度。默认为`False`。使用tensorrt或者paddle-tensorrt后端时可以设为`True`开启。 |
