English | [简体中文](export_CN.md)
# Diffusion Model Export

The project supports two methods of model export, [PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers) model export and [Diffusers](https://github.com/huggingface/diffusers) model export. Here we introduce each of these two methods. 

## PPDiffusers Model Export

[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers) is a Diffusion Model toolkit that supports cross-modal (e.g., image and speech) training and inference. It builds on the design of [Diffusers](https://github.com/huggingface/diffusers) by the 🤗 Huggingface team, and relies on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) framework and the [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) natural language processing library. The following describes how to use FastDeploy to deploy the Diffusion model provided by PPDiffusers for high performance.

### Dependency Installation

The model export depends on `paddlepaddle`, `paddlenlp` and `ppdiffusers`, which can be installed quickly by running the following command using `pip`.

```shell
pip install -r requirements_paddle.txt
```

### Model Export

___Note: The StableDiffusion model needs to be downloaded during the model export process. In order to use the model and weights, you must accept the License required. Please visit HuggingFace's [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), to read the License carefully, and then sign the agreement.___

___Tips: Stable Diffusion is based on these Licenses: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

You can run the following lines to export model.

```shell
python export_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path stable-diffusion-v1-4
```

The output model directory is as follows:
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

#### Parameters

Here is description of each command line parameter in `export_model.py`.

| Parameter |Description |
|----------|--------------|
|<div style="width: 230pt">--pretrained_model_name_or_path </div> | The diffusion pretrained model provided by ppdiffuers. Default is "CompVis/stable-diffusion-v1-4". For more diffusion pretrained models, please refer to [ppdiffuser model list](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/textual_inversion).|
|--output_path | Exported directory |


## Diffusers Model Export

[Diffusers](https://github.com/huggingface/diffusers) is a Diffusion Model toolkit built by HuggingFace to support cross-modal (e.g. image and speech) training and inference. The underlying model code is available in both a PyTorch implementation and a Flax implementation. This example shows how to use FastDeploy to deploy a PyTorch implementation of Diffusion Model for high performance. 

### Dependency Installation

The model export depends on `onnx`, `torch`, `diffusers` and `transformers`, which can be installed quickly by running the following command using `pip`.

```shell
pip install -r requirements_torch.txt
```

### Model Export

___Note: The StableDiffusion model needs to be downloaded during the model export process. In order to use the model and weights, you must accept the License required, and get the Token granted by HF Hub. Please visit HuggingFace's [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), to read the License carefully, and then sign the agreement.___

___Tips: Stable Diffusion is based on these Licenses: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

If you are exporting a model for the first time, you need to log in to the HuggingFace client first. Run the following command to log in:

```shell
huggingface-cli login
```

After finishing the login, you can run the following lines to export model.

```shell
python export_torch_to_onnx_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path torch_diffusion_model
```

The output model directory is as follows:

```shell
torch_diffusion_model/
├── text_encoder
│   └── inference.onnx
├── unet
│   └── inference.onnx
└── vae_decoder
    └── inference.onnx
```

#### Parameters

Here is description of each command line parameter in  `export_torch_to_onnx_model.py`.

| Parameter |Description |
|----------|--------------|
|<div style="width: 230pt">--pretrained_model_name_or_path </div> |The diffusion pretrained model provided by ppdiffuers, default is "CompVis/stable-diffusion-v1-4". For more diffusion pretrained models, please refer to [HuggingFace model list](https://huggingface.co/CompVis/stable-diffusion-v1-4).|
|--output_path |Exported directory |
