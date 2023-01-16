English |  [简体中文](README_CN.md)
# FastDeploy Diffusion Model High-Performance Deployment

This document completes the high-performance deployment of the Diffusion model with ⚡️`FastDeploy`, based on `DiffusionPipeline` in project [Diffusers](https://github.com/huggingface/diffusers) designed by Huggingface. 

### Preperation for Deployment

This example needs the deployment model after exporting the training model. Here are two ways to obtain the deployment model:

- Methods for model export. Please refer to [Model Export](export.md) to export deployment model.
- Download the deployment model. To facilitate developers to test the example, we have pre-exported some of the `Diffusion` models, so you can just download models and test them quickly:

| Model | Scheduler |
|----------|--------------|
| [CompVis/stable-diffusion-v1-4](https://bj.bcebos.com/fastdeploy/models/stable-diffusion/CompVis/stable-diffusion-v1-4.tgz) | PNDM |
| [runwayml/stable-diffusion-v1-5](https://bj.bcebos.com/fastdeploy/models/stable-diffusion/runwayml/stable-diffusion-v1-5.tgz) | EulerAncestral |

## Environment Dependency

In the example, the word splitter in CLIP model of PaddleNLP is required, so you need to run the following line to install the dependency.

```shell
pip install paddlenlp paddlepaddle-gpu
```

### Quick Experience

We are ready to start testing after model deployment. Here we will specify the model directory as well as the inference engine backend, and run the `infer.py` script to complete the inference.

```
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle
```

The image file is fd_astronaut_rides_horse.png. An example of the generated image is as follows (the generated image is different each time, the example is for reference only):

![fd_astronaut_rides_horse.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

If the stable-diffusion-v1-5 model is used, you can run these to complete the inference.

```
# Inference on GPU
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle

# Inference on KunlunXin XPU
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle-kunlunxin
```

#### Parameters

`infer.py` supports more command line parameters than the above example. The following is a description of each command line parameter.

| Parameter |Description |
|----------|--------------|
| --model_dir | Directory of the exported model. |
| --model_format | Model format. Default is `'paddle'`, optional list: `['paddle', 'onnx']`. |
| --backend | Inference engine backend. Default is`paddle`, optional list: `['onnx_runtime', 'paddle', 'paddle-kunlunxin']`, when the model format is `onnx`, optional list is`['onnx_runtime']`. |
| --scheduler | Scheduler in StableDiffusion model. Default is`'pndm'`, optional list `['pndm', 'euler_ancestral']`. The scheduler corresponding to the StableDiffusio model can be found in [ppdiffuser model list](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/textual_inversion).|
| --unet_model_prefix | UNet model prefix, default is `unet`. |
| --vae_model_prefix | VAE model prefix, defalut is `vae_decoder`. |
| --text_encoder_model_prefix | TextEncoder model prefix, default is `text_encoder`. |
| --inference_steps | Running times of UNet model, default is 100. |
| --image_path | Path to the generated images, defalut is `fd_astronaut_rides_horse.png`.  |
| --device_id | gpu id. If `device_id` is -1, cpu is used for inference. |
| --use_fp16 | Indicates if fp16 is used, default is `False`. Can be set to `True` when using tensorrt or paddle-tensorrt backend. |
