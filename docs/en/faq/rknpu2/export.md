English | [中文](../../../cn/faq/rknpu2/export.md) 

# Export Model

## Introduction

Fastdeploy has simply integrated the onnx->rknn conversion process. In this instruction, we first write yaml configuration files, then export models in `tools/export.py`.
Before you start the conversion, please check if the environment is installed successfully referring to [RKNN-Toolkit2 Installation](./install_rknn_toolkit2.md).


## Configuration Parameter in export.py

| Parameter            | Whether it can be NULL    |  Parameter Role               |
|-----------------|------------|--------------------|
| verbose         | Y(DEFAULT=TRUE) | Decide whether to output specific information when converting |
| config_path     | N          | Path to configuration file             |

## Config File Introduction

### Module of config yaml file

```yaml
model_path: ./portrait_pp_humansegv2_lite_256x144_pretrained.onnx
output_folder: ./
target_platform: RK3588
normalize:
  mean: [[0.5,0.5,0.5]]
  std: [[0.5,0.5,0.5]]
outputs: None
```

### Config parameters
* model_path: Model saving path.
* output_folder: Model saving folder name.
* target_platform: The device model runs on, only RK3588 or RK3568 can be chosen.
* normalize: Configure the normalize operation on NPU with two parameters std and mean.
  * std: If you do the normalize operation externally, please configure to [1/255,1/255,1/255].
  * mean: If you do the normalize operation externally, please configure to [0,0,0].
* outputs: Output node list, if you use default output node, please configure to None.

## How to convert model
Run the line in the root directory:

```bash
python tools/export.py  --config_path=./config.yaml
```

## Things to note in Model Export

* Please don't export models with softmax or argmax, calculate them externally instead.