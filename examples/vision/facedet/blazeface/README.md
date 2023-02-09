English | [简体中文](README_CN.md)
# BlazeFace Ready-to-deploy Model

- BlazeFace deployment model implementation comes from [BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection),and [Pre-training model based on WiderFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection)
  - （1）Provided in [Official library
](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/tools) *.params, could deploy after operation [export_model.py](#Export PADDLE model);
  - （2）Developers can train BlazeFace model based on their own data according to [export_model. py](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/tools/export_model.py)After exporting the model, complete the deployment。

## Export PADDLE model

Visit [BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection) Github library, download and install according to the instructions, download the `. yml` and `. params` model parameters, and use` export_ Model. py `gets the` pad `model file`. yml,. pdiparams,. pdmodel `.


* Download BlazeFace model parameter file

|Network structure | input size | number of pictures/GPU | learning rate strategy | Easy/Media/Hard Set | prediction delay (SD855) | model size (MB) | download | configuration file|
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|:----------:|:---------:|:--------:|
| BlazeFace  | 640  |    8    | 1000e     | 0.885 / 0.855 / 0.731 | - | 0.472 |[Download link](https://paddledet.bj.bcebos.com/models/blazeface_1000e.pdparams) | [Config file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection/blazeface_1000e.yml) |
| BlazeFace-FPN-SSH  | 640  |    8    | 1000e     | 0.907 / 0.883 / 0.793 | - | 0.479 |[Download link](https://paddledet.bj.bcebos.com/models/blazeface_fpn_ssh_1000e.pdparams) | [Config file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection/blazeface_fpn_ssh_1000e.yml) |

* Export paddle-format file
  ```bash
  python tools/export_model.py -c configs/face_detection/blazeface_1000e.yml -o weights=blazeface_1000e.pdparams --export_serving_model=True
  ```

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- This tutorial and related code are written based on [BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/face_detection)
