[简体中文](README_CN.md) | English

# FastDeploy Serving Deployment

## Introduction

FastDeploy builds an end-to-end serving deployment based on [Triton Inference Server](https://github.com/triton-inference-server/server). The underlying backend uses the FastDeploy high-performance Runtime module and integrates the FastDeploy pre- and post-processing modules to achieve end-to-end serving deployment. It can achieve fast deployment with easy-to-use process and excellent performance.

> FastDeploy also provides an easy-to-use Python service deployment method, refer [PaddleSeg deployment example](../examples/vision/segmentation/paddleseg/serving/simple_serving) for its usage.

## Prepare the environment

### Environment requirements

- Linux
- If using a GPU image, NVIDIA Driver >= 470 is required (for older Tesla architecture GPUs, such as T4, the NVIDIA Driver can be 418.40+, 440.33+, 450.51+, 460.27+)

### Obtain Image

#### CPU Image

CPU images only support Paddle/ONNX models for serving deployment on CPUs, and supported inference backends include OpenVINO, Paddle Inference, and ONNX Runtime

```shell
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.4-cpu-only-21.10
```

#### GPU Image

GPU images support Paddle/ONNX models for serving deployment on GPU and CPU, and supported inference backends including OpenVINO, TensorRT, Paddle Inference, and ONNX Runtime

```
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.4-gpu-cuda11.4-trt8.5-21.10
```

Users can also compile the image by themselves according to their own needs, referring to the following documents:

- [FastDeploy Serving Deployment Image Compilation](docs/EN/compile-en.md)

## Other Tutorials

- [How to Prepare Serving Model Repository](docs/EN/model_repository-en.md)
- [Serving Deployment Configuration for Runtime](docs/EN/model_configuration-en.md)
- [Demo of Serving Deployment](docs/EN/demo-en.md)
- [Client Access Instruction](docs/EN/client-en.md)
- [Serving deployment visualization](docs/EN/vdl_management-en.md)


### Serving Deployment Demo

| Task | Model  |
|---|---|
| Classification | [PaddleClas](../examples/vision/classification/paddleclas/serving/README.md) |
| Detection | [PaddleDetection](../examples/vision/detection/paddledetection/serving/README.md) |
| Detection | [ultralytics/YOLOv5](../examples/vision/detection/yolov5/serving/README.md) |
| NLP |	[PaddleNLP/ERNIE-3.0](../examples/text/ernie-3.0/serving/README.md)|
| NLP |	[PaddleNLP/UIE](../examples/text/uie/serving/README.md)|
| Speech |	[PaddleSpeech/PP-TTS](../examples/audio/pp-tts/serving/README.md)|
| OCR |	[PaddleOCR/PP-OCRv3](../examples/vision/ocr/PP-OCRv3/serving/README.md)|
