# PaddleClas CPU-GPU Python部署示例

本目录下提供`infer.py`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

## 1. 运行部署示例

```bash
# 找到部署包内的模型路径，例如ResNet50

# 准备一张测试图片，例如test.jpg

# 在CPU上使用Paddle Inference推理
python infer.py --model ResNet50 --image test.jpg --device cpu --backend paddle --topk 1
# 在CPU上使用OpenVINO推理
python infer.py --model ResNet50 --image test.jpg --device cpu --backend openvino --topk 1
# 在CPU上使用ONNX Runtime推理
python infer.py --model ResNet50 --image test.jpg --device cpu --backend ort --topk 1
# 在CPU上使用Paddle Lite推理
python infer.py --model ResNet50 --image test.jpg --device cpu --backend pplite --topk 1
# 在GPU上使用Paddle Inference推理
python infer.py --model ResNet50 --image test.jpg --device gpu --backend paddle --topk 1
# 在GPU上使用Paddle TensorRT推理
python infer.py --model ResNet50 --image test.jpg --device gpu --backend pptrt --topk 1
# 在GPU上使用ONNX Runtime推理
python infer.py --model ResNet50 --image test.jpg --device gpu --backend ort --topk 1
# 在GPU上使用Nvidia TensorRT推理
python infer.py --model ResNet50 --image test.jpg --device gpu --backend trt --topk 1
```

## 2. 部署示例选项说明  

|参数|含义|默认值
|---|---|---|  
|--model|指定模型文件夹所在的路径|None|
|--image|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu|
|--device_id|使用gpu时, 指定设备号|0|
|--backend|部署模型时使用的后端, 支持的值为`[paddle,pptrt,pplite,ort,openvino,trt]` |openvino|
|--topk|返回的前topk准确率, 支持的为`1,5` |1|

## 3. 更多指南
- [PaddleClas系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/image_classification.html)
