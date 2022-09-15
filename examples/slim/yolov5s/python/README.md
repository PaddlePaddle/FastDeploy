# YOLOv5s量化模型 Python部署示例
本目录下提供`infer.py`快速完成YOLOv5s量化模型在CPU/GPU上的部署推理加速. 执行以下脚本即可完成.

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)



```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/slim/yolov5/python

#下载FastDeloy提供的yolov5s量化模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# 在CPU上使用ONNXRuntime推理量化模型
python infer.py --model yolov5s_quant --image 000000014439.jpg --device cpu --backend ort
# 在CPU上使用Paddle-Inference推理量化模型
python infer.py --model yolov5s_quant --image 000000014439.jpg --device cpu --backend paddle
# 在GPU上使用TensorRT推理量化模型
python infer.py --model yolov5s_quant --image 000000014439.jpg --device gpu --backend trt
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184309358-d803347a-8981-44b6-b589-4608021ad0f4.jpg">
