# PaddleSeg Python部署示例

本目录下提供`infer.py`快速完成PaddleSeg模型在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。执行如下脚本即可完成

```bash
# 找到部署包内的模型路径，例如PP_LiteSeg

# 准备一张测试图片，例如test.jpg

# CPU推理
python infer.py --model PP_LiteSeg --image test.jpg --device cpu
# GPU推理
python infer.py --model PP_LiteSeg --image test.jpg --device gpu
# GPU上使用Paddle-TensorRT推理 （注意：Paddle-TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model PP_LiteSeg --image test.jpg --device gpu --use_trt True
```

## 快速链接
- [PaddleSeg python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/semantic_segmentation.html)
