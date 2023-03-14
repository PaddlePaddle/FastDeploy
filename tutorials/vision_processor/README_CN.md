中文 | [English](README.md)
# 多硬件图像处理库

多硬件图像处理库（Vision Processor）可用于实现模型的预处理、后处理等图像操作，底层封装了多个第三方图像处理库，包括：
- OpenCV，用于通用CPU图像处理
- FlyCV，主要针对ARM CPU加速
- CV-CUDA，用于NVIDIA GPU

## C++

待编写

## Python

Python API目前支持的算子如下：

- ResizeByShort
- NormalizeAndPermute

用户可通过继承PyProcessorManager类，实现自己的图像处理模块。基类PyProcessorManager实现了GPU内存管理、CUDA stream管理等，用户仅需要实现apply()函数，在其中调用多硬件图像处理库中的算子、实现处理逻辑即可，具体实现可参考示例代码。

### 示例代码

- [Python示例](python)

### CV-CUDA与OpenCV性能对比

CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

GPU: T4

CUDA: 11.6

Processing logic: Resize -> NormalizeAndPermute

Warmup 100 rounds，tested 1000 rounds and get avg. latency.

| Input Image Shape | Target shape | Batch Size | OpenCV | CV-CUDA | Gain |
| ----------- | -- | ---------- | ------- | ------ | ------ |
| 1920x1080   | 640x360 | 1 | 1.1572ms | 0.9067ms | 16.44% |
| 1280x720    | 640x360 | 1 | 2.7551ms | 0.5296ms | 80.78% |
| 360x240     | 640x360 | 1 | 3.3450ms | 0.2421ms | 92.76% |
