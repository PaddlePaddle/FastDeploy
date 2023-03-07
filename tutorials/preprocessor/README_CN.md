[English](README.md) | 中文
# Preprocessor API

## C++

待编写

---

## Python

预处理库Python API，目前支持的算子如下：

- ResizeByShort
- NormalizeAndPermute

### 示例代码：

- [Python示例](python)

### CVCUDA与OpenCV性能对比：

CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

GPU: T4

CUDA：11.6

Backend：ORT

Used OP：Resize to 640x360 -> NormalizeAndPermute

Warmup 100 rounds，tested 1000 rounds and get avg. latency.

| Image Shape | Batch Size | OpenCV               | CVCUDA               | Gain   |
| ----------- | ---------- | -------------------- | -------------------- | ------ |
| 1920x1080   |    1       | 1.1572ms             |     0.9067ms         | 16.44% |
| 1280x720    | 1          | 2.7551ms             | 0.5296ms             | 80.78% |
| 360x240     |  1         | 3.3450ms             |   0.2421ms           | 92.76% |
