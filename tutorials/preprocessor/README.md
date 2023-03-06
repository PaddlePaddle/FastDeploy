# Preprocessor API

## C++

TODO(guxukai)

---

## Python

Preprocessors library Python API，Currently supported operators are as follows:

- ResizeByShort
- NormalizeAndPermute

### Code Demo：

- [Python Demo](/python)

### Performance comparison between CVCUDA and OpenCV:

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
