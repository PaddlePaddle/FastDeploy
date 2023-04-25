English | [中文](README_CN.md)

# Vision Processor

Vision Processor is used to implement model preprocessing, postprocessing, etc. The following 3rd party vision libraries are integrated:
- OpenCV, general CPU image processing
- FlyCV, mainly optimized for ARM CPU
- CV-CUDA, for NVIDIA GPU

## C++

C++ API, Currently supported operators are as follows:

- Cast
- CenterCrop
- HWC2CHW
- Resize
- ResizeByShort
- NormalizeAndPermute
- Normalize
- Pad
- PadToSize
- StridePad

Users can inherit `ProcessorManager` when creating a `Preprocessor` class in the C++ deployment of the visual class model, and can choose to use OpenCV or CV-CUDA through `UseCuda()` in the `ProcessorManager` base class. The base class `ProcessorManager` implements GPU memory management, CUDA stream management, etc. Users only need to implement the `Apply()` function, in which operators in the multi-hardware image processing library are called to implement processing logic. For specific implementation, please refer to the sample code.

## Python

Python API, Currently supported operators are as follows:

- Cast
- CenterCrop
- HWC2CHW
- Resize
- ResizeByShort
- NormalizeAndPermute
- Normalize
- Pad
- PadToSize
- StridePad

Users can implement a image processing modules by inheriting the `PyProcessorManager` class. The base class `PyProcessorManager` implements GPU memory management, CUDA stream management, etc. Users only need to implement the apply() function by calling vision processors in this library and implements processing logic. For specific implementation, please refer to the demo code.

### Demo

- [Python Demo](python)
- [C++ Demo](cpp)

### Performance comparison between CV-CUDA and OpenCV:

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
