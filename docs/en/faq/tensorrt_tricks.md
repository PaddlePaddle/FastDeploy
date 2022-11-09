English | [中文](../../cn/faq/tensorrt_tricks.md)

# TensorRT Q&As

## 1. The following log may pop up when running TensorRT 
```
[WARNING] fastdeploy/backends/tensorrt/trt_backend.cc(552)::CreateTrtEngineFromOnnx	Cannot build engine right now, because there's dynamic input shape exists, list as below,
[WARNING] fastdeploy/backends/tensorrt/trt_backend.cc(556)::CreateTrtEngineFromOnnx	Input 0: TensorInfo(name: image, shape: [-1, 3, 320, 320], dtype: FDDataType::FP32)
[WARNING] fastdeploy/backends/tensorrt/trt_backend.cc(556)::CreateTrtEngineFromOnnx	Input 1: TensorInfo(name: scale_factor, shape: [1, 2], dtype: FDDataType::FP32)
[WARNING] fastdeploy/backends/tensorrt/trt_backend.cc(558)::CreateTrtEngineFromOnnx	FastDeploy will build the engine while inference with input data, and will also collect the input shape range information. You should be noticed that FastDeploy will rebuild the engine while new input shape is out of the collected shape range, this may bring some time consuming problem, refer https://github.com/PaddlePaddle/FastDeploy/docs/backends/tensorrt.md for more details.
[INFO] fastdeploy/fastdeploy_runtime.cc(270)::Init	Runtime initialized with Backend::TRT in device Device::GPU.
[INFO] fastdeploy/vision/detection/ppdet/ppyoloe.cc(65)::Initialize	Detected operator multiclass_nms3 in your model, will replace it with fastdeploy::backend::MultiClassNMS(background_label=-1, keep_top_k=100, nms_eta=1, nms_threshold=0.6, score_threshold=0.025, nms_top_k=1000, normalized=1).
[WARNING] fastdeploy/backends/tensorrt/utils.cc(40)::Update	[New Shape Out of Range] input name: image, shape: [1, 3, 320, 320], The shape range before: min_shape=[-1, 3, 320, 320], max_shape=[-1, 3, 320, 320].
[WARNING] fastdeploy/backends/tensorrt/utils.cc(52)::Update	[New Shape Out of Range] The updated shape range now: min_shape=[1, 3, 320, 320], max_shape=[1, 3, 320, 320].
[WARNING] fastdeploy/backends/tensorrt/trt_backend.cc(281)::Infer	TensorRT engine will be rebuilt once shape range information changed, this may take lots of time, you can set a proper shape range before loading model to avoid rebuilding process. refer https://github.com/PaddlePaddle/FastDeploy/docs/backends/tensorrt.md for more details.
[INFO] fastdeploy/backends/tensorrt/trt_backend.cc(416)::BuildTrtEngine	Start to building TensorRT Engine...
```

Most model shapes are dynamic, e.g. the classification model input [-1, 3, 224, 224] indicates that its first batch dimension is dynamic; the detection model input [-1, 3, -1, -1] indicates that its batch dimension, height and width are dynamic. TensorRT needs the range of these dynamic dimensions when building the engine. Therefore FastDeploy solves this problem in two ways

- 1. Automatically set dynamic Shape: If the loaded model contains a dynamic Shape, the TensorRT engine will not be created immediately. The engine will be built after obtaining the Shape data from actual inference data.
- - 1.1 Since most models are inferred with a stable Shape, it just postpones the construction to the inference process, with a limited impact on the whole task.
- - 1.2 If the Shape changes during the inference process, FastDeploy will keep collecting new Shapes to expand the dynamic dimension change range. Each time the model collects an out-of-ranged new Shape, the actual range will be updated in real-time, and it will take some time to rebuild the TensorRT engine, for instance, in the OCR models. With continuous inference, the engine will not be rebuilt after the data range of Shape finally stabilizes. 
- 2. Manually set the dynamic Shape: When developers know the dynamic Shape range before hand, they can set the dynamic range manually, so as to avoid reconstructing during inference.
- - 2.1 Python Interface calls `RuntimeOption.set_trt_input_shape`function. [Python API](https://baidu-paddle.github.io/fastdeploy-api/python/html)
- - 2.2 C++ Interface calls`RuntimeOption.SetTrtInputShape` function.[C++ API](https://baidu-paddle.github.io/fastdeploy-api/cpp/html)


## 2. It takes a long time to load model initialization on TensorRT

It takes a long time for TensorRT to build models. Therefore, FastDeploy provides a Cache mechanism to help developers cache the built models locally the model loading initialization can be completed quickly by loading the saved Cache.

- Python Interface calls`RuntimeOption.set_trt_cache_file` function[Python API](https://baidu-paddle.github.io/fastdeploy-api/python/html)
- C++ Interface calls`RuntimeOption.SetTrtCacheFile` function [C++ API](https://baidu-paddle.github.io/fastdeploy-api/cpp/html)

Interface inputs a file path string, and when the code is executed 

- If the input file path does not exist, the model will build a TensorRT engine. After the construction is completed, the engine will be converted to a binary stream and stored in this file path
- If the input file path does exist, the model will skip building the TensorRT engine and directly load this file and restore it to the TensorRT engine

Therefore, if there is a change in the model, inference configuration (for example, from Float32 to Float16), developers need to delete the local cache file first to avoid errors.


