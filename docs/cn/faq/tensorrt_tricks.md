# TensorRT使用问题

## 1. 运行TensorRT过程中，出现如下日志提示 
```bash
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

大部分模型会存在动态Shape，例如分类的输入为[-1, 3, 224, 224]，表示其第一维（batch维）是动态的； 检测的输入[-1, 3, -1, -1]，表示其batch维，以及高和宽是动态的。 而TensorRT在构建引擎时，需要知道这些动态维度的范围。 因此FastDeploy通过以下两种方式来解决

- 1. 自动设置动态Shape; 在加载模型时，如若遇到模型包含动态Shape，则不会立刻创建TensorRT引擎，而是在实际输入数据预测时，获取到数据的Shape，再进行构建。
- - 1.1 由于大部分模型在推理时，Shape都不会变，因此相当于只是将构建的过程推迟到预测阶段，整体没太大影响；
- - 1.2 如若预测过程中，Shape在变化，FastDeploy会不断收集新的Shape，扩大动态维度的变化范围。每次遇到新的Shape且超出范围的，则更新范围，并重新构建TensorRT引擎。 因此这样在遇到超过范围的Shape时，会重新花一定时间构建引擎，例如OCR模型存在这种现象，但随着不断预测，数据的Shape范围最终稳定后，便不会再重新构建。
- 2. 手动设置动态Shape；当知道模型存在动态Shape，先手动设置好其动态范围，这样可以避免预测时重新构建
- - 2.1 Python接口调用`RuntimeOption.set_trt_input_shape`函数。 [Python API文档](https://baidu-paddle.github.io/fastdeploy-api/python/html)
- - 2.2 C++接口调用`RuntimeOption.SetTrtInputShape`函数。[C++ API文档](https://baidu-paddle.github.io/fastdeploy-api/cpp/html)


## 2. 每次运行TensorRT，加载模型初始化耗时长

TensorRT每次构建模型的过程较长，FastDeploy提供了Cache机制帮助开发者将构建好的模型缓存在本地，这样在重新运行代码时，可以通过加载Cache，快速完成模型的加载初始化。

- Python接口调用`RuntimeOption.set_trt_cache_file`函数。[Python API文档](https://baidu-paddle.github.io/fastdeploy-api/python/html)
- C++接口调用`RuntimeOption.SetTrtCacheFile`函数。 [C++ API文档](https://baidu-paddle.github.io/fastdeploy-api/cpp/html)

接口传入文件路径字符串，当在执行代码时，
- 如若发现传入的文件路径不存在，则会构建TensorRT引擎，在构建完成后，将引擎转换为二进制流存储到此文件路径
- 如若发现传入的文件路径存在，则会跳过构建TensorRT引擎，直接加载此文件并还原成TensorRT引擎

因此，如若有修改模型，推理配置（例如Float32改成Float16)，需先删除本地的cache文件，避免出错。


