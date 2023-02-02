[English](README.md) | 中文

# FastDeploy模型多线程或多进程预测的使用

FastDeploy针对python和cpp开发者，提供了以下多线程或多进程的示例

- [python多线程以及多进程预测的使用示例](python)
- [cpp多线程预测的使用示例](cpp)

## 目前支持多线程以及多进程预测的模型

| 任务类型           | 说明                                  | 模型下载链接                                                                          |
|:-------------- |:----------------------------------- |:-------------------------------------------------------------------------------- |
| Detection      | 支持PaddleDetection系列模型 | [PaddleDetection](../../examples/vision/detection/paddledetection)       |
| Segmentation   | 支持PaddleSeg系列模型          | [PaddleSeg](../../examples/vision/segmentation/paddleseg) |
| Classification | 支持PaddleClas系列模型             | [PaddleClas](../../examples/vision/classification/paddleclas)   |
| OCR | 支持PaddleOCR系列模型             | [PaddleOCR](../../examples/vision/ocr/)   |
>> **注意**:
- 点击上方模型下载链接，至`下载预训练模型`模块下载模型
- OCR是多模型串联的模型，多线程示例请参考`pipeline`文件夹，其他单模型多线程示例在`single_model`文件夹中

## 多线程预测时克隆模型

针对一个视觉模型的推理包含3个环节
- 输入图像，图像经过预处理，最终得到要输入给模型Runtime的Tensor，即preprocess阶段
- 模型Runtime接收Tensor，进行推理，得到Runtime的输出Tensor，即infer阶段
- 对Runtime的输出Tensor做后处理，得到最后的结构化信息，如DetectionResult, SegmentationResult等等，即postprocess阶段

针对以上preprocess、infer、postprocess三个阶段，FastDeploy分别抽象出了三个对应的类，即Preprocessor、Runtime、PostProcessor

在多线程调用FastDeploy中的模型进行并行推理的时候，要考虑几个问题
- Preprocessor、Runtime、Postprocessor三个类能否分别支持并行处理
- 在支持多线程并发的前提下，能否最大限度的减少内存或显存占用

FastDeploy采用分别拷贝多个对象的方式，进行多线程推理，即每个线程都有一份独立的Preprocessor、Runtime、PostProcessor的实例化的对象。而为了减少内存的占用，对于Runtime的拷贝则采用共享模型权重的方式进行拷贝。因此，虽然复制了多个对象，但对于模型权重和参数在内存或显存中只有一份。
以此减少拷贝多个对象带来的内存占用。

FastDeploy提供如下接口，来进行模型的clone(以PaddleClas为例)

- Python: `PaddleClasModel.clone()`
- C++: `PaddleClasModel::Clone()`


### Python
```
import fastdeploy as fd
option = fd.RuntimeOption()
model = fd.vision.classification.PaddleClasModel(model_file,
                                                 params_file,
                                                 config_file,
                                                 runtime_option=option)
model2 = model.clone()
im = cv2.imread(image)
res = model.predict(im)
```

### C++
```
auto model = fastdeploy::vision::classification::PaddleClasModel(model_file,
                                                                 params_file,
                                                                 config_file,
                                                                 option);
auto model2 = model.Clone();
auto im = cv::imread(image_file);
fastdeploy::vision::ClassifyResult res;
model->Predict(im, &res)
```

>> **注意**:其他模型类似API接口可查阅[官方C++文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/index.html)以及[官方Python文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/index.html)

## Python多线程以及多进程

Python由于语言的限制即GIL锁的存在，在计算密集型的场景下，多线程无法充分利用硬件的性能。因此，Python上提供多进程和多线程两种示例。其异同点如下：

### FastDeploy模型多进程与多线程推理的比较

|     | 资源占用 | 计算密集型 | I/O密集型 | 进程或线程间通信 |
|:-------|:------|:----------|:----------|:----------|
| 多进程   | 大 | 快 | 快 | 慢|
| 多线程   | 小 | 慢 | 较快 |快|

>> **注意**:以上分析相对理论，实际上Python针对不同的计算任务也做出了一定的优化，像是numpy类的计算已经可以做到并行计算，同时由于多进程间的result汇总涉及到进程间通信，而且往往有时候很难鉴别该任务是计算密集型还是I/O密集型，所以一切都需要根据任务进行测试而定。


## C++多线程

C++的多线程，兼具了占用资源少，速度快的特点。因此，是使用多线程推理的最佳选择

### C++ 多线程Clone与不Clone内存占用对比

硬件：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz  
模型：ResNet50_vd_infer  
后端：CPU OPENVINO后端推理引擎

单进程内初始化多个模型，内存占用
| 模型数 | model.Clone()后 | Clone后model->predict()后    | 不Clone模型初始化后| 不Clone后model->predict()后 |
|:--- |:----- |:----- |:----- |:----- |
|1|322M |325M |322M|325M|
|2|322M|325M|559M|560M|
|3|322M|325M|771M|771M|

模型多线程预测内存占用
| 线程数 | model.Clone()后 | Clone后model->predict()后    | 不Clone模型初始化后| 不Clone后model->predict()后 |
|:--- |:----- |:----- |:----- |:----- |
|1|322M |337M |322M|337M|
|2|322M|343M|548M|566M|
|3|322M|347M|752M|784M|
