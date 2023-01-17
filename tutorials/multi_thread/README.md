English | [中文](README_CN.md)

# Usage of FastDeploy model multi-thread or multi-process prediction

FastDeploy provides the following multi-thread or multi-process examples for python and cpp developers

- [Example of using python multi-thread and multi-process prediction](python)
- [Example of using cpp multithreaded prediction](cpp)

## Models that currently support multi-thread and multi-process predictions

| task type           | illustrate      | model download link     |
|:-------------- |:---------------- |:------------------- |
| Detection      | support PaddleDetection series models | [PaddleDetection](../../examples/vision/detection/paddledetection)       |
| Segmentation   | support PaddleSeg series models       | [PaddleSeg](../../examples/vision/segmentation/paddleseg) |
| Classification | support PaddleClas series models      | [PaddleClas](../../examples/vision/classification/paddleclas)   |
| OCR            | support PaddleOCR series models       | [PaddleOCR](../../examples/vision/ocr/)   |

>> **Notice**:
- click the model download link above to download the model from the `Download pre-training model` module
- OCR is a pipeline model. For multi-thread examples, please refer to the `pipeline` folder. Other single-model multi-thread examples are in the `single_model` folder.

## Clone model when using multi-thread prediction

the inference process of vision model is consist of three stages
- load the image, then the image is preprocessed, finally get the Tensor to be input to the model Runtime, that is the preprocess stage
- the model Runtime receives Tensor, do the inference, and obtains the output tensor of Runtime, that is the infer stage
- process the output tensor of Runtime to get the final structured information, such as DetectionResult, SegmentationResult, etc., that is the postprocess stage

For the above three stages: preprocess, inference, and postprocess, FastDeploy abstracted three corresponding classes, namely Preprocessor, Runtime, and PostProcessor

When using FastDeploy for multi-thread inference, several issues should be considered
- Can the Preprocessor, Runtime, and Postprocessor support parallel processing respectively?
- 在支持多线程并发的前提下，能否最大限度的减少内存或显存占用
- Under the premise of supporting multi-thread concurrency, can the memory or video memory usage be minimized?

FastDeploy adopts the method of copying multiple objects separately for multi-thread inference, so each thread has an independent instance of Preprocessor, Runtime, and PostProcessor. In order to reduce the memory usage, the Runtime adopt sharing the model weights copy method. In this way, the memory usage caused by copying multiple objects is reduced.

FastDeploy provides the following interface to clone the model (take PaddleClas as an example)

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

>> **Notice**:Other models API refer to[官方C++文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/index.html) and [官方Python文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/index.html)

## Python multi-thread and multi-process

Due to language limitations, Python has the existence of GIL lock. In computing-intensive scenarios, multithreading cannot make full use of hardware resources. Therefore, two examples of multi-process and multi-thread are provided on Python. The similarities and differences are as follows:

### Comparison of multi-process and multi-thread inference in FastDeploy model

|     | resource usage | computationally intensive | I/O intensive | inter-process or inter-thread communication |
|:-------|:------|:----------|:----------|:----------|
| multi-process   | large | fast | fast | slow |
| multi-thread   | little | slow | relatively fast |fast|

>> **注意**: The above analysis is a theoretical analysis. In fact, Python has also made certain optimizations for different computing tasks. For example, the calculation of numpy can already be computed by multi-thread parallelly. In addition, the result aggregation between multiple processes involves time-consuming operation(inter-process communication), Besides, it is difficult to identify whether the task is computationally intensive or I/O intensive, so everything needs to be tested according to the task.



## C++ multi-thread

The C++ multi-thread has the characteristics of occupying less resources and high speed.Therefore, multi-threaded inference is the best choice in C++

### C++ comparition between multi-thread Clone and not Clone memory occupation

硬件：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz  
模型：ResNet50_vd_infer  
后端：CPU OPENVINO Backend

memory occupation of initializing multiple models in a single process
| number of models | after model.Clone() | after model->predict() with model.Clone()    | initializing model without model.Clone()| after model->predict() without model.Clone() |
|:--- |:----- |:----- |:----- |:----- |
|1|322M |325M |322M|325M|
|2|322M|325M|559M|560M|
|3|322M|325M|771M|771M|

memory occupation of multi-thread
| thread number | after model.Clone() | after model->predict() with model.Clone()    | initialize model without model.Clone() | after model->predict() without model.Clone() |
|:--- |:----- |:----- |:----- |:----- |
|1|322M |337M |322M|337M|
|2|322M|343M|548M|566M|
|3|322M|347M|752M|784M|
