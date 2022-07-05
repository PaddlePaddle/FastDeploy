# 模型开发

以`ultralytics/yolov5`为例，在`fastdeploy/vision`目录下新增`ultralytics`目录，并创建代码`yolov5.h`

定义`YOLOv5`类

```
class YOLOv5 : public FastDeployModel {
 public:
  // 构造函数指定模型路径，并默认为ONNX格式
  YOLOv5(const std::string& model_file)
    : FastDeployModel(model_file, "", Frontend::ONNX) {
    size = {640, 640}; // 图像预处理resize大小
    // 图像填充值
    padding_value = {114.0, 114.0, 114.0};
    // 是否只填充到满足stride的最小方框即可
    bool is_mini_pad = false;
    // 是否支持图像resize超过原图尺寸
    bool is_scale_up = true;
    // 步长，padding到长宽为stride的倍数
    stride = 32;

    // 通过下面的两个参数，来说明模型在CPU/GPU上支持的后端种类
    // 指定Device后，默认情况下，会优先选择最前的后端
    valid_cpu_backends = {Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  }

  std::string ModelName() const; // 返回模型名

  // 模型初始化, 须在此函数中主动调用基类的`InitBackend()`函数
  // 来初始化runtime
  // 一些模型前后处理的初始化也可在此函数中，如ppdet/ppcls创建一个
  // 数据预处理pipeline
  bool Init();

  // 预处理，其中输入是vision::Mat结构，输出是FDTensor
  // 输出提供给runtime进行推理使用
  bool Preprocess(Mat* mat, FDTensor* output);

  // 后处理，输入是runtime的输入FDTensor
  // 一些跟模型相关的预处理参数
  bool Postprocess(FDTensor& tensor, DetectionResult* res, float conf_thresh, float nms_iou_thresh);

  // 端到端的推理函数，包含前后处理
  // 因此一般也建议将后处理的部分参数放在这个接口中
  bool Predict(cv::Mat* im, DetectionResult* result, float conf_thresh = 0.25, float nms_iou_thresh = 0.5);
};
```

模型的实现上，并没有特别强的规范约束，但是
- 1. 一定要继承`FastDeployModel`
- 2. 确定可用的`valid_cpu_backends`和`valid_gpu_backends`
- 3. 要实现`Init()`/`ModelName()`/`Predict()`三个接口
- 4. 建议统一为`Preprocess`和`Postprocess`两个接口作为前后处理所用


## 其它

在`vision`中，会提供几类基础的数据结构使用，包括`vision::ClassifyResult`、`vision::DetectionResult`、`vision::SegmentationResult`等作为模型常见的输出结构。 但难免会遇到新的输出结构不在这几类中，对于一定要定制化的数据结构，默认按照下面方式处理

- 1. 如果是大量模型通用的结构，仍然实现在`vision/common.h`中，作为通用的输出结构
- 2. 如果只是某个模型需要，则实现在如`vision/ultralytics/yolov5.h`中，同时需要自行为此结构体进行pybind封装
