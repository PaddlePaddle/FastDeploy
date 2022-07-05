# Vision

Vision是FastDeploy中的视觉模型模块，包含`processors`和`utils`两个公共模块，以及模型模块。

## processors 图像处理模块

`processors`提供了常见的图像处理操作，并为各操作实现不同的后端，如当前支持的CPU以及GPU两种处理方式，在模型中预算中，开发者调用`processors`提供的API，即可快速在不同的处理后端进行切换。

默认在CPU上进行处理
```
namespace vis = fastdeploy::vision;

im = cv2.imread("test.jpg");

vis::Mat mat(im);
assert(vis::Resize::Run(&mat, 224, 224));
assert(vis::Normalize::Run(&mat, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}));
assert(vis::HWC2CHW::Run(&mat));
```

切换为CUDA GPU进行处理
```
namespace vis = fastdeploy::vision;
vis::Processor::default_lib = vis::ProcessorLib::OPENCV_CUDA;

im = cv2.imread("test.jpg");

vis::Mat mat(im);
assert(vis::Resize::Run(&mat, 224, 224));
assert(vis::Normalize::Run(&mat, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}));
assert(vis::HWC2CHW::Run(&mat));
```

在处理过程中，通过`fastdeploy::vision::Mat`作为传递的数据结构
```
struct Mat {
  Mat(cv::Mat); // 通过`cv::Mat`进行构造
  FDDataType Type(); // 数值类型
  int Channels(); // 通道数
  int Width(); // 宽
  int Height(); // 高

  // 获取图像，如Mat在GPU上，则会拷贝到CPU上再返回
  cv::Mat GetCpuMat();

  // 获取图像，如Mat在CPU上，则会拷贝到GPU上再返回
  cv::cuda::GpuMat GetGpuMat();

  void ShareWithTensor(FDTensor* tensor); // 构造一个FDTensor，并共享内存
  bool CopyToTensor(FDTensor* tensor); // 构造一个CPU上的FDTensor，并将数据拷贝过去

  Layout layout; // 数据排布，支持Layout::HWC / Layout::CHW
  Device device; // 数据存放设备，支持Device::CPU / Device::GPU
};
```

## utilities模块 工具模块

提供一些常见的函数，如分类模型常用的`TopK`选择，检测模型的`NMS`操作。同样后面可以考虑将后处理的实现也有不同后端


## visualize 可视化模块

提供一些可视化函数，如检测、分割、OCR都需要这种函数来看可视化的效果

## 模型模块

这个是`Vision`中最重要的模块，所有的模块均通过`域名` + `模型名`来划分，如

- vision::ppdet::YOLOv3  // PaddleDetection的YOLOv3模型
- vision::ppdet::RCNN  // PaddleDetection的RCNN类模型
- vision::ultralytics::YOLOv5 // https://github.com/ultralytics/yolov5 YOLOv5模型

模型的增加参考[模型开发](models.md)
