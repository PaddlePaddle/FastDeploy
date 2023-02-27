[English](../../en/faq/develop_c_sharp_api_for_a_new_model.md) | 中文

# FastDeploy给模型新增C# API

## 相关概念

FastDeploy的核心代码库的实现是基于C++开发的，为了提供给使用者C#接口进行调用，需要使用C接口作为沟通的桥梁。通常在给模型新增C# API时，需要先给模型新增C API。然后再开发C# API。C#会调用C接口，C接口再去调用C++接口，已实现具体的功能。关于如何给模型新增C API，请先参考文档[FastDeploy给模型新增C API](./develop_c_api_for_a_new_model.md)。

按照FastDeploy目前的实现结构，新增一个模型的API通常涉及以下三个部分：

- Model

  模型接口，提供给用户进行模型创建和载入、预测的功能。

- Result

  模型推理的结果

- Visualization

  对推理结果进行可视化的功能

由于C#是面向对象编程的高级语言，在设计C# API的过程中，各个部分的接口形式需要尽量和C++对应的接口保持一致，给使用者相同的使用体验。由于C#接口是通过调用底层的C接口来具体实现的，因此需要了解C#如何调用C接口以及两者数据结构之间的关系。具体可以参考[C# Marshal](https://learn.microsoft.com/en-us/dotnet/framework/interop/marshalling-data-with-platform-invoke)
。

## 实现流程

下面通过给ppseg系列模型提供C# API为示例讲述如何在当前框架下进行C# API的实现。

1. 提供表示分割模型结果的数据结构

打开文件fastdeploy/vision/common/result.h, 里面定义各种不同类别模型预测结果的数据结构，找到SegmentationResult，将数据结构对应用C#进行表示。
```c++
struct FASTDEPLOY_DECL SegmentationResult : public BaseResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  ResultType type = ResultType::SEGMENTATION;
}
```
在csharp/fastdeploy/vision/result.cs对应的定义一个C#的SegmentationResult结构

```c#
public class SegmentationResult{
  public List<byte> label_map;
  public List<float> score_map;
  public List<long> shape;
  public bool contain_score_map;
  public ResultType type;
}
```

该结构是提供给用户在C# API中实际使用的结构，但是为了能够在调用C接口时进行使用，需要使用C#按照C中定义的对应结构进行定义作为内部使用，和C语言中结构进行映射的C#结构定义在csharp/fastdeploy/types_internal_c.cs中。比如这里面再定义一个FD_SegmentationResult，结构如下

```c#
[StructLayout(LayoutKind.Sequential)]
public struct FD_SegmentationResult {
  public FD_OneDimArrayUint8 label_map;
  public FD_OneDimArrayFloat score_map;
  public FD_OneDimArrayInt64 shape;
  [MarshalAs(UnmanagedType.U1)]
  public bool contain_score_map;
  public FD_ResultType type;
}
```

接下来需要定义两个函数，负责在SegmentationResult和FD_SegmentationResult这两个结构间进行转换。这两个函数需要定义在ConvertResult类中。

```c#
public class ConvertResult {
  public static SegmentationResult
  ConvertCResultToSegmentationResult(FD_SegmentationResult fd_segmentation_result);
  public static FD_SegmentationResult
  ConvertSegmentationResultToCResult(SegmentationResult segmentation_result);
}
```

2. 提供模型接口的C# API

打开文件fastdeploy/vision/segmentation/ppseg/model.h，里面定义了分割模型的C++接口，即fastdeploy::vision::segmentation::PaddleSegModel类。在C#中定义一个类对应实现这些接口。

通过对照PaddleSegModel类里暴露的方法，提供对应的方法。一般在C#的类声明里分为两部分，第一部分是提供给用户的接口，第二部分是声明要调用的C接口。

```c#
public class PaddleSegModel {

  public PaddleSegModel(string model_file, string params_file,
                         string config_file, RuntimeOption custom_option = null,
                         ModelFormat model_format = ModelFormat.PADDLE)；
  ~PaddleSegModel()；

  public string ModelName()；

  public SegmentationResult Predict(Mat img)；

  public List<SegmentationResult> BatchPredict(List<Mat> imgs)；

  public bool Initialized()；

  // below are underlying C api
  private IntPtr fd_paddleseg_model_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreatePaddleSegModelWrapper")]
  private static extern IntPtr FD_C_CreatePaddleSegModelWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyPaddleSegModelWrapper")]
  private static extern void
  FD_C_DestroyPaddleSegModelWrapper(IntPtr fd_paddleseg_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleSegModelWrapperPredict")]
  private static extern bool
  FD_C_PaddleSegModelWrapperPredict(IntPtr fd_paddleseg_model_wrapper,
                                     IntPtr img,
                                     ref FD_SegmentationResult fd_segmentation_result);

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroySegmentationResult")]
  private static extern void
  FD_C_DestroySegmentationResult(ref FD_SegmentationResult fd_segmentation_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleSegModelWrapperInitialized")]
  private static extern bool
  FD_C_PaddleSegModelWrapperInitialized(IntPtr fd_paddleseg_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleSegModelWrapperBatchPredict")]
  private static extern bool
  FD_C_PaddleSegModelWrapperBatchPredict(IntPtr fd_paddleseg_model_wrapper,
                                          ref FD_OneDimMat imgs,
                                          ref FD_OneDimSegmentationResult results);

}
```


具体实现可以参考文件 csharp/fastdeploy/vision/segmentation/model.cs 。

3. 提供可视化函数C# API

打开文件fastdeploy/vision/visualize/visualize.h，里面有对于不同类型模型推理结果进行可视化的函数。这里用C#对照写一个可视化SegmentationResult的API在Visualize类中。同样，会调用底层的C接口实现具体功能，因此需要声明对应的C接口。

```c#
public class Visualize {
  public static Mat VisSegmentation(Mat im,
                                    SegmentationResult segmentation_result,
                                    float weight = 0.5);

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_VisSegmentation")]
  private static extern IntPtr
  FD_C_VisSegmentation(IntPtr im, ref FD_SegmentationResult fd_segmentation_result, float weight);
}
```

4. 创建example, 测试所添加的C# API

在examples目录下，根据所接入的模型的类别，在对应的文件夹下新增目录名csharp，里面创建csharp的示例代码和CMakeLists.txt，编译测试，确保使用新增的C# API能够正常工作。
