English | [中文](../../cn/faq/develop_c_sharp_api_for_a_new_model.md)

# Adds C# API to models

## Introduction

The core code library of FastDeploy is implemented based on C++ development. In order to provide C# interface for users to call, C interface needs to be used as a bridge for communication. Usually when adding C# API for a model, C API is needed to implement first, then develop C# API. C# will call C interface, and C interface will call C++ interface to implement specific functions. About how to add C API for a model, please refer to document [Adds C API to models](./develop_c_api_for_a_new_model.md).

According to FastDeploy’s current implementation structure, adding an API for a model usually involves following three parts:

- Model

Model interface, provide users with functions for model creation and loading, prediction.

- Result

Inference result of model

- Visualization

Function for visualizing inference result

Since C# is an object-oriented programming language,  C# interface forms of each part need to be consistent with corresponding interfaces of C++, giving users the same usage experience. Since C# interface is implemented by calling underlying C interface, it is necessary to understand how C# calls C interface and relationship between data structures of both. Please refer to [C# Marshal](https://learn.microsoft.com/en-us/dotnet/framework/interop/marshalling-data-with-platform-invoke).

## Implementation process

The following describes how to implement C# API for ppseg series models as an example of how to implement C# API under the current framework.

1. Provide a data structure that represents segmentation results

Open file fastdeploy/vision/common/result.h, which defines data structures for different types of model prediction results, find SegmentationResult, and use C# structure to represent the following data structure

```c++
struct FASTDEPLOY_DECL SegmentationResult : public BaseResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  ResultType type = ResultType::SEGMENTATION;
}
```

Define C# SegmentationResult structure in csharp/fastdeploy/vision/result.cs

```c#
public class SegmentationResult{
  public List<byte> label_map;
  public List<float> score_map;
  public List<long> shape;
  public bool contain_score_map;
  public ResultType type;
}
```

This structure is used by users in C# API, but in order to use it when calling C interface, we need to use C# to define corresponding structure defined in C as internal use, and C# structure defined in csharp/fastdeploy/types_internal_c.cs that maps with structure in C language. For example, define FD_SegmentationResult structure as follows

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

Next, we need to define two functions that are responsible for converting between the SegmentationResult and FD_SegmentationResult structures. These two functions need to be defined in the ConvertResult class.

```c#
public class ConvertResult {
  public static SegmentationResult
  ConvertCResultToSegmentationResult(FD_SegmentationResult fd_segmentation_result);
  public static FD_SegmentationResult
  ConvertSegmentationResultToCResult(SegmentationResult segmentation_result);
}
```

2. Provide C# API for model interface

Open the file fastdeploy/vision/segmentation/ppseg/model.h, which defines the C++ interface for the segmentation model, namely fastdeploy::vision::segmentation::PaddleSegModel class. Define a class in C# to implement these interfaces.
Generally, there are two parts in the C# class declaration. The first part is the interface for users, and the second part is the declaration of C interfaces to be called.

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

For more details, please refer to file 'csharp/fastdeploy/vision/segmentation/model.cs'.

3. Provide C# API for visualization function

Open the file fastdeploy/vision/visualize/visualize.h, which has functions for visualizing inference results of different types of models. Here, we write a C# API to visualize SegmentationResult in the Visualize class. Similarly, it will call the underlying C interface to implement specific functions, so you need to declare corresponding C interfaces.

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

4. Create example to test added C# API

In examples directory, according to category of model, create new directory named csharp in corresponding folder.Create csharp sample code and CMakeLists.txt inside, then compile and test it, to ensure that the added C# API can work normally.
