[English](../../en/faq/develop_c_api_for_a_new_model.md) | 中文

# FastDeploy给模型新增C API

## 相关概念

FastDeploy的核心代码库的实现是基于C++开发的，为了增强接口的可移植性以及提供多种不同开发语言的SDK，有必要提供一组C API，用来作为不同编程语言沟通的桥梁。

按照FastDeploy目前的实现结构，新增一个模型的API通常涉及以下三个部分：

- Model

  模型接口，提供给用户进行模型创建和载入、预测的功能。

- Result

  模型推理的结果

- Visualization

  对推理结果进行可视化的功能

对于Model, 已经基于C++对需要暴露给使用者的接口进行了实现，所需要做的只是使用C风格的接口基于C++接口再包裹一层。对于Result，需要使用C的结构对推理结果进行重新定义。对可视化函数，也只需要使用C风格的接口对C++接口进行包裹即可。

我们对命名规则做一个约定，所有和提供C API有关的结构和函数使用FD_C作为前缀进行命名。当使用C对C++的类进行封装时，采用FD_C_{类名}Wrapper的方式对结构进行命令。如果需要调用C++类中的某个方法，采用FD_C_{类名}Wrapper{方法名}的方式进行命名。
比如，对于C++中的fastdeploy::RuntimeOption类别，使用C进行封装的形式为
```c
struct FD_C_RuntimeOptionWrapper {
  std::unique_ptr<fastdeploy::RuntimeOption> runtime_option;
}
```
可以看到，这个结构里面用的其实是C++的内容，所以在C语言里，只使用FD_C_RuntimeOptionWrapper这个结构的指针，通过这个指针对C++的实际实现函数进行调用。比如想要调用RuntimeOption::UseCpu()这个函数，在C语言里的封装如下
```c
void FD_C_RuntimeOptionWrapperUseCpu(FD_C_RuntimeOptionWrapper fd_c_runtimeoption_wrapper){
  auto& runtime_option = fd_c_runtimeoption_wrapper->runtime_option;
  runtime_option->UseCpu();
}
```

通过这种方式，FD_C_RuntimeOptionWrapper负责持有C++里实际的类， FD_C_RuntimeOptionWrapper{方法名}负责调用C++里类的方法，实现用户在C语言里使用C API接口访问C++所实现的类和函数。


## 实现流程

下面通过给ppseg系列模型提供C API为示例讲述如何在当前框架下进行C API的实现。

1. 提供表示分割模型结果的数据结构

打开文件fastdeploy/vision/common/result.h, 里面定义各种不同类别模型预测结果的数据结构，找到SegmentationResult，将下列数据结构用纯C结构进行表示
```c++
struct FASTDEPLOY_DECL SegmentationResult : public BaseResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  ResultType type = ResultType::SEGMENTATION;
}
```
对应的定义一个C的FD_C_SegmentationResult结构进行表示
```c
typedef struct FD_C_SegmentationResult {
  FD_C_OneDimArrayUint8 label_map;
  FD_C_OneDimArrayFloat score_map;
  FD_C_OneDimArrayInt64 shape;
  FD_C_Bool contain_score_map;
  FD_C_ResultType type;
} FD_C_SegmentationResult;
```
关于FD_C_OneDimArrayUint8之类的表示，可以参考文件c_api/fastdeploy_capi/core/fd_type.h。

之后需要定义两个函数，用来从fastdeploy::SegmentationResult和FD_C_SegmentationResult之间进行相互转化。由于对C++的结构使用了对应的Wrapper结构进行包裹，所以实际定义的是FD_C_SegmentationResultWrapper和FD_C_SegmentationResult之间的转化，对应下面两个函数。
```c
FASTDEPLOY_CAPI_EXPORT extern FD_C_SegmentationResultWrapper*
FD_C_CreateSegmentationResultWrapperFromCResult(
     FD_C_SegmentationResult* fd_c_segmentation_result);

FASTDEPLOY_CAPI_EXPORT extern void
FD_C_SegmentationResultWrapperToCResult(
     FD_C_SegmentationResultWrapper* fd_c_segmentation_result_wrapper,
     FD_C_SegmentationResult* fd_c_segmentation_result);
```
还有其它的几个创建和销毁结构的API函数可以参考示例代码进行补充实现。

关于各种Result在C API中的实现位置为c_api/fastdeploy_capi/vision/result.cc。

关于声明各种Wrapper的结构可以参考文件c_api/fastdeploy_capi/internal/types_internal.h 。

2. 提供模型接口的C API

打开文件fastdeploy/vision/segmentation/ppseg/model.h，里面定义了分割模型的C++接口，即fastdeploy::vision::segmentation::PaddleSegModel类。在C中创建一个Wrapper来表示这个类，为了方便后续对同一类别的模型进行快速定义和实现，c_api/fastdeploy_capi/internal/types_internal.h中定义了宏来快速创建Wrapper，以及从Wrapper中取出所包裹的类的对象。例如定义创建分割类模型的Wrapper的宏为
```c
#define DEFINE_SEGMENTATION_MODEL_WRAPPER_STRUCT(typename, varname)  typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::vision::segmentation::typename> varname; \
} FD_C_##typename##Wrapper
```
可以按照那个文件已实现的宏的结构，对其它的宏定义进行补充。

声明了结构后，就需要在具体的模型类别目录下进行接口实现了。对应C++的目录结构，创建一个保存分割模型C API的目录c_api/fastdeploy_capi/vision/segmentation/ppseg, 创建文件model.h和model.cc分别声明和实现模型的C接口。

通过对照PaddleSegModel类里暴露的方法，目前主要需要实现如下五个接口
```
// 创建模型
FD_C_PaddleSegModelWrapper*
FD_C_CreatePaddleSegModelWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format);

// 销毁模型

void FD_C_DestroyPaddleSegModelWrapper(
    FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper);

// 判断初始化是否成功

FD_C_Bool FD_C_PaddleSegModelWrapperInitialized(
    FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper);

// 预测单张图
FD_C_Bool FD_C_PaddleSegModelWrapperPredict(
    FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper,
    FD_C_Mat img, FD_C_SegmentationResult* fd_c_segmentation_result);

// 成批预测
FD_C_Bool FD_C_PaddleSegModelWrapperBatchPredict(
            FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper,
            FD_C_OneDimMat imgs,
            FD_C_OneDimSegmentationResult* results);
```

3. 提供可视化函数C API

打开文件fastdeploy/vision/visualize/visualize.h，里面有对于不同类型模型推理结果进行可视化的函数。在c_api/fastdeploy_capi/vision/visualize.h中对其进行封装一下。例如在C API中需要定义并实现如下对分割结果进行可视化的函数

```c
FD_C_Mat FD_C_VisSegmentation(FD_C_Mat im,
                              FD_C_SegmentationResult* result,
                              float weight)
```

4. 创建example, 测试所添加的C API

在examples目录下，根据所接入的模型的类别，在对应的文件夹下新增目录名c，里面创建c的示例代码和CMakeLists.txt，编译测试，确保使用新增的C API能够正常工作。
