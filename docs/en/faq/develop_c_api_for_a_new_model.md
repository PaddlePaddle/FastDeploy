English | [中文](../../cn/faq/develop_c_api_for_a_new_model.md)

# Adds C API to models

## Introduction

The core code library of FastDeploy is implemented based on C++ development. In order to enhance the portability of the interface and provide SDKs for different development languages, it is necessary to provide a set of C APIs to serve as a bridge for communication between different programming languages.

According to FastDeploy’s current implementation structure, adding C APIs for a model usually involves the following three parts:

- Model

Model interface, providing users with functions for model creation and loading, prediction.

- Result

The result of model inference

- Visualization

Function for visualizing inference results

For Model, C++ has been used to implement the interfaces that need to be exposed to users. What needs to be done is to use C-style interfaces based on C++ interfaces and wraps them in another structure. For Result, C structures are used to define inference results. For visualization functions, you only need to use C-style interfaces to wrap C++ interfaces.

We make a convention on naming rules. All structures and functions related to provide C APIs are named with FD_C as a prefix. When using C to encapsulate C++ classes, use FD_C_{class name}Wrapper as the structure name. If you need to define a C interface to call a method in a C++ class, use FD_C_{class name}Wrapper{method name} as the name. For example, for the fastdeploy::RuntimeOption class in C++, use C encapsulation as follows

```c
struct FD_C_RuntimeOptionWrapper {
  std::unique_ptr<fastdeploy::RuntimeOption> runtime_option;
}
```

You can see that this structure actually uses C++ data, so in C language, only pointers of FD_C_RuntimeOptionWrapper are used. Through this pointer, call the actual implementation function in C++. For example, if you want to call RuntimeOption::UseCpu() function in C language,

```c
void FD_C_RuntimeOptionWrapperUseCpu(FD_C_RuntimeOptionWrapper fd_c_runtimeoption_wrapper){
  auto& runtime_option = fd_c_runtimeoption_wrapper->runtime_option;
  runtime_option->UseCpu();
}
```

In this way, FD_C_RuntimeOptionWrapper is responsible for holding the actual class in C++, and FD_C_RuntimeOptionWrapper{method name} is responsible for calling methods of classes in C++, helping user access classes and functions implemented by c++ using c api interface in c language.

## Implementation process

The following describes how to implement C API for ppseg series models as an example of how to implement C API under the current framework.

1. Provide a data structure that represents segmentation results

Open file fastdeploy/vision/common/result.h, which defines data structures for different types of model prediction results, find SegmentationResult, and use pure C structure to represent the following data structure

```c++
struct FASTDEPLOY_DECL SegmentationResult : public BaseResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  ResultType type = ResultType::SEGMENTATION;
}
```

Define a C FD_C_SegmentationResult structure for representation correspondingly

```c
typedef struct FD_C_SegmentationResult {
  FD_C_OneDimArrayUint8 label_map;
  FD_C_OneDimArrayFloat score_map;
  FD_C_OneDimArrayInt64 shape;
  FD_C_Bool contain_score_map;
  FD_C_ResultType type;
} FD_C_SegmentationResult;
```

For representations such as FD_C_OneDimArrayUint8, refer to file c_api/fastdeploy_capi/core/fd_type.h.

Then you need to define two functions that convert between fastdeploy::SegmentationResult and FD_C_SegmentationResult. Since a corresponding Wrapper structure in C is used for wrapping C++ structures, what is actually defined is conversion between FD_C_SegmentationResultWrapper and FD_C_SegmentationResult, corresponding to the following two functions.

```c
FASTDEPLOY_CAPI_EXPORT extern FD_C_SegmentationResultWrapper*
FD_C_CreateSegmentationResultWrapperFromCResult(
     FD_C_SegmentationResult* fd_c_segmentation_result);

FASTDEPLOY_CAPI_EXPORT extern void
FD_C_SegmentationResultWrapperToCResult(
     FD_C_SegmentationResultWrapper* fd_c_segmentation_result_wrapper,
     FD_C_SegmentationResult* fd_c_segmentation_result);
```

There are also other API functions for creating and destroying structures that can be implemented by referring to the sample code.

The implementation of various Results in C API is located at c_api/fastdeploy_capi/vision/result.cc.

For declaring various Wrapper structures, refer to file c_api/fastdeploy_capi/internal/types_internal.h .

2. Provide C API for model interface
Open file fastdeploy/vision/segmentation/ppseg/model.h, which defines the C++ interface for segmentation model, i.e. fastdeploy::vision::segmentation::PaddleSegModel class. Create a Wrapper in C to represent this class. For convenience of quick definition and implementation of models of the same category in the future, c_api/fastdeploy_capi/internal/types_internal.h defines macros to quickly create Wrapper and extract the wrapped class object from Wrapper. For example, define a macro to create a Wrapper for segmentation model as

```c
#define DEFINE_SEGMENTATION_MODEL_WRAPPER_STRUCT(typename, varname)  typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::vision::segmentation::typename> varname; \
} FD_C_##typename##Wrapper
```

You can supplement other macro definitions referring to the structure of the macros already implemented in that file.

After declaring the structure, you need to implement the interface in the specific model category directory. Corresponding to C++ directory structure, create a directory c_api/fastdeploy_capi/vision/segmentation/ppseg to save C API for segmentation model, create files model.h and model.cc respectively to declare and implement C API for model.

By comparing with methods exposed by PaddleSegModel class, currently mainly need to implement following five interfaces

```c
// Create model
FD_C_PaddleSegModelWrapper*
FD_C_CreatePaddleSegModelWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format);

// Destroy model

void FD_C_DestroyPaddleSegModelWrapper(
    FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper);

// Initilization

FD_C_Bool FD_C_PaddleSegModelWrapperInitialized(
    FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper);

// Predict
FD_C_Bool FD_C_PaddleSegModelWrapperPredict(
    FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper,
    FD_C_Mat img, FD_C_SegmentationResult* fd_c_segmentation_result);

// Batch prediction
FD_C_Bool FD_C_PaddleSegModelWrapperBatchPredict(
            FD_C_PaddleSegModelWrapper* fd_c_paddleseg_model_wrapper,
            FD_C_OneDimMat imgs,
            FD_C_OneDimSegmentationResult* results);
```

3. Provide C API for visualization function

Open file fastdeploy/vision/visualize/visualize.h, which has functions for visualizing inference results of different types of models. Wrap them in c_api/fastdeploy_capi/vision/visualize.h. For example, define and implement following function for visualizing segmentation results.

```c
FD_C_Mat FD_C_VisSegmentation(FD_C_Mat im,
                              FD_C_SegmentationResult* result,
                              float weight)
```

4. Create example to test added C API

In examples directory, according to category of model, create new directory named c in corresponding folder.Create c sample code and CMakeLists.txt inside, then compile and test it, to ensure that the added C API can work normally.
