English | [简体中文](classification_result_CN.md)
# Image Classification Result

The ClassifyResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the classification result and confidence level of the image.

## C++ Definition

`fastdeploy::vision::ClassifyResult`

```c++
struct ClassifyResult {
  std::vector<int32_t> label_ids;
  std::vector<float> scores;
  void Clear();
  std::string Str();
};
```

- **label_ids**: Member variable which indicates the classification results of a single image. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification results.
- **scores**: Member variable which indicates the confidence level of a single image on the corresponding classification result. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification confidence level.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.ClassifyResult`

- **label_ids**(list of int): Member variable which indicates the classification results of a single image. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification results.
- **scores**(list of float): Member variable which indicates the confidence level of a single image on the corresponding classification result. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification confidence level.

## C# Definition

`fastdeploy.vision.ClassifyResult`

```C#
public struct ClassifyResult {
  public List<int> label_ids;
  public List<float> scores;
}
```

- **label_ids**(list of int): Member variable which indicates the classification results of a single image. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification results.
- **scores**(list of float): Member variable which indicates the confidence level of a single image on the corresponding classification result. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification confidence level.

## C Definition

```c
typedef struct FD_C_ClassifyResult {
  FD_C_OneDimArrayInt32 label_ids;
  FD_C_OneDimArrayFloat scores;
} FD_C_ClassifyResult;
```

- **label_ids**(FD_C_OneDimArrayInt32): Member variable which indicates the classification results of a single image. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification results.FD_C_OneDimArrayInt32 includes two fields，i.e. size and data，in which size represents the number of elements，and data is the array to store elements.

```c
typedef struct FD_C_OneDimArrayInt32 {
  size_t size;
  int32_t* data;
} FD_C_OneDimArrayInt32;
```

- **scores**(FD_C_OneDimArrayFloat): Member variable which indicates the confidence level of a single image on the corresponding classification result. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification confidence level. FD_C_OneDimArrayFloat includes two fields，i.e. size and data，in which size represents the number of elements，and data is the array to store elements.

```c
typedef struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
} FD_C_OneDimArrayFloat;
```
