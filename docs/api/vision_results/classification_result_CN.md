简体中文 ｜ [English](classification_result.md)
# ClassifyResult 图像分类结果

ClassifyResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像的分类结果和置信度。

## C++ 定义

`fastdeploy::vision::ClassifyResult`

```c++
struct ClassifyResult {
  std::vector<int32_t> label_ids;
  std::vector<float> scores;
  void Clear();
  std::string Str();
};
```

- **label_ids**: 成员变量，表示单张图片的分类结果，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类结果
- **scores**: 成员变量，表示单张图片在相应分类结果上的置信度，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类置信度
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.vision.ClassifyResult`

- **label_ids**(list of int): 成员变量，表示单张图片的分类结果，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类结果
- **scores**(list of float): 成员变量，表示单张图片在相应分类结果上的置信度，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类置信度

## C# 定义

`fastdeploy.vision.ClassifyResult`

```C#
public struct ClassifyResult {
  public List<int> label_ids;
  public List<float> scores;
}
```

- **label_ids**(list of int): 成员变量，表示单张图片的分类结果，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类结果
- **scores**(list of float): 成员变量，表示单张图片在相应分类结果上的置信度，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类置信度

## C定义

```c
typedef struct FD_C_ClassifyResult {
  FD_C_OneDimArrayInt32 label_ids;
  FD_C_OneDimArrayFloat scores;
} FD_C_ClassifyResult;
```

- **label_ids**(FD_C_OneDimArrayInt32): 成员变量，表示单张图片的分类结果，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类结果。FD_C_OneDimArrayInt32包含两个字段，size和data，其中size表示数组的大小，data表示存储结果的数组。

```c
typedef struct FD_C_OneDimArrayInt32 {
  size_t size;
  int32_t* data;
} FD_C_OneDimArrayInt32;
```

- **scores**(FD_C_OneDimArrayFloat): 成员变量，表示单张图片在相应分类结果上的置信度，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类置信度。FD_C_OneDimArrayFloat包含两个字段，size和data，其中size表示数组的大小，data表示存储结果的数组。

```c
typedef struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
} FD_C_OneDimArrayFloat;
```
