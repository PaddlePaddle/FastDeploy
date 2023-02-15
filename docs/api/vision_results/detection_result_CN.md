简体中文 ｜ [English](detection_result.md)
# DetectionResult 目标检测结果

DetectionResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像检测出来的目标框、目标类别和目标置信度。

## C++ 定义

```c++
fastdeploy::vision::DetectionResult
```  

```c++
struct DetectionResult {
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
  std::vector<Mask> masks;
  bool contain_masks = false;
  void Clear();
  std::string Str();
};
```

- **boxes**: 成员变量，表示单张图片检测出来的所有目标框坐标，`boxes.size()`表示框的个数，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **scores**: 成员变量，表示单张图片检测出来的所有目标置信度，其元素个数与`boxes.size()`一致
- **label_ids**: 成员变量，表示单张图片检测出来的所有目标类别，其元素个数与`boxes.size()`一致
- **masks**: 成员变量，表示单张图片检测出来的所有实例mask，其元素个数及shape大小与`boxes`一致
- **contain_masks**: 成员变量，表示检测结果中是否包含实例mask，实例分割模型的结果此项一般为true.
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

```c++
fastdeploy::vision::Mask
```  
```c++
struct Mask {
  std::vector<int32_t> data;
  std::vector<int64_t> shape;  // (H,W) ...

  void Clear();
  std::string Str();
};
```  
- **data**: 成员变量，表示检测到的一个mask
- **shape**: 成员变量，表示mask的shape，如 (h,w)
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

```python
fastdeploy.vision.DetectionResult  
```

- **boxes**(list of list(float)): 成员变量，表示单张图片检测出来的所有目标框坐标。boxes是一个list，其每个元素为一个长度为4的list， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **scores**(list of float): 成员变量，表示单张图片检测出来的所有目标置信度
- **label_ids**(list of int): 成员变量，表示单张图片检测出来的所有目标类别
- **masks**: 成员变量，表示单张图片检测出来的所有实例mask，其元素个数及shape大小与`boxes`一致
- **contain_masks**: 成员变量，表示检测结果中是否包含实例mask，实例分割模型的结果此项一般为True.

```python
fastdeploy.vision.Mask  
```
- **data**: 成员变量，表示检测到的一个mask
- **shape**: 成员变量，表示mask的shape，如 (h,w)


## C# 定义

```c#
fastdeploy.vision.DetectionResult
```  

```C#
public struct DetectionResult {
  public List<float[]> boxes;
  public List<float> scores;
  public List<int> label_ids;
  public List<Mask> masks;
  public bool contain_masks;
}
```

- **boxes**(list of array(float)): 成员变量，表示单张图片检测出来的所有目标框坐标。boxes是一个list，其每个元素为一个长度为4的数组， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **scores**(list of float): 成员变量，表示单张图片检测出来的所有目标置信度
- **label_ids**(list of int): 成员变量，表示单张图片检测出来的所有目标类别
- **masks**: 成员变量，表示单张图片检测出来的所有实例mask，其元素个数及shape大小与`boxes`一致
- **contain_masks**: 成员变量，表示检测结果中是否包含实例mask，实例分割模型的结果此项一般为True.

```C#
public struct Mask {
  public List<byte> data;
  public List<long> shape;
}
```

- **data**: 成员变量，表示检测到的一个mask
- **shape**: 成员变量，表示mask的shape，如 (h,w)

## C定义

```c
typedef struct FD_C_DetectionResult {
  FD_C_TwoDimArrayFloat boxes;
  FD_C_OneDimArrayFloat scores;
  FD_C_OneDimArrayInt32 label_ids;
  FD_C_OneDimMask masks;
  FD_C_Bool contain_masks;
} FD_C_DetectionResult;
```

- **boxes**(FD_C_TwoDimArrayFloat): 成员变量，表示单张图片检测出来的所有目标框坐标。boxes是一个list，其每个元素为一个长度为4的数组， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标。FD_C_TwoDimArrayFloat表示一个二维数组，size表示所包含的一维数组的个数，data表示FD_C_OneDimArrayFloat的一维数组。

```c
typedef struct FD_C_TwoDimArrayFloat {
  size_t size;
  FD_C_OneDimArrayFloat* data;
}
```



- **scores**(FD_C_OneDimArrayFloat): 成员变量，表示单张图片检测出来的所有目标置信度。FD_C_OneDimArrayFloat包含两个字段，size和data，其中size表示数组的大小，data表示存储结果的数组。

```c
typedef struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
} FD_C_OneDimArrayFloat;
```

- **label_ids**(FD_C_OneDimArrayInt32): 成员变量，表示单张图片检测出来的所有目标类别。FD_C_OneDimArrayInt32包含两个字段，size和data，其中size表示数组的大小，data表示存储结果的数组。

```c
typedef struct FD_C_OneDimArrayInt32 {
  size_t size;
  int32_t* data;
} FD_C_OneDimArrayInt32;
```

- **masks**(FD_C_OneDimMask): 成员变量，表示单张图片检测出来的所有实例mask，其元素个数及shape大小与`boxes`一致

```c
typedef struct FD_C_OneDimMask {
  size_t size;
  FD_C_Mask* data;
} FD_C_OneDimMask;
```

```c
typedef struct FD_C_Mask {
  FD_C_OneDimArrayUint8 data;
  FD_C_OneDimArrayInt64 shape;
} FD_C_Mask;
```
- **contain_masks**: 成员变量，表示检测结果中是否包含实例mask，实例分割模型的结果此项一般为True.
