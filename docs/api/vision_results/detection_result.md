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
