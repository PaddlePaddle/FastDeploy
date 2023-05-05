中文 ｜ [English](segmentation_result.md)
# SegmentationResult 图像分割结果

SegmentationResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像中每个像素预测出来的分割类别和分割类别的概率值。

## C++ 定义

`fastdeploy::vision::SegmentationResult`

```c++
struct SegmentationResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  void Clear();
  void Free();
  std::string Str();
};
```

- **label_map**: 成员变量，表示单张图片每个像素点的分割类别，`label_map.size()`表示图片像素点的个数
- **score_map**: 成员变量，与label_map一一对应的所预测的分割类别概率值，只有导出PaddleSeg模型时指定`--output_op none`时，该成员变量才不为空，否则该成员变量为空
- **shape**: 成员变量，表示输出图片的shape，为H\*W
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Free()**: 成员函数，用于清除结构体中存储的结果并释放内存
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.vision.SegmentationResult`

- **label_map**(list of int): 成员变量，表示单张图片每个像素点的分割类别
- **score_map**(list of float): 成员变量，与label_map一一对应的所预测的分割类别概率值，只有导出PaddleSeg模型时指定`--output_op none`时，该成员变量才不为空，否则该成员变量为空
- **shape**(list of int): 成员变量，表示输出图片的shape，为H\*W

## C# 定义

`fastdeploy.vision.SegmentationResult`

```C#
public class SegmentationResult{
  public List<byte> label_map;
  public List<float> score_map;
  public List<long> shape;
  public bool contain_score_map;
  public ResultType type;
}
```

- **label_map**(list of byte): 成员变量，表示单张图片每个像素点的分割类别
- **score_map**(list of float): 成员变量，与label_map一一对应的所预测的分割类别概率值，只有导出PaddleSeg模型时指定`--output_op none`时，该成员变量才不为空，否则该成员变量为空
- **shape**(list of long): 成员变量，表示输出图片的shape，为H\*W


## C定义

```c
struct FD_C_SegmentationResult {
  FD_C_OneDimArrayUint8 label_map;
  FD_C_OneDimArrayFloat score_map;
  FD_C_OneDimArrayInt64 shape;
  FD_C_Bool contain_score_map;
  FD_C_ResultType type;
};
```

- **label_map**(FD_C_OneDimArrayUint8): 成员变量，表示单张图片每个像素点的分割类别

```c
struct FD_C_OneDimArrayUint8 {
  size_t size;
  uint8_t* data;
};
```

- **score_map**(FD_C_OneDimArrayFloat): 成员变量，与label_map一一对应的所预测的分割类别概率值，只有导出PaddleSeg模型时指定`--output_op none`时，该成员变量才不为空，否则该成员变量为空

```c
struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
};
```

- **shape**(FD_C_OneDimArrayInt64): 成员变量，表示输出图片的shape，为H\*W

```c
struct FD_C_OneDimArrayInt64 {
  size_t size;
  int64_t* data;
};
```
