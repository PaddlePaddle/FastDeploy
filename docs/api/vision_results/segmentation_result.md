# SegmentationResult 目标检测结果

SegmentationResult代码定义在`csrcs/fastdeploy/vision/common/result.h`中，用于表明图像中每个像素预测出来的分割类别和分割类别的概率值。

## C++ 定义

`fastdeploy::vision::DetectionResult`

```c++
struct DetectionResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  void Clear();
  std::string Str();
};
```

- **label_map**: 成员变量，表示单张图片每个像素点的分割类别，`label_map.size()`表示图片像素点的个数
- **score_map**: 成员变量，与label_map一一对应的所预测的分割类别概率值(当导出模型时指定`without_argmax`)或者经过softmax归一化化后的概率值(当导出模型时指定`without_argmax`以及`with_softmax`或者导出模型时指定`without_argmax`同时模型初始化的时候设置模型[类成员属性](../../../examples/vision/segmentation/paddleseg/cpp/)`with_softmax=True`)
- **shape**: 成员变量，表示输出图片的shape，为H\*W
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.vision.SegmentationResult`

- **label_map**(list of int): 成员变量，表示单张图片每个像素点的分割类别
- **score_map**(list of float): 成员变量，与label_map一一对应的所预测的分割类别概率值(当导出模型时指定`without_argmax`)或者经过softmax归一化化后的概率值(当导出模型时指定`without_argmax`以及`with_softmax`或者导出模型时指定`without_argmax`同时模型初始化的时候设置模型[类成员属性](../../../examples/vision/segmentation/paddleseg/python/)`with_softmax=true`)
- **shape**(list of int): 成员变量，表示输出图片的shape，为H\*W
