# DetectionResult 目标检测结果

DetectionResult代码定义在`csrcs/fastdeploy/vision/common/result.h`中，用于表明图像检测出来的目标框、目标类别和目标置信度。

## C++ 结构体

`fastdeploy::vision::DetectionResult`

```
struct MattingResult {
  std::vector<float> alpha;       // h x w
  std::vector<float> foreground;  // h x w x c (c=3 default)
  std::vector<int64_t> shape;
  bool contain_foreground = false;
  void Clear();
  std::string Str();
};
```

- **alpha**: 是一维向量，为预测的alpha透明度的值，值域为[0.,1.]，长度为hxw，h,w为输入图像的高和宽
- **foreground**: 是一维向量，为预测的前景，值域为[0.,255.]，长度为hxwxc，h,w为输入图像的高和宽，c一般为3，foreground不是一定有的，只有模型本身预测了前景，这个属性才会有效
- **contain_foreground**: 表示预测的结果是否包含前景
- **shape**: 表示输出结果的shape，当contain_foreground为false，shape只包含(h,w)，当contain_foreground为true，shape包含(h,w,c), c一般为3
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）


## Python结构体

`fastdeploy.vision.DetectionResult`

- **boxes**(list of list(float)): 成员变量，表示单张图片检测出来的所有目标框坐标。boxes是一个list，其每个元素为一个长度为4的list， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **scores**(list of float): 成员变量，表示单张图片检测出来的所有目标置信度
- **label_ids(list of int): 成员变量，表示单张图片检测出来的所有目标类别
