# ClassifyResult 图像分类结果

ClassifyResult代码定义在`csrcs/fastdeploy/vision/common/result.h`中，用于表明图像的分类结果和置信度。

## C++ 定义

`fastdeploy::vision::ClassifyResult`

```
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
