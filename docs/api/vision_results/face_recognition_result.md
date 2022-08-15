# FaceRecognitionResult 人脸检测结果

FaceRecognitionResult 代码定义在`csrcs/fastdeploy/vision/common/result.h`中，用于表明图像检测出来的目标框、目标类别和目标置信度。

## C++ 结构体

`fastdeploy::vision::FaceRecognitionResult`

```
struct FaceRecognitionResult {
  std::vector<float> embedding;
  void Clear();
  std::string Str();
};
```

- **embedding**: 成员变量，表示人脸识别模型最终的提取的特征embedding，可以用来计算人脸之间的特征相似度。
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python结构体

`fastdeploy.vision.FaceRecognitionResult`

- **embedding**: 成员变量，表示人脸识别模型最终的提取的特征embedding，可以用来计算人脸之间的特征相似度。
