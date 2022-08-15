# FaceRecognitionResult 人脸识别结果

FaceRecognitionResult 代码定义在`csrcs/fastdeploy/vision/common/result.h`中，用于表明人脸识别模型对图像特征的embedding。
## C++ 定义

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

## Python 定义

`fastdeploy.vision.FaceRecognitionResult`

- **embedding**: 成员变量，表示人脸识别模型最终提取的特征embedding，可以用来计算人脸之间的特征相似度。
