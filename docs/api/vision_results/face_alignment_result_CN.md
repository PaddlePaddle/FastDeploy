[English](face_alignment_result.md) | 简体中文

# FaceAlignmentResult 人脸对齐(人脸关键点检测)结果

FaceAlignmentResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明人脸landmarks。

## C++ 定义

`fastdeploy::vision::FaceAlignmentResult`

```c++
struct FaceAlignmentResult {
  std::vector<std::array<float, 2>> landmarks;
  void Clear();
  std::string Str();
};
```

- **landmarks**: 成员变量，表示单张人脸图片检测出来的所有关键点
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.vision.FaceAlignmentResult`

- **landmarks**(list of list(float)): 成员变量，表示单张人脸图片检测出来的所有关键点
