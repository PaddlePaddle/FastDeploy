# HeadPoseResult 头部姿态结果

HeadPoseResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明头部姿态结果。

## C++ 定义

`fastdeploy::vision::HeadPoseResult`

```c++
struct HeadPoseResult {
  std::vector<float> eulerangles;
  void Clear();
  std::string Str();
};
```

- **eulerangles**: 成员变量，表示单张人脸图片预测的欧拉角
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.vision.HeadPoseResult`

- **eulerangles**(list of float): 成员变量，表示单张人脸图片预测的欧拉角
