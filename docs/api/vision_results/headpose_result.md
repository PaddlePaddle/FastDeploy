# HeadPoseResult 头部姿态结果

HeadPoseResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明头部姿态结果。

## C++ 定义

`fastdeploy::vision::HeadPoseResult`

```c++
struct HeadPoseResult {
  std::vector<float> euler_angles;
  void Clear();
  std::string Str();
};
```

- **euler_angles**: 成员变量，表示单张人脸图片预测的欧拉角，存放的顺序是(yaw, pitch, roll)， yaw 代表水平转角，pitch 代表垂直角，roll 代表翻滚角，值域都为 [-90,+90]度
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.vision.HeadPoseResult`

- **euler_angles**(list of float): 成员变量，表示单张人脸图片预测的欧拉角，存放的顺序是(yaw, pitch, roll)， yaw 代表水平转角，pitch 代表垂直角，roll 代表翻滚角，值域都为 [-90,+90]度
