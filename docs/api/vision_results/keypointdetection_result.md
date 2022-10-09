# KeyPointDetectionResult 目标检测结果

KeyPointDetectionResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像中人物行为的各个关键点坐标和置信度。

## C++ 定义

`fastdeploy::vision::KeyPointDetectionResult`

```c++
struct KeyPointDetectionResult {
  std::vector<float> keypoints;
  int num_joints = -1;
  void Clear();
  std::string Str();
};
```

- **keypoints**: 成员变量，表示识别到的人物行为的关键点坐标和置信度。
                 `keypoints.size()= N * J * 3` 
                 `N`:图片中的人物数量 
                 `J`:num_joints（关键点数量） 
                 `3`:坐标信息[x, y, conf]
- **num_joints**: 成员变量，关键点的数量
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.vision.KeyPointDetectionResult`

- **keypoints**(list of float): 成员变量，表示识别到的人物行为的关键点坐标和置信度。
                                `keypoints.size()= N * J * 3` 
                                `N`:图片中的人物数量 
                                `J`:num_joints（关键点数量） 
                                `3`:坐标信息[x, y, conf]
- **num_joints**(int): 成员变量，关键点的数量
