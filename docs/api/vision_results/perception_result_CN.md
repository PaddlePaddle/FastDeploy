简体中文 ｜ [English](perception_result.md)
# PerceptionResult 目标检测结果

PerceptionResult`fastdeploy/vision/common/result.h`中，用于表明检测出来的3D目标的：二维目标框、目标框长宽高、目标类别和目标置信度、目标朝向角和观测角等。

## C++ 定义

```c++
fastdeploy::vision::PerceptionResult
```  

```c++
struct PerceptionResult {
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
  std::vector<std::array<float, 7>> boxes;
  std::vector<std::array<float, 3>> center;
  std::vector<float>observation_angle;
  std::vector<float>yaw_angle;
  std::vector<std::array<float, 3>>velocity;
  void Clear();
  std::string Str();
};
```

- **scores**: 成员变量，表示检测出来的所有目标置信度，`scores.size()`表示检测出来框的个数
- **label_ids**: 成员变量，表示检测出来的所有目标类别，其元素个数与`scores.size()`一致
- **boxes**: 成员变量，表示检测出来的所有目标框坐标，其元素个数与`scores.size()`一致，每个框以7个float数值依次表示xmin, ymin, xmax, ymax，h, w, l， 即左上角和右下角坐标以及3D框的长宽高
- **center**: 成员变量，表示检测出来的所有目标框中心点坐标，其元素个数与`scores.size()`一致，每个框以3个float数值依次表示框中心点坐标
- **observation_angle**: 成员变量，表示检测出来的框的观测角，其元素个数与`scores.size()`一致
- **yaw_angle**: 成员变量，表示检测出来的框的朝向角，其元素个数与`scores.size()`一致
- **velocity**: 成员变量，表示检测出来的框的速度，其元素个数与`scores.size()`一致
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

```python
fastdeploy.vision.PerceptionResult  
```

- **scores**(list of float): 成员变量，表示检测出来的所有目标置信度，`scores.size()`表示检测出来框的个数
- **label_ids**(list of int): 成员变量，表示检测出来的所有目标类别，其元素个数与`scores.size()`一致
- **boxes**(list of list(float)): 成员变量，表示检测出来的所有目标框坐标，其元素个数与`scores.size()`一致，每个框以7个float数值依次表示xmin, ymin, xmax, ymax，h, w, l， 即左上角和右下角坐标以及3D框的长宽高
- **center**(list of list(float)): 成员变量，表示检测出来的所有目标框中心点坐标，其元素个数与`scores.size()`一致，每个框以3个float数值依次表示框中心点坐标
- **observation_angle**: 成员变量，表示检测出来的框的朝向角，其元素个数与`scores.size()`一致
- **yaw_angle**: 成员变量，表示检测出来的框的朝向角，其元素个数与`scores.size()`一致
- **velocity**: 成员变量，表示检测出来的框的速度，其元素个数与`scores.size()`一致
