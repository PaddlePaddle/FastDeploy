# MOTResult 多目标跟踪结果

MOTResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明多目标跟踪中的检测出来的目标框、目标跟踪id、目标类别和目标置信度。

## C++ 定义

```c++
fastdeploy::vision::MOTResult
```  

```c++
struct MOTResult{
  // left top right bottom
  std::vector<std::array<int, 4>> boxes;
  std::vector<int> ids;
  std::vector<float> scores;
  std::vector<int> class_ids;
  void Clear();
  std::string Str();
};
```

- **boxes**: 成员变量，表示单帧画面中检测出来的所有目标框坐标，`boxes.size()`表示框的个数，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **ids**: 成员变量，表示单帧画面中所有目标的id，其元素个数与`boxes.size()`一致
- **scores**: 成员变量，表示单帧画面检测出来的所有目标置信度，其元素个数与`boxes.size()`一致
- **class_ids**: 成员变量，表示单帧画面出来的所有目标类别，其元素个数与`boxes.size()`一致
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

```python
fastdeploy.vision.MOTResult
```

- **boxes**(list of list(float)): 成员变量，表示单帧画面中检测出来的所有目标框坐标。boxes是一个list，其每个元素为一个长度为4的list， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **ids**(list of list(float)):成员变量，表示单帧画面中所有目标的id，其元素个数与`boxes`一致
- **scores**(list of float): 成员变量，表示单帧画面检测出来的所有目标置信度
- **class_ids**(list of int): 成员变量，表示单帧画面出来的所有目标类别

