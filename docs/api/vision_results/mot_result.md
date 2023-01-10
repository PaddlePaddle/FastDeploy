English | [中文](mot_result_CN.md)
# Multi-target Tracking Result

The MOTResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the detected target frame, target tracking id, target class and target confidence ratio in multi-target tracking task.

## C++ Definition

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

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single frame. `boxes.size()` indicates the number of boxes, each box is represented by 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **ids**: Member variable which indicates the ids of all targets in a single frame, where the element number is the same as `boxes.size()`.
- **scores**: Member variable which indicates the confidence level of all targets detected in a single frame, where the number of elements is the same as `boxes.size()`.
- **class_ids**: Member variable which indicates all target classes detected in a single frame, where the element number is the same as `boxes.size()`.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

```python
fastdeploy.vision.MOTResult
```

- **boxes**(list of list(float)): Member variable which indicates the coordinates of all detected target boxes in a single frame. It is a list, and each element in it is also a list of length 4, representing a box with 4 float values representing xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **ids**(list of list(float)): Member variable which indicates the ids of all targets in a single frame, where the element number is the same as `boxes`.
- **scores**(list of float): Member variable which indicates the confidence level of all targets detected in a single frame.
- **class_ids**(list of float): Member variable which indicates all target classes detected in a single frame.

