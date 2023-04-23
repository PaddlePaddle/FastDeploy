English | [简体中文](detection_result_CN.md)
# PerceptionResult target detection result

The PerceptionResult code is defined in `fastdeploy/vision/common/result.h`, which is used to indicate the detected 3D target: two-dimensional target frame, target frame length, width and height, target category and target confidence, target orientation angle and observation angle etc.

## C++ definition

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

- **scores**: Member variable, indicating the confidence of all detected targets, `scores.size()` indicates the number of detected boxes
- **label_ids**: Member variable, representing all detected target categories, the number of elements is consistent with `scores.size()`
- **boxes**: Member variable, representing the coordinates of all detected target boxes, the number of elements is consistent with `scores.size()`, and each box represents xmin, ymin, xmax, ymax in turn with 7 float values , h, w, l, the coordinates of the upper left and lower right corners and the length, width and height of the 3D box
- **center**: member variable, indicating the center point coordinates of all detected target frames, the number of elements is consistent with `scores.size()`, and each frame uses 3 float values to represent the center point coordinates of the frame in turn
- **observation_angle**: Member variable, indicating the observation angle of the detected frame, and the number of elements is consistent with `scores.size()`
- **yaw_angle**: Member variable, indicating the orientation angle of the detected frame, the number of elements is consistent with `scores.size()`
- **velocity**: Member variable, indicating the velocity of the detected frame, the number of elements is consistent with `scores.size()`
- **Clear()**: member function, used to clear the results stored in the structure
- **Str()**: member function, output the information in the structure as a string (for Debug)

## Python definition

```python
fastdeploy.vision.PerceptionResult
```

- **scores**(list of float): Member variable, indicating the confidence of all detected targets, `scores.size()` indicates the number of detected frames
- **label_ids**(list of int): Member variable, representing all detected target categories, the number of elements is consistent with `scores.size()`
- **boxes**(list of list(float)): Member variable, indicating the coordinates of all detected target boxes, the number of elements is the same as `scores.size()`, and each box is in order of 7 float values Indicates xmin, ymin, xmax, ymax, h, w, l, that is, the coordinates of the upper left and lower right corners and the length, width and height of the 3D box
- **center**(list of list(float)): Member variable, which represents the coordinates of the center points of all detected target frames, the number of elements is the same as `scores.size()`, and each frame is represented by 3 floats The values ​​in turn represent the coordinates of the center point of the box
- **observation_angle**: member variable, indicating the orientation angle of the detected frame, and the number of elements is consistent with `scores.size()`
- **yaw_angle**: Member variable, indicating the orientation angle of the detected frame, the number of elements is consistent with `scores.size()`
- **velocity**: Member variable, indicating the velocity of the detected frame, the number of elements is consistent with `scores.size()`
