English | [中文](face_detection_result_CN.md)
# Face Detection Result

The FaceDetectionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the target frames, face landmarks, target confidence and the number of landmark per face.

## C++ Definition

``fastdeploy::vision::FaceDetectionResult``

```c++
struct FaceDetectionResult {
  std::vector<std::array<float, 4>> boxes;
  std::vector<std::array<float, 2>> landmarks;
  std::vector<float> scores;
  int landmarks_per_face;
  void Clear();
  std::string Str();
};
```

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image. `boxes.size()` indicates the number of boxes, each box is represented by 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **scores**: Member variable which indicates the confidence level of all targets detected in a single image, where the number of elements is the same as `boxes.size()`.
- **landmarks**: Member variable which indicates the keypoints of all faces detected in a single image, where the number of elements is the same as `boxes.size()`.
- **landmarks_per_face**: Member variable which indicates the number of keypoints in each face box.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.FaceDetectionResult`

- **boxes**(list of list(float)): Member variable which indicates the coordinates of all detected target boxes in a single frame. It is a list, and each element in it is also a list of length 4, representing a box with 4 float values representing xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **scores**(list of float): Member variable which indicates the confidence level of all targets detected in a single image.
- **landmarks**(list of list(float)): Member variable which indicates the keypoints of all faces detected in a single image.
- **landmarks_per_face**(int): Member variable which indicates the number of keypoints in each face box.
