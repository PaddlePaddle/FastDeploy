English | [中文](headpose_result_CN.md)
# Head Pose Result

The HeadPoseResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the head pose result.

## C++ Definition

``fastdeploy::vision::HeadPoseResult`''

```c++
struct HeadPoseResult {
  std::vector<float> euler_angles;
  void Clear();
  std::string Str();
};
```

- **euler_angles**: Member variable which indicates the Euler angles predicted for a single face image, stored in the order (yaw, pitch, roll), with yaw representing the horizontal turn angle, pitch representing the vertical angle, and roll representing the roll angle, all with a value range of [-90,+90].
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.HeadPoseResult`

- **euler_angles**(list of float): Member variable which indicates the Euler angles predicted for a single face image, stored in the order (yaw, pitch, roll), with yaw representing the horizontal turn angle, pitch representing the vertical angle, and roll representing the roll angle, all with a value range of [-90,+90].
