English | [中文](keypointdetection_result_CN.md)
# Keypoint Detection Result

The KeyPointDetectionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the coordinates and confidence level of each keypoint of the target's behavior in the image.

## C++ Definition

``fastdeploy::vision::KeyPointDetectionResult``

```c++
struct KeyPointDetectionResult {
  std::vector<std::array<float, 2>> keypoints;
  std::vector<float> scores;
  int num_joints = -1;
  void Clear();
  std::string Str();
};
```

- **keypoints**: Member variable which indicates the coordinates of the identified target behavior keypoint. 
  ` keypoints.size() = N * J`:
    - `N`: the number of targets in the image
    - `J`: num_joints (the number of keypoints of a target)
- **scores**: Member variable which indicates the confidence level of the keypoint coordinates of the identified target behavior. 
  `scores.size() = N * J`:
    - `N`: the number of targets in the picture
    - `J`:num_joints (the number of keypoints of a target)
- **num_joints**: Member variable which indicates the number of keypoints of a target.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.KeyPointDetectionResult`

- **keypoints**(list of list(float)): Member variable which indicates the coordinates of the identified target behavior keypoint. 
  ` keypoints.size() = N * J`:
    - `N`: the number of targets in the image
    - `J`: num_joints (the number of keypoints of a target)
- **scores**(list of float): Member variable which indicates the confidence level of the keypoint coordinates of the identified target behavior. 
  `scores.size() = N * J`:
    - `N`: the number of targets in the picture
    - `J`:num_joints (the number of keypoints of a target)
- **num_joints**(int): Member variable which indicates the number of keypoints of a target.
