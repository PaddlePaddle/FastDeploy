English | [简体中文](face_alignment_result_CN.md)
# Face Alignment Result

The FaceAlignmentResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate face landmarks.

## C++ Definition

`fastdeploy::vision::FaceAlignmentResult`

```c++
struct FaceAlignmentResult {
  std::vector<std::array<float, 2>> landmarks;
  void Clear();
  std::string Str();
};
```

- **landmarks**: Member variable which indicates all the key points detected in a single face image.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.FaceAlignmentResult`

- **landmarks**(list of list(float)): Member variable which indicates all the key points detected in a single face image.
