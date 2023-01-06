English | [中文](face_recognition_result_CN.md)

# Face Recognition Result

The FaceRecognitionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the image features embedding in the face recognition model.
## C++ Definition

`fastdeploy::vision::FaceRecognitionResult`

```c++
struct FaceRecognitionResult {
  std::vector<float> embedding;
  void Clear();
  std::string Str();
};
```

- **embedding**: Member variable which indicates the final extracted feature embedding of the face recognition model, and can be used to calculate the facial feature similarity.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.FaceRecognitionResult`

- **embedding**(list of float): Member variable which indicates the final extracted feature embedding of the face recognition model, and can be used to calculate the facial feature similarity.
