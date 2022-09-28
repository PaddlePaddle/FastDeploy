# Face Recognition Results

The FaceRecognitionResult function is defined in `csrcs/fastdeploy/vision/common/result.h` , indicating the embedding of image features by the face recognition model.

## C++  Definition

`fastdeploy::vision::FaceRecognitionResult`

```c++
struct FaceRecognitionResult {
  std::vector<float> embedding;
  void Clear();
  std::string Str();
};
```

- **embedding**: Member variable that indicates the final abstracted feature embedding by the face recognition model, which can be used to calculate the feature similarity between faces.
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug).

## Python Definition

`fastdeploy.vision.FaceRecognitionResult`

- **embedding**(list of float): Member variable that indicates the final abstracted feature embedding by the face recognition model, which can be used to calculate the feature similarity between faces.
