# Face Detection Results

The FaceDetectionResult function is defined in `fastdeploy/vision/common/result.h` , indicating the object's frame, face landmarks , target confidence and the number of landmarks for each face detected.

## C++  Definition

`fastdeploy::vision::FaceDetectionResult`

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

- **boxes**: Member variable that indicates the coordinates of all object boxes detected from an image.`boxes.size()` indicates the number of boxes, and each box is represented by 4 float values in the order xmin, ymin, xmax, ymax, i.e. the top left and bottom right coordinates.
- **scores**: Member variable that indicates the confidence level of all objects detected from a single image, with the same number of elements as `boxes.size()`.
- **landmarks**: Member variable that indicates the key points of all faces detected in a single image, with the same number of elements as `boxes.size()`.
- **landmarks_per_face**: Member variable that indicates the number of key points in each face frame.
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug).

## Python Definition

`fastdeploy.vision.FaceDetectionResult`

- **boxes**(list of list(float)): Member variable that indicates the coordinates of all object boxes detected from an image. Boxes are a list, with each element being a 4-length list presented as a box with 4 float values for xmin, ymin, xmax, ymax, i.e. the top left and bottom right coordinates.
- **scores**(list of float): Member variable that indicates the confidence level of all objects detected from a single image.
- **landmarks**(list of list(float)): Member variable that indicates the key points of all faces detected in a single image.
- **landmarks_per_face**(int): Member variable that indicates the number of key points in each face frame.
