# Image Classification Results - ClassifyResult

The ClassifyResult function is defined in `csrcs/fastdeploy/vision/common/result.h` , indicating the classification results and confidence level of the image.

## C++ Definition

`fastdeploy::vision::ClassifyResult`

```c++
struct ClassifyResult {
  std::vector<int32_t> label_ids;
  std::vector<float> scores;
  void Clear();
  std::string Str();
};
```

- **label_ids**: Member variable, indicating the classification results for a single image. The number of member variables is determined by the topk input when using the classification model, e.g. the top 5 classification results.
- **scores**: Member variable, indicating the confidence level of a single image on the corresponding classification result. The number of member variables is determined by the topk input when using the classification model, e.g. the top 5 classification confidence level results.
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug)

## Python Definition

`fastdeploy.vision.ClassifyResult`

- **label_ids**(list of int): Member variable, indicating the classification results for a single image. The number of member variables is determined by the topk input when using the classification model, e.g. the top 5 classification results.
- **scores**(list of float): Member variable, indicating the confidence level of a single image on the corresponding classification result. The number of member variables is determined by the topk input when using the classification model, e.g. the top 5 classification confidence level results.
