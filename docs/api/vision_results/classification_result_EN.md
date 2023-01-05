English | [中文](classification_result.md)
# Image Classification Result

The ClassifyResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the classification result and confidence level of the image.

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

- **label_ids**: Member variable which indicates the classification results of a single image. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification results.
- **scores**: Member variable which indicates the confidence level of a single image on the corresponding classification result. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification confidence level.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.ClassifyResult`

- **label_ids**(list of int): Member variable which indicates the classification results of a single image. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification results.
- **scores**(list of float): Member variable which indicates the confidence level of a single image on the corresponding classification result. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification confidence level.
