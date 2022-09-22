# Segmentation Results

The SegmentationResult function is defined in `csrcs/fastdeploy/vision/common/result.h` , indicating the predicted segmentation class and the probability value of the segmentation class from each pixel in the image.

## C++  Definition

`fastdeploy::vision::DetectionResult`

```c++
struct DetectionResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  void Clear();
  std::string Str();
};
```

- **label_map**: Member variable that indicates the segmentation class for each pixel of a single image, and `label_map.size()` indicates the number of pixel points of the image
- **score_map**: Member variable that indicates the predicted probability value of the segmentation class corresponding to label_map (define `without_argmax` when exporting the model); or the probability value normalised by softmax (define `without_argmax` and `with_softmax` when exporting the model or define ` without_argmax` while setting the model [Class Member Attribute](../../../examples/vision/segmentation/paddleseg/cpp/)`with_softmax=True`) during initialization.
- **shape**: Member variable that indicates the shape of the output, e.g. (h,w)
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug)

## Python Definition

`fastdeploy.vision.SegmentationResult`

- **label_map**(list of int): Member variable that indicates the segmentation class for each pixel of a single image
- **score_map**(list of float): Member variable that indicates the predicted probability value of the segmentation class corresponding to label_map (define `without_argmax` when exporting the model); or the probability value normalised by softmax (define `without_argmax` and `with_softmax` when exporting the model or define `without_argmax` while setting the model [Class Member Attribute](../../../examples/vision/segmentation/paddleseg/cpp/)`with_softmax=True`) during initialization.
- **shape**(list of int): Member variable that indicates the shape of the output, e.g. (h,w)
