English | [中文](segmentation_result.md)
# Segmentation Result

The SegmentationResult code is defined in `fastdeploy/vision/common/result.h`, indicating the segmentation category and the segmentation category probability predicted in each pixel in the image.

## C++ Definition

``fastdeploy::vision::SegmentationResult``

```c++
struct SegmentationResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  void Clear();
  std::string Str();
};
```

- **label_map**: Member variable which indicates the segmentation category of each pixel in a single image. `label_map.size()` indicates the number of pixel points of a image.
- **score_map**: Member variable which indicates the predicted segmentation category probability value (specified as `--output_op argmax` when export) corresponding to label_map, or the probability value normalized by softmax (specified as `--output_op softmax` when export, or as `--output_op when exporting the model). none`  when export while setting the [class member attribute](../../../examples/vision/segmentation/paddleseg/cpp/) as `apply_softmax=True` during model initialization).
- **shape**: Member variable which indicates the shape of the output image as H\*W.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

`fastdeploy.vision.SegmentationResult`

- **label_map**(list of int): Member variable which indicates the segmentation category of each pixel in a single image.
- **score_map**(list of float): Member variable which indicates the predicted segmentation category probability value (specified as `--output_op argmax` when export) corresponding to label_map, or the probability value normalized by softmax (specified as `--output_op softmax` when export, or as `--output_op when exporting the model). none`  when export while setting the [class member attribute](../../../examples/vision/segmentation/paddleseg/cpp/) as `apply_softmax=True` during model initialization).
- **shape**(list of int): Member variable which indicates the shape of the output image as H\*W.
