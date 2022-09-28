# Detection Results

The DetectionResult function is defined in `csrcs/fastdeploy/vision/common/result.h` , indicating the object's frame, class and confidence level from the image detection.

## C++  Definition

```c++
fastdeploy::vision::DetectionResult
```

```c++
struct DetectionResult {
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
  std::vector<Mask> masks;
  bool contain_masks = false;
  void Clear();
  std::string Str();
};
```

- **boxes**: Member variable that indicates the coordinates of all object boxes detected from an image.`boxes.size()` indicates the number of boxes, and each box is represented by 4 float values in the order xmin, ymin, xmax, ymax, i.e. the top left and bottom right coordinates.
- **scores**: Member variable that indicates the confidence level of all objects detected from a single image, with the same number of elements as `boxes.size()`.
- **label_ids**: Member variable that indicates all object classes detected from a single image, with the same number of elements as `boxes.size()`
- **masks**: Member variable that indicates all cases of mask detected from a single image, with the same number of elements and shape size as `boxes`.
- **contain_masks**: Member variable that indicates whether the detection result contains a mask, whose result is generally true for segmentation models.
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug)

```c++
fastdeploy::vision::Mask
```

```c++
struct Mask {
  std::vector<int32_t> data;
  std::vector<int64_t> shape;  // (H,W) ...

  void Clear();
  std::string Str();
};
```

- **data**: Member variable that represents a detected mask
- **shape**: Member variable that indicates the shape of the mask, e.g. (h,w)
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug)

## Python Definition

```python
fastdeploy.vision.DetectionResult  
```

- **boxes**(list of list(float)): Member variable that indicates the coordinates of all object boxes detected from an image. Boxes are a list, with each element being a 4-length list presented as a box with 4 float values for xmin, ymin, xmax, ymax, i.e. the top left and bottom right coordinates.
- **scores**(list of float): Member variable that indicates the confidence level of all objects detected from a single image
- **label_ids**(list of int): Member variable that indicates all object classes detected from a single image
- **masks**: Member variable that indicates all cases of mask detected from a single image, with the same number of elements and shape size as `boxes`.
- **contain_masks**: Member variable that indicates whether the detection result contains a mask, whose result is generally true for segmentation models.

```python
fastdeploy.vision.Mask  
```

- **data**: Member variable that represents a detected mask
- **shape**: Member variable that indicates the shape of the mask, e.g. (h,w)
