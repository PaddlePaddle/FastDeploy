# OCR Results

The OCRResult function is defined in `fastdeploy/vision/common/result.h` , indicating the text box detected from the image, the text box direction classification, and the text content inside the text box.

## C++ Definition

```c++
fastdeploy::vision::OCRResult
```

```c++
struct OCRResult {
  std::vector<std::array<int, 8>> boxes;
  std::vector<std::string> text;
  std::vector<float> rec_scores;
  std::vector<float> cls_scores;
  std::vector<int32_t> cls_labels;
  ResultType type = ResultType::OCR;
  void Clear();
  std::string Str();
};
```

- **boxes**: Member variable that indicates the coordinates of all object boxes detected in a single image. `boxes.size()` indicates the number of boxes detected in a single image, with each box's 4 coordinate points being represented in order of 8 int values: lower left, lower right, upper right, upper left.
- **text**: Member variable that indicates the text content of multiple identified text boxes, with the same number of elements as `boxes.size()`.
- **rec_scores**: Member variable that indicates the confidence level of the text identified in the text box, with the same number of elements as `boxes.size()`.
- **cls_scores**: Member variable that indicates the confidence level of the classification result of the text box, with the same number of elements as `boxes.size()`.
- **cls_labels**: Member variable that indicates the direction classification of the text box, with the same number of elements as `boxes.size()`.
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug)

## Python Definition

```python
fastdeploy.vision.OCRResult  
```

- **boxes**: Member variable that indicates the coordinates of all object boxes detected in a single image. `boxes.size()` indicates the number of boxes detected in a single image, with each box's 4 coordinate points being represented in order of 8 int values: lower left, lower right, upper right, upper left.
- **text**: Member variable that indicates the text content of multiple identified text boxes, with the same number of elements as `boxes.size()`.
- **rec_scores**: Member variable that indicates the confidence level of the text identified in the text box, with the same number of elements as `boxes.size()`.
- **cls_scores**: Member variable that indicates the confidence level of the classification result of the text box, with the same number of elements as `boxes.size()`.
- **cls_labels**: Member variable that indicates the direction classification of the text box, with the same number of elements as `boxes.size()`.
