English | [中文](ocr_result_CN.md)
# OCR prediction result

The OCRResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the text box detected in the image, text box orientation classification, and the text content.

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

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image. `boxes.size()` indicates the number of detected boxes. Each box is represented by 8 int values to indicate the 4 coordinates of the box, in the order of lower left, lower right, upper right, upper left.
- **text**: Member variable which indicates the content of the recognized text in multiple text boxes, where the element number is the same as `boxes.size()`.
- **rec_scores**: Member variable which indicates the confidence level of the recognized text, where the element number is the same as `boxes.size()`.
- **cls_scores**: Member variable which indicates the confidence level of the classification result of the text box, where the element number is the same as `boxes.size()`.
- **cls_labels**: Member variable which indicates the directional category of the textbox, where the element number is the same as `boxes.size()`.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

```python
fastdeploy.vision.OCRResult  
```

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image. `boxes.size()` indicates the number of detected boxes. Each box is represented by 8 int values to indicate the 4 coordinates of the box, in the order of lower left, lower right, upper right, upper left.
- **text**: Member variable which indicates the content of the recognized text in multiple text boxes, where the element number is the same as `boxes.size()`.
- **rec_scores**: Member variable which indicates the confidence level of the recognized text, where the element number is the same as `boxes.size()`.
- **cls_scores**: Member variable which indicates the confidence level of the classification result of the text box, where the element number is the same as `boxes.size()`.
- **cls_labels**: Member variable which indicates the directional category of the textbox, where the element number is the same as `boxes.size()`.
