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


## C# Definition

`fastdeploy.vision.OCRResult`

```C#
public class OCRResult {
  public List<int[]> boxes;
  public List<string> text;
  public List<float> rec_scores;
  public List<float> cls_scores;
  public List<int> cls_labels;
  public ResultType type;
}
```

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image. `boxes.Count` indicates the number of detected boxes. Each box is represented by 8 int values to indicate the 4 coordinates of the box, in the order of lower left, lower right, upper right, upper left.
- **text**: Member variable which indicates the content of the recognized text in multiple text boxes, where the element number is the same as `boxes.Count`.
- **rec_scores**: Member variable which indicates the confidence level of the recognized text, where the element number is the same as `boxes.Count`.
- **cls_scores**: Member variable which indicates the confidence level of the classification result of the text box, where the element number is the same as `boxes.Count`.
- **cls_labels**: Member variable which indicates the directional category of the textbox, where the element number is the same as `boxes.Count`.

## C Definition

```c
struct FD_C_OCRResult {
  FD_C_TwoDimArrayInt32 boxes;
  FD_C_OneDimArrayCstr text;
  FD_C_OneDimArrayFloat rec_scores;
  FD_C_OneDimArrayFloat cls_scores;
  FD_C_OneDimArrayInt32 cls_labels;
  FD_C_ResultType type;
};
```

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image.

```c
typedef struct FD_C_TwoDimArrayInt32 {
  size_t size;
  FD_C_OneDimArrayInt32* data;
} FD_C_TwoDimArrayInt32;
```

```c
typedef struct FD_C_OneDimArrayInt32 {
  size_t size;
  int32_t* data;
} FD_C_OneDimArrayInt32;
```

- **text**: Member variable which indicates the content of the recognized text in multiple text boxes

```c
typedef struct FD_C_Cstr {
  size_t size;
  char* data;
} FD_C_Cstr;
typedef struct FD_C_OneDimArrayCstr {
  size_t size;
  FD_C_Cstr* data;
} FD_C_OneDimArrayCstr;
```

- **rec_scores**: Member variable which indicates the confidence level of the recognized text

```c
typedef struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
} FD_C_OneDimArrayFloat;
```
- **cls_scores**: Member variable which indicates the confidence level of the classification result of the text box
- **cls_labels**: Member variable which indicates the directional category of the textbox
