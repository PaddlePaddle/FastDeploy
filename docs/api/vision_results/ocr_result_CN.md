中文 ｜ [English](ocr_result.md)
# OCRResult OCR预测结果

OCRResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像检测和识别出来的文本框，文本框方向分类，以及文本框内的文本内容

## C++ 定义

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

- **boxes**: 成员变量，表示单张图片检测出来的所有目标框坐标，`boxes.size()`表示单张图内检测出的框的个数，每个框以8个int数值依次表示框的4个坐标点，顺序为左下，右下，右上，左上
- **text**: 成员变量，表示多个文本框内被识别出来的文本内容，其元素个数与`boxes.size()`一致
- **rec_scores**: 成员变量，表示文本框内识别出来的文本的置信度，其元素个数与`boxes.size()`一致
- **cls_scores**: 成员变量，表示文本框的分类结果的置信度，其元素个数与`boxes.size()`一致
- **cls_labels**: 成员变量，表示文本框的方向分类类别，其元素个数与`boxes.size()`一致
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

```python
fastdeploy.vision.OCRResult  
```

- **boxes**: 成员变量，表示单张图片检测出来的所有目标框坐标，`boxes.size()`表示单张图内检测出的框的个数，每个框以8个int数值依次表示框的4个坐标点，顺序为左下，右下，右上，左上
- **text**: 成员变量，表示多个文本框内被识别出来的文本内容，其元素个数与`boxes.size()`一致
- **rec_scores**: 成员变量，表示文本框内识别出来的文本的置信度，其元素个数与`boxes.size()`一致
- **cls_scores**: 成员变量，表示文本框的分类结果的置信度，其元素个数与`boxes.size()`一致
- **cls_labels**: 成员变量，表示文本框的方向分类类别，其元素个数与`boxes.size()`一致

## C# 定义

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

- **boxes**: 成员变量，表示单张图片检测出来的所有目标框坐标，`boxes.Count`表示单张图内检测出的框的个数，每个框以8个int数值依次表示框的4个坐标点，顺序为左下，右下，右上，左上
- **text**: 成员变量，表示多个文本框内被识别出来的文本内容，其元素个数与`boxes.Count`一致
- **rec_scores**: 成员变量，表示文本框内识别出来的文本的置信度，其元素个数与`boxes.Count`一致
- **cls_scores**: 成员变量，表示文本框的分类结果的置信度，其元素个数与`boxes.Count`一致
- **cls_labels**: 成员变量，表示文本框的方向分类类别，其元素个数与`boxes.Count`一致

## C定义

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

- **boxes**: 成员变量，表示单张图片检测出来的所有目标框坐标。

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

- **text**: 成员变量，表示多个文本框内被识别出来的文本内容。

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

- **rec_scores**: 成员变量，表示文本框内识别出来的文本的置信度。

```c
typedef struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
} FD_C_OneDimArrayFloat;
```
- **cls_scores**: 成员变量，表示文本框的分类结果的置信度。
- **cls_labels**: 成员变量，表示文本框的方向分类类别。
