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
