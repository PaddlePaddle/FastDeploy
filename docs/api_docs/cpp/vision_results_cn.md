[English](./vision_results_en.md) | 简体中文

# 视觉模型预测结果说明

## ClassifyResult 图像分类结果

ClassifyResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像的分类结果和置信度。

### C++ 定义

```c++
fastdeploy::vision::ClassifyResult
```

```c++
struct ClassifyResult {
  std::vector<int32_t> label_ids;
  std::vector<float> scores;
  void Clear();
  std::string Str();
};
```

- **label_ids**: 成员变量，表示单张图片的分类结果，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类结果
- **scores**: 成员变量，表示单张图片在相应分类结果上的置信度，其个数根据在使用分类模型时传入的topk决定，例如可以返回top 5的分类置信度
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## SegmentationResult 图像分割结果

SegmentationResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像中每个像素预测出来的分割类别和分割类别的概率值。

### C++ 定义

```c++
fastdeploy::vision::SegmentationResult
```

```c++
struct SegmentationResult {
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;
  void Clear();
  void Free();
  std::string Str();
};
```

- **label_map**: 成员变量，表示单张图片每个像素点的分割类别，`label_map.size()`表示图片像素点的个数
- **score_map**: 成员变量，与label_map一一对应的所预测的分割类别概率值，只有导出PaddleSeg模型时指定`--output_op none`时，该成员变量才不为空，否则该成员变量为空
- **shape**: 成员变量，表示输出图片的shape，为H\*W
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Free()**: 成员函数，用于清除结构体中存储的结果并释放内存
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## DetectionResult 目标检测结果

DetectionResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像检测出来的目标框、目标类别和目标置信度。

### C++ 定义

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

- **boxes**: 成员变量，表示单张图片检测出来的所有目标框坐标，`boxes.size()`表示框的个数，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **scores**: 成员变量，表示单张图片检测出来的所有目标置信度，其元素个数与`boxes.size()`一致
- **label_ids**: 成员变量，表示单张图片检测出来的所有目标类别，其元素个数与`boxes.size()`一致
- **masks**: 成员变量，表示单张图片检测出来的所有实例mask，其元素个数及shape大小与`boxes`一致
- **contain_masks**: 成员变量，表示检测结果中是否包含实例mask，实例分割模型的结果此项一般为true.
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

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
- **data**: 成员变量，表示检测到的一个mask
- **shape**: 成员变量，表示mask的shape，如 (h,w)
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## FaceAlignmentResult 人脸对齐(人脸关键点检测)结果

FaceAlignmentResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明人脸landmarks。

### C++ 定义

```c++
fastdeploy::vision::FaceAlignmentResult
```

```c++
struct FaceAlignmentResult {
  std::vector<std::array<float, 2>> landmarks;
  void Clear();
  std::string Str();
};
```

- **landmarks**: 成员变量，表示单张人脸图片检测出来的所有关键点
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## KeyPointDetectionResult 目标检测结果

KeyPointDetectionResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像中目标行为的各个关键点坐标和置信度。

### C++ 定义

```c++
fastdeploy::vision::KeyPointDetectionResult
```

```c++
struct KeyPointDetectionResult {
  std::vector<std::array<float, 2>> keypoints;
  std::vector<float> scores;
  int num_joints = -1;
  void Clear();
  std::string Str();
};
```

- **keypoints**: 成员变量，表示识别到的目标行为的关键点坐标。
  `keypoints.size()= N * J`
    - `N`：图片中的目标数量
    - `J`：num_joints（一个目标的关键点数量）
- **scores**: 成员变量，表示识别到的目标行为的关键点坐标的置信度。
  `scores.size()= N * J`
    - `N`：图片中的目标数量
    - `J`:num_joints（一个目标的关键点数量）
- **num_joints**: 成员变量，一个目标的关键点数量
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）


## FaceRecognitionResult 人脸识别结果

FaceRecognitionResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明人脸识别模型对图像特征的embedding。
### C++ 定义

```c++
fastdeploy::vision::FaceRecognitionResult
```

```c++
struct FaceRecognitionResult {
  std::vector<float> embedding;
  void Clear();
  std::string Str();
};
```

- **embedding**: 成员变量，表示人脸识别模型最终的提取的特征embedding，可以用来计算人脸之间的特征相似度。
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）



## MattingResult 抠图结果

MattingResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明模型预测的alpha透明度的值，预测的前景等。

### C++ 定义

```c++
fastdeploy::vision::MattingResult
```

```c++
struct MattingResult {
  std::vector<float> alpha;
  std::vector<float> foreground;
  std::vector<int64_t> shape;
  bool contain_foreground = false;
  void Clear();
  std::string Str();
};
```

- **alpha**: 是一维向量，为预测的alpha透明度的值，值域为[0.,1.]，长度为hxw，h,w为输入图像的高和宽
- **foreground**: 是一维向量，为预测的前景，值域为[0.,255.]，长度为hxwxc，h,w为输入图像的高和宽，c一般为3，foreground不是一定有的，只有模型本身预测了前景，这个属性才会有效
- **contain_foreground**: 表示预测的结果是否包含前景
- **shape**: 表示输出结果的shape，当contain_foreground为false，shape只包含(h,w)，当contain_foreground为true，shape包含(h,w,c), c一般为3
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## OCRResult OCR预测结果

OCRResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像检测和识别出来的文本框，文本框方向分类，以及文本框内的文本内容

### C++ 定义

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


## FaceDetectionResult 人脸检测结果

FaceDetectionResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明人脸检测出来的目标框、人脸landmarks，目标置信度和每张人脸的landmark数量。

### C++ 定义

```c++
fastdeploy::vision::FaceDetectionResult
```

```c++
struct FaceDetectionResult {
  std::vector<std::array<float, 4>> boxes;
  std::vector<std::array<float, 2>> landmarks;
  std::vector<float> scores;
  int landmarks_per_face;
  void Clear();
  std::string Str();
};
```

- **boxes**: 成员变量，表示单张图片检测出来的所有目标框坐标，`boxes.size()`表示框的个数，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **scores**: 成员变量，表示单张图片检测出来的所有目标置信度，其元素个数与`boxes.size()`一致
- **landmarks**: 成员变量，表示单张图片检测出来的所有人脸的关键点，其元素个数与`boxes.size()`一致
- **landmarks_per_face**: 成员变量，表示每个人脸框中的关键点的数量。
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## HeadPoseResult 头部姿态结果

HeadPoseResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明头部姿态结果。

### C++ 定义

```c++
fastdeploy::vision::HeadPoseResult
```

```c++
struct HeadPoseResult {
  std::vector<float> euler_angles;
  void Clear();
  std::string Str();
};
```

- **euler_angles**: 成员变量，表示单张人脸图片预测的欧拉角，存放的顺序是(yaw, pitch, roll)， yaw 代表水平转角，pitch 代表垂直角，roll 代表翻滚角，值域都为 [-90,+90]度
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）


API:`fastdeploy.vision.HeadPoseResult`, 该结果返回:
- **euler_angles**(list of float): 成员变量，表示单张人脸图片预测的欧拉角，存放的顺序是(yaw, pitch, roll)， yaw 代表水平转角，pitch 代表垂直角，roll 代表翻滚角，值域都为 [-90, +90]度

## MOTResult 多目标跟踪结果

MOTResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明多目标跟踪中的检测出来的目标框、目标跟踪id、目标类别和目标置信度。

### C++ 定义

```c++
fastdeploy::vision::MOTResult
```  

```c++
struct MOTResult{
  // left top right bottom
  std::vector<std::array<int, 4>> boxes;
  std::vector<int> ids;
  std::vector<float> scores;
  std::vector<int> class_ids;
  void Clear();
  std::string Str();
};
```

- **boxes**: 成员变量，表示单帧画面中检测出来的所有目标框坐标，`boxes.size()`表示框的个数，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **ids**: 成员变量，表示单帧画面中所有目标的id，其元素个数与`boxes.size()`一致
- **scores**: 成员变量，表示单帧画面检测出来的所有目标置信度，其元素个数与`boxes.size()`一致
- **class_ids**: 成员变量，表示单帧画面出来的所有目标类别，其元素个数与`boxes.size()`一致
- **Clear()**: 成员函数，用于清除结构体中存储的结果
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）
