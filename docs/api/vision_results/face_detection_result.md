# FaceDetectionResult 人脸检测结果

FaceDetectionResult 代码定义在`csrcs/fastdeploy/vision/common/result.h`中，用于表明人脸检测出来的目标框、人脸landmarks，目标置信度和每张人脸的landmark数量。

## C++ 定义

`fastdeploy::vision::FaceDetectionResult`

```
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

## Python 定义

`fastdeploy.vision.FaceDetectionResult`

- **boxes**(list of list(float)): 成员变量，表示单张图片检测出来的所有目标框坐标。boxes是一个list，其每个元素为一个长度为4的list， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标
- **scores**(list of float): 成员变量，表示单张图片检测出来的所有目标置信度
- **landmarks**(list of list(float)): 成员变量，表示单张图片检测出来的所有人脸的关键点
- **landmarks_per_face**(int): 成员变量，表示每个人脸框中的关键点的数量。
