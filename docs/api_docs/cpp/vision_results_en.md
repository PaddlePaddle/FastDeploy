English | [简体中文](./vision_results_cn.md)

# Description of Vision Results

## Image Classification Result

The ClassifyResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the classification result and confidence level of the image.

### C++ Definition

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

- **label_ids**: Member variable which indicates the classification results of a single image. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification results.
- **scores**: Member variable which indicates the confidence level of a single image on the corresponding classification result. Its number is determined by the topk passed in when using the classification model, e.g. it can return the top 5 classification confidence level.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).


## Segmentation Result

The SegmentationResult code is defined in `fastdeploy/vision/common/result.h`, indicating the segmentation category and the segmentation category probability predicted in each pixel in the image.

### C++ Definition

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
  std::string Str();
};
```

- **label_map**: Member variable which indicates the segmentation category of each pixel in a single image. `label_map.size()` indicates the number of pixel points of a image.
- **score_map**: Member variable which indicates the predicted segmentation category probability value corresponding to the label_map one-to-one, the member variable is not empty only when `--output_op none` is specified when exporting the PaddleSeg model, otherwise the member variable is empty.
- **shape**: Member variable which indicates the shape of the output image as H\*W.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Target Detection Result

The DetectionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the target frame, target class and target confidence level detected in the image.

### C++ Definition

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

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image. `boxes.size()` indicates the number of boxes, each box is represented by 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **scores**: Member variable which indicates the confidence level of all targets detected in a single image, where the number of elements is the same as `boxes.size()`.
- **label_ids**: Member variable which indicates all target categories detected in a single image, where the number of elements is the same as `boxes.size()`.
- **masks**: Member variable which indicates all detected instance masks of a single image, where the number of elements and the shape size are the same as `boxes`.
- **contain_masks**: Member variable which indicates whether the detected result contains instance masks, which is generally true for the instance segmentation model.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

```c++
fastdeploy::vision::Mask
```  
```c++
struct Mask {
  std::vector<int32_t> data;
  std::vector<int64_t> shape; // (H,W) ...

  void Clear();
  std::string Str();
};
```  
- **data**: Member variable which indicates a detected mask.
- **shape**: Member variable which indicates the shape of the mask, e.g. (h,w).
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).


## Face Detection Result

The FaceDetectionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the target frames, face landmarks, target confidence and the number of landmark per face.

### C++ Definition

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

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image. `boxes.size()` indicates the number of boxes, each box is represented by 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **scores**: Member variable which indicates the confidence level of all targets detected in a single image, where the number of elements is the same as `boxes.size()`.
- **landmarks**: Member variable which indicates the keypoints of all faces detected in a single image, where the number of elements is the same as `boxes.size()`.
- **landmarks_per_face**: Member variable which indicates the number of keypoints in each face box.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).


## Keypoint Detection Result

The KeyPointDetectionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the coordinates and confidence level of each keypoint of the target's behavior in the image.

### C++ Definition

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

- **keypoints**: Member variable which indicates the coordinates of the identified target behavior keypoint.
  ` keypoints.size() = N * J`:
    - `N`: the number of targets in the image
    - `J`: num_joints (the number of keypoints of a target)
- **scores**: Member variable which indicates the confidence level of the keypoint coordinates of the identified target behavior.
  `scores.size() = N * J`:
    - `N`: the number of targets in the picture
    - `J`:num_joints (the number of keypoints of a target)
- **num_joints**: Member variable which indicates the number of keypoints of a target.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).


## Face Recognition Result

The FaceRecognitionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the image features embedding in the face recognition model.
### C++ Definition

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

- **embedding**: Member variable which indicates the final extracted feature embedding of the face recognition model, and can be used to calculate the facial feature similarity.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Matting Result

The MattingResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the predicted value of alpha transparency predicted and the predicted foreground, etc.

### C++ Definition
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

- **alpha**: It is a one-dimensional vector, indicating the predicted value of alpha transparency. The value range is [0.,1.], and the length is hxw, in which h,w represent the height and the width of the input image seperately.
- **foreground**: It is a one-dimensional vector, indicating the predicted foreground. The value range is [0.,255.], and the length is hxwxc, in which h,w represent the height and the width of the input image, and c is generally 3. This vector is valid only when the model itself predicts the foreground.
- **contain_foreground**: Used to indicate whether the result contains foreground.
- **shape**: Used to indicate the shape of the output. When contain_foreground is false, the shape only contains (h,w), while when contain_foreground is true, the shape contains (h,w,c), in which c is generally 3.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).


## OCR prediction result

The OCRResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the text box detected in the image, text box orientation classification, and the text content.

### C++ Definition

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


## Face Alignment Result

The FaceAlignmentResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate face landmarks.

### C++ Definition
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

- **landmarks**: Member variable which indicates all the key points detected in a single face image.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).


## Head Pose Result

The HeadPoseResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the head pose result.

### C++ Definition

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

- **euler_angles**: Member variable which indicates the Euler angles predicted for a single face image, stored in the order (yaw, pitch, roll), with yaw representing the horizontal turn angle, pitch representing the vertical angle, and roll representing the roll angle, all with a value range of [-90,+90].
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).
