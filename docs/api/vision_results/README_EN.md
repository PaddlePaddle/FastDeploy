[简体中文](README_CN.md)| English
# Prediction Results of the Vision Model

FastDeploy defines different structures (`fastdeploy/vision/common/result.h`) to express the model prediction results according to the vision model task.

| Structure | Document | Description | Corresponding Model |
|:------------------------|:----------------------------------------------|:------------------|:------------------------|
| ClassifyResult | [C++/Python document](./classification_result_EN.md) | Image classification return results | ResNet50, MobileNetV3, etc. |
| SegmentationResult | [C++/Python document](./segmentation_result_EN.md) | Image segmentation result | PP-HumanSeg, PP-LiteSeg, etc. |
| DetectionResult | [C++/Python document](./detection_result_EN.md) | Target detection result | PP-YOLOE, YOLOv7, etc. |
| FaceDetectionResult | [C++/Python document](./face_detection_result_EN.md) |  Result of face detection | SCRFD, RetinaFace, etc. |
| FaceAlignmentResult | [C++/Python document](./face_alignment_result_EN.md) | Face alignment result(Face keypoint detection) | PFLD model, etc. |
| KeyPointDetectionResult | [C++/Python document](./keypointdetection_result_EN.md) | Result of keypoint detection | PP-Tinypose model, etc. |
| FaceRecognitionResult | [C++/Python document](./face_recognition_result_EN.md) | Result of face recognition | ArcFace, CosFace, etc. |
| MattingResult | [C++/Python document](./matting_result_EN.md) | Image/video keying result | MODNet, RVM, etc. |
| OCRResult | [C++/Python document](./ocr_result_EN.md) | Text box detection, classification and text recognition result | OCR, etc. |
| MOTResult | [C++/Python document](./mot_result_EN.md) | Multi-target tracking result | pptracking, etc. |
| HeadPoseResult | [C++/Python document](./headpose_result_EN.md) | Head pose estimation result | FSANet, etc. |
