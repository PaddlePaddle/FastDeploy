# 视觉模型预测结果说明

FastDeploy根据视觉模型的任务类型，定义了不同的结构体(`csrcs/fastdeploy/vision/common/result.h`)来表达模型预测结果，具体如下表所示

| 结构体 | 文档 | 说明 | 相应模型 |
| :----- | :--- | :---- | :------- |
| ClassificationResult | [C++/Python文档](./classification_result.md) | 图像分类返回结果 | ResNet50、MobileNetV3等 |
| DetectionResult | [C++/Python文档](./detection_result.md) | 目标检测返回结果 | PPYOLOE、YOLOv7系列模型等 |
| FaceDetectionResult | [C++/Python文档](./face_detection_result.md) | 目标检测返回结果 | SCRFD、RetinaFace系列模型等 |
| FaceRecognitionResult | [C++/Python文档](./face_recognition_result.md) | 目标检测返回结果 | ArcFace、CosFace系列模型等 |
| MattingResult | [C++/Python文档](./matting_result.md) | 目标检测返回结果 | MODNet系列模型等 |
