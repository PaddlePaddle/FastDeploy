# 视觉模型部署

本目录下提供了各类视觉模型的部署，主要涵盖以下任务类型

| 任务类型           | 说明                                  | 预测结果结构体                                                                          |
|:-------------- |:----------------------------------- |:-------------------------------------------------------------------------------- |
| Detection      | 目标检测，输入图像，检测图像中物体位置，并返回检测框坐标及类别和置信度 | [DetectionResult](../../docs/api/vision_results/detection_result.md)       |
| Segmentation   | 语义分割，输入图像，给出图像中每个像素的分类及置信度          | [SegmentationResult](../../docs/api/vision_results/segmentation_result.md) |
| Classification | 图像分类，输入图像，给出图像的分类结果和置信度             | [ClassifyResult](../../docs/api/vision_results/classification_result.md)   |
| FaceDetection | 人脸检测，输入图像，检测图像中人脸位置，并返回检测框坐标及人脸关键点             | [FaceDetectionResult](../../docs/api/vision_results/face_detection_result.md)   |
| FaceRecognition | 人脸识别，输入图像，返回可用于相似度计算的人脸特征的embedding            | [FaceRecognitionResult](../../docs/api/vision_results/face_recognition_result.md)   |
| Matting | 抠图，输入图像，返回图片的前景每个像素点的Alpha值            | [MattingResult](../../docs/api/vision_results/matting_result.md)   |
## FastDeploy API设计

视觉模型具有较有统一任务范式，在设计API时（包括C++/Python），FastDeploy将视觉模型的部署拆分为四个步骤

- 模型加载
- 图像预处理
- 模型推理
- 推理结果后处理

FastDeploy针对飞桨的视觉套件，以及外部热门模型，提供端到端的部署服务，用户只需准备模型，按以下步骤即可完成整个模型的部署

- 加载模型
- 调用`predict`接口
