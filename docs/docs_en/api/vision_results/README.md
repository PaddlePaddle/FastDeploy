# Vision Model Inference Results

FastDeploy defines different structs (`csrcs/fastdeploy/vision/common/result.h`) to demonstrate the model inference results according to the task types of vision models. The details are as follows

| Struct                | Doc                                        | Description                                                                  | Related Models          |
|:--------------------- |:------------------------------------------ |:---------------------------------------------------------------------------- |:----------------------- |
| ClassifyResult        | [C++/Python](./classification_result.md)   | Image classification results                                                 | ResNet50、MobileNetV3    |
| SegmentationResult    | [C++/Python](./segmentation_result.md)     | Image segmentation results                                                   | PP-HumanSeg、PP-LiteSeg  |
| DetectionResult       | [C++/Python](./detection_result.md)        | Object detection results                                                     | PPYOLOE、YOLOv7 Series   |
| FaceDetectionResult   | [C++/Python](./face_detection_result.md)   | Object detection results                                                     | SCRFD、RetinaFace Series |
| FaceRecognitionResult | [C++/Python](./face_recognition_result.md) | Object detection results                                                     | ArcFace、CosFace Series  |
| MattingResult         | [C++/Python](./matting_result.md)          | Object detection results                                                     | MODNet Series           |
| OCRResult             | [C++/Python](./ocr_result.md)              | Text box detection, classification and optical character recognition results | OCR Series              |
