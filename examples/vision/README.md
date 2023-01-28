English | [简体中文](README_CN.md) 

# Visual Model Deployment

This directory provides the deployment of various visual models, including the following task types

| Task Type           |  Description                               | Predicted Structure                                                                         |
|:-------------- |:----------------------------------- |:-------------------------------------------------------------------------------- |
| Detection      | Target detection. Input the image, detect the object’s position in the image, and return the detected box coordinates, category, and confidence coefficient | [DetectionResult](../../docs/api/vision_results/detection_result.md)       |
| Segmentation   | Semantic segmentation. Input the image and output the classification and confidence coefficient of each pixel         | [SegmentationResult](../../docs/api/vision_results/segmentation_result.md) |
| Classification | Image classification. Input the image and output the classification result and confidence coefficient of the image             | [ClassifyResult](../../docs/api/vision_results/classification_result.md)   |
| FaceDetection | Face detection. Input the image, detect the position of faces in the image, and return detected box coordinates and key points of faces            | [FaceDetectionResult](../../docs/api/vision_results/face_detection_result.md)   |
| FaceAlignment |  Face alignment(key points detection).Input the image and return face key points           | [FaceAlignmentResult](../../docs/api/vision_results/face_alignment_result.md)   |
| KeypointDetection   | Key point detection. Input the image and return the coordinates and confidence coefficient of the key points of the person's behavior in the image         | [KeyPointDetectionResult](../../docs/api/vision_results/keypointdetection_result.md) |
| FaceRecognition | Face recognition. Input the image and return an embedding of facial features that can be used for similarity calculation            | [FaceRecognitionResult](../../docs/api/vision_results/face_recognition_result.md)   |
| Matting | Matting. Input the image and return the Alpha value of each pixel in the foreground of the image           | [MattingResult](../../docs/api/vision_results/matting_result.md)   |
| OCR | Text box detection, classification, and text box content recognition. Input the image and return the text box’s coordinates, orientation category, and content         | [OCRResult](../../docs/api/vision_results/ocr_result.md)   |
| MOT | Multi-objective tracking. Input the image and detect the position of objects in the image, and return detected box coordinates, object id, and class confidence        | [MOTResult](../../docs/api/vision_results/mot_result.md)   |
| HeadPose | Head posture estimation. Return head Euler angle            | [HeadPoseResult](../../docs/api/vision_results/headpose_result.md)   |

## FastDeploy API Design

Generally, visual models have a uniform task paradigm. When designing API (including C++/Python), FastDeploy conducts four steps to deploy visual models

- Model loading
- Image pre-processing
- Model Inference
- Post-processing of inference results

Targeted at the vision suite of PaddlePaddle and external popular models, FastDeploy provides an end-to-end deployment service. Users merely prepare the model and follow these steps to complete the deployment

- Model Loading
- Calling the `predict`interface

When deploying visual models, FastDeploy supports one-click switching of the backend inference engine. Please refer to [How to switch model inference engine](../../docs/en/faq/how_to_change_backend.md).

