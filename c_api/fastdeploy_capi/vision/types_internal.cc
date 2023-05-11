// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy_capi/vision/types_internal.h"

namespace fastdeploy {

#ifdef ENABLE_VISION

// results:

// ClassifyResult
DECL_AND_IMPLEMENT_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    ClassifyResult, fd_classify_result_wrapper, classify_result)
// DetectionResult
DECL_AND_IMPLEMENT_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    DetectionResult, fd_detection_result_wrapper, detection_result)

// OCRResult
DECL_AND_IMPLEMENT_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(OCRResult,
                                                        fd_ocr_result_wrapper,
                                                        ocr_result)
// SegmentationResult
DECL_AND_IMPLEMENT_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    SegmentationResult, fd_segmentation_result_wrapper, segmentation_result)

// Models:

// Classification

// PaddleClasModel
DECL_AND_IMPLEMENT_CLASSIFICATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleClasModel, fd_paddleclas_model_wrapper, paddleclas_model)

// detection models:

// PPYOLOE

DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PPYOLOE, fd_ppyoloe_wrapper, ppyoloe_model)

// PicoDet
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PicoDet, fd_picodet_wrapper, picodet_model)

// PPYOLO
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PPYOLO, fd_ppyolo_wrapper, ppyolo_model)

// YOLOv3
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    YOLOv3, fd_yolov3_wrapper, yolov3_model)

// PaddleYOLOX
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleYOLOX, fd_paddleyolox_wrapper, paddleyolox_model)

// FasterRCNN
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    FasterRCNN, fd_fasterrcnn_wrapper, fasterrcnn_model)

// MaskRCNN
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    MaskRCNN, fd_maskrcnn_wrapper, maskrcnn_model)

// SSD
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(SSD,
                                                                 fd_ssd_wrapper,
                                                                 ssd_model)

// PaddleYOLOv5
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleYOLOv5, fd_paddleyolov5_wrapper, paddleyolov5_model)

// PaddleYOLOv6
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleYOLOv6, fd_paddleyolov6_wrapper, paddleyolov6_model)

// PaddleYOLOv7
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleYOLOv7, fd_paddleyolov7_wrapper, paddleyolov7_model)

// PaddleYOLOv8
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleYOLOv8, fd_paddleyolov8_wrapper, paddleyolov8_model)

// RTMDet
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    RTMDet, fd_rtmdet_wrapper, rtmdet_model)

// CascadeRCNN
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    CascadeRCNN, fd_cascadercnn_wrapper, cascadercnn_model)

// PSSDet
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PSSDet, fd_pssdet_wrapper, pssdet_model)

// RetinaNet
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    RetinaNet, fd_retinanet_wrapper, retinanet_model)

// FCOS
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    FCOS, fd_fcos_wrapper, fcos_model)

// TTFNet
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    TTFNet, fd_ttfnet_wrapper, ttfnet_model)

// TOOD
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    TOOD, fd_tood_wrapper, tood_model)

// GFL
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(GFL,
                                                                 fd_gfl_wrapper,
                                                                 gfl_model)

// YOLOv5
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    YOLOv5, fd_yolov5_wrapper, yolov5_model)

// YOLOv7
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    YOLOv7, fd_yolov7_wrapper, yolov7_model)

// YOLOv8
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    YOLOv8, fd_yolov8_wrapper, yolov8_model)

// YOLOv6
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    YOLOv6, fd_yolov6_wrapper, yolov6_model)

// YOLOR
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    YOLOR, fd_yolor_wrapper, yolor_model)

// YOLOX
DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    YOLOX, fd_yolox_wrapper, yolox_model)

// OCR models

// Recognizer
DECL_AND_IMPLEMENT_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    Recognizer, fd_recognizer_wrapper, recognizer_model);

// DBDetector
DECL_AND_IMPLEMENT_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    DBDetector, fd_dbdetector_wrapper, dbdetector_model);

// Classifier
DECL_AND_IMPLEMENT_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    Classifier, fd_classifier_wrapper, classifier_model);

// Table
DECL_AND_IMPLEMENT_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    StructureV2Table, fd_structurev2_table_wrapper, table_model);

// PPOCRv2
DECL_AND_IMPLEMENT_PIPELINE_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PPOCRv2, fd_ppocrv2_wrapper, ppocrv2_model);

// PPOCRv3
DECL_AND_IMPLEMENT_PIPELINE_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PPOCRv3, fd_ppocrv3_wrapper, ppocrv3_model);

// PPStructureV2Table
DECL_AND_IMPLEMENT_PIPELINE_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PPStructureV2Table, fd_ppstructurev2_table_wrapper,
    ppstructurev2table_model);

// Segmentation models

// PaddleSegModel
DECL_AND_IMPLEMENT_SEGMENTATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleSegModel, fd_paddleseg_model_wrapper, segmentation_model);

#endif

}  // namespace fastdeploy