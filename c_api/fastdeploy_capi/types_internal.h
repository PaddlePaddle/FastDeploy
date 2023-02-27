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

#pragma once

#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy_capi/fd_type.h"
#include <memory>

#ifdef ENABLE_VISION
#include "fastdeploy/vision/classification/ppcls/model.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/detection/ppdet/model.h"
#include "fastdeploy/vision/ocr/ppocr/classifier.h"
#include "fastdeploy/vision/ocr/ppocr/dbdetector.h"
#include "fastdeploy/vision/ocr/ppocr/recognizer.h"
#include "fastdeploy/vision/ocr/ppocr/ppocr_v2.h"
#include "fastdeploy/vision/ocr/ppocr/ppocr_v3.h"
#include "fastdeploy/vision/segmentation/ppseg/model.h"

#define DEFINE_RESULT_WRAPPER_STRUCT(typename, varname) typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::vision::typename> varname; \
} FD_C_##typename##Wrapper

#define DEFINE_CLASSIFICATION_MODEL_WRAPPER_STRUCT(typename, varname)  typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::vision::classification::typename> \
      varname; \
} FD_C_##typename##Wrapper

#define DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(typename, varname)  typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::vision::detection::typename> varname; \
} FD_C_##typename##Wrapper

#define DEFINE_OCR_MODEL_WRAPPER_STRUCT(typename, varname)  typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::vision::ocr::typename> varname; \
} FD_C_##typename##Wrapper

#define DEFINE_PIPELINE_MODEL_WRAPPER_STRUCT(typename, varname)  typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::pipeline::typename> varname; \
} FD_C_##typename##Wrapper

#define DEFINE_SEGMENTATION_MODEL_WRAPPER_STRUCT(typename, varname)  typedef struct FD_C_##typename##Wrapper { \
  std::unique_ptr<fastdeploy::vision::segmentation::typename> varname; \
} FD_C_##typename##Wrapper

// -------------  belows are wrapper struct define --------------------- //

// Results:

// ClassifyResult
DEFINE_RESULT_WRAPPER_STRUCT(ClassifyResult, classify_result);

// DetectionResult
DEFINE_RESULT_WRAPPER_STRUCT(DetectionResult, detection_result);


// OCRResult
DEFINE_RESULT_WRAPPER_STRUCT(OCRResult, ocr_result);

// Segmentation Result
DEFINE_RESULT_WRAPPER_STRUCT(SegmentationResult, segmentation_result);

// Models:

// Classification

// PaddleClasModel

DEFINE_CLASSIFICATION_MODEL_WRAPPER_STRUCT(PaddleClasModel, paddleclas_model);

// Detection

// PPYOLOE
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PPYOLOE, ppyoloe_model);


// PicoDet
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PicoDet, picodet_model);

// PPYOLO
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PPYOLO, ppyolo_model);

// YOLOv3
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(YOLOv3, yolov3_model);

// PaddleYOLOX
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PaddleYOLOX, paddleyolox_model);

// FasterRCNN
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(FasterRCNN, fasterrcnn_model);

// MaskRCNN
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(MaskRCNN, maskrcnn_model);

// SSD
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(SSD, ssd_model);

// PaddleYOLOv5
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PaddleYOLOv5, paddleyolov5_model);

// PaddleYOLOv6
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PaddleYOLOv6, paddleyolov6_model);

// PaddleYOLOv7
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PaddleYOLOv7, paddleyolov7_model);

// PaddleYOLOv8
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PaddleYOLOv8, paddleyolov8_model);

// RTMDet
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(RTMDet, rtmdet_model);

// CascadeRCNN
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(CascadeRCNN, cascadercnn_model);

// PSSDet
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(PSSDet, pssdet_model);

// RetinaNet
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(RetinaNet, retinanet_model);


// FCOS
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(FCOS, fcos_model);

// TTFNet
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(TTFNet, ttfnet_model);

// TOOD
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(TOOD, tood_model);

// GFL
DEFINE_DETECTION_MODEL_WRAPPER_STRUCT(GFL, gfl_model);

// OCR models

// Recognizer
DEFINE_OCR_MODEL_WRAPPER_STRUCT(Recognizer, recognizer_model);

// DBDetector
DEFINE_OCR_MODEL_WRAPPER_STRUCT(DBDetector, dbdetector_model);

// Classifier
DEFINE_OCR_MODEL_WRAPPER_STRUCT(Classifier, classifier_model);

// PPOCRv2
DEFINE_PIPELINE_MODEL_WRAPPER_STRUCT(PPOCRv2, ppocrv2_model);

// PPOCRv3
DEFINE_PIPELINE_MODEL_WRAPPER_STRUCT(PPOCRv3, ppocrv3_model);

// Segmentation models

// PaddleSegModel
DEFINE_SEGMENTATION_MODEL_WRAPPER_STRUCT(PaddleSegModel, segmentation_model);

// -------------  belows are function declaration for get ptr from wrapper --------------------- //

#define DECLARE_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, varname) std::unique_ptr<fastdeploy::vision::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* varname)

#define DECLARE_CLASSIFICATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, varname) std::unique_ptr<fastdeploy::vision::classification::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* varname)

#define DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, varname) std::unique_ptr<fastdeploy::vision::detection::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* varname)


#define DECLARE_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, varname) std::unique_ptr<fastdeploy::vision::ocr::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* varname)

#define DECLARE_PIPELINE_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, varname) std::unique_ptr<fastdeploy::pipeline::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* varname)

#define DECLARE_SEGMENTATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, varname) std::unique_ptr<fastdeploy::vision::segmentation::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* varname)


namespace fastdeploy {

// results:

// ClassifyResult
DECLARE_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(ClassifyResult,
                                             fd_classify_result_wrapper);
// DetectionResult
DECLARE_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(DetectionResult,
                                             fd_detection_result_wrapper);


// OCRResult
DECLARE_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(OCRResult,
                                             fd_ocr_result_wrapper);

// SegementationResult
DECLARE_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(SegmentationResult,
                                             fd_segmentation_result_wrapper);


// Models:

// Classification

// PaddleClasModel

DECLARE_CLASSIFICATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleClasModel, fd_paddleclas_model_wrapper);


// detection models:

// PPYOLOE

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PPYOLOE,
                                                      fd_ppyoloe_wrapper);

// PicoDet

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PicoDet,
                                                      fd_picodet_wrapper);

// PPYOLO

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PPYOLO,
                                                      fd_ppyolo_wrapper);

// YOLOv3

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(YOLOv3,
                                                      fd_yolov3_wrapper);

// PaddleYOLOX

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PaddleYOLOX,
                                                      fd_paddleyolox_wrapper);

// FasterRCNN

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(FasterRCNN,
                                                      fd_fasterrcnn_wrapper);

// MaskRCNN

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(MaskRCNN,
                                                      fd_maskrcnn_wrapper);

// SSD

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(SSD,
                                                      fd_ssd_wrapper);

// PaddleYOLOv5

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PaddleYOLOv5,
                                                      fd_paddleyolov5_wrapper);

// PaddleYOLOv6

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PaddleYOLOv6,
                                                      fd_paddleyolov6_wrapper);

// PaddleYOLOv7

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PaddleYOLOv7,
                                                      fd_paddleyolov7_wrapper);

// PaddleYOLOv8

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PaddleYOLOv8,
                                                      fd_paddleyolov8_wrapper);

// RTMDet

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(RTMDet,
                                                      fd_rtmdet_wrapper);

// CascadeRCNN

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(CascadeRCNN,
                                                      fd_cascadercnn_wrapper);

// PSSDet

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PSSDet,
                                                      fd_pssdet_wrapper);

// RetinaNet

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(RetinaNet,
                                                      fd_retinanet_wrapper);

// FCOS

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(FCOS,
                                                      fd_fcos_wrapper);

// TTFNet

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(TTFNet,
                                                      fd_ttfnet_wrapper);

// TOOD

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(TOOD,
                                                      fd_tood_wrapper);

// GFL

DECLARE_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(GFL,
                                                      fd_gfl_wrapper);

// OCR models

// Recognizer
DECLARE_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(Recognizer, fd_recognizer_wrapper);

// DBDetector
DECLARE_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(DBDetector, fd_dbdetector_wrapper);

// Classifier
DECLARE_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(Classifier, fd_classifier_wrapper);

// PPOCRv2
DECLARE_PIPELINE_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PPOCRv2, fd_ppocrv2_wrapper);

// PPOCRv3
DECLARE_PIPELINE_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(PPOCRv3, fd_ppocrv3_wrapper);

// Segmentation models

// PaddleSegModel
DECLARE_SEGMENTATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(
    PaddleSegModel, fd_paddleseg_model_wrapper);

}  // namespace fastdeploy

#endif



typedef struct FD_C_RuntimeOptionWrapper {
  std::unique_ptr<fastdeploy::RuntimeOption> runtime_option;
} FD_C_RuntimeOptionWrapper;

namespace fastdeploy {
std::unique_ptr<fastdeploy::RuntimeOption>&
FD_C_CheckAndConvertRuntimeOptionWrapper(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper);
}

#define CHECK_AND_CONVERT_FD_TYPE(TYPENAME, variable_name)                     \
  fastdeploy::FD_C_CheckAndConvert##TYPENAME(variable_name)

#define DECL_AND_IMPLEMENT_RESULT_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, var_wrapper_name, var_ptr_name) std::unique_ptr<fastdeploy::vision::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* var_wrapper_name) { \
  FDASSERT(var_wrapper_name != nullptr, \
           "The pointer of " #var_wrapper_name " shouldn't be nullptr."); \
  return var_wrapper_name->var_ptr_name; \
}

#define DECL_AND_IMPLEMENT_CLASSIFICATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, var_wrapper_name, var_ptr_name) std::unique_ptr<fastdeploy::vision::classification::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* var_wrapper_name) { \
  FDASSERT(var_wrapper_name != nullptr, \
           "The pointer of " #var_wrapper_name " shouldn't be nullptr."); \
  return var_wrapper_name->var_ptr_name; \
}

#define DECL_AND_IMPLEMENT_DETECTION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, var_wrapper_name, var_ptr_name) std::unique_ptr<fastdeploy::vision::detection::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* var_wrapper_name) { \
  FDASSERT(var_wrapper_name != nullptr, \
           "The pointer of " #var_wrapper_name " shouldn't be nullptr."); \
  return var_wrapper_name->var_ptr_name; \
}


#define DECL_AND_IMPLEMENT_OCR_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, var_wrapper_name, var_ptr_name) std::unique_ptr<fastdeploy::vision::ocr::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* var_wrapper_name) { \
  FDASSERT(var_wrapper_name != nullptr, \
           "The pointer of " #var_wrapper_name " shouldn't be nullptr."); \
  return var_wrapper_name->var_ptr_name; \
}

#define DECL_AND_IMPLEMENT_PIPELINE_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, var_wrapper_name, var_ptr_name) std::unique_ptr<fastdeploy::pipeline::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* var_wrapper_name) { \
  FDASSERT(var_wrapper_name != nullptr, \
           "The pointer of " #var_wrapper_name " shouldn't be nullptr."); \
  return var_wrapper_name->var_ptr_name; \
}

#define DECL_AND_IMPLEMENT_SEGMENTATION_MODEL_FUNC_FOR_GET_PTR_FROM_WRAPPER(typename, var_wrapper_name, var_ptr_name) std::unique_ptr<fastdeploy::vision::segmentation::typename>& \
FD_C_CheckAndConvert##typename##Wrapper( \
    FD_C_##typename##Wrapper* var_wrapper_name) { \
  FDASSERT(var_wrapper_name != nullptr, \
           "The pointer of " #var_wrapper_name " shouldn't be nullptr."); \
  return var_wrapper_name->var_ptr_name; \
}
