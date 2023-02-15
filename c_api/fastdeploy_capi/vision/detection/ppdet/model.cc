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

#include "fastdeploy_capi/vision/detection/ppdet/model.h"

#include "fastdeploy_capi/types_internal.h"
#include "fastdeploy_capi/vision/visualize.h"

#ifdef __cplusplus
extern "C" {
#endif

// PPYOLOE

FD_C_PPYOLOEWrapper* FD_C_CreatesPPYOLOEWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  IMPLEMENT_CREATE_WRAPPER_FUNCTION(PPYOLOE, ppyoloe_model);
}

void FD_C_DestroyPPYOLOEWrapper(
    __fd_take FD_C_PPYOLOEWrapper* fd_ppyoloe_wrapper) {
  IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper);
}

FD_C_Bool FD_C_PPYOLOEWrapperPredict(
    FD_C_PPYOLOEWrapper* fd_ppyoloe_wrapper, FD_C_Mat img,
    FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper) {
  IMPLEMENT_PREDICT_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper);
}

FD_C_Bool FD_C_PPYOLOEWrapperInitialized(
    FD_C_PPYOLOEWrapper* fd_ppyoloe_wrapper) {
  IMPLEMENT_INITIALIZED_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper);
}

FD_C_DetectionResult* FD_C_DetectionResultToC(
    fastdeploy::vision::DetectionResult* detection_result) {
  // Internal use, transfer fastdeploy::vision::DetectionResult to
  // FD_C_DetectionResult
  FD_C_DetectionResult* fd_c_detection_result = new FD_C_DetectionResult();
  // copy boxes
  const int boxes_coordinate_dim = 4;
  fd_c_detection_result->boxes.size = detection_result->boxes.size();
  fd_c_detection_result->boxes.data =
      new FD_C_OneDimArrayFloat[fd_c_detection_result->boxes.size];
  for (size_t i = 0; i < detection_result->boxes.size(); i++) {
    fd_c_detection_result->boxes.data[i].size = boxes_coordinate_dim;
    fd_c_detection_result->boxes.data[i].data = new float[boxes_coordinate_dim];
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      fd_c_detection_result->boxes.data[i].data[j] =
          detection_result->boxes[i][j];
    }
  }
  // copy scores
  fd_c_detection_result->scores.size = detection_result->scores.size();
  fd_c_detection_result->scores.data =
      new float[fd_c_detection_result->scores.size];
  memcpy(fd_c_detection_result->scores.data, detection_result->scores.data(),
         sizeof(float) * fd_c_detection_result->scores.size);
  // copy label_ids
  fd_c_detection_result->label_ids.size = detection_result->label_ids.size();
  fd_c_detection_result->label_ids.data =
      new int32_t[fd_c_detection_result->label_ids.size];
  memcpy(fd_c_detection_result->label_ids.data,
         detection_result->label_ids.data(),
         sizeof(int32_t) * fd_c_detection_result->label_ids.size);
  // copy masks
  fd_c_detection_result->masks.size = detection_result->masks.size();
  fd_c_detection_result->masks.data =
      new FD_C_Mask[fd_c_detection_result->masks.size];
  for (size_t i = 0; i < detection_result->masks.size(); i++) {
    // copy data in mask
    fd_c_detection_result->masks.data[i].data.size =
        detection_result->masks[i].data.size();
    fd_c_detection_result->masks.data[i].data.data =
        new uint8_t[detection_result->masks[i].data.size()];
    memcpy(fd_c_detection_result->masks.data[i].data.data,
           detection_result->masks[i].data.data(),
           sizeof(uint8_t) * detection_result->masks[i].data.size());
    // copy shape in mask
    fd_c_detection_result->masks.data[i].shape.size =
        detection_result->masks[i].shape.size();
    fd_c_detection_result->masks.data[i].shape.data =
        new int64_t[detection_result->masks[i].shape.size()];
    memcpy(fd_c_detection_result->masks.data[i].shape.data,
           detection_result->masks[i].shape.data(),
           sizeof(int64_t) * detection_result->masks[i].shape.size());
    fd_c_detection_result->masks.data[i].type =
        static_cast<FD_C_ResultType>(detection_result->masks[i].type);
  }
  fd_c_detection_result->contain_masks = detection_result->contain_masks;
  fd_c_detection_result->type =
      static_cast<FD_C_ResultType>(detection_result->type);
  return fd_c_detection_result;
}

FD_C_Bool FD_C_PPYOLOEWrapperBatchPredict(
    FD_C_PPYOLOEWrapper* fd_ppyoloe_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimDetectionResult* results) {
  IMPLEMENT_BATCH_PREDICT_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper);
}

// PicoDet
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PicoDet, picodet_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PicoDet, fd_picodet_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PicoDet, fd_picodet_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PicoDet, fd_picodet_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PicoDet, fd_picodet_wrapper)

// PPYOLO
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PPYOLO, ppyolo_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPYOLO, fd_ppyolo_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PPYOLO, fd_ppyolo_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PPYOLO, fd_ppyolo_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PPYOLO, fd_ppyolo_wrapper)

// YOLOv3
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(YOLOv3, yolov3_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(YOLOv3, fd_yolov3_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(YOLOv3, fd_yolov3_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(YOLOv3, fd_yolov3_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(YOLOv3, fd_yolov3_wrapper)

// PaddleYOLOX
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PaddleYOLOX, paddleyolox_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PaddleYOLOX,
                                               fd_paddleyolox_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PaddleYOLOX, fd_paddleyolox_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PaddleYOLOX, fd_paddleyolox_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PaddleYOLOX,
                                             fd_paddleyolox_wrapper)

// FasterRCNN
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(FasterRCNN, fasterrcnn_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(FasterRCNN,
                                               fd_fasterrcnn_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(FasterRCNN, fd_fasterrcnn_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(FasterRCNN, fd_fasterrcnn_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(FasterRCNN, fd_fasterrcnn_wrapper)

// MaskRCNN
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(MaskRCNN, maskrcnn_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(MaskRCNN, fd_maskrcnn_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(MaskRCNN, fd_maskrcnn_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(MaskRCNN, fd_maskrcnn_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(MaskRCNN, fd_maskrcnn_wrapper)

// SSD
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(SSD, ssd_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(SSD, fd_ssd_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(SSD, fd_ssd_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(SSD, fd_ssd_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(SSD, fd_ssd_wrapper)

// PaddleYOLOv5
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PaddleYOLOv5, paddleyolov5_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv5,
                                               fd_paddleyolov5_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PaddleYOLOv5, fd_paddleyolov5_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PaddleYOLOv5,
                                           fd_paddleyolov5_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PaddleYOLOv5,
                                             fd_paddleyolov5_wrapper)

// PaddleYOLOv6
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PaddleYOLOv6, paddleyolov6_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv6,
                                               fd_paddleyolov6_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PaddleYOLOv6, fd_paddleyolov6_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PaddleYOLOv6,
                                           fd_paddleyolov6_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PaddleYOLOv6,
                                             fd_paddleyolov6_wrapper)

// PaddleYOLOv7
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PaddleYOLOv7, paddleyolov7_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv7,
                                               fd_paddleyolov7_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PaddleYOLOv7, fd_paddleyolov7_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PaddleYOLOv7,
                                           fd_paddleyolov7_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PaddleYOLOv7,
                                             fd_paddleyolov7_wrapper)

// PaddleYOLOv8
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PaddleYOLOv8, paddleyolov8_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv8,
                                               fd_paddleyolov8_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PaddleYOLOv8, fd_paddleyolov8_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PaddleYOLOv8,
                                           fd_paddleyolov8_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PaddleYOLOv8,
                                             fd_paddleyolov8_wrapper)

// RTMDet
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(RTMDet, rtmdet_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(RTMDet, fd_rtmdet_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(RTMDet, fd_rtmdet_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(RTMDet, fd_rtmdet_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(RTMDet, fd_rtmdet_wrapper)

// CascadeRCNN
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(CascadeRCNN, cascadercnn_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(CascadeRCNN,
                                               fd_cascadercnn_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(CascadeRCNN, fd_cascadercnn_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(CascadeRCNN, fd_cascadercnn_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(CascadeRCNN,
                                             fd_cascadercnn_wrapper)

// PSSDet
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PSSDet, pssdet_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PSSDet, fd_pssdet_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PSSDet, fd_pssdet_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PSSDet, fd_pssdet_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PSSDet, fd_pssdet_wrapper)

// RetinaNet
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(RetinaNet, retinanet_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(RetinaNet, fd_retinanet_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(RetinaNet, fd_retinanet_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(RetinaNet, fd_retinanet_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(RetinaNet, fd_retinanet_wrapper)

// FCOS
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(FCOS, fcos_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(FCOS, fd_fcos_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(FCOS, fd_fcos_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(FCOS, fd_fcos_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(FCOS, fd_fcos_wrapper)

// TTFNet
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(TTFNet, ttfnet_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(TTFNet, fd_ttfnet_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(TTFNet, fd_ttfnet_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(TTFNet, fd_ttfnet_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(TTFNet, fd_ttfnet_wrapper)

// TOOD
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(TOOD, tood_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(TOOD, fd_tood_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(TOOD, fd_tood_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(TOOD, fd_tood_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(TOOD, fd_tood_wrapper)

// GFL
DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(GFL, gfl_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(GFL, fd_gfl_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(GFL, fd_gfl_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(GFL, fd_gfl_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(GFL, fd_gfl_wrapper)

#ifdef __cplusplus
}
#endif
