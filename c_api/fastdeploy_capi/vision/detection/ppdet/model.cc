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

#include "fastdeploy_capi/internal/types_internal.h"
#include "fastdeploy_capi/vision/visualize.h"

#ifdef __cplusplus
extern "C" {
#endif

// PPYOLOE

DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(PPYOLOE, ppyoloe_model)
DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper)
DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper)
DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper)
DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(PPYOLOE, fd_ppyoloe_wrapper)

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
