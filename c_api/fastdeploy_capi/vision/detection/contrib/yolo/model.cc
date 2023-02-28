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

#include "fastdeploy_capi/vision/detection/contrib/yolo/model.h"

#include "fastdeploy_capi/internal/types_internal.h"
#include "fastdeploy_capi/vision/visualize.h"

#ifdef __cplusplus
extern "C" {
#endif

// YOLOv5
YOLO_DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(YOLOv5, yolov5_model)
YOLO_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(YOLOv5, fd_yolov5_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(YOLOv5, fd_yolov5_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(YOLOv5, fd_yolov5_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(YOLOv5, fd_yolov5_wrapper)

// YOLOv7
YOLO_DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(YOLOv7, yolov7_model)
YOLO_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(YOLOv7, fd_yolov7_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(YOLOv7, fd_yolov7_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(YOLOv7, fd_yolov7_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(YOLOv7, fd_yolov7_wrapper)

// YOLOv8
YOLO_DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(YOLOv8, yolov8_model)
YOLO_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(YOLOv8, fd_yolov8_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(YOLOv8, fd_yolov8_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(YOLOv8, fd_yolov8_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(YOLOv8, fd_yolov8_wrapper)

// YOLOv6
YOLO_DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(YOLOv6, yolov6_model)
YOLO_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(YOLOv6, fd_yolov6_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION_WITH_THREASHOLD(YOLOv6,
                                                            fd_yolov6_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(YOLOv6, fd_yolov6_wrapper)

// YOLOR
YOLO_DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(YOLOR, yolor_model)
YOLO_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(YOLOR, fd_yolor_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION_WITH_THREASHOLD(YOLOR,
                                                            fd_yolor_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(YOLOR, fd_yolor_wrapper)

// YOLOX
YOLO_DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(YOLOX, yolox_model)
YOLO_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(YOLOX, fd_yolox_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION_WITH_THREASHOLD(YOLOX,
                                                            fd_yolox_wrapper)
YOLO_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(YOLOX, fd_yolox_wrapper)

#ifdef __cplusplus
}
#endif