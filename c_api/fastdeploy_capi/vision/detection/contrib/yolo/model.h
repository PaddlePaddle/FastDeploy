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

#include "fastdeploy_capi/core/fd_common.h"
#include "fastdeploy_capi/core/fd_type.h"
#include "fastdeploy_capi/runtime/runtime_option.h"
#include "fastdeploy_capi/vision/result.h"
#include "fastdeploy_capi/vision/detection/contrib/yolo/base_define.h"

typedef struct FD_C_RuntimeOptionWrapper FD_C_RuntimeOptionWrapper;

#ifdef __cplusplus
extern "C" {
#endif

// YOLOv5

typedef struct FD_C_YOLOv5Wrapper FD_C_YOLOv5Wrapper;

/** \brief Create a new FD_C_YOLOv5Wrapper object
 *
 * \param[in] model_file Path of model file, e.g ./yolov5.onnx
 * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLOv5Wrapper object
 */

YOLO_DECLARE_CREATE_WRAPPER_FUNCTION(YOLOv5);

/** \brief Destroy a FD_C_YOLOv5Wrapper object
 *
 * \param[in] fd_c_yolov5_wrapper pointer to FD_C_YOLOv5Wrapper object
 */

YOLO_DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOv5, fd_c_yolov5_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolov5_wrapper pointer to FD_C_YOLOv5Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

YOLO_DECLARE_PREDICT_FUNCTION(YOLOv5, fd_c_yolov5_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_yolov5_wrapper pointer to FD_C_YOLOv5Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

YOLO_DECLARE_INITIALIZED_FUNCTION(YOLOv5, fd_c_yolov5_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_yolov5_wrapper pointer to FD_C_YOLOv5Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

YOLO_DECLARE_BATCH_PREDICT_FUNCTION(YOLOv5, fd_c_yolov5_wrapper);

// YOLOv7

typedef struct FD_C_YOLOv7Wrapper FD_C_YOLOv7Wrapper;

/** \brief Create a new FD_C_YOLOv7Wrapper object
 *
 * \param[in] model_file Path of model file, e.g ./yolov7.onnx
 * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLOv7Wrapper object
 */

YOLO_DECLARE_CREATE_WRAPPER_FUNCTION(YOLOv7);

/** \brief Destroy a FD_C_YOLOv7Wrapper object
 *
 * \param[in] fd_c_yolov7_wrapper pointer to FD_C_YOLOv7Wrapper object
 */

YOLO_DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOv7, fd_c_yolov7_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolov7_wrapper pointer to FD_C_YOLOv7Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

YOLO_DECLARE_PREDICT_FUNCTION(YOLOv7, fd_c_yolov7_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_yolov7_wrapper pointer to FD_C_YOLOv7Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

YOLO_DECLARE_INITIALIZED_FUNCTION(YOLOv7, fd_c_yolov7_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_yolov7_wrapper pointer to FD_C_YOLOv7Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

YOLO_DECLARE_BATCH_PREDICT_FUNCTION(YOLOv7, fd_c_yolov7_wrapper);

// YOLOv8

typedef struct FD_C_YOLOv8Wrapper FD_C_YOLOv8Wrapper;

/** \brief Create a new FD_C_YOLOv8Wrapper object
 *
 * \param[in] model_file Path of model file, e.g ./yolov8.onnx
 * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLOv8Wrapper object
 */

YOLO_DECLARE_CREATE_WRAPPER_FUNCTION(YOLOv8);

/** \brief Destroy a FD_C_YOLOv8Wrapper object
 *
 * \param[in] fd_c_yolov8_wrapper pointer to FD_C_YOLOv8Wrapper object
 */

YOLO_DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOv8, fd_c_yolov8_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolov8_wrapper pointer to FD_C_YOLOv8Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

YOLO_DECLARE_PREDICT_FUNCTION(YOLOv8, fd_c_yolov8_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_yolov8_wrapper pointer to FD_C_YOLOv8Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

YOLO_DECLARE_INITIALIZED_FUNCTION(YOLOv8, fd_c_yolov8_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_yolov8_wrapper pointer to FD_C_YOLOv8Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

YOLO_DECLARE_BATCH_PREDICT_FUNCTION(YOLOv8, fd_c_yolov8_wrapper);

// YOLOv6

typedef struct FD_C_YOLOv6Wrapper FD_C_YOLOv6Wrapper;

/** \brief Create a new FD_C_YOLOv6Wrapper object
 *
 * \param[in] model_file Path of model file, e.g ./yolov6.onnx
 * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLOv6Wrapper object
 */

YOLO_DECLARE_CREATE_WRAPPER_FUNCTION(YOLOv6);

/** \brief Destroy a FD_C_YOLOv6Wrapper object
 *
 * \param[in] fd_c_yolov6_wrapper pointer to FD_C_YOLOv6Wrapper object
 */

YOLO_DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOv6, fd_c_yolov6_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolov6_wrapper pointer to FD_C_YOLOv6Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] conf_threshold confidence threashold for postprocessing
 * \param[in] nms_iou_threshold iou threashold for NMS
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

YOLO_DECLARE_PREDICT_FUNCTION_WITH_THRESHOLD(YOLOv6, fd_c_yolov6_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_yolov6_wrapper pointer to FD_C_YOLOv6Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

YOLO_DECLARE_INITIALIZED_FUNCTION(YOLOv6, fd_c_yolov6_wrapper);


// YOLOR

typedef struct FD_C_YOLORWrapper FD_C_YOLORWrapper;

/** \brief Create a new FD_C_YOLORWrapper object
 *
 * \param[in] model_file Path of model file, e.g ./yolor.onnx
 * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLORWrapper object
 */

YOLO_DECLARE_CREATE_WRAPPER_FUNCTION(YOLOR);

/** \brief Destroy a FD_C_YOLORWrapper object
 *
 * \param[in] fd_c_yolor_wrapper pointer to FD_C_YOLORWrapper object
 */

YOLO_DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOR, fd_c_yolor_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolor_wrapper pointer to FD_C_YOLORWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] conf_threshold confidence threashold for postprocessing
 * \param[in] nms_iou_threshold iou threashold for NMS
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

YOLO_DECLARE_PREDICT_FUNCTION_WITH_THRESHOLD(YOLOR, fd_c_yolor_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_yolor_wrapper pointer to FD_C_YOLORWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

YOLO_DECLARE_INITIALIZED_FUNCTION(YOLOR, fd_c_yolor_wrapper);



// YOLOX

typedef struct FD_C_YOLOXWrapper FD_C_YOLOXWrapper;

/** \brief Create a new FD_C_YOLOXWrapper object
 *
 * \param[in] model_file Path of model file, e.g ./yolox.onnx
 * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLOXWrapper object
 */

YOLO_DECLARE_CREATE_WRAPPER_FUNCTION(YOLOX);

/** \brief Destroy a FD_C_YOLOXWrapper object
 *
 * \param[in] fd_c_yolox_wrapper pointer to FD_C_YOLOXWrapper object
 */

YOLO_DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOX, fd_c_yolox_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolox_wrapper pointer to FD_C_YOLOXWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] conf_threshold confidence threashold for postprocessing
 * \param[in] nms_iou_threshold iou threashold for NMS
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

YOLO_DECLARE_PREDICT_FUNCTION_WITH_THRESHOLD(YOLOX, fd_c_yolox_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_yolox_wrapper pointer to FD_C_YOLOXWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

YOLO_DECLARE_INITIALIZED_FUNCTION(YOLOX, fd_c_yolox_wrapper);

#ifdef __cplusplus
}  // extern "C"
#endif
