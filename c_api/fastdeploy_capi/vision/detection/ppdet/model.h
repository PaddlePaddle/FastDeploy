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
#include "fastdeploy_capi/vision/detection/ppdet/base_define.h"

typedef struct FD_C_RuntimeOptionWrapper FD_C_RuntimeOptionWrapper;

#ifdef __cplusplus
extern "C" {
#endif

// PPYOLOE

typedef struct FD_C_PPYOLOEWrapper FD_C_PPYOLOEWrapper;

/** \brief Create a new FD_C_PPYOLOEWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PPYOLOEWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PPYOLOE);

/** \brief Destroy a FD_C_PPYOLOEWrapper object
 *
 * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PPYOLOE, fd_c_ppyoloe_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PPYOLOE, fd_c_ppyoloe_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PPYOLOE, fd_c_ppyoloe_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PPYOLOE, fd_c_ppyoloe_wrapper);

// PicoDet

typedef struct FD_C_PicoDetWrapper FD_C_PicoDetWrapper;

/** \brief Create a new FD_C_PicoDetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PicoDetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PicoDet);

/** \brief Destroy a FD_C_PicoDetWrapper object
 *
 * \param[in] fd_c_picodet_wrapper pointer to FD_C_PicoDetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PicoDet, fd_c_picodet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_picodet_wrapper pointer to FD_C_PicoDetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PicoDet, fd_c_picodet_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_picodet_wrapper pointer to FD_C_PicoDetWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PicoDet, fd_c_picodet_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_picodet_wrapper pointer to FD_C_PicoDetWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PicoDet, fd_c_picodet_wrapper);


// PPYOLO

typedef struct FD_C_PPYOLOWrapper FD_C_PPYOLOWrapper;

/** \brief Create a new FD_C_PPYOLOWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PPYOLOWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PPYOLO);

/** \brief Destroy a FD_C_PPYOLOWrapper object
 *
 * \param[in] fd_c_ppyolo_wrapper pointer to FD_C_PPYOLOWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PPYOLO, fd_c_ppyolo_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ppyolo_wrapper pointer to FD_C_PPYOLOWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PPYOLO, fd_c_ppyolo_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_ppyolo_wrapper pointer to FD_C_PPYOLOWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PPYOLO, fd_c_ppyolo_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_ppyolo_wrapper pointer to FD_C_PPYOLOWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PPYOLO, fd_c_ppyolo_wrapper);

// YOLOv3

typedef struct FD_C_YOLOv3Wrapper FD_C_YOLOv3Wrapper;

/** \brief Create a new FD_C_YOLOv3Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLOv3Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(YOLOv3);

/** \brief Destroy a FD_C_YOLOv3Wrapper object
 *
 * \param[in] fd_c_yolov3_wrapper pointer to FD_C_YOLOv3Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOv3, fd_c_yolov3_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolov3_wrapper pointer to FD_C_YOLOv3Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(YOLOv3, fd_c_yolov3_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_yolov3_wrapper pointer to FD_C_YOLOv3Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(YOLOv3, fd_c_yolov3_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_yolov3_wrapper pointer to FD_C_YOLOv3Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(YOLOv3, fd_c_yolov3_wrapper);

// PaddleYOLOX

typedef struct FD_C_PaddleYOLOXWrapper FD_C_PaddleYOLOXWrapper;

/** \brief Create a new FD_C_PaddleYOLOXWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOXWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOX);

/** \brief Destroy a FD_C_PaddleYOLOXWrapper object
 *
 * \param[in] fd_c_paddleyolox_wrapper pointer to FD_C_PaddleYOLOXWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOX, fd_c_paddleyolox_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolox_wrapper pointer to FD_C_PaddleYOLOXWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOX, fd_c_paddleyolox_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_paddleyolox_wrapper pointer to FD_C_PaddleYOLOXWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PaddleYOLOX, fd_c_paddleyolox_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_paddleyolox_wrapper pointer to FD_C_PaddleYOLOXWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PaddleYOLOX, fd_c_paddleyolox_wrapper);

// FasterRCNN

typedef struct FD_C_FasterRCNNWrapper FD_C_FasterRCNNWrapper;

/** \brief Create a new FD_C_FasterRCNNWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_FasterRCNNWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(FasterRCNN);

/** \brief Destroy a FD_C_FasterRCNNWrapper object
 *
 * \param[in] fd_c_fasterrcnn_wrapper pointer to FD_C_FasterRCNNWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(FasterRCNN, fd_c_fasterrcnn_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_fasterrcnn_wrapper pointer to FD_C_FasterRCNNWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(FasterRCNN, fd_c_fasterrcnn_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_fasterrcnn_wrapper pointer to FD_C_FasterRCNNWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(FasterRCNN, fd_c_fasterrcnn_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_fasterrcnn_wrapper pointer to FD_C_FasterRCNNWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(FasterRCNN, fd_c_fasterrcnn_wrapper);

// MaskRCNN

typedef struct FD_C_MaskRCNNWrapper FD_C_MaskRCNNWrapper;

/** \brief Create a new FD_C_MaskRCNNWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_MaskRCNNWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(MaskRCNN);

/** \brief Destroy a FD_C_MaskRCNNWrapper object
 *
 * \param[in] fd_c_maskrcnn_wrapper pointer to FD_C_MaskRCNNWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(MaskRCNN, fd_c_maskrcnn_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_maskrcnn_wrapper pointer to FD_C_MaskRCNNWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(MaskRCNN, fd_c_maskrcnn_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_maskrcnn_wrapper pointer to FD_C_MaskRCNNWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(MaskRCNN, fd_c_maskrcnn_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_maskrcnn_wrapper pointer to FD_C_MaskRCNNWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(MaskRCNN, fd_c_maskrcnn_wrapper);

// SSD

typedef struct FD_C_SSDWrapper FD_C_SSDWrapper;

/** \brief Create a new FD_C_SSDWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_SSDWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(SSD);

/** \brief Destroy a FD_C_SSDWrapper object
 *
 * \param[in] fd_c_ssd_wrapper pointer to FD_C_SSDWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(SSD, fd_c_ssd_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ssd_wrapper pointer to FD_C_SSDWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(SSD, fd_c_ssd_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_ssd_wrapper pointer to FD_C_SSDWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(SSD, fd_c_ssd_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_ssd_wrapper pointer to FD_C_SSDWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(SSD, fd_c_ssd_wrapper);

// PaddleYOLOv5

typedef struct FD_C_PaddleYOLOv5Wrapper FD_C_PaddleYOLOv5Wrapper;

/** \brief Create a new FD_C_PaddleYOLOv5Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv5Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv5);

/** \brief Destroy a FD_C_PaddleYOLOv5Wrapper object
 *
 * \param[in] fd_c_paddleyolov5_wrapper pointer to FD_C_PaddleYOLOv5Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv5, fd_c_paddleyolov5_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov5_wrapper pointer to FD_C_PaddleYOLOv5Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv5, fd_c_paddleyolov5_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_paddleyolov5_wrapper pointer to FD_C_PaddleYOLOv5Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PaddleYOLOv5, fd_c_paddleyolov5_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_paddleyolov5_wrapper pointer to FD_C_PaddleYOLOv5Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PaddleYOLOv5, fd_c_paddleyolov5_wrapper);

// PaddleYOLOv6

typedef struct FD_C_PaddleYOLOv6Wrapper FD_C_PaddleYOLOv6Wrapper;

/** \brief Create a new FD_C_PaddleYOLOv6Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv6Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv6);

/** \brief Destroy a FD_C_PaddleYOLOv6Wrapper object
 *
 * \param[in] fd_c_paddleyolov6_wrapper pointer to FD_C_PaddleYOLOv6Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv6, fd_c_paddleyolov6_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov6_wrapper pointer to FD_C_PaddleYOLOv6Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv6, fd_c_paddleyolov6_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_paddleyolov6_wrapper pointer to FD_C_PaddleYOLOv6Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PaddleYOLOv6, fd_c_paddleyolov6_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_paddleyolov6_wrapper pointer to FD_C_PaddleYOLOv6Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PaddleYOLOv6, fd_c_paddleyolov6_wrapper);

// PaddleYOLOv7

typedef struct FD_C_PaddleYOLOv7Wrapper FD_C_PaddleYOLOv7Wrapper;

/** \brief Create a new FD_C_PaddleYOLOv7Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv7Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv7);

/** \brief Destroy a FD_C_PaddleYOLOv7Wrapper object
 *
 * \param[in] fd_c_paddleyolov7_wrapper pointer to FD_C_PaddleYOLOv7Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv7, fd_c_paddleyolov7_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov7_wrapper pointer to FD_C_PaddleYOLOv7Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv7, fd_c_paddleyolov7_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_paddleyolov7_wrapper pointer to FD_C_PaddleYOLOv7Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PaddleYOLOv7, fd_c_paddleyolov7_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_paddleyolov7_wrapper pointer to FD_C_PaddleYOLOv7Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PaddleYOLOv7, fd_c_paddleyolov7_wrapper);

// PaddleYOLOv8

typedef struct FD_C_PaddleYOLOv8Wrapper FD_C_PaddleYOLOv8Wrapper;

/** \brief Create a new FD_C_PaddleYOLOv8Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv8Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv8);

/** \brief Destroy a FD_C_PaddleYOLOv8Wrapper object
 *
 * \param[in] fd_c_paddleyolov8_wrapper pointer to FD_C_PaddleYOLOv8Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv8, fd_c_paddleyolov8_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov8_wrapper pointer to FD_C_PaddleYOLOv8Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv8, fd_c_paddleyolov8_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_paddleyolov8_wrapper pointer to FD_C_PaddleYOLOv8Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PaddleYOLOv8, fd_c_paddleyolov8_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_paddleyolov8_wrapper pointer to FD_C_PaddleYOLOv8Wrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PaddleYOLOv8, fd_c_paddleyolov8_wrapper);

// RTMDet

typedef struct FD_C_RTMDetWrapper FD_C_RTMDetWrapper;

/** \brief Create a new FD_C_RTMDetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_RTMDetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(RTMDet);

/** \brief Destroy a FD_C_RTMDetWrapper object
 *
 * \param[in] fd_c_rtmdet_wrapper pointer to FD_C_RTMDetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(RTMDet, fd_c_rtmdet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_rtmdet_wrapper pointer to FD_C_RTMDetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(RTMDet, fd_c_rtmdet_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_rtmdet_wrapper pointer to FD_C_RTMDetWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(RTMDet, fd_c_rtmdet_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_rtmdet_wrapper pointer to FD_C_RTMDetWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(RTMDet, fd_c_rtmdet_wrapper);

// CascadeRCNN

typedef struct FD_C_CascadeRCNNWrapper FD_C_CascadeRCNNWrapper;

/** \brief Create a new FD_C_CascadeRCNNWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_CascadeRCNNWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(CascadeRCNN);

/** \brief Destroy a FD_C_CascadeRCNNWrapper object
 *
 * \param[in] fd_c_cascadercnn_wrapper pointer to FD_C_CascadeRCNNWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(CascadeRCNN, fd_c_cascadercnn_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_cascadercnn_wrapper pointer to FD_C_CascadeRCNNWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(CascadeRCNN, fd_c_cascadercnn_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_cascadercnn_wrapper pointer to FD_C_CascadeRCNNWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(CascadeRCNN, fd_c_cascadercnn_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_cascadercnn_wrapper pointer to FD_C_CascadeRCNNWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(CascadeRCNN, fd_c_cascadercnn_wrapper);

// PSSDet

typedef struct FD_C_PSSDetWrapper FD_C_PSSDetWrapper;

/** \brief Create a new FD_C_PSSDetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PSSDetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PSSDet);

/** \brief Destroy a FD_C_PSSDetWrapper object
 *
 * \param[in] fd_c_pssdet_wrapper pointer to FD_C_PSSDetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PSSDet, fd_c_pssdet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_pssdet_wrapper pointer to FD_C_PSSDetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PSSDet, fd_c_pssdet_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_pssdet_wrapper pointer to FD_C_PSSDetWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(PSSDet, fd_c_pssdet_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_pssdet_wrapper pointer to FD_C_PSSDetWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(PSSDet, fd_c_pssdet_wrapper);

// RetinaNet

typedef struct FD_C_RetinaNetWrapper FD_C_RetinaNetWrapper;

/** \brief Create a new FD_C_RetinaNetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_RetinaNetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(RetinaNet);

/** \brief Destroy a FD_C_RetinaNetWrapper object
 *
 * \param[in] fd_c_retinanet_wrapper pointer to FD_C_RetinaNetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(RetinaNet, fd_c_retinanet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_retinanet_wrapper pointer to FD_C_RetinaNetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(RetinaNet, fd_c_retinanet_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_retinanet_wrapper pointer to FD_C_RetinaNetWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(RetinaNet, fd_c_retinanet_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_retinanet_wrapper pointer to FD_C_RetinaNetWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(RetinaNet, fd_c_retinanet_wrapper);

// FCOS

typedef struct FD_C_FCOSWrapper FD_C_FCOSWrapper;

/** \brief Create a new FD_C_FCOSWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_FCOSWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(FCOS);

/** \brief Destroy a FD_C_FCOSWrapper object
 *
 * \param[in] fd_c_fcos_wrapper pointer to FD_C_FCOSWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(FCOS, fd_c_fcos_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_fcos_wrapper pointer to FD_C_FCOSWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(FCOS, fd_c_fcos_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_fcos_wrapper pointer to FD_C_FCOSWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(FCOS, fd_c_fcos_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_fcos_wrapper pointer to FD_C_FCOSWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(FCOS, fd_c_fcos_wrapper);

// TTFNet

typedef struct FD_C_TTFNetWrapper FD_C_TTFNetWrapper;

/** \brief Create a new FD_C_TTFNetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_TTFNetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(TTFNet);

/** \brief Destroy a FD_C_TTFNetWrapper object
 *
 * \param[in] fd_c_ttfnet_wrapper pointer to FD_C_TTFNetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(TTFNet, fd_c_ttfnet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ttfnet_wrapper pointer to FD_C_TTFNetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(TTFNet, fd_c_ttfnet_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_ttfnet_wrapper pointer to FD_C_TTFNetWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(TTFNet, fd_c_ttfnet_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_ttfnet_wrapper pointer to FD_C_TTFNetWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(TTFNet, fd_c_ttfnet_wrapper);

// TOOD

typedef struct FD_C_TOODWrapper FD_C_TOODWrapper;

/** \brief Create a new FD_C_TOODWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_TOODWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(TOOD);

/** \brief Destroy a FD_C_TOODWrapper object
 *
 * \param[in] fd_c_tood_wrapper pointer to FD_C_TOODWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(TOOD, fd_c_tood_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_tood_wrapper pointer to FD_C_TOODWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(TOOD, fd_c_tood_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_tood_wrapper pointer to FD_C_TOODWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(TOOD, fd_c_tood_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_tood_wrapper pointer to FD_C_TOODWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(TOOD, fd_c_tood_wrapper);

// GFL

typedef struct FD_C_GFLWrapper FD_C_GFLWrapper;

/** \brief Create a new FD_C_GFLWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_GFLWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(GFL);

/** \brief Destroy a FD_C_GFLWrapper object
 *
 * \param[in] fd_c_gfl_wrapper pointer to FD_C_GFLWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(GFL, fd_c_gfl_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_gfl_wrapper pointer to FD_C_GFLWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(GFL, fd_c_gfl_wrapper);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_gfl_wrapper pointer to FD_C_GFLWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

DECLARE_INITIALIZED_FUNCTION(GFL, fd_c_gfl_wrapper);

/** \brief Predict the detection results for a batch of input images
   *
   * \param[in] fd_c_gfl_wrapper pointer to FD_C_GFLWrapper object
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   *
   * \return true if the prediction successed, otherwise false
   */

DECLARE_BATCH_PREDICT_FUNCTION(GFL, fd_c_gfl_wrapper);







#ifdef __cplusplus
}  // extern "C"
#endif
