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
#include "fastdeploy_capi/vision/ocr/ppocr/base_define.h"

#ifdef __cplusplus
extern "C" {
#endif

// Recognizer

typedef struct FD_C_RecognizerWrapper FD_C_RecognizerWrapper;

/** \brief Create a new FD_C_RecognizerWrapper object
 *
 * \param[in] model_file Path of model file, e.g ./ch_PP-OCRv3_rec_infer/model.pdmodel.
 * \param[in] params_file Path of parameter file, e.g ./ch_PP-OCRv3_rec_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
 * \param[in] label_path Path of label file used by OCR recognition model. e.g ./ppocr_keys_v1.txt
 * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
 * \param[in] model_format Model format of the loaded model, default is Paddle format.
 *
 * \return Return a pointer to FD_C_RecognizerWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_RecognizerWrapper*
FD_C_CreateRecognizerWrapper(
    const char* model_file, const char* params_file, const char* label_path,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format);

/** \brief Destroy a FD_C_RecognizerWrapper object
 *
 * \param[in] fd_c_recognizer_wrapper pointer to FD_C_RecognizerWrapper object
 */

OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(Recognizer, fd_c_recognizer_wrapper);

/** \brief Predict the ocr result for an input image
 *
 * \param[in] fd_c_recognizer_wrapper pointer to FD_C_RecognizerWrapper object
 * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] text The text result of rec model will be written into this parameter.
 * \param[in] rec_score The score result of rec model will be written into this parameter.
 * \return true if the prediction is successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_RecognizerWrapperPredict(
    __fd_keep FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_Mat img,
    FD_C_Cstr* text, float* rec_score);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_recognizer_wrapper pointer to FD_C_RecognizerWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

OCR_DECLARE_INITIALIZED_FUNCTION(Recognizer, fd_c_recognizer_wrapper);

/** \brief Predict the ocr results for a batch of input images
 *
 * \param[in] fd_c_recognizer_wrapper pointer to FD_C_RecognizerWrapper object
 * \param[in] imgs The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] texts The list of text results of rec model will be written into this vector.
 * \param[in] rec_scores The list of score result of rec model will be written into this vector.
 *
 * \return true if the prediction successed, otherwise false
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_RecognizerWrapperBatchPredict(
    __fd_keep FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayCstr* texts, FD_C_OneDimArrayFloat* rec_scores);

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_RecognizerWrapperBatchPredictWithIndex(
    __fd_keep FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayCstr* texts, FD_C_OneDimArrayFloat* rec_scores,
    size_t start_index, size_t end_index,
    FD_C_OneDimArrayInt32 indices);


// Classifier

typedef struct FD_C_ClassifierWrapper FD_C_ClassifierWrapper;

/** \brief Create a new FD_C_ClassifierWrapper object
 *
 * \param[in] model_file Path of model file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel.
 * \param[in] params_file Path of parameter file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
 * \param[in] model_format Model format of the loaded model, default is Paddle format.
 *
 * \return Return a pointer to FD_C_ClassifierWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_ClassifierWrapper*
FD_C_CreateClassifierWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format);

/** \brief Destroy a FD_C_ClassifierWrapper object
 *
 * \param[in] fd_c_classifier_wrapper pointer to FD_C_ClassifierWrapper object
 */

OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(Classifier, fd_c_classifier_wrapper);

/** \brief Predict the input image and get OCR classification model cls_result.
 *
 * \param[in] fd_c_classifier_wrapper pointer to FD_C_ClassifierWrapper object
 * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] cls_label The label result of cls model will be written in to this param.
 * \param[in] cls_score The score result of cls model will be written in to this param.
 * \return true if the prediction is successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_ClassifierWrapperPredict(
    __fd_keep FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_Mat img,
    int32_t* cls_label, float* cls_score);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_classifier_wrapper pointer to FD_C_ClassifierWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

OCR_DECLARE_INITIALIZED_FUNCTION(Classifier, fd_c_classifier_wrapper);

/** \brief BatchPredict the input image and get OCR classification model cls_result.
 *
 * \param[in] fd_c_classifier_wrapper pointer to FD_C_ClassifierWrapper object
 * \param[in] imgs The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] cls_labels The label results of cls model will be written in to this vector.
 * \param[in] cls_scores The score results of cls model will be written in to this vector.
 * \return true if the prediction is successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_ClassifierWrapperBatchPredict(
    __fd_keep FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayInt32* cls_labels, FD_C_OneDimArrayFloat* cls_scores);

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_ClassifierWrapperBatchPredictWithIndex(
    __fd_keep FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayInt32* cls_labels, FD_C_OneDimArrayFloat* cls_scores,
    size_t start_index, size_t end_index);


// DBDetector

typedef struct FD_C_DBDetectorWrapper FD_C_DBDetectorWrapper;

/** \brief Create a new FD_C_DBDetectorWrapper object
 *
 * \param[in] model_file Path of model file, e.g ./ch_PP-OCRv3_det_infer/model.pdmodel.
 * \param[in] params_file Path of parameter file, e.g ./ch_PP-OCRv3_det_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
 * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
 * \param[in] model_format Model format of the loaded model, default is Paddle format.
 *
 * \return Return a pointer to FD_C_DBDetectorWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_DBDetectorWrapper*
FD_C_CreateDBDetectorWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format);

/** \brief Destroy a FD_C_DBDetectorWrapper object
 *
 * \param[in] fd_c_dbdetector_wrapper pointer to FD_C_DBDetectorWrapper object
 */

OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(DBDetector, fd_c_dbdetector_wrapper);

/** \brief Predict the input image and get OCR detection model result.
 *
 * \param[in] fd_c_dbdetector_wrapper pointer to FD_C_DBDetectorWrapper object
 * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] boxes_result The output of OCR detection model result will be written to this structure.
 * \return true if the prediction is successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_DBDetectorWrapperPredict(
    __fd_keep FD_C_DBDetectorWrapper* fd_c_dbdetector_wrapper, FD_C_Mat img,
    FD_C_TwoDimArrayInt32* boxes_result);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_dbdetector_wrapper pointer to FD_C_DBDetectorWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

OCR_DECLARE_INITIALIZED_FUNCTION(DBDetector, fd_c_dbdetector_wrapper);

/** \brief BatchPredict the input image and get OCR detection model result.
 *
 * \param[in] fd_c_dbdetector_wrapper pointer to FD_C_DBDetectorWrapper object
 * \param[in] imgs The list input of image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] det_results The output of OCR detection model result will be written to this structure.
 *
 * \return true if the prediction is successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_DBDetectorWrapperBatchPredict(
    __fd_keep FD_C_DBDetectorWrapper* fd_c_dbdetector_wrapper, FD_C_OneDimMat imgs,
    FD_C_ThreeDimArrayInt32* det_results);


// StructureV2Table

typedef struct FD_C_StructureV2TableWrapper FD_C_StructureV2TableWrapper;

/** \brief Create a new FD_C_StructureV2TableWrapper object
 *
 * \param[in] model_file Path of model file, e.g ./en_ppstructure_mobile_v2.0_SLANet_infer/model.pdmodel.
 * \param[in] params_file Path of parameter file, e.g ./en_ppstructure_mobile_v2.0_SLANet_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
 * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
 * \param[in] model_format Model format of the loaded model, default is Paddle format.
 *
 * \return Return a pointer to FD_C_StructureV2TableWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_StructureV2TableWrapper*
FD_C_CreateStructureV2TableWrapper(
    const char* model_file, const char* params_file, const char* table_char_dict_path,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format);

/** \brief Destroy a FD_C_StructureV2TableWrapper object
 *
 * \param[in] fd_c_structurev2table_wrapper pointer to FD_C_DBDetectorWrapper object
 */

OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(StructureV2Table, fd_c_structurev2table_wrapper);

/** \brief Predict the input image and get OCR table model result.
 *
 * \param[in] fd_c_structurev2table_wrapper pointer to FD_C_StructureV2TableWrapper object
 * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] boxes_result The output of OCR table model result will be written to this structure.
 * \return true if the prediction is successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_StructureV2TableWrapperPredict(
    __fd_keep FD_C_StructureV2TableWrapper* fd_c_structurev2table_wrapper, FD_C_Mat img,
    FD_C_TwoDimArrayInt32* boxes_result, FD_C_OneDimArrayCstr* structure_result);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_dbdetector_wrapper pointer to FD_C_StructureV2TableWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

OCR_DECLARE_INITIALIZED_FUNCTION(StructureV2Table, fd_c_structurev2table_wrapper);

/** \brief BatchPredict the input image and get OCR table model result.
 *
 * \param[in] fd_c_structurev2table_wrapper pointer to FD_C_StructureV2TableWrapper object
 * \param[in] imgs The list input of image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] det_results The output of OCR table model result will be written to this structure.
 *
 * \return true if the prediction is successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_StructureV2TableWrapperBatchPredict(
    __fd_keep FD_C_StructureV2TableWrapper* fd_c_structurev2table_wrapper, FD_C_OneDimMat imgs,
    FD_C_ThreeDimArrayInt32* det_results, FD_C_TwoDimArrayCstr* structure_results);


// PPOCRv2


typedef struct FD_C_PPOCRv2Wrapper FD_C_PPOCRv2Wrapper;

/** \brief Set up the detection model path, classification model path and recognition model path respectively.
 *
 * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv2_det_infer
 * \param[in] cls_model Path of classification model, e.g ./ch_ppocr_mobile_v2.0_cls_infer
 * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv2_rec_infer
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_PPOCRv2Wrapper*
FD_C_CreatePPOCRv2Wrapper(
    FD_C_DBDetectorWrapper* det_model,
    FD_C_ClassifierWrapper* cls_model,
    FD_C_RecognizerWrapper* rec_model);

/** \brief Destroy a FD_C_PPOCRv2Wrapper object
 *
 * \param[in] fd_c_ppocrv2_wrapper pointer to FD_C_PPOCRv2Wrapper object
 */

OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(PPOCRv2, fd_c_ppocrv2_wrapper);

/** \brief Predict the input image and get OCR result.
 *
 * \param[in] fd_c_ppocrv2_wrapper pointer to FD_C_PPOCRv2Wrapper object
 * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] result The output OCR result will be written to this structure.
 * \return true if the prediction successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_PPOCRv2WrapperPredict(
    __fd_keep FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper, FD_C_Mat img,
    FD_C_OCRResult* result);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_ppocrv2_wrapper pointer to FD_C_PPOCRv2Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

OCR_DECLARE_INITIALIZED_FUNCTION(PPOCRv2, fd_c_ppocrv2_wrapper);

/** \brief BatchPredict the input image and get OCR result.
 *
 * \param[in] fd_c_ppocrv2_wrapper pointer to FD_C_PPOCRv2Wrapper object
 * \param[in] imgs The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] batch_result The output list of OCR result will be written to this structure.
 * \return true if the prediction successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_PPOCRv2WrapperBatchPredict(
    __fd_keep FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* batch_result);



// PPOCRv3

typedef struct FD_C_PPOCRv3Wrapper FD_C_PPOCRv3Wrapper;

/** \brief Set up the detection model path, classification model path and recognition model path respectively.
 *
 * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv2_det_infer
 * \param[in] cls_model Path of classification model, e.g ./ch_ppocr_mobile_v2.0_cls_infer
 * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv2_rec_infer
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_PPOCRv3Wrapper*
FD_C_CreatePPOCRv3Wrapper(
    FD_C_DBDetectorWrapper* det_model,
    FD_C_ClassifierWrapper* cls_model,
    FD_C_RecognizerWrapper* rec_model);

/** \brief Destroy a FD_C_PPOCRv3Wrapper object
 *
 * \param[in] fd_c_ppocrv3_wrapper pointer to FD_C_PPOCRv3Wrapper object
 */

OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(PPOCRv3, fd_c_ppocrv3_wrapper);

/** \brief Predict the input image and get OCR result.
 *
 * \param[in] fd_c_ppocrv3_wrapper pointer to FD_C_PPOCRv3Wrapper object
 * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] result The output OCR result will be written to this structure.
 * \return true if the prediction successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_PPOCRv3WrapperPredict(
    __fd_keep FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper, FD_C_Mat img,
    FD_C_OCRResult* result);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_ppocrv3_wrapper pointer to FD_C_PPOCRv3Wrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

OCR_DECLARE_INITIALIZED_FUNCTION(PPOCRv3, fd_c_ppocrv3_wrapper);

/** \brief BatchPredict the input image and get OCR result.
 *
 * \param[in] fd_c_ppocrv3_wrapper pointer to FD_C_PPOCRv3Wrapper object
 * \param[in] imgs The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] batch_result The output list of OCR result will be written to this structure.
 * \return true if the prediction successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_PPOCRv3WrapperBatchPredict(
    __fd_keep FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* batch_result);


// PPStructureV2Table

typedef struct FD_C_PPStructureV2TableWrapper FD_C_PPStructureV2TableWrapper;

/** \brief Set up the detection model path, classification model path and table recognition model path respectively.
 *
 * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv3_det_infer
 * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv3_rec_infer
 * \param[in] table_model Path of table model, e.g ./en_ppstructure_mobile_v2.0_SLANet_infer
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_PPStructureV2TableWrapper*
FD_C_CreatePPStructureV2TableWrapper(
    FD_C_DBDetectorWrapper* det_model,
    FD_C_RecognizerWrapper* rec_model,
    FD_C_StructureV2TableWrapper* table_model);

/** \brief Destroy a FD_C_PPTableWrapper object
 *
 * \param[in] fd_c_ppstructurev2table_wrapper pointer to FD_C_PPStructureV2TableWrapper object
 */

OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(PPStructureV2Table, fd_c_ppstructurev2table_wrapper);

/** \brief Predict the input image and get OCR result.
 *
 * \param[in] fd_c_ppstructurev2table_wrapper pointer to FD_C_PPStructureV2TableWrapper object
 * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] result The output OCR result will be written to this structure.
 * \return true if the prediction successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_PPStructureV2TableWrapperPredict(
    __fd_keep FD_C_PPStructureV2TableWrapper* fd_c_ppstructurev2table_wrapper, FD_C_Mat img,
    FD_C_OCRResult* result);

/** \brief Check if the model is initialized successfully
 *
 * \param[in] fd_c_ppstructurev2table_wrapper pointer to FD_C_PPStructureV2TableWrapper object
 *
 * \return Return a bool of value true if initialized successfully
 */

OCR_DECLARE_INITIALIZED_FUNCTION(PPStructureV2Table, fd_c_ppstructurev2table_wrapper);

/** \brief BatchPredict the input image and get OCR result.
 *
 * \param[in] fd_c_ppstructurev2table_wrapper pointer to FD_C_PPStructureV2TableWrapper object
 * \param[in] imgs The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
 * \param[in] batch_result The output list of OCR result will be written to this structure.
 * \return true if the prediction successed, otherwise false.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_PPStructureV2TableWrapperBatchPredict(
    __fd_keep FD_C_PPStructureV2TableWrapper* fd_c_ppstructurev2table_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* batch_result);

#ifdef __cplusplus
}  // extern "C"
#endif
