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

#include "fastdeploy_capi/vision/ocr/ppocr/model.h"

#include "fastdeploy_capi/internal/types_internal.h"
#include "fastdeploy_capi/vision/visualize.h"

#ifdef __cplusplus
extern "C" {
#endif

// Recognizer

FD_C_RecognizerWrapper* FD_C_CreateRecognizerWrapper(
    const char* model_file, const char* params_file, const char* label_path,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  FD_C_RecognizerWrapper* fd_c_recognizer_wrapper =
      new FD_C_RecognizerWrapper();
  fd_c_recognizer_wrapper->recognizer_model =
      std::unique_ptr<fastdeploy::vision::ocr::Recognizer>(
          new fastdeploy::vision::ocr::Recognizer(
              std::string(model_file), std::string(params_file),
              std::string(label_path), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_c_recognizer_wrapper;
}

OCR_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(Recognizer,
                                                   fd_c_recognizer_wrapper)

FD_C_Bool FD_C_RecognizerWrapperPredict(
    FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_Mat img,
    FD_C_Cstr* text, float* rec_score) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(RecognizerWrapper, fd_c_recognizer_wrapper);

  std::string res_string;

  bool successful = model->Predict(*im, &res_string, rec_score);
  if (successful) {
    text->size = res_string.size();
    text->data = new char[res_string.size() + 1];
    strcpy(text->data, res_string.c_str());
  }
  return successful;
}

OCR_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(Recognizer,
                                               fd_c_recognizer_wrapper)

FD_C_Bool FD_C_RecognizerWrapperBatchPredict(
    FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayCstr* texts, FD_C_OneDimArrayFloat* rec_scores) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<std::string> texts_out;
  std::vector<float> rec_scores_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
  }
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(RecognizerWrapper, fd_c_recognizer_wrapper);
  bool successful = model->BatchPredict(imgs_vec, &texts_out, &rec_scores_out);
  if (successful) {
    // copy results back to FD_C_OneDimArrayCstr and FD_C_OneDimArrayFloat
    texts->size = texts_out.size();
    texts->data = new FD_C_Cstr[texts->size];
    for (int i = 0; i < texts_out.size(); i++) {
      texts->data[i].size = texts_out[i].length();
      texts->data[i].data = new char[texts_out[i].length() + 1];
      strncpy(texts->data[i].data, texts_out[i].c_str(), texts_out[i].length());
    }

    rec_scores->size = rec_scores_out.size();
    rec_scores->data = new float[rec_scores->size];
    memcpy(rec_scores->data, rec_scores_out.data(),
           sizeof(float) * rec_scores->size);
  }
  return successful;
}

FD_C_Bool FD_C_RecognizerWrapperBatchPredictWithIndex(
    FD_C_RecognizerWrapper* fd_c_recognizer_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayCstr* texts, FD_C_OneDimArrayFloat* rec_scores,
    size_t start_index, size_t end_index, FD_C_OneDimArrayInt32 indices) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<std::string> texts_out;
  std::vector<float> rec_scores_out;
  std::vector<int> indices_in;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
  }
  for (int i = 0; i < indices.size; i++) {
    indices_in.push_back(indices.data[i]);
  }
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(RecognizerWrapper, fd_c_recognizer_wrapper);
  bool successful = model->BatchPredict(imgs_vec, &texts_out, &rec_scores_out,
                                        start_index, end_index, indices_in);
  if (successful) {
    // copy results back to FD_C_OneDimArrayCstr and FD_C_OneDimArrayFloat
    texts->size = texts_out.size();
    texts->data = new FD_C_Cstr[texts->size];
    for (int i = 0; i < texts_out.size(); i++) {
      texts->data[i].size = texts_out[i].length();
      texts->data[i].data = new char[texts_out[i].length() + 1];
      strncpy(texts->data[i].data, texts_out[i].c_str(), texts_out[i].length());
    }

    rec_scores->size = rec_scores_out.size();
    rec_scores->data = new float[rec_scores->size];
    memcpy(rec_scores->data, rec_scores_out.data(),
           sizeof(float) * rec_scores->size);
  }
  return successful;
}

// Classifier

FD_C_ClassifierWrapper* FD_C_CreateClassifierWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  FD_C_ClassifierWrapper* fd_c_classifier_wrapper =
      new FD_C_ClassifierWrapper();
  fd_c_classifier_wrapper->classifier_model =
      std::unique_ptr<fastdeploy::vision::ocr::Classifier>(
          new fastdeploy::vision::ocr::Classifier(
              std::string(model_file), std::string(params_file),
              *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_c_classifier_wrapper;
}

OCR_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(Classifier,
                                                   fd_c_classifier_wrapper)

FD_C_Bool FD_C_ClassifierWrapperPredict(
    FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_Mat img,
    int32_t* cls_label, float* cls_score) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(ClassifierWrapper, fd_c_classifier_wrapper);
  bool successful = model->Predict(*im, cls_label, cls_score);
  return successful;
}

OCR_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(Classifier,
                                               fd_c_classifier_wrapper)

FD_C_Bool FD_C_ClassifierWrapperBatchPredict(
    FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayInt32* cls_labels, FD_C_OneDimArrayFloat* cls_scores) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<int> cls_labels_out;
  std::vector<float> cls_scores_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
  }
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(ClassifierWrapper, fd_c_classifier_wrapper);
  bool successful =
      model->BatchPredict(imgs_vec, &cls_labels_out, &cls_scores_out);
  if (successful) {
    // copy results back to FD_C_OneDimArrayInt32 and FD_C_OneDimArrayFloat
    cls_labels->size = cls_labels_out.size();
    cls_labels->data = new int[cls_labels->size];
    memcpy(cls_labels->data, cls_labels_out.data(),
           sizeof(int) * cls_labels->size);

    cls_scores->size = cls_scores_out.size();
    cls_scores->data = new float[cls_scores->size];
    memcpy(cls_scores->data, cls_scores_out.data(),
           sizeof(int) * cls_scores->size);
  }
  return successful;
}

FD_C_Bool FD_C_ClassifierWrapperBatchPredictWithIndex(
    FD_C_ClassifierWrapper* fd_c_classifier_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimArrayInt32* cls_labels, FD_C_OneDimArrayFloat* cls_scores,
    size_t start_index, size_t end_index) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<int> cls_labels_out;
  std::vector<float> cls_scores_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
  }
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(ClassifierWrapper, fd_c_classifier_wrapper);
  bool successful = model->BatchPredict(
      imgs_vec, &cls_labels_out, &cls_scores_out, start_index, end_index);
  if (successful) {
    // copy results back to FD_C_OneDimArrayInt32 and FD_C_OneDimArrayFloat
    cls_labels->size = cls_labels_out.size();
    cls_labels->data = new int[cls_labels->size];
    memcpy(cls_labels->data, cls_labels_out.data(),
           sizeof(int) * cls_labels->size);

    cls_scores->size = cls_scores_out.size();
    cls_scores->data = new float[cls_scores->size];
    memcpy(cls_scores->data, cls_scores_out.data(),
           sizeof(int) * cls_scores->size);
  }
  return successful;
}

// DBDetector
FD_C_DBDetectorWrapper* FD_C_CreateDBDetectorWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  FD_C_DBDetectorWrapper* fd_c_dbdetector_wrapper =
      new FD_C_DBDetectorWrapper();
  fd_c_dbdetector_wrapper->dbdetector_model =
      std::unique_ptr<fastdeploy::vision::ocr::DBDetector>(
          new fastdeploy::vision::ocr::DBDetector(
              std::string(model_file), std::string(params_file),
              *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_c_dbdetector_wrapper;
}

OCR_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(DBDetector,
                                                   fd_c_dbdetector_wrapper)

FD_C_Bool FD_C_DBDetectorWrapperPredict(
    FD_C_DBDetectorWrapper* fd_c_dbdetector_wrapper, FD_C_Mat img,
    FD_C_TwoDimArrayInt32* boxes_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  std::vector<std::array<int, 8>> boxes_result_out;
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(DBDetectorWrapper, fd_c_dbdetector_wrapper);
  bool successful = model->Predict(*im, &boxes_result_out);
  if (successful) {
    // copy boxes
    const int boxes_coordinate_dim = 8;
    boxes_result->size = boxes_result_out.size();
    boxes_result->data = new FD_C_OneDimArrayInt32[boxes_result->size];
    for (size_t i = 0; i < boxes_result_out.size(); i++) {
      boxes_result->data[i].size = boxes_coordinate_dim;
      boxes_result->data[i].data = new int[boxes_coordinate_dim];
      for (size_t j = 0; j < boxes_coordinate_dim; j++) {
        boxes_result->data[i].data[j] = boxes_result_out[i][j];
      }
    }
  }
  return successful;
}

OCR_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(DBDetector,
                                               fd_c_dbdetector_wrapper)

FD_C_Bool FD_C_DBDetectorWrapperBatchPredict(
    FD_C_DBDetectorWrapper* fd_c_dbdetector_wrapper, FD_C_OneDimMat imgs,
    FD_C_ThreeDimArrayInt32* det_results) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<std::vector<std::array<int, 8>>> det_results_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
  }
  auto& model =
      CHECK_AND_CONVERT_FD_TYPE(DBDetectorWrapper, fd_c_dbdetector_wrapper);
  bool successful = model->BatchPredict(imgs_vec, &det_results_out);
  if (successful) {
    // copy results back to FD_C_ThreeDimArrayInt32
    det_results->size = det_results_out.size();
    det_results->data = new FD_C_TwoDimArrayInt32[det_results->size];
    for (int batch_indx = 0; batch_indx < det_results->size; batch_indx++) {
      const int boxes_coordinate_dim = 8;
      det_results->data[batch_indx].size = det_results_out[batch_indx].size();
      det_results->data[batch_indx].data =
          new FD_C_OneDimArrayInt32[det_results->data[batch_indx].size];
      for (size_t i = 0; i < det_results_out[batch_indx].size(); i++) {
        det_results->data[batch_indx].data[i].size = boxes_coordinate_dim;
        det_results->data[batch_indx].data[i].data =
            new int[boxes_coordinate_dim];
        for (size_t j = 0; j < boxes_coordinate_dim; j++) {
          det_results->data[batch_indx].data[i].data[j] =
              det_results_out[batch_indx][i][j];
        }
      }
    }
  }
  return successful;
}

// PPOCRv2

FD_C_PPOCRv2Wrapper* FD_C_CreatePPOCRv2Wrapper(
    FD_C_DBDetectorWrapper* fd_c_det_model_wrapper,
    FD_C_ClassifierWrapper* fd_c_cls_model_wrapper,
    FD_C_RecognizerWrapper* fd_c_rec_model_wrapper) {
  FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper = new FD_C_PPOCRv2Wrapper();
  auto& det_model =
      CHECK_AND_CONVERT_FD_TYPE(DBDetectorWrapper, fd_c_det_model_wrapper);
  auto& cls_model =
      CHECK_AND_CONVERT_FD_TYPE(ClassifierWrapper, fd_c_cls_model_wrapper);
  auto& rec_model =
      CHECK_AND_CONVERT_FD_TYPE(RecognizerWrapper, fd_c_rec_model_wrapper);
  fd_c_ppocrv2_wrapper->ppocrv2_model =
      std::unique_ptr<fastdeploy::pipeline::PPOCRv2>(
          new fastdeploy::pipeline::PPOCRv2(det_model.get(), cls_model.get(),
                                            rec_model.get()));
  return fd_c_ppocrv2_wrapper;
}

PIPELINE_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPOCRv2,
                                                        fd_c_ppocrv2_wrapper)

FD_C_Bool FD_C_PPOCRv2WrapperPredict(FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper,
                                     FD_C_Mat img,
                                     FD_C_OCRResult* fd_c_ocr_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& model = CHECK_AND_CONVERT_FD_TYPE(PPOCRv2Wrapper, fd_c_ppocrv2_wrapper);
  FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper =
      FD_C_CreateOCRResultWrapper();
  auto& ocr_result =
      CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, fd_c_ocr_result_wrapper);

  bool successful = model->Predict(im, ocr_result.get());
  if (successful) {
    FD_C_OCRResultWrapperToCResult(fd_c_ocr_result_wrapper, fd_c_ocr_result);
  }
  FD_C_DestroyOCRResultWrapper(fd_c_ocr_result_wrapper);
  return successful;
}

PIPELINE_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PPOCRv2,
                                                    fd_c_ppocrv2_wrapper)

FD_C_Bool FD_C_PPOCRv2WrapperBatchPredict(
    FD_C_PPOCRv2Wrapper* fd_c_ppocrv2_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* results) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<FD_C_OCRResultWrapper*> results_wrapper_out;
  std::vector<fastdeploy::vision::OCRResult> results_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
    FD_C_OCRResultWrapper* fd_ocr_result_wrapper =
        FD_C_CreateOCRResultWrapper();
    results_wrapper_out.push_back(fd_ocr_result_wrapper);
  }
  auto& model = CHECK_AND_CONVERT_FD_TYPE(PPOCRv2Wrapper, fd_c_ppocrv2_wrapper);
  bool successful = model->BatchPredict(imgs_vec, &results_out);
  if (successful) {
    // copy results back to FD_C_OneDimOCRResult
    results->size = results_out.size();
    results->data = new FD_C_OCRResult[results->size];
    for (int i = 0; i < results_out.size(); i++) {
      (*CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, results_wrapper_out[i])) =
          std::move(results_out[i]);
      FD_C_OCRResultWrapperToCResult(results_wrapper_out[i], &results->data[i]);
    }
  }
  for (int i = 0; i < results_out.size(); i++) {
    FD_C_DestroyOCRResultWrapper(results_wrapper_out[i]);
  }
  return successful;
}

// PPOCRv3

FD_C_PPOCRv3Wrapper* FD_C_CreatePPOCRv3Wrapper(
    FD_C_DBDetectorWrapper* fd_c_det_model_wrapper,
    FD_C_ClassifierWrapper* fd_c_cls_model_wrapper,
    FD_C_RecognizerWrapper* fd_c_rec_model_wrapper) {
  FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper = new FD_C_PPOCRv3Wrapper();
  auto& det_model =
      CHECK_AND_CONVERT_FD_TYPE(DBDetectorWrapper, fd_c_det_model_wrapper);
  auto& cls_model =
      CHECK_AND_CONVERT_FD_TYPE(ClassifierWrapper, fd_c_cls_model_wrapper);
  auto& rec_model =
      CHECK_AND_CONVERT_FD_TYPE(RecognizerWrapper, fd_c_rec_model_wrapper);
  fd_c_ppocrv3_wrapper->ppocrv3_model =
      std::unique_ptr<fastdeploy::pipeline::PPOCRv3>(
          new fastdeploy::pipeline::PPOCRv3(det_model.get(), cls_model.get(),
                                            rec_model.get()));
  return fd_c_ppocrv3_wrapper;
}

PIPELINE_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPOCRv3,
                                                        fd_c_ppocrv3_wrapper)

FD_C_Bool FD_C_PPOCRv3WrapperPredict(FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper,
                                     FD_C_Mat img,
                                     FD_C_OCRResult* fd_c_ocr_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& model = CHECK_AND_CONVERT_FD_TYPE(PPOCRv3Wrapper, fd_c_ppocrv3_wrapper);
  FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper =
      FD_C_CreateOCRResultWrapper();
  auto& ocr_result =
      CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, fd_c_ocr_result_wrapper);

  bool successful = model->Predict(im, ocr_result.get());
  if (successful) {
    FD_C_OCRResultWrapperToCResult(fd_c_ocr_result_wrapper, fd_c_ocr_result);
  }
  FD_C_DestroyOCRResultWrapper(fd_c_ocr_result_wrapper);
  return successful;
}

PIPELINE_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PPOCRv3,
                                                    fd_c_ppocrv3_wrapper)

FD_C_Bool FD_C_PPOCRv3WrapperBatchPredict(
    FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* results) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<FD_C_OCRResultWrapper*> results_wrapper_out;
  std::vector<fastdeploy::vision::OCRResult> results_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
    FD_C_OCRResultWrapper* fd_ocr_result_wrapper =
        FD_C_CreateOCRResultWrapper();
    results_wrapper_out.push_back(fd_ocr_result_wrapper);
  }
  auto& model = CHECK_AND_CONVERT_FD_TYPE(PPOCRv3Wrapper, fd_c_ppocrv3_wrapper);
  bool successful = model->BatchPredict(imgs_vec, &results_out);
  if (successful) {
    // copy results back to FD_C_OneDimOCRResult
    results->size = results_out.size();
    results->data = new FD_C_OCRResult[results->size];
    for (int i = 0; i < results_out.size(); i++) {
      (*CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, results_wrapper_out[i])) =
          std::move(results_out[i]);
      FD_C_OCRResultWrapperToCResult(results_wrapper_out[i], &results->data[i]);
    }
  }
  for (int i = 0; i < results_out.size(); i++) {
    FD_C_DestroyOCRResultWrapper(results_wrapper_out[i]);
  }
  return successful;
}

// PPTable
FD_C_PPTableWrapper* FD_C_CreateTableWrapper(
    FD_C_DBDetectorWrapper* fd_c_det_model_wrapper,
    FD_C_RecognizerWrapper* fd_c_rec_model_wrapper,
    FD_C_PPTableWrapper* fd_c_table_model_wrapper) {
  FD_C_PPTableWrapper* fd_c_pptable_wrapper = new FD_C_PPTableWrapper();
  auto& det_model =
      CHECK_AND_CONVERT_FD_TYPE(DBDetectorWrapper, fd_c_det_model_wrapper);
  auto& rec_model =
      CHECK_AND_CONVERT_FD_TYPE(RecognizerWrapper, fd_c_rec_model_wrapper);
  auto& table_model =
      CHECK_AND_CONVERT_FD_TYPE(PPTableWrapper, fd_c_table_model_wrapper);
  fd_c_pptable_wrapper->pptable_model =
      std::unique_ptr<fastdeploy::pipeline::PPTable>(
          new fastdeploy::pipeline::PPTable(det_model.get(), rec_model.get(),
                                            table_model.get()));
  return fd_c_pptable_wrapper;
}

PIPELINE_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(PPTable,
                                                        fd_c_pptable_wrapper)

FD_C_Bool FD_C_PPTableWrapperPredict(FD_C_PPTableWrapper* fd_c_pptable_wrapper,
                                     FD_C_Mat img,
                                     FD_C_OCRResult* fd_c_table_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& model = CHECK_AND_CONVERT_FD_TYPE(PPTableWrapper, fd_c_pptable_wrapper);
  FD_C_OCRResultWrapper* fd_c_table_result_wrapper =
      FD_C_CreateOCRResultWrapper();
  auto& table_result =
      CHECK_AND_CONVERT_FD_TYPE(TableResultWrapper, fd_c_table_result_wrapper);

  bool successful = model->Predict(im, table_result.get());
  if (successful) {
    FD_C_OCRResultWrapperToCResult(fd_c_table_result_wrapper,
                                   fd_c_table_result);
  }
  FD_C_DestroyOCRResultWrapper(fd_c_table_result_wrapper);
  return successful;
}

PIPELINE_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(PPTable,
                                                    fd_c_pptable_wrapper)

FD_C_Bool FD_C_PPTableWrapperBatchPredict(
    FD_C_PPTableWrapper* fd_c_pptable_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimOCRResult* results) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<FD_C_OCRResultWrapper*> results_wrapper_out;
  std::vector<fastdeploy::vision::TableResult> results_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
    FD_C_OCRResultWrapper* fd_table_result_wrapper =
        FD_C_CreateOCRResultWrapper();
    results_wrapper_out.push_back(fd_table_result_wrapper);
  }
  auto& model = CHECK_AND_CONVERT_FD_TYPE(PPTableWrapper, fd_c_pptable_wrapper);
  bool successful = model->BatchPredict(imgs_vec, &results_out);
  if (successful) {
    // copy results back to FD_C_OneDimOCRResult
    results->size = results_out.size();
    results->data = new FD_C_OCRResult[results->size];
    for (int i = 0; i < results_out.size(); i++) {
      (*CHECK_AND_CONVERT_FD_TYPE(TableResultWrapper, results_wrapper_out[i])) =
          std::move(results_out[i]);
      FD_C_OCRResultWrapperToCResult(results_wrapper_out[i], &results->data[i]);
    }
  }
  for (int i = 0; i < results_out.size(); i++) {
    FD_C_DestroyOCRResultWrapper(results_wrapper_out[i]);
  }
  return successful;
}
#ifdef __cplusplus
}
#endif