// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "fastdeploy/vision/segmentation/ppseg/postprocessor.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace segmentation {

PaddleSegPostprocessor::PaddleSegPostprocessor(const std::string& config_file) {
  FDASSERT(ReadFromConfig(config_file), "Failed to create PaddleDetPreprocessor.");
  initialized_ = true;
}

bool PaddleSegPostprocessor::ReadFromConfig(const std::string& config_file) {
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
    return false;
  }   

  if (cfg["Deploy"]["output_op"]) {
    std::string output_op = cfg["Deploy"]["output_op"].as<std::string>();
    if (output_op == "softmax") {
      is_with_softmax_ = true;
      is_with_argmax_ = false;
    } else if (output_op == "argmax") {
      is_with_softmax_ = false;
      is_with_argmax_ = true;
    } else if (output_op == "none") {
      is_with_softmax_ = false;
      is_with_argmax_ = false;
    } else {
      FDERROR << "Unexcepted output_op operator in deploy.yml: " << output_op
              << "." << std::endl;
      return false;
    }
  }
  return true;
}

bool PaddleSegPostprocessor::CopyFromInferResults(const FDTensor& infer_results,
                                                  FDTensor* infer_result,
                                                  const std::vector<int64_t>& infer_result_shape,
                                                  const int64_t start_idx,
                                                  const int64_t offset,
                                                  std::vector<int32_t>* int32_copy_result_buffer,
                                                  std::vector<int64_t>* int64_copy_result_buffer,
                                                  std::vector<float_t>* fp32_copy_result_buffer) {
  int64_t infer_batch = infer_results.shape[0];
  if(infer_batch == 1) {
    *infer_result = infer_results;
    // batch is 1, so ignore
    infer_result->shape = infer_result_shape;
  } else {
    if (infer_results.dtype == FDDataType::FP32) {
      const float_t* infer_results_buffer =
          static_cast<const float_t*>(infer_results.Data()) + start_idx;
      fp32_copy_result_buffer = new std::vector<float_t>(
            infer_results_buffer, infer_results_buffer + offset);
      infer_result->Resize(infer_result_shape, FDDataType::FP32);
      infer_result->SetExternalData(
          infer_result->shape, FDDataType::FP32,
          static_cast<void*>(fp32_copy_result_buffer->data()));
    }
    else if (infer_results.dtype == FDDataType::INT64) {
      const int64_t* infer_results_buffer =
          static_cast<const int64_t*>(infer_results.Data()) + start_idx;
      int64_copy_result_buffer = new std::vector<int64_t>(
            infer_results_buffer, infer_results_buffer + offset);
      infer_result->Resize(infer_result_shape, FDDataType::INT64);
      infer_result->SetExternalData(
          infer_result->shape, FDDataType::INT64,
          static_cast<void*>(int64_copy_result_buffer->data()));
    }else if (infer_results.dtype == FDDataType::INT32) {
      const int32_t* infer_results_buffer =
          static_cast<const int32_t*>(infer_results.Data()) + start_idx;
      int32_copy_result_buffer = new std::vector<int32_t>(
            infer_results_buffer, infer_results_buffer + offset);
      infer_result->Resize(infer_result_shape, FDDataType::INT32);
      infer_result->SetExternalData(
          infer_result->shape, FDDataType::INT32,
          static_cast<void*>(int32_copy_result_buffer->data()));
    } else {
      FDERROR << "Don't support infer_results FDDataType." << std::endl;
      return false;
    }
  }
  return true;
}

bool PaddleSegPostprocessor::ProcessWithScoreResult(const FDTensor& infer_result,
                                                    const int64_t out_num,
                                                    SegmentationResult* result) {
  int32_t* argmax_infer_result_buffer = nullptr;
  float_t* score_infer_result_buffer = nullptr;
  FDTensor argmax_infer_result;
  FDTensor max_score_result;
  std::vector<int64_t> reduce_dim{-1};
  function::ArgMax(infer_result, &argmax_infer_result, -1, FDDataType::INT32);
  if (is_store_score_map_) {
    function::Max(infer_result, &max_score_result, reduce_dim);
    score_infer_result_buffer = static_cast<float_t*>(max_score_result.Data());
    std::memcpy(result->score_map.data(), score_infer_result_buffer,
              out_num * sizeof(float_t));
  }
  
  argmax_infer_result_buffer =
      static_cast<int32_t*>(argmax_infer_result.Data());
  
  for (int i = 0; i < out_num; i++) {
    result->label_map[i] =
        static_cast<uint8_t>(*(argmax_infer_result_buffer + i));
  }
  
  return true;
}

bool PaddleSegPostprocessor::ProcessWithLabelResult(FDTensor& infer_result,
                                                    const int64_t out_num,
                                                    SegmentationResult* result) {
  if (infer_result.dtype == FDDataType::FP32) {
    float_t* infer_result_buffer =
        static_cast<float_t*>(infer_result.Data());
    for (int i = 0; i < out_num; i++) {
      result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
    }
  } else if (infer_result.dtype == FDDataType::INT64) {
    const int64_t* infer_result_buffer =
        static_cast<const int64_t*>(infer_result.Data());
    for (int i = 0; i < out_num; i++) {
      result->label_map[i] =
          static_cast<uint8_t>(*(infer_result_buffer + i));
    }
  } else if (infer_result.dtype == FDDataType::INT32) {
    const int32_t* infer_result_buffer =
        static_cast<const int32_t*>(infer_result.Data());
    for (int i = 0; i < out_num; i++) {
      result->label_map[i] =
          static_cast<uint8_t>(*(infer_result_buffer + i));
    }
  } else if (infer_result.dtype == FDDataType::UINT8) {
    const uint8_t* infer_result_buffer =
        static_cast<const uint8_t*>(infer_result.Data());
    memcpy(result->label_map.data(), infer_result_buffer, out_num * sizeof(uint8_t));
  }
  else {
    FDERROR << "Don't support FDDataType for processing label result." << std::endl;
    return false;
  }
  return true;
}

bool PaddleSegPostprocessor::ResizeInferResult(FDTensor& infer_result,
                                               const int64_t offset, 
                                               const std::array<int, 2> resize_info,
		                                           FDTensor* new_infer_result,
                                               std::vector<uint8_t>* uint8_result_buffer, 
					                                     Mat* mat) {
  FDDataType infer_results_dtype = infer_result.dtype;
  FDASSERT(infer_results_dtype == FDDataType::INT64 ||
           infer_results_dtype == FDDataType::FP32 ||
           infer_results_dtype == FDDataType::INT32,
           "Don't support FDDataType : %s for resizing operation in PaddleSeg.",
           Str(infer_results_dtype).c_str());
  if (infer_result.dtype == FDDataType::INT64 ||
      infer_result.dtype == FDDataType::INT32 ) {
    if (infer_result.dtype == FDDataType::INT64) {
      int64_t* infer_result_buffer =
          static_cast<int64_t*>(infer_result.Data());
      // cv::resize don't support `CV_8S` or `CV_32S`
      // refer to https://github.com/opencv/opencv/issues/20991
      // https://github.com/opencv/opencv/issues/7862
      uint8_result_buffer = new std::vector<uint8_t>(
          infer_result_buffer, infer_result_buffer + offset);
    }
    if (infer_result.dtype == FDDataType::INT32) {
      int32_t* infer_result_buffer =
          static_cast<int32_t*>(infer_result.Data());
      // cv::resize don't support `CV_8S` or `CV_32S`
      // refer to https://github.com/opencv/opencv/issues/20991
      // https://github.com/opencv/opencv/issues/7862
      uint8_result_buffer = new std::vector<uint8_t>(
          infer_result_buffer, infer_result_buffer + offset);
    }
    
    infer_result.Resize(infer_result.shape, FDDataType::FP32);
    infer_result.SetExternalData(
        infer_result.shape, FDDataType::UINT8,
        static_cast<void*>(uint8_result_buffer->data()));
  } 
  mat = new Mat(Mat::Create(infer_result, ProcLib::OPENCV));
  Resize::Run(mat, resize_info[1], resize_info[0], -1.0f, -1.0f, 1, false, ProcLib::OPENCV);
  mat->ShareWithTensor(new_infer_result);
  return true;
}

bool PaddleSegPostprocessor::Run(
    const std::vector<FDTensor>& infer_results,
    std::vector<SegmentationResult>* results,
    const std::map<std::string, std::vector<std::array<int, 2>>>& imgs_info) {
  // PaddleSeg has three types of inference output:
  //     1. output with argmax and without softmax. 3-D matrix N(C)HW, Channel
  //     is batch_size, the element in matrix is classified label_id INT64 type.
  //     2. output without argmax and without softmax. 4-D matrix NCHW, N(batch)
  //     is batch_size, Channel is the num of classes. The element is the logits 
  //     of classes FP32 type
  //     3. output without argmax and with softmax. 4-D matrix NCHW, the result
  //     of 2 with softmax layer
  // Fastdeploy output:
  //     1. label_map
  //     2. score_map(optional)
  //     3. shape: 2-D HW
  if (!initialized_) {
    FDERROR << "Postprocessor is not initialized." << std::endl;
    return false;
  }

  FDDataType infer_results_dtype = infer_results[0].dtype;
  FDASSERT(infer_results_dtype == FDDataType::INT64 ||
           infer_results_dtype == FDDataType::FP32 ||
           infer_results_dtype == FDDataType::INT32,
           "Require the data type of output is int64, fp32 or int32, but now "
           "it's %s.",
           Str(infer_results_dtype).c_str());

  auto iter_input_imgs_shape_list = imgs_info.find("shape_info");
  FDASSERT(iter_input_imgs_shape_list != imgs_info.end(), "Cannot find shape_info from imgs_info.");

  // For Argmax Softmax function below
  FDTensor transform_infer_results;
  bool is_transform = false;
  int64_t infer_batch = infer_results[0].shape[0];
  int64_t infer_channel = 0;
  int64_t infer_height = 0;
  int64_t infer_width = 0;  

  if (is_with_argmax_) {
    infer_channel = 1;
    infer_height = infer_results[0].shape[1];
    infer_width = infer_results[0].shape[2];
  } else {
    // output without argmax
    infer_channel = 1;
    infer_height = infer_results[0].shape[2];
    infer_width = infer_results[0].shape[3];
    is_transform = true;
    if (is_store_score_map_) {
      infer_channel = infer_results[0].shape[1];
      std::vector<int64_t> dim{0, 2, 3, 1};
      function::Transpose(infer_results[0], &transform_infer_results, dim);
      if (!is_with_softmax_ && apply_softmax_) {
        function::Softmax(transform_infer_results, &transform_infer_results, 1);
      }
    } else {
      function::ArgMax(infer_results[0], &transform_infer_results, 1, FDDataType::INT32);
      infer_results_dtype = transform_infer_results.dtype;
    }
  }

  int64_t infer_chw = infer_channel * infer_height * infer_width;

  results->resize(infer_batch);
  for (int i = 0; i < infer_batch; i++) {
    SegmentationResult* result = &((*results)[i]);
    result->Clear();
    int64_t start_idx = i * infer_chw;

    FDTensor infer_result;
    std::vector<int64_t> infer_result_shape = {infer_height, infer_width, infer_channel};
    std::vector<int32_t>* int32_copy_result_buffer = nullptr;
    std::vector<int64_t>* int64_copy_result_buffer = nullptr;
    std::vector<float_t>* fp32_copy_result_buffer = nullptr;
    
    if (is_transform) {
      CopyFromInferResults(transform_infer_results, 
                           &infer_result,
                           infer_result_shape,
                           start_idx, 
                           infer_chw,
                           int32_copy_result_buffer,
                           int64_copy_result_buffer,
                           fp32_copy_result_buffer);
    } else {
      CopyFromInferResults(infer_results[0], 
                           &infer_result,
                           infer_result_shape,
                           start_idx, 
                           infer_chw,
                           int32_copy_result_buffer,
                           int64_copy_result_buffer,
                           fp32_copy_result_buffer);
    }
    bool is_resized = false;
    int input_height = iter_input_imgs_shape_list->second[i][0];
    int input_width = iter_input_imgs_shape_list->second[i][1];
    if (input_height != infer_height || input_width != infer_width) {
      is_resized = true;
    }
    // for resize mat below
    FDTensor new_infer_result;
    Mat* mat = nullptr;
    std::vector<uint8_t>* uint8_result_buffer = nullptr;
    if (is_resized) {
      ResizeInferResult(infer_result, infer_chw, {input_height, input_width}, &new_infer_result, uint8_result_buffer, mat);
      result->shape = new_infer_result.shape;
    } else {
      result->shape = infer_result.shape;
    }
    // output shape is 2-D HW layout, so out_num = H * W
    int out_num =
        std::accumulate(result->shape.begin(), result->shape.begin() + 2, 1,
                        std::multiplies<int>());
  
    if (!is_with_argmax_ && is_store_score_map_) {
      // output with label_map and score_map
      result->contain_score_map = true;
      result->Resize(out_num);
      if (is_resized) {
        ProcessWithScoreResult(new_infer_result, out_num, result);
      } else {
        ProcessWithScoreResult(infer_result, out_num, result);
      }
    } else {
      result->Resize(out_num);
      // output only with label_map
      if (is_resized) {
        ProcessWithLabelResult(new_infer_result, out_num, result);
      } else {
        ProcessWithLabelResult(infer_result, out_num, result);
      }
    }
    // HWC remove C
    result->shape.erase(result->shape.begin() + 2);
    delete int32_copy_result_buffer;
    delete int64_copy_result_buffer;
    delete fp32_copy_result_buffer;
    delete uint8_result_buffer;
    delete mat;
    mat = nullptr;
  }
  return true;
}
}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
