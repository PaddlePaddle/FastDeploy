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
  this->config_file_ = config_file;
  if (!ReadFromConfig()) {
    FDERROR << "Failed to read postprocess configuration from config file."
            << std::endl;
  }
}

bool PaddleSegPostprocessor::ReadFromConfig() {
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
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
    }
  }
}

bool PaddleSegPostprocessor::Run(
    std::vector<FDTensor> infer_result, SegmentationResult* result,
    const std::map<std::string, std::array<int, 2>>& im_info) {
  // PaddleSeg has three types of inference output:
  //     1. output with argmax and without softmax. 3-D matrix N(C)HW, Channel
  //     always 1, the element in matrix is classified label_id INT64 Type.
  //     2. output without argmax and without softmax. 4-D matrix NCHW, N(batch)
  //     always
  //     1(only support batch size 1), Channel is the num of classes. The
  //     element is the logits of classes
  //     FP32
  //     3. output without argmax and with softmax. 4-D matrix NCHW, the result
  //     of 2 with softmax layer
  // Fastdeploy output:
  //     1. label_map
  //     2. score_map(optional)
  //     3. shape: 2-D HW
  FDASSERT(infer_result[0].shape[0] == 1, "Only support batch size = 1.");
  FDTensor* infer_result_ptr = &(infer_result[0]);
  FDASSERT(infer_result_ptr->dtype == FDDataType::INT64 ||
           infer_result_ptr->dtype == FDDataType::FP32 ||
           infer_result_ptr->dtype == FDDataType::INT32,
           "Require the data type of output is int64, fp32 or int32, but now "
           "it's %s.",
           Str(infer_result_ptr->dtype).c_str());
  result->Clear();
  

  int64_t infer_batch = infer_result_ptr->shape[0];
  int64_t infer_channel = 0;
  int64_t infer_height = 0;
  int64_t infer_width = 0;

  if (is_with_argmax_) {
    infer_channel = 1;
    infer_height = infer_result_ptr->shape[1];
    infer_width = infer_result_ptr->shape[2];
  } else {
    infer_channel = infer_result_ptr->shape[1];
    infer_height = infer_result_ptr->shape[2];
    infer_width = infer_result_ptr->shape[3];
  }
  int64_t infer_chw = infer_channel * infer_height * infer_width;

  bool is_resized = false;
  auto iter_ipt = im_info.find("input_shape");
  FDASSERT(iter_ipt != im_info.end(), "Cannot find input_shape from im_info.");
  int ipt_h = iter_ipt->second[0];
  int ipt_w = iter_ipt->second[1];
  if (ipt_h != infer_height || ipt_w != infer_width) {
    is_resized = true;
  }

  if (!is_with_softmax_ && apply_softmax_) {
    Softmax(*infer_result_ptr, infer_result_ptr, 1);
  }

  if (!is_with_argmax_) {
    // output without argmax
    result->contain_score_map = true;

    std::vector<int64_t> dim{0, 2, 3, 1};
    Transpose(*infer_result_ptr, infer_result_ptr, dim);
  }
  // batch always 1, so ignore
  infer_result_ptr->shape = {infer_height, infer_width, infer_channel};

  // for resize mat below
  FDTensor new_infer_result;
  Mat* mat = nullptr;
  std::vector<float_t>* fp32_result_buffer = nullptr;
  if (is_resized) {
    if (infer_result_ptr->dtype == FDDataType::INT64 ||
        infer_result_ptr->dtype == FDDataType::INT32) {
      if (infer_result_ptr->dtype == FDDataType::INT64) {
        int64_t* infer_result_buffer =
            static_cast<int64_t*>(infer_result_ptr->Data());
        // cv::resize don't support `CV_8S` or `CV_32S`
        // refer to https://github.com/opencv/opencv/issues/20991
        // https://github.com/opencv/opencv/issues/7862
        fp32_result_buffer = new std::vector<float_t>(
            infer_result_buffer, infer_result_buffer + infer_chw);
      }
      if (infer_result_ptr->dtype == FDDataType::INT32) {
        int32_t* infer_result_buffer =
            static_cast<int32_t*>(infer_result_ptr->Data());
        // cv::resize don't support `CV_8S` or `CV_32S`
        // refer to https://github.com/opencv/opencv/issues/20991
        // https://github.com/opencv/opencv/issues/7862
        fp32_result_buffer = new std::vector<float_t>(
            infer_result_buffer, infer_result_buffer + infer_chw);
      }
      infer_result_ptr->Resize(infer_result_ptr->shape, FDDataType::FP32);
      infer_result_ptr->SetExternalData(
          infer_result_ptr->shape, FDDataType::FP32,
          static_cast<void*>(fp32_result_buffer->data()));
    }
    mat = new Mat(CreateFromTensor(*infer_result_ptr));
    Resize::Run(mat, ipt_w, ipt_h, -1.0f, -1.0f, 1);
    mat->ShareWithTensor(&new_infer_result);
    result->shape = new_infer_result.shape;
  } else {
    result->shape = infer_result_ptr->shape;
  }
  // output shape is 2-D HW layout, so out_num = H * W
  int out_num =
      std::accumulate(result->shape.begin(), result->shape.begin() + 2, 1,
                      std::multiplies<int>());
  result->Resize(out_num);
  if (result->contain_score_map) {
    // output with label_map and score_map
    int32_t* argmax_infer_result_buffer = nullptr;
    float_t* score_infer_result_buffer = nullptr;
    FDTensor argmax_infer_result;
    FDTensor max_score_result;
    std::vector<int64_t> reduce_dim{-1};
    // argmax
    if (is_resized) {
      ArgMax(new_infer_result, &argmax_infer_result, -1, FDDataType::INT32);
      Max(new_infer_result, &max_score_result, reduce_dim);
    } else {
      ArgMax(*infer_result_ptr, &argmax_infer_result, -1, FDDataType::INT32);
      Max(*infer_result_ptr, &max_score_result, reduce_dim);
    }
    argmax_infer_result_buffer =
        static_cast<int32_t*>(argmax_infer_result.Data());
    score_infer_result_buffer = static_cast<float_t*>(max_score_result.Data());
    for (int i = 0; i < out_num; i++) {
      result->label_map[i] =
          static_cast<uint8_t>(*(argmax_infer_result_buffer + i));
    }
    std::memcpy(result->score_map.data(), score_infer_result_buffer,
                out_num * sizeof(float_t));

  } else {
    // output only with label_map
    if (is_resized) {
      float_t* infer_result_buffer =
          static_cast<float_t*>(new_infer_result.Data());
      for (int i = 0; i < out_num; i++) {
        result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
      }
    } else {
      if (infer_result_ptr->dtype == FDDataType::INT64) {
        const int64_t* infer_result_buffer =
            static_cast<const int64_t*>(infer_result_ptr->Data());
        for (int i = 0; i < out_num; i++) {
          result->label_map[i] =
              static_cast<uint8_t>(*(infer_result_buffer + i));
        }
      }
      if (infer_result_ptr->dtype == FDDataType::INT32) {
        const int32_t* infer_result_buffer =
            static_cast<const int32_t*>(infer_result_ptr->Data());
        for (int i = 0; i < out_num; i++) {
          result->label_map[i] =
              static_cast<uint8_t>(*(infer_result_buffer + i));
        }
      }
    }
  }
  // HWC remove C
  result->shape.erase(result->shape.begin() + 2);
  delete fp32_result_buffer;
  delete mat;
  mat = nullptr;
  return true;
}
}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy