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

#include "rkyolov5.h"
#include <array>
namespace fastdeploy {
namespace vision {
namespace detection {

RKYOLOv5::RKYOLOv5(const std::string& model_file,
                   const std::string& params_file,
                   const fastdeploy::RuntimeOption& custom_option,
                   const fastdeploy::ModelFormat& model_format) {
  valid_cpu_backends = {Backend::ORT};
  valid_rknpu_backends = {Backend::RKNPU2};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool RKYOLOv5::Initialize() {
  // parameters for preprocess
  reused_input_tensors.resize(1);

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  return true;
}

bool RKYOLOv5::Preprocess(fastdeploy::vision::Mat* mat,
                          std::vector<FDTensor>* outputs) {

  std::vector<float> input_shape = {static_cast<float>(mat->Height()),
                                    static_cast<float>(mat->Width())};
  std::vector<float> output_shape = {640.0, 640.0};
  std::vector<float> padding_value = {114.0, 114.0, 114.0};

  // process after image load
  float ratio = (output_shape[0]) / std::max(static_cast<float>(mat->Height()),
                                             static_cast<float>(mat->Width()));
  if (ratio != 1.0) {
    int interp = cv::INTER_AREA;
    if (ratio > 1.0) {
      interp = cv::INTER_LINEAR;
    }
    int resize_h = int(mat->Height() * ratio);
    int resize_w = int(mat->Width() * ratio);
    Resize::Run(mat, resize_w, resize_h, -1, -1, interp);
  }
  BGR2RGB::Run(mat);
  PadToSize::Run(mat, output_shape[0], output_shape[1], padding_value);
  Cast::Run(mat, "float");
  outputs->resize(1);
  (*outputs)[0].name = InputInfoOfRuntime(0).name;
  mat->ShareWithTensor(&((*outputs)[0]));
  // reshape to [1, c, h, w]
  (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);
  return true;
}

int RKYOLOv5::Process(FDTensor &input_tensor,
                      std::vector<int> &anchor,
                      int &stride,
                      DetectionResult* result) {
  int validCount = 0;
  auto* input = static_cast<int8_t *>(input_tensor.MutableData());
  int prob_box_size = input_tensor.shape[2];
  int obj_class_num = prob_box_size - 5;
  int grid_h = input_tensor.shape[3];
  int grid_w = input_tensor.shape[4];
  int grid_len = grid_h * grid_w;
  float thres = unsigmoid(threshold);
  int8_t thres_i8 = qnt_f32_to_affine(thres,
                                      input_tensor.rknpu2_zp,
                                      input_tensor.rknpu2_scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        int8_t box_confidence =
            input[(prob_box_size * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8) {
          int     offset = (prob_box_size * a) * grid_len + i * grid_w + j;
          int8_t* in_ptr = input + offset;
          float   box_x  = sigmoid(deqnt_affine_to_f32(*in_ptr, input_tensor.rknpu2_zp, input_tensor.rknpu2_scale)) * 2.0 - 0.5;
          float   box_y  = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], input_tensor.rknpu2_zp, input_tensor.rknpu2_scale)) * 2.0 - 0.5;
          float   box_w  = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], input_tensor.rknpu2_zp, input_tensor.rknpu2_scale)) * 2.0;
          float   box_h  = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], input_tensor.rknpu2_zp, input_tensor.rknpu2_scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < obj_class_num; ++k) {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs > thres_i8) {
            result->scores.push_back(
                sigmoid(deqnt_affine_to_f32(maxClassProbs,
                                      input_tensor.rknpu2_zp,
                                      input_tensor.rknpu2_scale)) *
                sigmoid(deqnt_affine_to_f32(box_confidence,
                                      input_tensor.rknpu2_zp,
                                      input_tensor.rknpu2_scale)));
            result->label_ids.push_back(maxClassId);
            validCount++;
            result->boxes.emplace_back(std::array<float, 4>{
                box_x, box_y,
                box_x+box_w, box_y+box_h});
          }
        }
      }
    }
  }
  return validCount;
}

bool RKYOLOv5::Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result) {
  for (int i = 0; i < infer_result.size(); ++i) {
    std::cout << Process(infer_result[i],
                         anchors[i],
                         strides[i],
                         result) << std::endl;
  }
  utils::NMS(result);
  return true;
}

bool RKYOLOv5::Predict(cv::Mat* im, DetectionResult* result) {
  Mat mat(*im);

  std::vector<FDTensor> processed_data;
  if (!Preprocess(&mat, &processed_data)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  float* tmp = static_cast<float*>(processed_data[1].Data());
  std::vector<FDTensor> infer_result;
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  if (!Postprocess(infer_result, result)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}
} // namespace detection
} // namespace vision
} // namespace fastdeploy