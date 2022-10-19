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

#include "fastdeploy/vision/matting/contrib/rvm.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {

namespace vision {

namespace matting {

RobustVideoMatting::RobustVideoMatting(const std::string& model_file, const std::string& params_file,
               const RuntimeOption& custom_option,
               const ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT}; 
    valid_gpu_backends = {Backend::ORT}; 
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool RobustVideoMatting::Initialize() {
  // parameters for preprocess
  size = {1080, 1920};

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool RobustVideoMatting::Preprocess(Mat* mat, FDTensor* output,
                        std::map<std::string, std::array<int, 2>>* im_info) {
  // Resize
  int resize_w = size[0];
  int resize_h = size[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }
  BGR2RGB::Run(mat);
  
  // Normalize
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  Convert::Run(mat, alpha, beta);
  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {mat->Height(), mat->Width()};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool RobustVideoMatting::Postprocess(
    std::vector<FDTensor>& infer_result, MattingResult* result,
    const std::map<std::string, std::array<int, 2>>& im_info) {
  FDASSERT((infer_result.size() == 6),
           "The default number of output tensor must be 6 according to "
           "RobustVideoMatting.");
  FDTensor& fgr = infer_result.at(0); // fgr (1, 3, h, w) 0.~1.
  FDTensor& alpha = infer_result.at(1);  // alpha (1, 1, h, w) 0.~1.
  FDASSERT((fgr.shape[0] == 1), "Only support batch = 1 now.");
  FDASSERT((alpha.shape[0] == 1), "Only support batch = 1 now.");
  if (fgr.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  if (alpha.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  for (size_t i = 0; i < 4; ++i) {
    FDTensor& rki = infer_result.at(i+2);
    dynamic_inputs_dims_[i] = rki.shape;
    dynamic_inputs_datas_[i].resize(rki.Numel());
    memcpy(dynamic_inputs_datas_[i].data(), rki.Data(),
                        rki.Numel() * FDDataTypeSize(rki.dtype));
  }

  auto iter_in = im_info.find("input_shape");
  auto iter_out = im_info.find("output_shape");
  FDASSERT(iter_out != im_info.end() && iter_in != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  int out_h = iter_out->second[0];
  int out_w = iter_out->second[1];
  int in_h = iter_in->second[0];
  int in_w = iter_in->second[1];

  float* alpha_ptr = static_cast<float*>(alpha.Data());
  cv::Mat alpha_zero_copy_ref(out_h, out_w, CV_32FC1, alpha_ptr);
  Mat alpha_resized(alpha_zero_copy_ref);  // ref-only, zero copy.
  if ((out_h != in_h) || (out_w != in_w)) {
    // already allocated a new continuous memory after resize.
    Resize::Run(&alpha_resized, in_w, in_h, -1, -1);
  }

  result->Clear();
  result->contain_foreground = false;
  result->shape = {static_cast<int64_t>(in_h), static_cast<int64_t>(in_w)};
  int numel = in_h * in_w;
  int nbytes = numel * sizeof(float);
  result->Resize(numel);
  memcpy(result->alpha.data(), alpha_resized.GetOpenCVMat()->data, nbytes);
  return true;
}

bool RobustVideoMatting::Predict(cv::Mat* im, MattingResult* result) {
  Mat mat(*im);
  int inputs_nums = NumInputsOfRuntime();
  std::vector<FDTensor> input_tensors(inputs_nums);
  std::map<std::string, std::array<int, 2>> im_info;
  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {mat.Height(), mat.Width()};
  im_info["output_shape"] = {mat.Height(), mat.Width()};
  // convert vector to FDTensor
  for (size_t i = 1; i < inputs_nums; ++i) {
    input_tensors[i].SetExternalData(dynamic_inputs_dims_[i-1], FDDataType::FP32, dynamic_inputs_datas_[i-1].data());
    input_tensors[i].device = Device::CPU;
  }
  if (!Preprocess(&mat, &input_tensors[0], &im_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  for (size_t i = 0; i < inputs_nums; ++i) {
    input_tensors[i].name = InputInfoOfRuntime(i).name;
  }
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors, result, im_info)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy