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

#include "fastdeploy/vision/detection/contrib/yolov5.h"

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOv5::YOLOv5(const std::string& model_file, const std::string& params_file,
               const RuntimeOption& custom_option,
               const ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
#ifdef ENABLE_CUDA_PREPROCESS
  cudaSetDevice(runtime_option.device_id);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cuda_stream_ = reinterpret_cast<void*>(stream);
  runtime_option.SetExternalStream(cuda_stream_);
#endif  // ENABLE_CUDA_PREPROCESS
  initialized = Initialize();
}

bool YOLOv5::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

YOLOv5::~YOLOv5() {
#ifdef ENABLE_CUDA_PREPROCESS
  if (use_cuda_preprocessing_) {
    CUDA_CHECK(cudaFreeHost(input_img_cuda_buffer_host_));
    CUDA_CHECK(cudaFree(input_img_cuda_buffer_device_));
    CUDA_CHECK(cudaFree(input_tensor_cuda_buffer_device_));
    CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cuda_stream_)));
  }
#endif  // ENABLE_CUDA_PREPROCESS
}

bool YOLOv5::Predict(cv::Mat* im, DetectionResult* result, float conf_threshold, float nms_threshold) {
  postprocessor_.SetConfThreshold(conf_threshold);
  postprocessor_.SetNMSThreshold(nms_threshold);

}

YOLOv5::Predict(const cv::Mat* im, DetectionResult* result) {

                     }

YOLOv5::BatchPredict(const std::vector<cv::Mat>& images, std::vector<DetectionResult>* results) {
  std::map<std::string, std::array<float, 2>> im_info;

  std::vector<FDMat> fd_images = WrapMat(images);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_, &im_info)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results, &im_info)) {
    FDERROR << "Failed to postprocess the inference results by runtime." << std::endl;
    return false;
  }

  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
