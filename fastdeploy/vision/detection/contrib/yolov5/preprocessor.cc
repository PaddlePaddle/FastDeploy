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

#include "fastdeploy/vision/detection/contrib/yolov5/preprocessor.h"
#include "fastdeploy/function/concat.h"

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOv5Preprocessor::YOLOv5Preprocessor() {
  size_ = {640, 640};
  padding_value_ = {114.0, 114.0, 114.0};
  is_mini_pad_ = false;
  is_no_pad_ = false;
  is_scale_up_ = false;
  stride_ = 32;
  max_wh_ = 7680.0;
  initialized_ = true;
}

void YOLOv5Preprocessor::LetterBox(FDMat* mat) {
  float scale =
      std::min(size_[1] * 1.0 / mat->Height(), size_[0] * 1.0 / mat->Width());
  if (!is_scale_up_) {
    scale = std::min(scale, 1.0f);
  }

  int resize_h = int(round(mat->Height() * scale));
  int resize_w = int(round(mat->Width() * scale));

  int pad_w = size_[0] - resize_w;
  int pad_h = size_[1] - resize_h;
  if (is_mini_pad_) {
    pad_h = pad_h % stride_;
    pad_w = pad_w % stride_;
  } else if (is_no_pad_) {
    pad_h = 0;
    pad_w = 0;
    resize_h = size_[1];
    resize_w = size_[0];
  }
  Resize::Run(mat, resize_w, resize_h);
  if (pad_h > 0 || pad_w > 0) {
    float half_h = pad_h * 1.0 / 2;
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    float half_w = pad_w * 1.0 / 2;
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));
    Pad::Run(mat, top, bottom, left, right, padding_value_);
  }
}

bool YOLOv5Preprocessor::Preprocess(FDMat* mat, FDTensor* output,
            std::map<std::string, std::array<float, 2>>* im_info) {
  // Record the shape of image and the shape of preprocessed image
  (*im_info)["input_shape"] = {static_cast<float>(mat->Height()),
                               static_cast<float>(mat->Width())};

  // process after image load
  double ratio = (size_[0] * 1.0) / std::max(static_cast<float>(mat->Height()),
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
  // yolov5's preprocess steps
  // 1. letterbox
  // 2. BGR->RGB
  // 3. HWC->CHW
  LetterBox(mat);
  BGR2RGB::Run(mat);
  // Compute `result = mat * alpha + beta` directly by channel
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  Convert::Run(mat, alpha, beta);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->ExpandDim(0);  // reshape to n, h, w, c
  return true;
}

void YOLOv5Preprocessor::UseCudaPreprocessing(int max_image_size) {
#ifdef ENABLE_CUDA_PREPROCESS
  use_cuda_preprocessing_ = true;
  is_scale_up_ = true;
  if (input_img_cuda_buffer_host_ == nullptr) {
    // prepare input data cache in GPU pinned memory
    CUDA_CHECK(cudaMallocHost((void**)&input_img_cuda_buffer_host_,
                              max_image_size * 3));
    // prepare input data cache in GPU device memory
    CUDA_CHECK(
        cudaMalloc((void**)&input_img_cuda_buffer_device_, max_image_size * 3));
    CUDA_CHECK(cudaMalloc((void**)&input_tensor_cuda_buffer_device_,
                          3 * size_[0] * size_[1] * sizeof(float)));
  }
#else
  FDWARNING << "The FastDeploy didn't compile with BUILD_CUDA_SRC=ON."
            << std::endl;
  use_cuda_preprocessing_ = false;
#endif
}

bool YOLOv5Preprocessor::CudaPreprocess(FDMat* mat, FDTensor* output,
                std::map<std::string, std::array<float, 2>>* im_info) {
#ifdef ENABLE_CUDA_PREPROCESS
  if (is_mini_pad_ != false || is_no_pad_ != false || is_scale_up_ != true) {
    FDERROR << "Preprocessing with CUDA is only available when the arguments "
               "satisfy (is_mini_pad_=false, is_no_pad_=false, is_scale_up_=true)."
            << std::endl;
    return false;
  }

  // Record the shape of image and the shape of preprocessed image
  (*im_info)["input_shape"] = {static_cast<float>(mat->Height()),
                               static_cast<float>(mat->Width())};

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  int src_img_buf_size = mat->Height() * mat->Width() * mat->Channels();
  memcpy(input_img_cuda_buffer_host_, mat->Data(), src_img_buf_size);
  CUDA_CHECK(cudaMemcpyAsync(input_img_cuda_buffer_device_,
                             input_img_cuda_buffer_host_, src_img_buf_size,
                             cudaMemcpyHostToDevice, stream));
  utils::CudaYoloPreprocess(input_img_cuda_buffer_device_, mat->Width(),
                            mat->Height(), input_tensor_cuda_buffer_device_,
                            size_[0], size_[1], padding_value_, stream);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(size_[0]),
                                static_cast<float>(size_[1])};

  output->SetExternalData({mat->Channels(), size_[0], size_[1]}, FDDataType::FP32,
                          input_tensor_cuda_buffer_device_);
  output->device = Device::GPU;
  output->ExpandDim(0);  // reshape to n, h, w, c
  return true;
#else
  FDERROR << "CUDA src code was not enabled." << std::endl;
  return false;
#endif  // ENABLE_CUDA_PREPROCESS
}

bool YOLOv5Preprocessor::Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs,
                             std::map<std::string, std::array<float, 2>>* im_info) {
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  if (images->size() == 0) {
    FDERROR << "The size of input images should be greater than 0." << std::endl;
    return false;
  }
  outputs->resize(1);
  // Concat all the preprocessed data to a batch tensor
  std::vector<FDTensor> tensors(images->size()); 
  for (size_t i = 0; i < images->size(); ++i) {
    if (use_cuda_preprocessing_) {
      if (!CudaPreprocess(&(*images)[i], &tensors[i], im_info)) {
      FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
      }
    } else {
      if (!Preprocess(&(*images)[i], &tensors[i], im_info)) {
        FDERROR << "Failed to preprocess input image." << std::endl;
        return false;
      }
    }
  }

  if (tensors.size() == 1) {
    (*outputs)[0] = std::move(tensors[0]);
  } else {
    function::Concat(tensors, &((*outputs)[0]), 0);
  }
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
