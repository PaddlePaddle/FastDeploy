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

#pragma once

#include "./scheduler.h"
#include "fast_tokenizer/tokenizers/clip_fast_tokenizer.h"
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/runtime.h"
#include "opencv2/core/core.hpp"
#include <memory>
#include <string>
#include <vector>

namespace fastdeploy {

class StableDiffusionInpaintPipeline {
 public:
  typedef void (*callback_ptr)(int, int, FDTensor*);

  StableDiffusionInpaintPipeline(
      std::unique_ptr<Runtime> vae_encoder,
      std::unique_ptr<Runtime> vae_decoder,
      std::unique_ptr<Runtime> text_encoder, std::unique_ptr<Runtime> unet,
      std::unique_ptr<Scheduler> scheduler,
      const paddlenlp::fast_tokenizer::tokenizers_impl::ClipFastTokenizer&
          tokenizer);
  void Predict(const std::vector<std::string>& prompts, const cv::Mat& image,
               const cv::Mat& mask_image, std::vector<FDTensor>* output_images,
               int height = 512, int width = 512, int num_inference_steps = 50,
               float guidance_scale = 7.5,
               const std::vector<std::string>& negative_prompt = {},
               int num_images_per_prompt = 1, float eta = 0.0,
               uint32_t max_length = 77, const FDTensor* latents = nullptr,
               bool output_cv_mat = true, callback_ptr callback = nullptr,
               int callback_steps = 1);

 private:
  void PrepareMaskAndMaskedImage(const cv::Mat& image, const cv::Mat& mask_mat,
                                 const std::vector<int64_t>& shape,
                                 FDTensor* mask, FDTensor* mask_image);
  std::unique_ptr<Runtime> vae_encoder_;
  std::unique_ptr<Runtime> vae_decoder_;
  std::unique_ptr<Runtime> text_encoder_;
  std::unique_ptr<Runtime> unet_;
  std::unique_ptr<Scheduler> scheduler_;
  paddlenlp::fast_tokenizer::tokenizers_impl::ClipFastTokenizer tokenizer_;
};

}  // namespace fastdeploy
