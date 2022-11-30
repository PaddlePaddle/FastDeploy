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

#include "pipeline_stable_diffusion_inpaint.h"

using namespace paddlenlp;

namespace fastdeploy {

StableDiffusionInpaintPipeline::StableDiffusionInpaintPipeline(
    std::unique_ptr<Runtime> vae_encoder, std::unique_ptr<Runtime> vae_decoder,
    std::unique_ptr<Runtime> text_encoder, std::unique_ptr<Runtime> unet,
    std::unique_ptr<Scheduler> scheduler,
    const paddlenlp::fast_tokenizer::tokenizers_impl::ClipFastTokenizer&
        tokenizer)
    : vae_encoder_(std::move(vae_encoder)),
      vae_decoder_(std::move(vae_decoder)),
      text_encoder_(std::move(text_encoder)), unet_(std::move(unet)),
      scheduler_(std::move(scheduler)), tokenizer_(tokenizer) {}

void StableDiffusionInpaintPipeline::Predict(
    const std::vector<std::string>& prompts, cv::Mat* image,
    cv::Mat* mask_image, FDTensor* output_image, int height, int width,
    int num_inference_steps, float guidance_scale,
    const std::vector<std::string>& negative_prompt, int num_images_per_prompt,
    float eta, uint32_t max_length, const FDTensor* latents,
    callback_ptr callback, int callback_steps) {
  int batch_size = prompts.size();
  FDASSERT(batch_size >= 1, "prompts should not be empty");
  FDASSERT(
      height % 8 != 0 or width % 8 != 0,
      "`height` and `width` have to be divisible by 8 but are {%d} and {%d}.",
      height, width);
  FDASSERT(callback_steps <= 0,
           "`callback_steps` has to be a positive integer but is {%d}",
           callback_steps);

  scheduler_->SetTimesteps(num_inference_steps);

  // Setting tokenizer attr
  if (max_length == 0) {
    tokenizer_.EnablePadMethod(fast_tokenizer::core::RIGHT,
                               tokenizer_.GetPadTokenId(), 0,
                               tokenizer_.GetPadToken(), nullptr, nullptr);
    tokenizer_.DisableTruncMethod();
  } else {
    tokenizer_.EnablePadMethod(fast_tokenizer::core::RIGHT,
                               tokenizer_.GetPadTokenId(), 0,
                               tokenizer_.GetPadToken(), &max_length, nullptr);
    tokenizer_.EnableTruncMethod(max_length, 0, fast_tokenizer::core::RIGHT,
                                 fast_tokenizer::core::LONGEST_FIRST);
  }
  std::vector<fast_tokenizer::core::Encoding> encodings;
  tokenizer_.EncodeBatchStrings(prompts, &encodings);
}
}  // namespace fastdeploy
