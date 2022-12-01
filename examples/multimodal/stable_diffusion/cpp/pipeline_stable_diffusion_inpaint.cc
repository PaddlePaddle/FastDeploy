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
#include "fastdeploy/function/functions.h"
#include <algorithm>

using namespace paddlenlp;

namespace fastdeploy {

static constexpr int NUM_LATENT_CHANNELS = 4;
static constexpr int NUM_UNET_INPUT_CHANNELS = 9;

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

  std::vector<int64_t> input_ids;
  for (auto& encoding : encodings) {
    auto curr_ids = encoding.GetIds();
    input_ids.insert(input_ids.end(), curr_ids.begin(), curr_ids.end());
  }
  encodings.clear();
  // Get text encoder output
  FDTensor text_intput_ids;
  std::vector<FDTensor> text_inputs(1);
  text_inputs[0].SetExternalData({batch_size, max_length}, FDDataType::INT64,
                                 input_ids.data());

  TensorInfo text_info = text_encoder_->GetInputInfo(0);
  text_inputs[0].name = text_info.name;
  int output_size = text_encoder_->GetOutputInfos().size();
  std::vector<FDTensor> text_outputs(output_size);
  text_encoder_->Infer(text_inputs, &text_outputs);

  FDTensor text_embeddings;
  function::Tile(text_outputs[0], {num_images_per_prompt, 1, 1},
                 &text_embeddings);

  //    here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
  //    of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
  //    corresponds to doing no classifier free guidance.
  bool do_classifier_free_guidance = guidance_scale > 1.0;
  if (do_classifier_free_guidance) {
    std::vector<std::string> uncond_tokens;
    if (negative_prompt.size() == 0) {
      uncond_tokens = {""};
    } else if (negative_prompt.size() != batch_size) {
      FDASSERT(false,
               "negative_prompt has batch size %d, but prompt has batch size "
               "%d. Please make sure that passed `negative_prompt` matches the "
               "batch size of `prompt`.",
               prompts.size(), negative_prompt.size());
    } else {
      uncond_tokens = negative_prompt;
    }
    tokenizer_.EncodeBatchStrings(uncond_tokens, &encodings);
    input_ids.clear();
    for (auto& encoding : encodings) {
      auto curr_ids = encoding.GetIds();
      input_ids.insert(input_ids.end(), curr_ids.begin(), curr_ids.end());
    }
    text_inputs[0].SetExternalData({batch_size, max_length}, FDDataType::INT64,
                                   input_ids.data());
    text_encoder_->Infer(text_inputs, &text_outputs);
    FDTensor uncond_embeddings;
    function::Tile(text_outputs[0], {num_images_per_prompt, 1, 1},
                   &uncond_embeddings);
    function::Concat({uncond_embeddings, text_embeddings}, &text_embeddings);
  }
  std::vector<int64_t> latents_shape = {batch_size * num_images_per_prompt,
                                        NUM_LATENT_CHANNELS, height / 8,
                                        width / 8};
  auto latents_dtype = text_embeddings.Dtype();
  if (latents == nullptr) {

  } else if {
    bool result = std::equals(latents_shape.begin(), latents_shape.end(),
                              latents->Shape().begin());
    FDASSERT(result, "Unexpected latents shape, got %s, expected %s",
             Str(latents_shape).c_str(), Str(latents->Shape()).c_str());
  }
}
}  // namespace fastdeploy
