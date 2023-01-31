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
#include "fastdeploy/vision/common/processors/color_space_convert.h"
#include "fastdeploy/vision/common/processors/mat.h"
#include "fastdeploy/vision/common/processors/resize.h"
#include <algorithm>

using namespace paddlenlp;

namespace fastdeploy {

static constexpr int NUM_LATENT_CHANNELS = 4;
static constexpr int NUM_UNET_INPUT_CHANNELS = 9;

void StableDiffusionInpaintPipeline::PrepareMaskAndMaskedImage(
    const cv::Mat& image, const cv::Mat& mask_mat,
    const std::vector<int64_t>& shape, FDTensor* mask, FDTensor* mask_image) {
  vision::FDMat image_fdmat(image);
  vision::BGR2RGB::Run(&image_fdmat, vision::ProcLib::OPENCV);
  vision::Resize::Run(&image_fdmat, shape[1] * 8, shape[0] * 8, -1.0f, -1.0f,
                      cv::INTER_NEAREST, false, vision::ProcLib::OPENCV);
  image_fdmat.ShareWithTensor(mask_image);

  vision::FDMat mask_fdmat(mask_mat);
  vision::BGR2GRAY::Run(&mask_fdmat, vision::ProcLib::OPENCV);
  vision::Resize::Run(&mask_fdmat, shape[1] * 8, shape[0] * 8, -1.0f, -1.0f,
                      cv::INTER_NEAREST, false, vision::ProcLib::OPENCV);
  FDTensor image_mask;
  mask_fdmat.ShareWithTensor(&image_mask);
  function::Cast(image_mask, &image_mask, FDDataType::FP32);
  std::vector<float> float_mask(image_mask.Numel(), 0);
  float* image_mask_ptr = reinterpret_cast<float*>(image_mask.Data());
  for (int i = 0; i < image_mask.Numel(); ++i) {
    if (image_mask_ptr[i] < 127.5) {
      float_mask[i] = 1;
    }
  }
  // NCHW format
  image_mask.SetExternalData({1, 1, shape[0] * 8, shape[1] * 8},
                             FDDataType::FP32, float_mask.data());

  // Set mask_image
  mask_image->ExpandDim();
  function::Transpose(*mask_image, mask_image, {0, 3, 1, 2});
  function::Cast(*mask_image, mask_image, FDDataType::FP32);
  *mask_image = *mask_image / 127.5f - 1.0f;
  *mask_image = *mask_image * image_mask;

  // Set mask
  vision::FDMat mask_fdmat_t(mask_mat);
  vision::BGR2GRAY::Run(&mask_fdmat_t, vision::ProcLib::OPENCV);
  vision::Resize::Run(&mask_fdmat_t, shape[1], shape[0], -1.0f, -1.0f,
                      cv::INTER_NEAREST, false, vision::ProcLib::OPENCV);
  mask_fdmat_t.ShareWithTensor(mask);
  function::Cast(*mask, mask, FDDataType::FP32);
  *mask = *mask / 255.0f;
  mask->ExpandDim();
  function::Transpose(*mask, mask, {0, 3, 1, 2});
  float* mask_data = reinterpret_cast<float*>(mask->Data());
  for (int i = 0; i < mask->Numel(); ++i) {
    if (mask_data[i] < 0.5) {
      mask_data[i] = 0;
    } else {
      mask_data[i] = 1;
    }
  }
}

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
    const std::vector<std::string>& prompts, const cv::Mat& image,
    const cv::Mat& mask_image, std::vector<FDTensor>* output_images, int height,
    int width, int num_inference_steps, float guidance_scale,
    const std::vector<std::string>& negative_prompt, int num_images_per_prompt,
    float eta, uint32_t max_length, const FDTensor* latents, bool output_cv_mat,
    callback_ptr callback, int callback_steps) {
  int batch_size = prompts.size();
  FDASSERT(batch_size >= 1, "prompts should not be empty");
  FDASSERT(
      height % 8 == 0 && width % 8 == 0,
      "`height` and `width` have to be divisible by 8 but are {%d} and {%d}.",
      height, width);
  FDASSERT(callback_steps > 0,
           "`callback_steps` has to be a positive integer but is {%d}",
           callback_steps);

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
  std::vector<FDTensor> inputs(1);
  inputs[0].SetExternalData({batch_size, max_length}, FDDataType::INT64,
                            input_ids.data());

  TensorInfo text_info = text_encoder_->GetInputInfo(0);
  inputs[0].name = text_info.name;
  int output_size = text_encoder_->GetOutputInfos().size();
  std::vector<FDTensor> outputs(output_size);
  text_encoder_->Infer(inputs, &outputs);

  FDTensor text_embeddings;
  function::Tile(outputs[0], {num_images_per_prompt, 1, 1}, &text_embeddings);

  // here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
  // of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
  // corresponds to doing no classifier free guidance.
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
    inputs[0].SetExternalData({batch_size, max_length}, FDDataType::INT64,
                              input_ids.data());
    text_encoder_->Infer(inputs, &outputs);
    FDTensor uncond_embeddings;
    function::Tile(outputs[0], {num_images_per_prompt, 1, 1},
                   &uncond_embeddings);
    function::Concat({uncond_embeddings, text_embeddings}, &text_embeddings);
  }
  std::vector<int64_t> latents_shape = {batch_size * num_images_per_prompt,
                                        NUM_LATENT_CHANNELS, height / 8,
                                        width / 8};
  auto latents_dtype = text_embeddings.Dtype();
  FDTensor actual_latents;
  if (latents == nullptr) {
    function::GaussianRandom(latents_shape, &actual_latents, latents_dtype);
  } else {
    bool result = std::equal(latents_shape.begin(), latents_shape.end(),
                             latents->Shape().begin());
    FDASSERT(result, "Unexpected latents shape, got %s, expected %s",
             Str(latents_shape).c_str(), Str(latents->Shape()).c_str());
    actual_latents = *latents;
  }
  FDTensor mask_t, mask_image_t;
  PrepareMaskAndMaskedImage(image, mask_image, {height / 8, width / 8}, &mask_t,
                            &mask_image_t);
  function::Cast(mask_t, &mask_t, actual_latents.Dtype());
  function::Cast(mask_image_t, &mask_image_t, actual_latents.Dtype());

  // Get vae encoder output
  TensorInfo vae_encoder_info = vae_encoder_->GetInputInfo(0);
  mask_image_t.name = vae_encoder_info.name;
  outputs.resize(vae_encoder_->GetOutputInfos().size());
  inputs = {mask_image_t};
  vae_encoder_->Infer(inputs, &outputs);
  FDTensor masked_image_latents = 0.18215f * outputs[0];

  std::vector<int64_t> mask_shape(mask_t.Shape().size(), 1);
  mask_shape[0] = batch_size * num_images_per_prompt;
  function::Tile(mask_t, mask_shape, &mask_t);

  std::vector<int64_t> mask_image_shape(masked_image_latents.Shape().size(), 1);
  mask_image_shape[0] = batch_size * num_images_per_prompt;
  function::Tile(masked_image_latents, mask_image_shape, &masked_image_latents);

  if (do_classifier_free_guidance) {
    function::Concat({mask_t, mask_t}, &mask_t);
    function::Concat({masked_image_latents, masked_image_latents},
                     &masked_image_latents);
  }
  int num_channels_mask = mask_t.Shape()[1];
  int num_channels_masked_image = masked_image_latents.Shape()[1];
  FDASSERT(
      NUM_LATENT_CHANNELS + num_channels_mask + num_channels_masked_image ==
          NUM_UNET_INPUT_CHANNELS,
      "Incorrect configuration settings! The config of `pipeline.unet` expects"
      " %d but received `num_channels_latents`: %d + `num_channels_mask`: %d "
      "+ `num_channels_masked_image`: %d"
      " = %d. Please verify the config of `pipeline.unet` or your `mask_image` "
      "or `image` input.",
      NUM_UNET_INPUT_CHANNELS, NUM_LATENT_CHANNELS, num_channels_mask,
      num_channels_masked_image,
      NUM_LATENT_CHANNELS + num_channels_mask + num_channels_masked_image);

  // set timesteps
  scheduler_->SetTimesteps(num_inference_steps);

  // scale the initial noise by the standard deviation required by the scheduler
  actual_latents = actual_latents * scheduler_->InitNoiseSigma();

  auto timestep = scheduler_->GetTimesteps();
  int64_t* timestep_data = reinterpret_cast<int64_t*>(timestep.Data());
  outputs.resize(unet_->GetOutputInfos().size());
  inputs.resize(unet_->GetInputInfos().size());
  inputs[2] = std::move(text_embeddings);
  auto unet_infos = unet_->GetInputInfos();
  for (int i = 0; i < timestep.Numel(); ++i) {
    FDTensor t;
    function::Slice(timestep, {0}, {i}, &t);
    inputs[1] = t;
    // expand the latents if we are doing classifier free guidance
    FDTensor latent_model_input;
    if (do_classifier_free_guidance) {
      function::Concat({actual_latents, actual_latents}, &latent_model_input);
    } else {
      latent_model_input = actual_latents;
    }
    // concat latents, mask, masked_image_latnets in the channel dimension
    function::Concat({latent_model_input, mask_t, masked_image_latents},
                     &latent_model_input, 1);
    scheduler_->ScaleModelInput(latent_model_input, &latent_model_input, {t});
    inputs[0] = std::move(latent_model_input);
    // predict the noise residual
    for (int i = 0; i < unet_infos.size(); ++i) {
      inputs[i].name = unet_infos[i].name;
    }
    unet_->Infer(inputs, &outputs);
    FDTensor noise_pred = std::move(outputs[0]);
    // perform guidance
    if (do_classifier_free_guidance) {
      std::vector<FDTensor> noise_preds;
      int dim0 = noise_pred.Shape()[0];
      function::Split(noise_pred, {dim0 - dim0 / 2, dim0 / 2}, &noise_preds);
      noise_pred =
          noise_preds[0] + guidance_scale * (noise_preds[1] - noise_preds[0]);
    }

    // compute the previous noisy sample x_t -> x_t-1
    int64_t time = reinterpret_cast<int64_t*>(t.Data())[0];
    scheduler_->Step(noise_pred, time, actual_latents, &actual_latents);

    // call the callback, if provided
    if (callback != nullptr && i % callback_steps == 0) {
      callback(i, time, &actual_latents);
    }
  }
  actual_latents = (1.0f / 0.18215f) * actual_latents;

  // Get vae decoder output
  int actual_latents_bs = actual_latents.Shape()[0];
  TensorInfo vae_decoder_info = vae_decoder_->GetInputInfo(0);
  inputs.resize(1);
  outputs.resize(vae_decoder_->GetOutputInfos().size());
  std::vector<FDTensor> decoder_reuslt;
  for (int i = 0; i < actual_latents_bs; ++i) {
    function::Slice(actual_latents, {0}, {i}, {i + 1}, &inputs[0]);
    inputs[0].name = vae_decoder_info.name;
    vae_decoder_->Infer(inputs, &outputs);
    decoder_reuslt.emplace_back(std::move(outputs[0]));
  }
  FDTensor output_image;
  function::Concat(decoder_reuslt, &output_image);

  function::Clip(output_image / 2.0f + 0.5f, 0, 1, &output_image);
  function::Transpose(output_image, &output_image, {0, 2, 3, 1});

  if (output_cv_mat) {
    output_image = output_image * 255.0f;
    function::Round(output_image, &output_image);
    function::Cast(output_image, &output_image, FDDataType::UINT8);
  }
  int output_batch_size = output_image.Shape()[0];
  output_images->resize(output_batch_size);
  for (int i = 0; i < output_batch_size; ++i) {
    function::Slice(output_image, {0}, {i}, &(*output_images)[i]);
    vision::FDMat mask_fdmat_t = vision::FDMat::Create((*output_images)[i]);
    vision::RGB2BGR::Run(&mask_fdmat_t, vision::ProcLib::OPENCV);
    mask_fdmat_t.CopyToTensor(&(*output_images)[i]);
  }
}
}  // namespace fastdeploy
