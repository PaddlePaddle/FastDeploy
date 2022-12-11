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

#include "./dpm_solver_multistep_scheduler.h"
#include "./pipeline_stable_diffusion_inpaint.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/common/processors/mat.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

template <typename T> std::string Str(const T* value, int size) {
  std::ostringstream oss;
  oss << "[ " << value[0];
  for (int i = 1; i < size; ++i) {
    oss << " ," << value[i];
  }
  oss << " ]";
  return oss.str();
}

std::unique_ptr<fastdeploy::Runtime> CreateRuntime(
    const std::string& model_file, const std::string& params_file,
    bool use_trt_backend = false, bool use_fp16 = false,
    const std::unordered_map<std::string, std::vector<std::vector<int>>>&
        dynamic_shapes = {},
    const std::vector<std::string>& disable_paddle_trt_ops = {}) {
  fastdeploy::RuntimeOption runtime_option;
  runtime_option.SetModelPath(model_file, params_file,
                              fastdeploy::ModelFormat::PADDLE);
  runtime_option.UseGpu();
  if (!use_trt_backend) {
    runtime_option.UsePaddleBackend();
  } else {
    runtime_option.UseTrtBackend();
    runtime_option.EnablePaddleToTrt();
    for (auto it = dynamic_shapes.begin(); it != dynamic_shapes.end(); ++it) {
      if (it->second.size() != 3) {
        std::cerr << "The size of dynamic_shapes of input `" << it->first
                  << "` should be 3, but receive " << it->second.size()
                  << std::endl;
        continue;
      }
      std::vector<int> min_shape = (it->second)[0];
      std::vector<int> opt_shape = (it->second)[1];
      std::vector<int> max_shape = (it->second)[2];
      runtime_option.SetTrtInputShape(it->first, min_shape, opt_shape,
                                      max_shape);
    }
    runtime_option.SetTrtCacheFile("paddle.trt");
    runtime_option.EnablePaddleTrtCollectShape();
    runtime_option.DisablePaddleTrtOPs(disable_paddle_trt_ops);
    if (use_fp16) {
      runtime_option.EnableTrtFP16();
    }
  }
  std::unique_ptr<fastdeploy::Runtime> runtime =
      std::unique_ptr<fastdeploy::Runtime>(new fastdeploy::Runtime());
  if (!runtime->Init(runtime_option)) {
    std::cerr << "--- Init FastDeploy Runitme Failed! "
              << "\n--- Model:  " << model_file << std::endl;
    return nullptr;
  } else {
    std::cout << "--- Init FastDeploy Runitme Done! "
              << "\n--- Model:  " << model_file << std::endl;
  }
  return runtime;
}

int main() {
  // 0. Init all configs
  std::string model_dir = "sd15_inpaint";
  int max_length = 77;
  bool use_trt_backend = true;
  bool use_fp16 = true;
  int batch_size = 1;
  int num_images_per_prompt = 1;
  int num_inference_steps = 50;

  int height = 512;
  int width = 512;
  constexpr int unet_inpaint_channels = 9;
  constexpr int latents_channels = 4;

  // 1. Init scheduler
  std::unique_ptr<fastdeploy::Scheduler> dpm(
      new fastdeploy::DPMSolverMultistepScheduler(
          /* num_train_timesteps */ 1000,
          /* beta_start = */ 0.00085,
          /* beta_end = */ 0.012,
          /* beta_schedule = */ "scaled_linear",
          /* trained_betas = */ {},
          /* solver_order = */ 2,
          /* predict_epsilon = */ true,
          /* thresholding = */ false,
          /* dynamic_thresholding_ratio = */ 0.995,
          /* sample_max_value = */ 1.0,
          /* algorithm_type = */ "dpmsolver++",
          /* solver_type = */ "midpoint",
          /* lower_order_final = */ true));

  // 2. Init text encoder runtime
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      text_dynamic_shape = {{"input_ids",
                             {/* min_shape */ {1, max_length},
                              /* opt_shape */ {batch_size, max_length},
                              /* max_shape */ {2 * batch_size, max_length}}}};
  std::string text_model_dir = model_dir + sep + "text_encoder";
  std::string text_model_file = text_model_dir + sep + "inference.pdmodel";
  std::string text_params_file = text_model_dir + sep + "inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> text_encoder_runtime =
      CreateRuntime(text_model_file, text_params_file, use_trt_backend,
                    use_fp16, text_dynamic_shape);

  // 3. Init vae encoder runtime
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      vae_encoder_dynamic_shape = {
          {"sample",
           {/* min_shape */ {1, 3, height, width},
            /* opt_shape */ {2 * batch_size, 3, height, width},
            /* max_shape */ {2 * batch_size, 3, height, width}}}};
  std::string vae_encoder_model_dir = model_dir + sep + "vae_encoder";
  std::string vae_encoder_model_file =
      vae_encoder_model_dir + sep + "inference.pdmodel";
  std::string vae_encoder_params_file =
      vae_encoder_model_dir + sep + "inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> vae_encoder_runtime =
      CreateRuntime(vae_encoder_model_file, vae_encoder_params_file,
                    use_trt_backend, use_fp16, vae_encoder_dynamic_shape);

  // 4. Init vae decoder runtime
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      vae_decoder_dynamic_shape = {
          {"latent_sample",
           {/* min_shape */ {1, latents_channels, height / 8, width / 8},
            /* opt_shape */
            {2 * batch_size, latents_channels, height / 8, width / 8},
            /* max_shape */
            {2 * batch_size, latents_channels, height / 8, width / 8}}}};
  std::string vae_decoder_model_dir = model_dir + sep + "vae_decoder";
  std::string vae_decoder_model_file =
      vae_decoder_model_dir + sep + "inference.pdmodel";
  std::string vae_decoder_params_file =
      vae_decoder_model_dir + sep + "inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> vae_decoder_runtime =
      CreateRuntime(vae_decoder_model_file, vae_decoder_params_file,
                    use_trt_backend, use_fp16, vae_decoder_dynamic_shape);

  // 5. Init unet runtime
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      unet_dynamic_shape = {
          {"sample",
           {/* min_shape */ {1, unet_inpaint_channels, height / 8, width / 8},
            /* opt_shape */
            {2 * batch_size, unet_inpaint_channels, height / 8, width / 8},
            /* max_shape */
            {2 * batch_size, unet_inpaint_channels, height / 8, width / 8}}},
          {"timesteps", {{1}, {1}, {1}}},
          {"encoder_hidden_states",
           {{1, max_length, 768},
            {2 * batch_size, max_length, 768},
            {2 * batch_size, max_length, 768}}}};
  std::vector<std::string> unet_disable_paddle_trt_ops = {"sin", "cos"};
  std::string unet_model_dir = model_dir + sep + "unet";
  std::string unet_model_file = unet_model_dir + sep + "inference.pdmodel";
  std::string unet_params_file = unet_model_dir + sep + "inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> unet_runtime =
      CreateRuntime(unet_model_file, unet_params_file, use_trt_backend,
                    use_fp16, unet_dynamic_shape, unet_disable_paddle_trt_ops);

  // 6. Init fast tokenizer
  paddlenlp::fast_tokenizer::tokenizers_impl::ClipFastTokenizer tokenizer(
      "clip/vocab.json", "clip/merges.txt", /* max_length = */ max_length);
  fastdeploy::StableDiffusionInpaintPipeline pipe(
      /* vae_encoder = */ std::move(vae_encoder_runtime),
      /* vae_decoder = */ std::move(vae_decoder_runtime),
      /* text_encoder = */ std::move(text_encoder_runtime),
      /* unet = */ std::move(unet_runtime),
      /* scheduler = */ std::move(dpm),
      /* tokenizer = */ tokenizer);

  // 7. Read images
  auto image = cv::imread("overture-creations.png");
  auto mask_image = cv::imread("overture-creations-mask.png");

  // 8. Predict
  /*
   * One may need to pass the initial noise to predict api.
   * There's an example:
   * std::vector<float> latents_data = {xxxx};
   * fastdeploy::FDTensor latents;
   * latents.SetExternalData({batch_size * num_images_per_prompt, latents_channels, height / 8, width / 8},fastdeploy::FDDataType::FP32, latents_data.data());
   * pipe.Predict(..., /* latents = *\/ &latents, ....);
   */
  std::vector<std::string> prompts = {
      "Face of a yellow cat, high resolution, sitting on a park bench"};
  std::vector<fastdeploy::FDTensor> outputs;
  fastdeploy::TimeCounter tc;
  tc.Start();
  pipe.Predict(prompts, image, mask_image, &outputs,
               /* height = */ height,
               /* width = */ width,
               /* num_inference_steps = */ num_inference_steps,
               /* guidance_scale = */ 7.5,
               /* negative_prompt = */ {},
               /* num_images_per_prompt = */ num_images_per_prompt,
               /* eta = */ 1.0,
               /* max_length = */ max_length,
               /* latents = */ nullptr,
               /* output_cv_mat = */ true,
               /* callback = */ nullptr,
               /* callback_steps = */ 1);
  tc.End();
  tc.PrintInfo();
  fastdeploy::vision::FDMat mat = fastdeploy::vision::FDMat::Create(outputs[0]);
  cv::imwrite("cat_on_bench_new.png", *mat.GetOpenCVMat());
  return 0;
}