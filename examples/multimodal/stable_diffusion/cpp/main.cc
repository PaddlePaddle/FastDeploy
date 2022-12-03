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

#include "dpm_solver_multistep_scheduler.h"
#include "fastdeploy/vision/common/processors/mat.h"
#include "fastdeploy/utils/perf.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "pipeline_stable_diffusion_inpaint.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

template <typename T> std::string Str(const T* value, int size) {
  std::ostringstream oss;
  oss << "[ " << value[0];
  for (int i = 1; i < size; ++i) {
    oss << " ," << value[i];
  }
  oss << " ]";
  return oss.str();
}

std::unique_ptr<fastdeploy::Runtime>
CreateRuntime(const std::string& model_file, const std::string& params_file,
              bool use_paddle_backend = true) {
  fastdeploy::RuntimeOption runtime_option;
  runtime_option.SetModelPath(model_file, params_file,
                              fastdeploy::ModelFormat::PADDLE);
  runtime_option.UseGpu();
  if (use_paddle_backend) {
    runtime_option.UsePaddleBackend();
  } else {
    runtime_option.UseOrtBackend();
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
  std::string text_model_file = "sd15_inpaint/text_encoder/inference.pdmodel";
  std::string text_params_file =
      "sd15_inpaint/text_encoder/inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> text_encoder_runtime =
      CreateRuntime(text_model_file, text_params_file, false);

  // 3. Init vae encoder runtime
  std::string vae_encoder_model_file =
      "sd15_inpaint/vae_encoder/inference.pdmodel";
  std::string vae_encoder_params_file =
      "sd15_inpaint/vae_encoder/inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> vae_encoder_runtime =
      CreateRuntime(vae_encoder_model_file, vae_encoder_params_file);

  // 4. Init vae decoder runtime
  std::string vae_decoder_model_file =
      "sd15_inpaint/vae_decoder/inference.pdmodel";
  std::string vae_decoder_params_file =
      "sd15_inpaint/vae_decoder/inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> vae_decoder_runtime =
      CreateRuntime(vae_decoder_model_file, vae_decoder_params_file);

  // 5. Init unet runtime
  std::string unet_model_file = "sd15_inpaint/unet/inference.pdmodel";
  std::string unet_params_file = "sd15_inpaint/unet/inference.pdiparams";
  std::unique_ptr<fastdeploy::Runtime> unet_runtime =
      CreateRuntime(unet_model_file, unet_params_file);

  // 6. Init fast tokenizer
  paddlenlp::fast_tokenizer::tokenizers_impl::ClipFastTokenizer tokenizer(
      "clip/vocab.json", "clip/merges.txt", /* max_length = */ 77);
  fastdeploy::StableDiffusionInpaintPipeline pipe(
      std::move(vae_encoder_runtime), std::move(vae_decoder_runtime),
      std::move(text_encoder_runtime), std::move(unet_runtime),
      /* scheduler = */ std::move(dpm), tokenizer);

  // 7. Read images
  auto image = cv::imread("overture-creations.png");
  auto mask_image = cv::imread("overture-creations-mask.png");

  // 8. Predict
  std::vector<std::string> prompts = {
      "Face of a yellow cat, high resolution, sitting on a park bench"};
  std::vector<fastdeploy::FDTensor> outputs;
  fastdeploy::TimeCounter tc;
  tc.Start();
  pipe.Predict(prompts, image, mask_image, &outputs, /* height = */ 512,
               /* width = */ 512, /* num_inference_steps = */ 50);
  tc.End();
  tc.PrintInfo();
  fastdeploy::vision::FDMat mat = fastdeploy::vision::FDMat::Create(outputs[0]);
  cv::imwrite("cat_on_bench_new.png", *mat.GetOpenCVMat());
  return 0;
}