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
#include "fastdeploy/core/fd_tensor.h"

namespace fastdeploy {

class DPMSolverMultistepScheduler : public Scheduler {
 public:
  DPMSolverMultistepScheduler(int num_train_timesteps = 1000,
                              float beta_start = 0.0001, float beta_end = 0.02,
                              const std::string& beta_schedule = "linear",
                              const std::vector<float>& trained_betas = {},
                              int solver_order = 2, bool predict_epsilon = true,
                              bool thresholding = false,
                              float dynamic_thresholding_ratio = 0.995,
                              float sample_max_value = 1.0,
                              const std::string& algorithm_type = "dpmsolver++",
                              const std::string& solver_type = "midpoint",
                              bool lower_order_final = true);
  void BetaForAlphaBar(FDTensor* out, int num_diffusion_timesteps,
                       float max_beta = 0.999);
  void ConvertModelOutput(const FDTensor& model_output, int timestep,
                          const FDTensor& sample, FDTensor* out);
  void DPMSolverFirstOrderUpdate(const FDTensor& model_output, int timestep,
                                 int prev_timestep, const FDTensor& sample,
                                 FDTensor* out);
  void MultiStepDPMSolverSecondOrderUpdate(
      const std::vector<FDTensor>& model_output_list,
      const std::vector<int>& timestep_list, int prev_timestep,
      const FDTensor& sample, FDTensor* out);
  void MultiStepDPMSolverThirdOrderUpdate(
      const std::vector<FDTensor>& model_output_list,
      const std::vector<int>& timestep_list, int prev_timestep,
      const FDTensor& sample, FDTensor* out);
  void SetTimesteps(int num_inference_steps) override;
  void Step(const FDTensor& model_output, int timestep, const FDTensor& sample,
            FDTensor* prev_sample) override;
  void ScaleModelInput(const FDTensor& sample, FDTensor* out,
                       const std::vector<FDTensor>& timesteps = {}) override;
  void AddNoise(const FDTensor& original_samples, const FDTensor& noise,
                const FDTensor& timesteps, FDTensor* out) override;
  float InitNoiseSigma() override;
  FDTensor GetTimesteps() override;
  struct Config {
    int num_train_timesteps_;
    float beta_start_;
    float beta_end_;
    std::string beta_schedule_;
    int solver_order_;
    bool predict_epsilon_;
    bool thresholding_;
    float dynamic_thresholding_ratio_;
    float sample_max_value_;
    std::string algorithm_type_;
    std::string solver_type_;
    bool lower_order_final_;
  } config;

 private:
  FDTensor betas_;
  FDTensor alphas_;
  FDTensor alphas_cumprod_;
  FDTensor alpha_t_;
  FDTensor sigma_t_;
  FDTensor lambda_t_;
  int num_inference_steps_;
  FDTensor timesteps_;
  int lower_order_nums_;
  std::vector<FDTensor> model_outputs_;
};

}  // namespace fastdeploy
