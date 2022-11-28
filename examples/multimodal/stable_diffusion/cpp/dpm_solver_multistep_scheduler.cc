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
#include "fastdeploy/core/fd_scalar.h"
#include "fastdeploy/function/functions.h"
#include <algorithm>
#include <cmath>

namespace fastdeploy {

void DPMSolverMultistepScheduler::BetaForAlphaBar(FDTensor* out,
                                                  int num_diffusion_timesteps,
                                                  float max_beta) {
  auto alpha_bar = [](float time_step) -> float {
    constexpr float pi = 3.14159265358979323846;
    return std::pow(std::cos((time_step + 0.008) / 1.008 * pi / 2), 2);
  };
  std::vector<FDTensor> betas;
  for (int i = 0; i < num_diffusion_timesteps; ++i) {
    float t1 = i / num_diffusion_timesteps;
    float t2 = (i + 1) / num_diffusion_timesteps;
    float beta_val = (std::min)(1 - alpha_bar(t1) / alpha_bar(t2), max_beta);
    betas.emplace_back(Scalar(beta_val));
  }
  function::Concat(betas, out);
}

DPMSolverMultistepScheduler::DPMSolverMultistepScheduler(
    int num_train_timesteps, float beta_start, float beta_end,
    const std::string& beta_schedule, const std::vector<float>& trained_betas,
    int solver_order, bool predict_epsilon, bool thresholding,
    float dynamic_thresholding_ratio, float sample_max_value,
    const std::string& algorithm_type, const std::string& solver_type,
    bool lower_order_final)
    : num_train_timesteps_(num_train_timesteps), beta_start_(beta_start),
      beta_end_(beta_end), beta_schedule_(beta_schedule),
      solver_order_(solver_order), predict_epsilon_(predict_epsilon),
      thresholding_(thresholding),
      dynamic_thresholding_ratio_(dynamic_thresholding_ratio),
      sample_max_value_(sample_max_value), algorithm_type_(algorithm_type),
      solver_type_(solver_type), lower_order_final_(lower_order_final) {
  int beta_size = trained_betas.size();
  if (beta_size > 0) {
    betas_.Allocate({beta_size}, FDDataType::FP32);
    std::copy(trained_betas.data(), trained_betas.data() + beta_size,
              reinterpret_cast<float*>(betas_.Data()));
  } else if (beta_schedule == "linear") {
    function::Linspace(beta_start, beta_end, num_train_timesteps, &betas_,
                       FDDataType::FP32);
  } else if (beta_schedule == "scaled_linear") {
    function::Linspace(beta_start, beta_end, num_train_timesteps, &betas_,
                       FDDataType::FP32);
    betas_ = betas_ * betas_;
  } else if (beta_schedule == "squaredcos_cap_v2") {
    BetaForAlphaBar(&betas_, num_train_timesteps);
  } else {
    FDASSERT(false, "%s is not implemented for DPMSolverMultistepScheduler",
             beta_schedule.c_str());
  }

  Scalar one = static_cast<float>(1.0);
  alphas_ = FDTensor(one) - betas_;
  function::Cumprod(alphas_, &alphas_cumprod_);
  function::Sqrt(alphas_cumprod_, &alpha_t_);
  function::Sqrt(FDTensor(one) - alphas_cumprod_, &sigma_t_);
  FDTensor alpha_t_log, sigma_t_log;
  function::Log(alpha_t_, &alpha_t_log);
  function::Log(sigma_t_, &sigma_t_log);
  lambda_t_ = alpha_t_log - sigma_t_log;

  FDASSERT(algorithm_type_ == "dpmsolver" || algorithm_type_ == "dpmsolver++",
           "%s does is not implemented for DPMSolverMultistepScheduler",
           algorithm_type_.c_str());
  FDASSERT(solver_type_ == "midpoint" || solver_type_ == "heun",
           "%s does is not implemented for DPMSolverMultistepScheduler",
           solver_type_.c_str());
  num_inference_steps_ = -1;

  function::Linspace(0, num_train_timesteps_ - 1, num_train_timesteps_,
                     &timesteps_);
  // Reverse timesteps
  float* timesteps_data = reinterpret_cast<float*>(timesteps_.Data());
  std::reverse(timesteps_data, timesteps_data + timesteps_.Numel());

  model_outputs_.resize(solver_order_);
  lower_order_nums_ = 0;
}

void DPMSolverMultistepScheduler::ConvertModelOutput(
    const FDTensor& model_output, int timestep, const FDTensor& sample,
    FDTensor* out) {
  if (algorithm_type_ == "dpmsolver++") {
    FDTensor x0_pred;
    if (predict_epsilon_) {
      FDTensor alpha_t, sigma_t;
      function::Slice(alpha_t_, {0}, {timestep}, {timestep + 1}, &alpha_t);
      function::Slice(sigma_t_, {0}, {timestep}, {timestep + 1}, &sigma_t);
      alpha_t.Squeeze();
      sigma_t_.Squeeze();
      x0_pred = (sample - sigma_t * model_output) / alpha_t;
    } else {
      x0_pred = model_output;
    }
    if (thresholding_) {
      FDTensor dynamic_max_val, x0_pred_abs;
      function::Abs(x0_pred, &x0_pred_abs);
      x0_pred_abs.Reshape({x0_pred_abs.Shape()[0], -1});
      function::Quantile(x0_pred_abs, {dynamic_thresholding_ratio_}, {1},
                         &dynamic_max_val);

      FDTensor max_value, dy_max_val;
      function::FullLike(dynamic_max_val, sample_max_value_, &max_value,
                         dynamic_max_val.Dtype());
      function::Maximum(dynamic_max_val, max_value, &dy_max_val);
      int expand_dims = x0_pred.Shape().size() - 1;
      for (int i = 0; i < expand_dims; ++i) {
        dy_max_val.ExpandDim(dy_max_val.Shape().size());
      }
      float clip_max = reinterpret_cast<float*>(dy_max_val.Data())[0];
      function::Clip(x0_pred, -clip_max, clip_max, &x0_pred);
      x0_pred = x0_pred / dy_max_val;
    }
    *out = std::move(x0_pred);
  } else if (algorithm_type_ == "dpmsolver") {
    if (predict_epsilon_) {
      *out = model_output;
    } else {
      FDTensor alpha_t, sigma_t;
      function::Slice(alpha_t_, {0}, {timestep}, {timestep + 1}, &alpha_t);
      function::Slice(sigma_t_, {0}, {timestep}, {timestep + 1}, &sigma_t);
      alpha_t.Squeeze();
      sigma_t_.Squeeze();
      *out = (sample - alpha_t * model_output) / sigma_t;
    }
  }
}

void DPMSolverMultistepScheduler::SetTimesteps(int num_inference_steps) {
  num_inference_steps_ = num_inference_steps;
  function::Linspace(0, num_train_timesteps_ - 1, num_inference_steps + 1,
                     &timesteps_);
  function::Round(timesteps_, &timesteps_);
  // Reverse timesteps
  float* timesteps_data = reinterpret_cast<float*>(timesteps_.Data());
  std::reverse(timesteps_data, timesteps_data + timesteps_.Numel());
  FDTensor timestep_tmp;
  timestep_tmp.Allocate({num_inference_steps}, timesteps_.Dtype());
  float* timestep_tmp_data = reinterpret_cast<float*>(timestep_tmp.Data());
  std::copy(timesteps_data, timesteps_data + num_inference_steps,
            timestep_tmp_data);
  timesteps_ = std::move(timestep_tmp);

  function::Cast(timesteps_, &timesteps_, FDDataType::INT64);

  model_outputs_.clear();
  model_outputs_.resize(solver_order_);

  lower_order_nums_ = 0;
}

void DPMSolverMultistepScheduler::Step(const FDTensor& model_output,
                                       int timestep, const FDTensor& sample,
                                       FDTensor* prev_sample) {}

}  // namespace fastdeploy
