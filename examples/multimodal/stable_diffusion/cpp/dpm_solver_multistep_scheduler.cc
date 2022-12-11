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
    : config({num_train_timesteps, beta_start, beta_end, beta_schedule,
              solver_order, predict_epsilon, thresholding,
              dynamic_thresholding_ratio, sample_max_value, algorithm_type,
              solver_type, lower_order_final}) {
  int beta_size = trained_betas.size();
  if (beta_size > 0) {
    betas_.Allocate({beta_size}, FDDataType::FP32);
    std::copy(trained_betas.data(), trained_betas.data() + beta_size,
              reinterpret_cast<float*>(betas_.Data()));
  } else if (beta_schedule == "linear") {
    function::Linspace(beta_start, beta_end, num_train_timesteps, &betas_,
                       FDDataType::FP32);
  } else if (beta_schedule == "scaled_linear") {
    function::Linspace(std::sqrt(beta_start), std::sqrt(beta_end),
                       num_train_timesteps, &betas_, FDDataType::FP32);
    betas_ = betas_ * betas_;
  } else if (beta_schedule == "squaredcos_cap_v2") {
    BetaForAlphaBar(&betas_, num_train_timesteps);
  } else {
    FDASSERT(false, "%s is not implemented for DPMSolverMultistepScheduler",
             beta_schedule.c_str());
  }

  alphas_ = 1.0f - betas_;
  function::Cumprod(alphas_, &alphas_cumprod_);
  function::Sqrt(alphas_cumprod_, &alpha_t_);
  function::Sqrt(1.0f - alphas_cumprod_, &sigma_t_);
  FDTensor alpha_t_log, sigma_t_log;
  function::Log(alpha_t_, &alpha_t_log);
  function::Log(sigma_t_, &sigma_t_log);
  lambda_t_ = alpha_t_log - sigma_t_log;

  FDASSERT(config.algorithm_type_ == "dpmsolver" ||
               config.algorithm_type_ == "dpmsolver++",
           "%s does is not implemented for DPMSolverMultistepScheduler",
           config.algorithm_type_.c_str());
  FDASSERT(config.solver_type_ == "midpoint" || config.solver_type_ == "heun",
           "%s does is not implemented for DPMSolverMultistepScheduler",
           config.solver_type_.c_str());
  num_inference_steps_ = -1;

  function::Linspace(0, config.num_train_timesteps_ - 1,
                     config.num_train_timesteps_, &timesteps_);
  function::Cast(timesteps_, &timesteps_, FDDataType::INT64);
  // Reverse timesteps
  int64_t* timesteps_data = reinterpret_cast<int64_t*>(timesteps_.Data());
  std::reverse(timesteps_data, timesteps_data + timesteps_.Numel());

  model_outputs_.resize(config.solver_order_);
  lower_order_nums_ = 0;
}

float DPMSolverMultistepScheduler::InitNoiseSigma() { return 1.0; }

void DPMSolverMultistepScheduler::ConvertModelOutput(
    const FDTensor& model_output, int timestep, const FDTensor& sample,
    FDTensor* out) {
  if (config.algorithm_type_ == "dpmsolver++") {
    FDTensor x0_pred;
    if (config.predict_epsilon_) {
      FDTensor alpha_t, sigma_t;
      function::Slice(alpha_t_, {0}, {timestep}, &alpha_t);
      function::Slice(sigma_t_, {0}, {timestep}, &sigma_t);
      x0_pred = (sample - sigma_t * model_output) / alpha_t;
    } else {
      x0_pred = model_output;
    }
    if (config.thresholding_) {
      FDTensor dynamic_max_val, x0_pred_abs;
      function::Abs(x0_pred, &x0_pred_abs);
      x0_pred_abs.Reshape({x0_pred_abs.Shape()[0], -1});
      function::Quantile(x0_pred_abs, {config.dynamic_thresholding_ratio_}, {1},
                         &dynamic_max_val);

      FDTensor max_value, dy_max_val;
      function::FullLike(dynamic_max_val, config.sample_max_value_, &max_value,
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
  } else if (config.algorithm_type_ == "dpmsolver") {
    if (config.predict_epsilon_) {
      *out = model_output;
    } else {
      FDTensor alpha_t, sigma_t;
      function::Slice(alpha_t_, {0}, {timestep}, &alpha_t);
      function::Slice(sigma_t_, {0}, {timestep}, &sigma_t);
      *out = (sample - (alpha_t * model_output)) / sigma_t;
    }
  }
}

void DPMSolverMultistepScheduler::DPMSolverFirstOrderUpdate(
    const FDTensor& model_output, int timestep, int prev_timestep,
    const FDTensor& sample, FDTensor* out) {
  FDTensor lambda_t, lambda_s;
  function::Slice(lambda_t_, {0}, {prev_timestep}, &lambda_t);
  function::Slice(lambda_t_, {0}, {timestep}, &lambda_s);

  FDTensor alpha_t, alpha_s;
  function::Slice(alpha_t_, {0}, {prev_timestep}, &alpha_t);
  function::Slice(alpha_t_, {0}, {timestep}, &alpha_s);

  FDTensor sigma_t, sigma_s;
  function::Slice(sigma_t_, {0}, {prev_timestep}, &sigma_t);
  function::Slice(sigma_t_, {0}, {timestep}, &sigma_s);

  FDTensor h = lambda_t - lambda_s;
  if (config.algorithm_type_ == "dpmsolver++") {
    function::Exp(0.0f - h, &h);
    *out = (sigma_t / sigma_s) * sample - (alpha_t * (h - 1.0f)) * model_output;
  } else if (config.algorithm_type_ == "dpmsolver") {
    function::Exp(h, &h);
    *out = (alpha_t / alpha_s) * sample - (sigma_t * (h - 1.0f)) * model_output;
  }
}

void DPMSolverMultistepScheduler::MultiStepDPMSolverSecondOrderUpdate(
    const std::vector<FDTensor>& model_output_list,
    const std::vector<int>& timestep_list, int prev_timestep,
    const FDTensor& sample, FDTensor* out) {
  int timestep_size = timestep_list.size();
  int model_output_size = model_output_list.size();
  int t = prev_timestep;
  int s0 = timestep_list[timestep_size - 1];
  int s1 = timestep_list[timestep_size - 2];
  const FDTensor& m0 = model_output_list[model_output_size - 1];
  const FDTensor& m1 = model_output_list[model_output_size - 2];
  FDTensor lambda_t, lambda_s0, lambda_s1;
  function::Slice(lambda_t_, {0}, {t}, &lambda_t);
  function::Slice(lambda_t_, {0}, {s0}, &lambda_s0);
  function::Slice(lambda_t_, {0}, {s1}, &lambda_s1);

  FDTensor alpha_t, alpha_s0, sigma_t, sigma_s0;
  function::Slice(alpha_t_, {0}, {t}, &alpha_t);
  function::Slice(alpha_t_, {0}, {s0}, &alpha_s0);
  function::Slice(sigma_t_, {0}, {t}, &sigma_t);
  function::Slice(sigma_t_, {0}, {s0}, &sigma_s0);

  FDTensor h = lambda_t - lambda_s0;
  FDTensor h0 = lambda_s0 - lambda_s1;
  FDTensor r0 = h0 / h;
  FDTensor D0 = m0;
  FDTensor D1 = (1.0f / r0) * (m0 - m1);
  if (config.algorithm_type_ == "dpmsolver++") {
    if (config.solver_type_ == "midpoint") {
      function::Exp(0.0f - h, &h);
      *out = (sigma_t / sigma_s0 * sample) - (alpha_t * (h - 1.0f) * D0) -
             (0.5f * alpha_t * (h - 1.0f) * D1);
    } else if (config.solver_type_ == "heun") {
      FDTensor h_exp;
      function::Exp(0.0f - h, &h_exp);
      *out = (sigma_t / sigma_s0 * sample) - (alpha_t * (h_exp - 1.0f) * D0) +
             (alpha_t * ((h_exp - 1.0f) / h + 1.0f) * D1);
    }
  } else if (config.algorithm_type_ == "dpmsolver") {
    FDTensor h_exp;
    function::Exp(h, &h_exp);
    if (config.solver_type_ == "midpoint") {
      *out = alpha_t / alpha_s0 * sample - sigma_t * (h_exp - 1.0f) * D0 -
             0.5 * (sigma_t * (h_exp - 1.0f) * D1);
    } else if (config.solver_type_ == "heun") {
      *out = alpha_t / alpha_s0 * sample - sigma_t * (h_exp - 1.0f) * D0 -
             (sigma_t * ((h_exp - 1.0f) / h - 1.0f) * D1);
    }
  }
}

void DPMSolverMultistepScheduler::MultiStepDPMSolverThirdOrderUpdate(
    const std::vector<FDTensor>& model_output_list,
    const std::vector<int>& timestep_list, int prev_timestep,
    const FDTensor& sample, FDTensor* out) {
  int timestep_size = timestep_list.size();
  int model_output_size = model_output_list.size();
  int t = prev_timestep;

  int s0 = timestep_list[timestep_size - 1];
  int s1 = timestep_list[timestep_size - 2];
  int s2 = timestep_list[timestep_size - 3];
  const FDTensor& m0 = model_output_list[model_output_size - 1];
  const FDTensor& m1 = model_output_list[model_output_size - 2];
  const FDTensor& m2 = model_output_list[model_output_size - 3];

  FDTensor lambda_t, lambda_s0, lambda_s1, lambda_s2;
  function::Slice(lambda_t_, {0}, {t}, &lambda_t);
  function::Slice(lambda_t_, {0}, {s0}, &lambda_s0);
  function::Slice(lambda_t_, {0}, {s1}, &lambda_s1);
  function::Slice(lambda_t_, {0}, {s2}, &lambda_s2);

  FDTensor alpha_t, alpha_s0, sigma_t, sigma_s0;
  function::Slice(alpha_t_, {0}, {t}, &alpha_t);
  function::Slice(alpha_t_, {0}, {s0}, &alpha_s0);
  function::Slice(sigma_t_, {0}, {t}, &sigma_t);
  function::Slice(sigma_t_, {0}, {s0}, &sigma_s0);

  FDTensor h = lambda_t - lambda_s0;
  FDTensor h0 = lambda_s0 - lambda_s1;
  FDTensor h1 = lambda_s1 - lambda_s2;

  FDTensor r0 = h0 / h;
  FDTensor r1 = h1 / h;
  FDTensor D0 = m0;
  FDTensor D1_0 = (1.0f / r0) * (m0 - m1);
  FDTensor D1_1 = (1.0f / r1) * (m1 - m2);
  FDTensor D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1);
  FDTensor D2 = (1.0f / (r0 + r1)) * (D1_0 - D1_1);

  if (config.algorithm_type_ == "dpmsolver++") {
    FDTensor h_exp;
    function::Exp(0.0f - h, &h_exp);
    *out = (sigma_t / sigma_s0) * sample - (alpha_t * (h_exp - 1.0f)) * D0 +
           (alpha_t * ((h_exp - 1.0) / h + 1.0)) * D1 -
           (alpha_t * ((h_exp - 1.0 + h) / (h * h) - 0.5)) * D2;

  } else if (config.algorithm_type_ == "dpmsolver") {
    FDTensor h_exp;
    function::Exp(h, &h_exp);
    *out = (alpha_t / alpha_s0) * sample - (sigma_t * (h_exp - 1.0f)) * D0 +
           (sigma_t * ((h_exp - 1.0) / h - 1.0)) * D1 -
           (sigma_t * ((h_exp - 1.0 - h) / (h * h) - 0.5)) * D2;
  }
}

void DPMSolverMultistepScheduler::ScaleModelInput(
    const FDTensor& sample, FDTensor* out,
    const std::vector<FDTensor>& timesteps) {
  *out = sample;
}

void DPMSolverMultistepScheduler::SetTimesteps(int num_inference_steps) {
  num_inference_steps_ = num_inference_steps;
  function::Linspace(0, config.num_train_timesteps_ - 1,
                     num_inference_steps + 1, &timesteps_);
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
  model_outputs_.resize(config.solver_order_);

  lower_order_nums_ = 0;
}

void DPMSolverMultistepScheduler::Step(const FDTensor& model_output,
                                       int timestep, const FDTensor& sample,
                                       FDTensor* prev_sample) {
  FDASSERT(num_inference_steps_ > -1,
           "Number of inference steps is -1, you need to run SetTimesteps "
           "after creating the scheduler");
  int64_t step_index = timesteps_.Numel() - 1;
  int64_t* timesteps_data = reinterpret_cast<int64_t*>(timesteps_.Data());
  int64_t* timesteps_iter =
      std::find(timesteps_data, timesteps_data + timesteps_.Numel(), timestep);
  if (timesteps_iter - timesteps_data < timesteps_.Numel()) {
    step_index = timesteps_iter - timesteps_data;
  }
  int64_t prev_timestep = 0;
  if (step_index != timesteps_.Numel() - 1) {
    prev_timestep = timesteps_data[step_index + 1];
  }
  bool lower_order_final = (step_index == timesteps_.Numel() - 1) &&
                           config.lower_order_final_ &&
                           (timesteps_.Numel() < 15);
  bool lower_order_second = (step_index == timesteps_.Numel() - 2) &&
                            config.lower_order_final_ &&
                            (timesteps_.Numel() < 15);
  FDTensor model_out;
  ConvertModelOutput(model_output, timestep, sample, &model_out);
  for (int i = 0; i < config.solver_order_ - 1; ++i) {
    model_outputs_[i] = std::move(model_outputs_[i + 1]);
  }
  model_outputs_[config.solver_order_ - 1] = std::move(model_out);

  if (config.solver_order_ == 1 || lower_order_nums_ < 1 || lower_order_final) {
    DPMSolverFirstOrderUpdate(model_outputs_[config.solver_order_ - 1],
                              timestep, prev_timestep, sample, prev_sample);
  } else if (config.solver_order_ == 2 || lower_order_nums_ < 2 ||
             lower_order_second) {
    int t0 = reinterpret_cast<int64_t*>(timesteps_.Data())[step_index - 1];
    std::vector<int> timestep_list = {t0, timestep};
    MultiStepDPMSolverSecondOrderUpdate(model_outputs_, timestep_list,
                                        prev_timestep, sample, prev_sample);
  } else {
    int t0 = reinterpret_cast<int64_t*>(timesteps_.Data())[step_index - 1];
    int t1 = reinterpret_cast<int64_t*>(timesteps_.Data())[step_index - 2];
    std::vector<int> timestep_list = {t1, t0, timestep};
    MultiStepDPMSolverThirdOrderUpdate(model_outputs_, timestep_list,
                                       prev_timestep, sample, prev_sample);
  }

  if (lower_order_nums_ < config.solver_order_) {
    lower_order_nums_ += 1;
  }
}

void DPMSolverMultistepScheduler::AddNoise(const FDTensor& original_samples,
                                           const FDTensor& noise,
                                           const FDTensor& timesteps,
                                           FDTensor* out) {
  function::Cast(alphas_cumprod_, &alphas_cumprod_, original_samples.Dtype());

  const int64_t* timesteps_data =
      reinterpret_cast<const int64_t*>(timesteps.Data());
  std::vector<int64_t> timesteps_vec;
  for (int i = 0; i < timesteps.Numel(); ++i) {
    timesteps_vec.push_back(timesteps_data[i]);
  }
  FDTensor sqrt_alpha_prod;
  function::Slice(alphas_cumprod_, {0}, timesteps_vec, &sqrt_alpha_prod);
  function::Sqrt(sqrt_alpha_prod, &sqrt_alpha_prod);
  sqrt_alpha_prod.Reshape({-1});
  int rank_diff =
      original_samples.Shape().size() - sqrt_alpha_prod.Shape().size();
  for (int i = 0; i < rank_diff; ++i) {
    int curr_rank = sqrt_alpha_prod.Shape().size();
    sqrt_alpha_prod.ExpandDim(curr_rank - 1);
  }

  FDTensor sqrt_one_minus_alpha_prod;
  function::Slice(alphas_cumprod_, {0}, timesteps_vec,
                  &sqrt_one_minus_alpha_prod);
  sqrt_one_minus_alpha_prod = 1.0f - sqrt_one_minus_alpha_prod;
  function::Sqrt(sqrt_one_minus_alpha_prod, &sqrt_one_minus_alpha_prod);
  sqrt_one_minus_alpha_prod.Reshape({-1});
  rank_diff = original_samples.Shape().size() -
              sqrt_one_minus_alpha_prod.Shape().size();
  for (int i = 0; i < rank_diff; ++i) {
    int curr_rank = sqrt_one_minus_alpha_prod.Shape().size();
    sqrt_one_minus_alpha_prod.ExpandDim(curr_rank - 1);
  }
  *out = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise;
}

FDTensor DPMSolverMultistepScheduler::GetTimesteps() { return timesteps_; }

}  // namespace fastdeploy
