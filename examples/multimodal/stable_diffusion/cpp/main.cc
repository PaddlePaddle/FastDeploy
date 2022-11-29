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
#include <iostream>

int main() {
  fastdeploy::DPMSolverMultistepScheduler dpm(
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
      /* lower_order_final = */ true);

  return 0;
}