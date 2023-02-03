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
#include "fastdeploy/core/config.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/benchmark/option.h"
#include "fastdeploy/benchmark/results.h"
#include <map>

#ifdef ENABLE_BENCHMARK
  #define PROFILE_LOOP_INFO                                             \
    FDINFO << "__p_enable_profile:" << __p_enable_profile << ", "       \
           << "__p_include_h2d_d2h:" << __p_include_h2d_d2h << ", "     \
           << "__p_repeats:" << __p_repeats << ", "                     \
           << "__p_warmup:" << __p_warmup << ", "                       \
           << "__p_loop:" << __p_loop << ""                             \
           << std::endl;                  

  #define PROFILE_LOOP_BEGIN(option)                                    \
    int __p_loop = 1;                                                   \
    const bool __p_enable_profile = option.enable_profile;              \
    const bool __p_include_h2d_d2h = option.include_h2d_d2h;            \
    const int __p_repeats = option.repeats;                             \
    const int __p_warmup = option.warmup;                               \
    if ((__p_enable_profile && (!__p_include_h2d_d2h))) {               \
      __p_loop = (__p_repeats) + (__p_warmup);                          \
    }                                                                   \
    PROFILE_LOOP_INFO                                                   \
    TimeCounter __p_tc;                                                 \
    bool __p_tc_start = false;                                          \
    for (int __p_i = 0; __p_i < __p_loop; ++__p_i) {                    \
      if (__p_i >= (__p_warmup) && (!__p_tc_start)) {                   \
        __p_tc.Start();                                                 \
        __p_tc_start = true;                                            \
      }                                                                 \

  #define PROFILE_LOOP_END(result)                                      \
    }                                                                   \
    if ((__p_enable_profile && (!__p_include_h2d_d2h))) {               \
      if (__p_tc_start) {                                               \
        __p_tc.End();                                                   \
        double __p_tc_duration = __p_tc.Duration();                     \
        result.time_of_runtime =                                        \
          __p_tc_duration / static_cast<double>(__p_repeats);           \
      }                                                                 \
    }

  #define PROFILE_LOOP_H2D_D2H_INFO                                     \
    FDINFO << "__p_enable_profile_h:" << __p_enable_profile_h << ", "   \
           << "__p_include_h2d_d2h_h:" << __p_include_h2d_d2h_h << ", " \
           << "__p_repeats_h:" << __p_repeats_h << ", "                 \
           << "__p_warmup_h:" << __p_warmup_h << ", "                   \
           << "__p_loop_h:" << __p_loop_h << ""                         \
           << std::endl;                                                

  #define PROFILE_LOOP_H2D_D2H_BEGIN(option)                            \
    int __p_loop_h = 1;                                                 \
    const bool __p_enable_profile_h = option.enable_profile;            \
    const bool __p_include_h2d_d2h_h = option.include_h2d_d2h;          \
    const int __p_repeats_h = option.repeats;                           \
    const int __p_warmup_h = option.warmup;                             \
    if ((__p_enable_profile_h && __p_include_h2d_d2h_h)) {              \
      __p_loop_h = (__p_repeats_h) + (__p_warmup_h);                    \
    }                                                                   \
    PROFILE_LOOP_H2D_D2H_INFO                                           \
    TimeCounter __p_tc_h;                                               \
    bool __p_tc_start_h = false;                                        \
    for (int __p_i_h = 0; __p_i_h < __p_loop_h; ++__p_i_h) {            \
      if (__p_i_h >= (__p_warmup_h) && (!__p_tc_start_h)) {             \
        __p_tc_h.Start();                                               \
        __p_tc_start_h = true;                                          \
      }                                                                 \

  #define PROFILE_LOOP_H2D_D2H_END(result)                              \
    }                                                                   \
    if ((__p_enable_profile_h && __p_include_h2d_d2h_h)) {              \
      if (__p_tc_start_h) {                                             \
         __p_tc_h.End();                                                \
        double __p_tc_duration_h = __p_tc_h.Duration();                 \
        result.time_of_runtime =                                        \
          __p_tc_duration_h / static_cast<double>(__p_repeats_h);       \
      }                                                                 \
    }  
#else
  #define PROFILE_LOOP_BEGIN(option) {}
  #define PROFILE_LOOP_END(result) {}
  #define PROFILE_LOOP_H2D_D2H_BEGIN(option) {}
  #define PROFILE_LOOP_H2D_D2H_END(result) {}
#endif