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
//
// Part of the following code in this file refs to
// https://github.com/gatagat/lap/blob/master/lap/lapjv.h
//
// Copyright (c) 2012-2017, Tomas Kazmar
// Licensed under The BSD 2-Clause "Simplified" License

#ifndef STREAMER_SRC_GSTREAMER_PLUGIN_FDTRACKER_INCLUDE_LAPJV_H_
#define STREAMER_SRC_GSTREAMER_PLUGIN_FDTRACKER_INCLUDE_LAPJV_H_
#define LARGE 1000000

#if !defined TRUE
#define TRUE 1
#endif
#if !defined FALSE
#define FALSE 0
#endif

#define NEW(x, t, n)                                               \
  if ((x = reinterpret_cast<t *>(malloc(sizeof(t) * (n)))) == 0) { \
    return -1;                                                     \
  }
#define FREE(x) \
  if (x != 0) { \
    free(x);    \
    x = 0;      \
  }
#define SWAP_INDICES(a, b) \
  {                        \
    int_t _temp_index = a; \
    a = b;                 \
    b = _temp_index;       \
  }
#include <opencv2/opencv.hpp>

typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;
typedef char boolean;
typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;

int lapjv_internal(const cv::Mat &cost,
                   const bool extend_cost,
                   const float cost_limit,
                   int *x,
                   int *y);


#endif  // STREAMER_SRC_GSTREAMER_PLUGIN_FDTRACKER_INCLUDE_LAPJV_H_
