/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

// Note: The shapes are in the format MxNxK. The K shape of the runtime config
enum class CutlassTileConfig {
  // Signals that we should run heuristics do choose a config
  Undefined, // 0

  // Signals that we should run heuristics do choose a config
  ChooseWithHeuristic, // 1

  // SiMT config
  CtaShape128x128x8_WarpShape64x64x8, // 2

  // TensorCore configs CTA_N = 128, CTA_K = 64
  // Warp configs for M=16
  CtaShape16x128x64_WarpShape16x32x64, // 3
  CtaShape16x256x64_WarpShape16x64x64, // 4

  // Warp configs for M=32
  CtaShape32x128x64_WarpShape32x32x64, // 5

  // Warp configs for M=64
  CtaShape64x128x64_WarpShape32x64x64, // 6
  CtaShape64x128x64_WarpShape64x32x64, // 7

  // Warp configs for M=128
  CtaShape128x128x64_WarpShape64x32x64, // 8
  CtaShape128x128x64_WarpShape128x32x64, // 9

  // configs for large M in encoder
  CtaShape128x256x64_WarpShape64x64x64, // 10
  CtaShape256x128x64_WarpShape64x64x64, // 11

  CtaShape32x256x64_WarpShape32x64x64,  // 12
  CtaShape64x256x64_WarpShape64x64x64, // 13
  CtaShape128x256x64_WarpShape128x64x64, // 14
  CtaShape32x512x64_WarpShape32x128x64, // 15
};

enum class SplitKStyle {
  NO_SPLIT_K, //0
  SPLIT_K_SERIAL, //1
  SPLIT_K_STREAM, //2
  // SPLIT_K_PARALLEL // Not supported yet
};

struct CutlassGemmConfig {
  CutlassTileConfig tile_config = CutlassTileConfig::ChooseWithHeuristic;
  SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
  int split_k_factor = -1;
  int stages = -1;
};

struct TileShape {
  int m;
  int n;
};
