// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "helper.h"
#include "moe_fast_hardamard_impl_common.h"

template <typename T, int VecSize>
__device__ __forceinline__ void hadamard_mult_thread_28_transpose(
    T x[28][VecSize]) {  // 35
  T out[28];
#pragma unroll
  for (int vi = 0; vi < VecSize; vi++) {
    out[0] = +x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] +
             x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] + x[11][vi] +
             x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] + x[16][vi] +
             x[17][vi] + x[18][vi] + x[19][vi] + x[20][vi] + x[21][vi] +
             x[22][vi] + x[23][vi] + x[24][vi] + x[25][vi] + x[26][vi] +
             x[27][vi];
    out[1] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
             x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] + x[10][vi] + x[11][vi] -
             x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] + x[16][vi] -
             x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] - x[21][vi] -
             x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] - x[26][vi] +
             x[27][vi];
    out[2] = +x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] +
             x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] + x[11][vi] +
             x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] - x[16][vi] +
             x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] - x[21][vi] -
             x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] + x[26][vi] -
             x[27][vi];
    out[3] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] +
             x[6][vi] + x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] - x[11][vi] +
             x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] + x[16][vi] -
             x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] + x[21][vi] -
             x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] + x[26][vi] +
             x[27][vi];
    out[4] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] -
             x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] - x[10][vi] - x[11][vi] -
             x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] - x[16][vi] +
             x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] + x[21][vi] +
             x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] - x[26][vi] +
             x[27][vi];
    out[5] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] +
             x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] - x[11][vi] -
             x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] + x[16][vi] -
             x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] - x[21][vi] +
             x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] - x[26][vi] -
             x[27][vi];
    out[6] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] +
             x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] - x[11][vi] -
             x[12][vi] - x[13][vi] + x[14][vi] - x[15][vi] + x[16][vi] +
             x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] + x[21][vi] -
             x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] - x[26][vi] -
             x[27][vi];
    out[7] = +x[0][vi] - x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] +
             x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] + x[11][vi] -
             x[12][vi] - x[13][vi] + x[14][vi] - x[15][vi] - x[16][vi] +
             x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] - x[21][vi] +
             x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] - x[26][vi] -
             x[27][vi];
    out[8] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
             x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] + x[11][vi] +
             x[12][vi] - x[13][vi] + x[14][vi] - x[15][vi] - x[16][vi] -
             x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] + x[21][vi] -
             x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] + x[26][vi] -
             x[27][vi];
    out[9] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] + x[5][vi] +
             x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] - x[11][vi] +
             x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] - x[16][vi] -
             x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] - x[21][vi] +
             x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] + x[26][vi] +
             x[27][vi];
    out[10] = +x[0][vi] + x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] +
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] -
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] +
              x[21][vi] - x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] -
              x[26][vi] + x[27][vi];
    out[11] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] +
              x[11][vi] + x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] +
              x[21][vi] + x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] +
              x[26][vi] - x[27][vi];
    out[12] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] + x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] +
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] + x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] -
              x[26][vi] + x[27][vi];
    out[13] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] -
              x[16][vi] + x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] +
              x[26][vi] - x[27][vi];
    out[14] = -x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] +
              x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] +
              x[11][vi] + x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] -
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] -
              x[26][vi] - x[27][vi];
    out[15] = +x[0][vi] - x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
              x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] + x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] -
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] +
              x[21][vi] + x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] +
              x[26][vi] - x[27][vi];
    out[16] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] +
              x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] +
              x[11][vi] + x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] -
              x[16][vi] - x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] +
              x[21][vi] + x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] -
              x[26][vi] + x[27][vi];
    out[17] = +x[0][vi] - x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] - x[5][vi] +
              x[6][vi] + x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] -
              x[11][vi] + x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] -
              x[16][vi] - x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] -
              x[21][vi] + x[22][vi] + x[23][vi] + x[24][vi] + x[25][vi] -
              x[26][vi] - x[27][vi];
    out[18] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] -
              x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] - x[10][vi] -
              x[11][vi] - x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] -
              x[21][vi] - x[22][vi] + x[23][vi] + x[24][vi] + x[25][vi] +
              x[26][vi] - x[27][vi];
    out[19] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] - x[5][vi] +
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] -
              x[11][vi] - x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] -
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] +
              x[21][vi] - x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] +
              x[26][vi] + x[27][vi];
    out[20] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] -
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] -
              x[11][vi] - x[12][vi] - x[13][vi] - x[14][vi] + x[15][vi] -
              x[16][vi] - x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] + x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] +
              x[26][vi] + x[27][vi];
    out[21] = +x[0][vi] - x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] +
              x[6][vi] - x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] +
              x[11][vi] - x[12][vi] - x[13][vi] - x[14][vi] + x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] +
              x[26][vi] + x[27][vi];
    out[22] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] + x[12][vi] - x[13][vi] - x[14][vi] + x[15][vi] +
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] -
              x[26][vi] + x[27][vi];
    out[23] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] + x[5][vi] +
              x[6][vi] - x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] +
              x[16][vi] + x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] +
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] -
              x[26][vi] - x[27][vi];
    out[24] = +x[0][vi] + x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] +
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] +
              x[16][vi] + x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] -
              x[21][vi] + x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] +
              x[26][vi] - x[27][vi];
    out[25] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] -
              x[16][vi] + x[17][vi] + x[18][vi] + x[19][vi] + x[20][vi] -
              x[21][vi] - x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] -
              x[26][vi] + x[27][vi];
    out[26] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] -
              x[16][vi] - x[17][vi] + x[18][vi] + x[19][vi] + x[20][vi] +
              x[21][vi] - x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] -
              x[26][vi] - x[27][vi];
    out[27] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] +
              x[21][vi] + x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] -
              x[26][vi] - x[27][vi];
#pragma unroll
    for (int i = 0; i < 28; i++) {
      x[i][vi] = out[i];
    }
  }
}

template <typename T, int VecSize>
__device__ __forceinline__ void hadamard_mult_thread_36_transpose(
    T x[36][VecSize]) {  // 4t
  T out[36];
#pragma unroll
  for (int vi = 0; vi < VecSize; vi++) {
    out[0] = +x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] +
             x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] + x[11][vi] +
             x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] + x[16][vi] +
             x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] + x[21][vi] +
             x[22][vi] + x[23][vi] + x[24][vi] + x[25][vi] + x[26][vi] +
             x[27][vi] + x[28][vi] + x[29][vi] + x[30][vi] + x[31][vi] +
             x[32][vi] + x[33][vi] + x[34][vi] + x[35][vi];
    out[1] = +x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] -
             x[6][vi] - x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] - x[11][vi] -
             x[12][vi] - x[13][vi] + x[14][vi] - x[15][vi] + x[16][vi] +
             x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] + x[21][vi] -
             x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] - x[26][vi] +
             x[27][vi] + x[28][vi] - x[29][vi] - x[30][vi] - x[31][vi] +
             x[32][vi] - x[33][vi] + x[34][vi] + x[35][vi];
    out[2] = +x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] +
             x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] + x[10][vi] + x[11][vi] -
             x[12][vi] - x[13][vi] - x[14][vi] + x[15][vi] - x[16][vi] +
             x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] + x[21][vi] +
             x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] - x[26][vi] -
             x[27][vi] + x[28][vi] + x[29][vi] - x[30][vi] - x[31][vi] -
             x[32][vi] + x[33][vi] - x[34][vi] + x[35][vi];
    out[3] = +x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] -
             x[6][vi] + x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] + x[11][vi] +
             x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] + x[16][vi] -
             x[17][vi] + x[18][vi] + x[19][vi] + x[20][vi] - x[21][vi] +
             x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] - x[26][vi] -
             x[27][vi] - x[28][vi] + x[29][vi] + x[30][vi] - x[31][vi] -
             x[32][vi] - x[33][vi] + x[34][vi] - x[35][vi];
    out[4] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] +
             x[6][vi] - x[7][vi] + x[8][vi] - x[9][vi] - x[10][vi] - x[11][vi] +
             x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] - x[16][vi] +
             x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] + x[21][vi] -
             x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] + x[26][vi] -
             x[27][vi] - x[28][vi] - x[29][vi] + x[30][vi] + x[31][vi] -
             x[32][vi] - x[33][vi] - x[34][vi] + x[35][vi];
    out[5] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] +
             x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] - x[10][vi] - x[11][vi] -
             x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] - x[16][vi] -
             x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] + x[21][vi] +
             x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] - x[26][vi] +
             x[27][vi] - x[28][vi] - x[29][vi] - x[30][vi] + x[31][vi] +
             x[32][vi] - x[33][vi] - x[34][vi] - x[35][vi];
    out[6] = +x[0][vi] - x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] +
             x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] - x[11][vi] -
             x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] - x[16][vi] -
             x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] - x[21][vi] +
             x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] + x[26][vi] -
             x[27][vi] + x[28][vi] - x[29][vi] - x[30][vi] - x[31][vi] +
             x[32][vi] + x[33][vi] - x[34][vi] - x[35][vi];
    out[7] = +x[0][vi] - x[1][vi] - x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] +
             x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] + x[11][vi] -
             x[12][vi] - x[13][vi] - x[14][vi] + x[15][vi] + x[16][vi] -
             x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] + x[21][vi] -
             x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] + x[26][vi] +
             x[27][vi] - x[28][vi] + x[29][vi] - x[30][vi] - x[31][vi] -
             x[32][vi] + x[33][vi] + x[34][vi] - x[35][vi];
    out[8] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] + x[4][vi] - x[5][vi] +
             x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] - x[11][vi] +
             x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] + x[16][vi] +
             x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] - x[21][vi] +
             x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] - x[26][vi] +
             x[27][vi] + x[28][vi] - x[29][vi] + x[30][vi] - x[31][vi] -
             x[32][vi] - x[33][vi] + x[34][vi] + x[35][vi];
    out[9] = +x[0][vi] + x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] + x[5][vi] -
             x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] + x[11][vi] -
             x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] - x[16][vi] +
             x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] - x[21][vi] -
             x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] + x[26][vi] -
             x[27][vi] + x[28][vi] + x[29][vi] - x[30][vi] + x[31][vi] -
             x[32][vi] - x[33][vi] - x[34][vi] + x[35][vi];
    out[10] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] +
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] +
              x[11][vi] + x[12][vi] - x[13][vi] + x[14][vi] - x[15][vi] -
              x[16][vi] - x[17][vi] + x[18][vi] + x[19][vi] + x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] +
              x[26][vi] + x[27][vi] - x[28][vi] + x[29][vi] + x[30][vi] -
              x[31][vi] + x[32][vi] - x[33][vi] - x[34][vi] - x[35][vi];
    out[11] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] +
              x[11][vi] + x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] -
              x[16][vi] - x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] +
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] -
              x[26][vi] + x[27][vi] + x[28][vi] - x[29][vi] + x[30][vi] +
              x[31][vi] - x[32][vi] + x[33][vi] - x[34][vi] - x[35][vi];
    out[12] = +x[0][vi] - x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] +
              x[11][vi] + x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] +
              x[16][vi] - x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] +
              x[21][vi] + x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] +
              x[26][vi] - x[27][vi] + x[28][vi] + x[29][vi] - x[30][vi] +
              x[31][vi] + x[32][vi] - x[33][vi] + x[34][vi] - x[35][vi];
    out[13] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
              x[6][vi] - x[7][vi] - x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] + x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] -
              x[16][vi] + x[17][vi] + x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] + x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] -
              x[26][vi] + x[27][vi] - x[28][vi] + x[29][vi] + x[30][vi] -
              x[31][vi] + x[32][vi] + x[33][vi] - x[34][vi] + x[35][vi];
    out[14] = +x[0][vi] + x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] + x[5][vi] +
              x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] +
              x[16][vi] - x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] -
              x[26][vi] - x[27][vi] + x[28][vi] - x[29][vi] + x[30][vi] +
              x[31][vi] - x[32][vi] + x[33][vi] + x[34][vi] - x[35][vi];
    out[15] = +x[0][vi] - x[1][vi] + x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] +
              x[6][vi] + x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] +
              x[16][vi] + x[17][vi] + x[18][vi] - x[19][vi] + x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] -
              x[26][vi] - x[27][vi] - x[28][vi] + x[29][vi] - x[30][vi] +
              x[31][vi] + x[32][vi] - x[33][vi] + x[34][vi] + x[35][vi];
    out[16] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] - x[10][vi] -
              x[11][vi] + x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] +
              x[16][vi] + x[17][vi] + x[18][vi] + x[19][vi] - x[20][vi] +
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] +
              x[26][vi] - x[27][vi] - x[28][vi] - x[29][vi] + x[30][vi] -
              x[31][vi] + x[32][vi] + x[33][vi] - x[34][vi] + x[35][vi];
    out[17] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] -
              x[11][vi] - x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] +
              x[16][vi] + x[17][vi] + x[18][vi] + x[19][vi] + x[20][vi] -
              x[21][vi] + x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] +
              x[26][vi] + x[27][vi] - x[28][vi] - x[29][vi] - x[30][vi] +
              x[31][vi] - x[32][vi] + x[33][vi] + x[34][vi] - x[35][vi];
    out[18] = -x[0][vi] + x[1][vi] + x[2][vi] + x[3][vi] + x[4][vi] + x[5][vi] +
              x[6][vi] + x[7][vi] + x[8][vi] + x[9][vi] + x[10][vi] +
              x[11][vi] + x[12][vi] + x[13][vi] + x[14][vi] + x[15][vi] +
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] -
              x[26][vi] - x[27][vi] - x[28][vi] - x[29][vi] - x[30][vi] -
              x[31][vi] - x[32][vi] - x[33][vi] - x[34][vi] - x[35][vi];
    out[19] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] -
              x[6][vi] - x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] -
              x[11][vi] - x[12][vi] - x[13][vi] + x[14][vi] - x[15][vi] +
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] + x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] +
              x[26][vi] - x[27][vi] - x[28][vi] + x[29][vi] + x[30][vi] +
              x[31][vi] - x[32][vi] + x[33][vi] - x[34][vi] - x[35][vi];
    out[20] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] +
              x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] + x[10][vi] +
              x[11][vi] - x[12][vi] - x[13][vi] - x[14][vi] + x[15][vi] -
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] +
              x[26][vi] + x[27][vi] - x[28][vi] - x[29][vi] + x[30][vi] +
              x[31][vi] + x[32][vi] - x[33][vi] + x[34][vi] - x[35][vi];
    out[21] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
              x[6][vi] + x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] +
              x[11][vi] + x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] +
              x[26][vi] + x[27][vi] + x[28][vi] - x[29][vi] - x[30][vi] +
              x[31][vi] + x[32][vi] + x[33][vi] - x[34][vi] + x[35][vi];
    out[22] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] +
              x[6][vi] - x[7][vi] + x[8][vi] - x[9][vi] - x[10][vi] -
              x[11][vi] + x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] -
              x[16][vi] + x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] -
              x[26][vi] + x[27][vi] + x[28][vi] + x[29][vi] - x[30][vi] -
              x[31][vi] + x[32][vi] + x[33][vi] + x[34][vi] - x[35][vi];
    out[23] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] +
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] - x[10][vi] -
              x[11][vi] - x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] -
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] -
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] +
              x[26][vi] - x[27][vi] + x[28][vi] + x[29][vi] + x[30][vi] -
              x[31][vi] - x[32][vi] + x[33][vi] + x[34][vi] + x[35][vi];
    out[24] = +x[0][vi] - x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
              x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] -
              x[11][vi] - x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] -
              x[16][vi] - x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] +
              x[21][vi] - x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] -
              x[26][vi] + x[27][vi] - x[28][vi] + x[29][vi] + x[30][vi] +
              x[31][vi] - x[32][vi] - x[33][vi] + x[34][vi] + x[35][vi];
    out[25] = +x[0][vi] - x[1][vi] - x[2][vi] + x[3][vi] - x[4][vi] + x[5][vi] +
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] - x[12][vi] - x[13][vi] - x[14][vi] + x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] -
              x[21][vi] + x[22][vi] - x[23][vi] - x[24][vi] - x[25][vi] -
              x[26][vi] - x[27][vi] + x[28][vi] - x[29][vi] + x[30][vi] +
              x[31][vi] + x[32][vi] - x[33][vi] - x[34][vi] + x[35][vi];
    out[26] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] + x[4][vi] - x[5][vi] +
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] - x[13][vi] - x[14][vi] - x[15][vi] +
              x[16][vi] + x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] +
              x[21][vi] - x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] -
              x[26][vi] - x[27][vi] - x[28][vi] + x[29][vi] - x[30][vi] +
              x[31][vi] + x[32][vi] + x[33][vi] - x[34][vi] - x[35][vi];
    out[27] = +x[0][vi] + x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] + x[5][vi] -
              x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] - x[14][vi] - x[15][vi] -
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] +
              x[21][vi] + x[22][vi] - x[23][vi] + x[24][vi] - x[25][vi] -
              x[26][vi] - x[27][vi] - x[28][vi] - x[29][vi] + x[30][vi] -
              x[31][vi] + x[32][vi] + x[33][vi] + x[34][vi] - x[35][vi];
    out[28] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] +
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] + x[12][vi] - x[13][vi] + x[14][vi] - x[15][vi] -
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] +
              x[21][vi] + x[22][vi] + x[23][vi] - x[24][vi] + x[25][vi] -
              x[26][vi] - x[27][vi] - x[28][vi] - x[29][vi] - x[30][vi] +
              x[31][vi] - x[32][vi] + x[33][vi] + x[34][vi] + x[35][vi];
    out[29] = +x[0][vi] - x[1][vi] + x[2][vi] + x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] + x[7][vi] - x[8][vi] + x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] -
              x[16][vi] - x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] -
              x[21][vi] + x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] +
              x[26][vi] - x[27][vi] - x[28][vi] - x[29][vi] - x[30][vi] -
              x[31][vi] + x[32][vi] - x[33][vi] + x[34][vi] + x[35][vi];
    out[30] = +x[0][vi] - x[1][vi] - x[2][vi] + x[3][vi] + x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] + x[8][vi] - x[9][vi] + x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] -
              x[21][vi] - x[22][vi] + x[23][vi] + x[24][vi] + x[25][vi] -
              x[26][vi] + x[27][vi] - x[28][vi] - x[29][vi] - x[30][vi] -
              x[31][vi] - x[32][vi] + x[33][vi] - x[34][vi] + x[35][vi];
    out[31] = +x[0][vi] - x[1][vi] - x[2][vi] - x[3][vi] + x[4][vi] + x[5][vi] -
              x[6][vi] - x[7][vi] - x[8][vi] + x[9][vi] - x[10][vi] +
              x[11][vi] + x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] -
              x[16][vi] + x[17][vi] - x[18][vi] + x[19][vi] + x[20][vi] +
              x[21][vi] - x[22][vi] - x[23][vi] + x[24][vi] + x[25][vi] +
              x[26][vi] - x[27][vi] + x[28][vi] - x[29][vi] - x[30][vi] -
              x[31][vi] - x[32][vi] - x[33][vi] + x[34][vi] - x[35][vi];
    out[32] = +x[0][vi] + x[1][vi] - x[2][vi] - x[3][vi] - x[4][vi] + x[5][vi] +
              x[6][vi] - x[7][vi] - x[8][vi] - x[9][vi] + x[10][vi] -
              x[11][vi] + x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] +
              x[21][vi] + x[22][vi] - x[23][vi] - x[24][vi] + x[25][vi] +
              x[26][vi] + x[27][vi] - x[28][vi] + x[29][vi] - x[30][vi] -
              x[31][vi] - x[32][vi] - x[33][vi] - x[34][vi] + x[35][vi];
    out[33] = +x[0][vi] - x[1][vi] + x[2][vi] - x[3][vi] - x[4][vi] - x[5][vi] +
              x[6][vi] + x[7][vi] - x[8][vi] - x[9][vi] - x[10][vi] +
              x[11][vi] - x[12][vi] + x[13][vi] + x[14][vi] - x[15][vi] +
              x[16][vi] + x[17][vi] - x[18][vi] + x[19][vi] - x[20][vi] +
              x[21][vi] + x[22][vi] + x[23][vi] - x[24][vi] - x[25][vi] +
              x[26][vi] + x[27][vi] + x[28][vi] - x[29][vi] + x[30][vi] -
              x[31][vi] - x[32][vi] - x[33][vi] - x[34][vi] - x[35][vi];
    out[34] = +x[0][vi] + x[1][vi] - x[2][vi] + x[3][vi] - x[4][vi] - x[5][vi] -
              x[6][vi] + x[7][vi] + x[8][vi] - x[9][vi] - x[10][vi] -
              x[11][vi] + x[12][vi] - x[13][vi] + x[14][vi] + x[15][vi] -
              x[16][vi] + x[17][vi] - x[18][vi] - x[19][vi] + x[20][vi] -
              x[21][vi] + x[22][vi] + x[23][vi] + x[24][vi] - x[25][vi] -
              x[26][vi] + x[27][vi] + x[28][vi] + x[29][vi] - x[30][vi] +
              x[31][vi] - x[32][vi] - x[33][vi] - x[34][vi] - x[35][vi];
    out[35] = +x[0][vi] + x[1][vi] + x[2][vi] - x[3][vi] + x[4][vi] - x[5][vi] -
              x[6][vi] - x[7][vi] + x[8][vi] + x[9][vi] - x[10][vi] -
              x[11][vi] - x[12][vi] + x[13][vi] - x[14][vi] + x[15][vi] +
              x[16][vi] - x[17][vi] - x[18][vi] - x[19][vi] - x[20][vi] +
              x[21][vi] - x[22][vi] + x[23][vi] + x[24][vi] + x[25][vi] -
              x[26][vi] - x[27][vi] + x[28][vi] + x[29][vi] + x[30][vi] -
              x[31][vi] + x[32][vi] - x[33][vi] - x[34][vi] - x[35][vi];
#pragma unroll
    for (int i = 0; i < 36; i++) {
      x[i][vi] = out[i];
    }
  }
}

template <typename T>
__device__ __forceinline__ void hadamard_mult_thread_28(T x[28]) {  // 35
  T out[28];
  out[0] = +x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] +
           x[9] + x[10] + x[11] + x[12] + x[13] - x[14] + x[15] + x[16] +
           x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] +
           x[25] + x[26] + x[27];
  out[1] = +x[0] + x[1] + x[2] - x[3] + x[4] + x[5] - x[6] - x[7] - x[8] -
           x[9] + x[10] + x[11] - x[12] + x[13] + x[14] - x[15] + x[16] -
           x[17] + x[18] + x[19] - x[20] - x[21] - x[22] - x[23] + x[24] +
           x[25] - x[26] + x[27];
  out[2] = +x[0] + x[1] + x[2] + x[3] - x[4] + x[5] + x[6] - x[7] - x[8] -
           x[9] - x[10] + x[11] + x[12] - x[13] + x[14] + x[15] - x[16] +
           x[17] - x[18] + x[19] + x[20] - x[21] - x[22] - x[23] - x[24] +
           x[25] + x[26] - x[27];
  out[3] = +x[0] - x[1] + x[2] + x[3] + x[4] - x[5] + x[6] + x[7] - x[8] -
           x[9] - x[10] - x[11] + x[12] + x[13] + x[14] - x[15] + x[16] -
           x[17] + x[18] - x[19] + x[20] + x[21] - x[22] - x[23] - x[24] -
           x[25] + x[26] + x[27];
  out[4] = +x[0] + x[1] - x[2] + x[3] + x[4] + x[5] - x[6] + x[7] + x[8] -
           x[9] - x[10] - x[11] - x[12] + x[13] + x[14] + x[15] - x[16] +
           x[17] - x[18] + x[19] - x[20] + x[21] + x[22] - x[23] - x[24] -
           x[25] - x[26] + x[27];
  out[5] = +x[0] + x[1] + x[2] - x[3] + x[4] + x[5] + x[6] - x[7] + x[8] +
           x[9] - x[10] - x[11] - x[12] - x[13] + x[14] + x[15] + x[16] -
           x[17] + x[18] - x[19] + x[20] - x[21] + x[22] + x[23] - x[24] -
           x[25] - x[26] - x[27];
  out[6] = +x[0] - x[1] + x[2] + x[3] - x[4] + x[5] + x[6] + x[7] - x[8] +
           x[9] + x[10] - x[11] - x[12] - x[13] + x[14] - x[15] + x[16] +
           x[17] - x[18] + x[19] - x[20] + x[21] - x[22] + x[23] + x[24] -
           x[25] - x[26] - x[27];
  out[7] = +x[0] - x[1] - x[2] + x[3] + x[4] - x[5] + x[6] + x[7] + x[8] -
           x[9] + x[10] + x[11] - x[12] - x[13] + x[14] - x[15] - x[16] +
           x[17] + x[18] - x[19] + x[20] - x[21] + x[22] - x[23] + x[24] +
           x[25] - x[26] - x[27];
  out[8] = +x[0] - x[1] - x[2] - x[3] + x[4] + x[5] - x[6] + x[7] + x[8] +
           x[9] - x[10] + x[11] + x[12] - x[13] + x[14] - x[15] - x[16] -
           x[17] + x[18] + x[19] - x[20] + x[21] - x[22] + x[23] - x[24] +
           x[25] + x[26] - x[27];
  out[9] = +x[0] - x[1] - x[2] - x[3] - x[4] + x[5] + x[6] - x[7] + x[8] +
           x[9] + x[10] - x[11] + x[12] + x[13] + x[14] - x[15] - x[16] -
           x[17] - x[18] + x[19] + x[20] - x[21] + x[22] - x[23] + x[24] -
           x[25] + x[26] + x[27];
  out[10] = +x[0] + x[1] - x[2] - x[3] - x[4] - x[5] + x[6] + x[7] - x[8] +
            x[9] + x[10] + x[11] - x[12] + x[13] + x[14] + x[15] - x[16] -
            x[17] - x[18] - x[19] + x[20] + x[21] - x[22] + x[23] - x[24] +
            x[25] - x[26] + x[27];
  out[11] = +x[0] + x[1] + x[2] - x[3] - x[4] - x[5] - x[6] + x[7] + x[8] -
            x[9] + x[10] + x[11] + x[12] - x[13] + x[14] + x[15] + x[16] -
            x[17] - x[18] - x[19] - x[20] + x[21] + x[22] - x[23] + x[24] -
            x[25] + x[26] - x[27];
  out[12] = +x[0] - x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7] + x[8] +
            x[9] - x[10] + x[11] + x[12] + x[13] + x[14] - x[15] + x[16] +
            x[17] - x[18] - x[19] - x[20] - x[21] + x[22] + x[23] - x[24] +
            x[25] - x[26] + x[27];
  out[13] = +x[0] + x[1] - x[2] + x[3] + x[4] - x[5] - x[6] - x[7] - x[8] +
            x[9] + x[10] - x[11] + x[12] + x[13] + x[14] + x[15] - x[16] +
            x[17] + x[18] - x[19] - x[20] - x[21] - x[22] + x[23] + x[24] -
            x[25] + x[26] - x[27];
  out[14] = -x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] +
            x[9] + x[10] + x[11] + x[12] + x[13] - x[14] - x[15] - x[16] -
            x[17] - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] - x[24] -
            x[25] - x[26] - x[27];
  out[15] = +x[0] - x[1] + x[2] - x[3] + x[4] + x[5] - x[6] - x[7] - x[8] -
            x[9] + x[10] + x[11] - x[12] + x[13] - x[14] - x[15] - x[16] +
            x[17] - x[18] - x[19] + x[20] + x[21] + x[22] + x[23] - x[24] -
            x[25] + x[26] - x[27];
  out[16] = +x[0] + x[1] - x[2] + x[3] - x[4] + x[5] + x[6] - x[7] - x[8] -
            x[9] - x[10] + x[11] + x[12] - x[13] - x[14] - x[15] - x[16] -
            x[17] + x[18] - x[19] - x[20] + x[21] + x[22] + x[23] + x[24] -
            x[25] - x[26] + x[27];
  out[17] = +x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] + x[7] - x[8] -
            x[9] - x[10] - x[11] + x[12] + x[13] - x[14] + x[15] - x[16] -
            x[17] - x[18] + x[19] - x[20] - x[21] + x[22] + x[23] + x[24] +
            x[25] - x[26] - x[27];
  out[18] = +x[0] + x[1] - x[2] + x[3] - x[4] + x[5] - x[6] + x[7] + x[8] -
            x[9] - x[10] - x[11] - x[12] + x[13] - x[14] - x[15] + x[16] -
            x[17] - x[18] - x[19] + x[20] - x[21] - x[22] + x[23] + x[24] +
            x[25] + x[26] - x[27];
  out[19] = +x[0] + x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] +
            x[9] - x[10] - x[11] - x[12] - x[13] - x[14] - x[15] - x[16] +
            x[17] - x[18] - x[19] - x[20] + x[21] - x[22] - x[23] + x[24] +
            x[25] + x[26] + x[27];
  out[20] = +x[0] - x[1] + x[2] + x[3] - x[4] + x[5] - x[6] + x[7] - x[8] +
            x[9] + x[10] - x[11] - x[12] - x[13] - x[14] + x[15] - x[16] -
            x[17] + x[18] - x[19] - x[20] - x[21] + x[22] - x[23] - x[24] +
            x[25] + x[26] + x[27];
  out[21] = +x[0] - x[1] - x[2] + x[3] + x[4] - x[5] + x[6] - x[7] + x[8] -
            x[9] + x[10] + x[11] - x[12] - x[13] - x[14] + x[15] + x[16] -
            x[17] - x[18] + x[19] - x[20] - x[21] - x[22] + x[23] - x[24] -
            x[25] + x[26] + x[27];
  out[22] = +x[0] - x[1] - x[2] - x[3] + x[4] + x[5] - x[6] + x[7] - x[8] +
            x[9] - x[10] + x[11] + x[12] - x[13] - x[14] + x[15] + x[16] +
            x[17] - x[18] - x[19] + x[20] - x[21] - x[22] - x[23] + x[24] -
            x[25] - x[26] + x[27];
  out[23] = +x[0] - x[1] - x[2] - x[3] - x[4] + x[5] + x[6] - x[7] + x[8] -
            x[9] + x[10] - x[11] + x[12] + x[13] - x[14] + x[15] + x[16] +
            x[17] + x[18] - x[19] - x[20] + x[21] - x[22] - x[23] - x[24] +
            x[25] - x[26] - x[27];
  out[24] = +x[0] + x[1] - x[2] - x[3] - x[4] - x[5] + x[6] + x[7] - x[8] +
            x[9] - x[10] + x[11] - x[12] + x[13] - x[14] - x[15] + x[16] +
            x[17] + x[18] + x[19] - x[20] - x[21] + x[22] - x[23] - x[24] -
            x[25] + x[26] - x[27];
  out[25] = +x[0] + x[1] + x[2] - x[3] - x[4] - x[5] - x[6] + x[7] + x[8] -
            x[9] + x[10] - x[11] + x[12] - x[13] - x[14] - x[15] - x[16] +
            x[17] + x[18] + x[19] + x[20] - x[21] - x[22] + x[23] - x[24] -
            x[25] - x[26] + x[27];
  out[26] = +x[0] - x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7] + x[8] +
            x[9] - x[10] + x[11] - x[12] + x[13] - x[14] + x[15] - x[16] -
            x[17] + x[18] + x[19] + x[20] + x[21] - x[22] - x[23] + x[24] -
            x[25] - x[26] - x[27];
  out[27] = +x[0] + x[1] - x[2] + x[3] + x[4] - x[5] - x[6] - x[7] - x[8] +
            x[9] + x[10] - x[11] + x[12] - x[13] - x[14] - x[15] + x[16] -
            x[17] - x[18] + x[19] + x[20] + x[21] + x[22] - x[23] - x[24] +
            x[25] - x[26] - x[27];
#pragma unroll
  for (int i = 0; i < 28; i++) {
    x[i] = out[i];
  }
}

template <typename T>
__device__ __forceinline__ void hadamard_mult_thread_36(T x[36]) {  // 4t
  T out[36];
  out[0] = +x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] +
           x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] +
           x[17] - x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] +
           x[25] + x[26] + x[27] + x[28] + x[29] + x[30] + x[31] + x[32] +
           x[33] + x[34] + x[35];
  out[1] = +x[0] + x[1] + x[2] + x[3] - x[4] + x[5] - x[6] - x[7] - x[8] +
           x[9] + x[10] - x[11] - x[12] - x[13] + x[14] - x[15] + x[16] +
           x[17] + x[18] - x[19] + x[20] + x[21] - x[22] + x[23] - x[24] -
           x[25] - x[26] + x[27] + x[28] - x[29] - x[30] - x[31] + x[32] -
           x[33] + x[34] + x[35];
  out[2] = +x[0] + x[1] + x[2] + x[3] + x[4] - x[5] + x[6] - x[7] - x[8] -
           x[9] + x[10] + x[11] - x[12] - x[13] - x[14] + x[15] - x[16] +
           x[17] + x[18] + x[19] - x[20] + x[21] + x[22] - x[23] + x[24] -
           x[25] - x[26] - x[27] + x[28] + x[29] - x[30] - x[31] - x[32] +
           x[33] - x[34] + x[35];
  out[3] = +x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - x[6] + x[7] - x[8] -
           x[9] - x[10] + x[11] + x[12] - x[13] - x[14] - x[15] + x[16] -
           x[17] + x[18] + x[19] + x[20] - x[21] + x[22] + x[23] - x[24] +
           x[25] - x[26] - x[27] - x[28] + x[29] + x[30] - x[31] - x[32] -
           x[33] + x[34] - x[35];
  out[4] = +x[0] - x[1] + x[2] + x[3] + x[4] + x[5] + x[6] - x[7] + x[8] -
           x[9] - x[10] - x[11] + x[12] + x[13] - x[14] - x[15] - x[16] +
           x[17] + x[18] - x[19] + x[20] + x[21] - x[22] + x[23] + x[24] -
           x[25] + x[26] - x[27] - x[28] - x[29] + x[30] + x[31] - x[32] -
           x[33] - x[34] + x[35];
  out[5] = +x[0] + x[1] - x[2] + x[3] + x[4] + x[5] + x[6] + x[7] - x[8] +
           x[9] - x[10] - x[11] - x[12] + x[13] + x[14] - x[15] - x[16] -
           x[17] + x[18] + x[19] - x[20] + x[21] + x[22] - x[23] + x[24] +
           x[25] - x[26] + x[27] - x[28] - x[29] - x[30] + x[31] + x[32] -
           x[33] - x[34] - x[35];
  out[6] = +x[0] - x[1] + x[2] - x[3] + x[4] + x[5] + x[6] + x[7] + x[8] -
           x[9] + x[10] - x[11] - x[12] - x[13] + x[14] + x[15] - x[16] -
           x[17] + x[18] - x[19] + x[20] - x[21] + x[22] + x[23] - x[24] +
           x[25] + x[26] - x[27] + x[28] - x[29] - x[30] - x[31] + x[32] +
           x[33] - x[34] - x[35];
  out[7] = +x[0] - x[1] - x[2] + x[3] - x[4] + x[5] + x[6] + x[7] + x[8] +
           x[9] - x[10] + x[11] - x[12] - x[13] - x[14] + x[15] + x[16] -
           x[17] + x[18] - x[19] - x[20] + x[21] - x[22] + x[23] + x[24] -
           x[25] + x[26] + x[27] - x[28] + x[29] - x[30] - x[31] - x[32] +
           x[33] + x[34] - x[35];
  out[8] = +x[0] - x[1] - x[2] - x[3] + x[4] - x[5] + x[6] + x[7] + x[8] +
           x[9] + x[10] - x[11] + x[12] - x[13] - x[14] - x[15] + x[16] +
           x[17] + x[18] - x[19] - x[20] - x[21] + x[22] - x[23] + x[24] +
           x[25] - x[26] + x[27] + x[28] - x[29] + x[30] - x[31] - x[32] -
           x[33] + x[34] + x[35];
  out[9] = +x[0] + x[1] - x[2] - x[3] - x[4] + x[5] - x[6] + x[7] + x[8] +
           x[9] + x[10] + x[11] - x[12] + x[13] - x[14] - x[15] - x[16] +
           x[17] + x[18] + x[19] - x[20] - x[21] - x[22] + x[23] - x[24] +
           x[25] + x[26] - x[27] + x[28] + x[29] - x[30] + x[31] - x[32] -
           x[33] - x[34] + x[35];
  out[10] = +x[0] + x[1] + x[2] - x[3] - x[4] - x[5] + x[6] - x[7] + x[8] +
            x[9] + x[10] + x[11] + x[12] - x[13] + x[14] - x[15] - x[16] -
            x[17] + x[18] + x[19] + x[20] - x[21] - x[22] - x[23] + x[24] -
            x[25] + x[26] + x[27] - x[28] + x[29] + x[30] - x[31] + x[32] -
            x[33] - x[34] - x[35];
  out[11] = +x[0] - x[1] + x[2] + x[3] - x[4] - x[5] - x[6] + x[7] - x[8] +
            x[9] + x[10] + x[11] + x[12] + x[13] - x[14] + x[15] - x[16] -
            x[17] + x[18] - x[19] + x[20] + x[21] - x[22] - x[23] - x[24] +
            x[25] - x[26] + x[27] + x[28] - x[29] + x[30] + x[31] - x[32] +
            x[33] - x[34] - x[35];
  out[12] = +x[0] - x[1] - x[2] + x[3] + x[4] - x[5] - x[6] - x[7] + x[8] -
            x[9] + x[10] + x[11] + x[12] + x[13] + x[14] - x[15] + x[16] -
            x[17] + x[18] - x[19] - x[20] + x[21] + x[22] - x[23] - x[24] -
            x[25] + x[26] - x[27] + x[28] + x[29] - x[30] + x[31] + x[32] -
            x[33] + x[34] - x[35];
  out[13] = +x[0] - x[1] - x[2] - x[3] + x[4] + x[5] - x[6] - x[7] - x[8] +
            x[9] - x[10] + x[11] + x[12] + x[13] + x[14] + x[15] - x[16] +
            x[17] + x[18] - x[19] - x[20] - x[21] + x[22] + x[23] - x[24] -
            x[25] - x[26] + x[27] - x[28] + x[29] + x[30] - x[31] + x[32] +
            x[33] - x[34] + x[35];
  out[14] = +x[0] + x[1] - x[2] - x[3] - x[4] + x[5] + x[6] - x[7] - x[8] -
            x[9] + x[10] - x[11] + x[12] + x[13] + x[14] + x[15] + x[16] -
            x[17] + x[18] + x[19] - x[20] - x[21] - x[22] + x[23] + x[24] -
            x[25] - x[26] - x[27] + x[28] - x[29] + x[30] + x[31] - x[32] +
            x[33] + x[34] - x[35];
  out[15] = +x[0] - x[1] + x[2] - x[3] - x[4] - x[5] + x[6] + x[7] - x[8] -
            x[9] - x[10] + x[11] - x[12] + x[13] + x[14] + x[15] + x[16] +
            x[17] + x[18] - x[19] + x[20] - x[21] - x[22] - x[23] + x[24] +
            x[25] - x[26] - x[27] - x[28] + x[29] - x[30] + x[31] + x[32] -
            x[33] + x[34] + x[35];
  out[16] = +x[0] + x[1] - x[2] + x[3] - x[4] - x[5] - x[6] + x[7] + x[8] -
            x[9] - x[10] - x[11] + x[12] - x[13] + x[14] + x[15] + x[16] +
            x[17] + x[18] + x[19] - x[20] + x[21] - x[22] - x[23] - x[24] +
            x[25] + x[26] - x[27] - x[28] - x[29] + x[30] - x[31] + x[32] +
            x[33] - x[34] + x[35];
  out[17] = +x[0] + x[1] + x[2] - x[3] + x[4] - x[5] - x[6] - x[7] + x[8] +
            x[9] - x[10] - x[11] - x[12] + x[13] - x[14] + x[15] + x[16] +
            x[17] + x[18] + x[19] + x[20] - x[21] + x[22] - x[23] - x[24] -
            x[25] + x[26] + x[27] - x[28] - x[29] - x[30] + x[31] - x[32] +
            x[33] + x[34] - x[35];
  out[18] = -x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] +
            x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] +
            x[17] - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] - x[24] -
            x[25] - x[26] - x[27] - x[28] - x[29] - x[30] - x[31] - x[32] -
            x[33] - x[34] - x[35];
  out[19] = +x[0] - x[1] + x[2] + x[3] - x[4] + x[5] - x[6] - x[7] - x[8] +
            x[9] + x[10] - x[11] - x[12] - x[13] + x[14] - x[15] + x[16] +
            x[17] - x[18] - x[19] - x[20] - x[21] + x[22] - x[23] + x[24] +
            x[25] + x[26] - x[27] - x[28] + x[29] + x[30] + x[31] - x[32] +
            x[33] - x[34] - x[35];
  out[20] = +x[0] + x[1] - x[2] + x[3] + x[4] - x[5] + x[6] - x[7] - x[8] -
            x[9] + x[10] + x[11] - x[12] - x[13] - x[14] + x[15] - x[16] +
            x[17] - x[18] - x[19] - x[20] - x[21] - x[22] + x[23] - x[24] +
            x[25] + x[26] + x[27] - x[28] - x[29] + x[30] + x[31] + x[32] -
            x[33] + x[34] - x[35];
  out[21] = +x[0] + x[1] + x[2] - x[3] + x[4] + x[5] - x[6] + x[7] - x[8] -
            x[9] - x[10] + x[11] + x[12] - x[13] - x[14] - x[15] + x[16] -
            x[17] - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] + x[24] -
            x[25] + x[26] + x[27] + x[28] - x[29] - x[30] + x[31] + x[32] +
            x[33] - x[34] + x[35];
  out[22] = +x[0] - x[1] + x[2] + x[3] - x[4] + x[5] + x[6] - x[7] + x[8] -
            x[9] - x[10] - x[11] + x[12] + x[13] - x[14] - x[15] - x[16] +
            x[17] - x[18] + x[19] - x[20] - x[21] - x[22] - x[23] - x[24] +
            x[25] - x[26] + x[27] + x[28] + x[29] - x[30] - x[31] + x[32] +
            x[33] + x[34] - x[35];
  out[23] = +x[0] + x[1] - x[2] + x[3] + x[4] - x[5] + x[6] + x[7] - x[8] +
            x[9] - x[10] - x[11] - x[12] + x[13] + x[14] - x[15] - x[16] -
            x[17] - x[18] - x[19] + x[20] - x[21] - x[22] - x[23] - x[24] -
            x[25] + x[26] - x[27] + x[28] + x[29] + x[30] - x[31] - x[32] +
            x[33] + x[34] + x[35];
  out[24] = +x[0] - x[1] + x[2] - x[3] + x[4] + x[5] - x[6] + x[7] + x[8] -
            x[9] + x[10] - x[11] - x[12] - x[13] + x[14] + x[15] - x[16] -
            x[17] - x[18] + x[19] - x[20] + x[21] - x[22] - x[23] - x[24] -
            x[25] - x[26] + x[27] - x[28] + x[29] + x[30] + x[31] - x[32] -
            x[33] + x[34] + x[35];
  out[25] = +x[0] - x[1] - x[2] + x[3] - x[4] + x[5] + x[6] - x[7] + x[8] +
            x[9] - x[10] + x[11] - x[12] - x[13] - x[14] + x[15] + x[16] -
            x[17] - x[18] + x[19] + x[20] - x[21] + x[22] - x[23] - x[24] -
            x[25] - x[26] - x[27] + x[28] - x[29] + x[30] + x[31] + x[32] -
            x[33] - x[34] + x[35];
  out[26] = +x[0] - x[1] - x[2] - x[3] + x[4] - x[5] + x[6] + x[7] - x[8] +
            x[9] + x[10] - x[11] + x[12] - x[13] - x[14] - x[15] + x[16] +
            x[17] - x[18] + x[19] + x[20] + x[21] - x[22] + x[23] - x[24] -
            x[25] - x[26] - x[27] - x[28] + x[29] - x[30] + x[31] + x[32] +
            x[33] - x[34] - x[35];
  out[27] = +x[0] + x[1] - x[2] - x[3] - x[4] + x[5] - x[6] + x[7] + x[8] -
            x[9] + x[10] + x[11] - x[12] + x[13] - x[14] - x[15] - x[16] +
            x[17] - x[18] - x[19] + x[20] + x[21] + x[22] - x[23] + x[24] -
            x[25] - x[26] - x[27] - x[28] - x[29] + x[30] - x[31] + x[32] +
            x[33] + x[34] - x[35];
  out[28] = +x[0] + x[1] + x[2] - x[3] - x[4] - x[5] + x[6] - x[7] + x[8] +
            x[9] - x[10] + x[11] + x[12] - x[13] + x[14] - x[15] - x[16] -
            x[17] - x[18] - x[19] - x[20] + x[21] + x[22] + x[23] - x[24] +
            x[25] - x[26] - x[27] - x[28] - x[29] - x[30] + x[31] - x[32] +
            x[33] + x[34] + x[35];
  out[29] = +x[0] - x[1] + x[2] + x[3] - x[4] - x[5] - x[6] + x[7] - x[8] +
            x[9] + x[10] - x[11] + x[12] + x[13] - x[14] + x[15] - x[16] -
            x[17] - x[18] + x[19] - x[20] - x[21] + x[22] + x[23] + x[24] -
            x[25] + x[26] - x[27] - x[28] - x[29] - x[30] - x[31] + x[32] -
            x[33] + x[34] + x[35];
  out[30] = +x[0] - x[1] - x[2] + x[3] + x[4] - x[5] - x[6] - x[7] + x[8] -
            x[9] + x[10] + x[11] - x[12] + x[13] + x[14] - x[15] + x[16] -
            x[17] - x[18] + x[19] + x[20] - x[21] - x[22] + x[23] + x[24] +
            x[25] - x[26] + x[27] - x[28] - x[29] - x[30] - x[31] - x[32] +
            x[33] - x[34] + x[35];
  out[31] = +x[0] - x[1] - x[2] - x[3] + x[4] + x[5] - x[6] - x[7] - x[8] +
            x[9] - x[10] + x[11] + x[12] - x[13] + x[14] + x[15] - x[16] +
            x[17] - x[18] + x[19] + x[20] + x[21] - x[22] - x[23] + x[24] +
            x[25] + x[26] - x[27] + x[28] - x[29] - x[30] - x[31] - x[32] -
            x[33] + x[34] - x[35];
  out[32] = +x[0] + x[1] - x[2] - x[3] - x[4] + x[5] + x[6] - x[7] - x[8] -
            x[9] + x[10] - x[11] + x[12] + x[13] - x[14] + x[15] + x[16] -
            x[17] - x[18] - x[19] + x[20] + x[21] + x[22] - x[23] - x[24] +
            x[25] + x[26] + x[27] - x[28] + x[29] - x[30] - x[31] - x[32] -
            x[33] - x[34] + x[35];
  out[33] = +x[0] - x[1] + x[2] - x[3] - x[4] - x[5] + x[6] + x[7] - x[8] -
            x[9] - x[10] + x[11] - x[12] + x[13] + x[14] - x[15] + x[16] +
            x[17] - x[18] + x[19] - x[20] + x[21] + x[22] + x[23] - x[24] -
            x[25] + x[26] + x[27] + x[28] - x[29] + x[30] - x[31] - x[32] -
            x[33] - x[34] - x[35];
  out[34] = +x[0] + x[1] - x[2] + x[3] - x[4] - x[5] - x[6] + x[7] + x[8] -
            x[9] - x[10] - x[11] + x[12] - x[13] + x[14] + x[15] - x[16] +
            x[17] - x[18] - x[19] + x[20] - x[21] + x[22] + x[23] + x[24] -
            x[25] - x[26] + x[27] + x[28] + x[29] - x[30] + x[31] - x[32] -
            x[33] - x[34] - x[35];
  out[35] = +x[0] + x[1] + x[2] - x[3] + x[4] - x[5] - x[6] - x[7] + x[8] +
            x[9] - x[10] - x[11] - x[12] + x[13] - x[14] + x[15] + x[16] -
            x[17] - x[18] - x[19] - x[20] + x[21] - x[22] + x[23] + x[24] +
            x[25] - x[26] - x[27] + x[28] + x[29] + x[30] - x[31] + x[32] -
            x[33] - x[34] - x[35];
#pragma unroll
  for (int i = 0; i < 36; i++) {
    x[i] = out[i];
  }
}

template <int kNChunks, typename T>
__device__ __forceinline__ void hadamard_mult_thread_chunk_28(
    T x[kNChunks][28]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_28(x[c]);
  }
}

template <int kNChunks, typename T>
__device__ __forceinline__ void hadamard_mult_thread_chunk_36(
    T x[kNChunks][36]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_36(x[c]);
  }
}

template <int kNChunks, int VecSize, bool UseDiagonalBlockMatrix, typename T>
inline __device__ void load_input(const T *x,
                                  T x_vals[kNChunks][VecSize],
                                  int dim) {
  using vec_t = typename BytesToType<sizeof(T) * VecSize>::Type;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    int offset;
    if constexpr (UseDiagonalBlockMatrix) {
      static_assert(kNChunks == 1);
      offset = blockIdx.y * blockDim.x + threadIdx.x;
    } else {
      offset = c * blockDim.x + threadIdx.x;
    }
    if (offset * VecSize < dim) {
      reinterpret_cast<vec_t *>(x_vals)[c] =
          reinterpret_cast<const vec_t *>(x)[offset];
    }
  }
}

template <typename InType, typename OutType>
__forceinline__ __device__ OutType QuantHelperFunc(const InType input,
                                                   const float scale,
                                                   const int round_type,
                                                   const float max_bound,
                                                   const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(rint(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  return static_cast<OutType>(
      ClipFunc<float>(quant_value, min_bound, max_bound));
}

template <int kNChunks,
          int VecSize,
          bool UseDiagonalBlockMatrix,
          typename T,
          typename OutT>
inline __device__ void smooth_quant_store_output(OutT *out,
                                                 const T *shift,
                                                 const T *smooth,
                                                 T out_vals[kNChunks][VecSize],
                                                 const float quant_scale,
                                                 const int quant_round_type,
                                                 const float quant_max_bound,
                                                 const float quant_min_bound,
                                                 const int dim) {
  using DstVec = AlignedVector<OutT, VecSize>;
  using Vec = AlignedVector<T, VecSize>;
  DstVec dst_vec;
  Vec shift_vec;
  Vec smooth_vec;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    int base_idx;
    if constexpr (UseDiagonalBlockMatrix) {
      base_idx = blockIdx.y * blockDim.x + threadIdx.x;
    } else {
      base_idx = c * blockDim.x + threadIdx.x;
    }
    const int idx = base_idx * VecSize;
    if (idx < dim) {
      Load<T, VecSize>(shift + idx, &shift_vec);
      Load<T, VecSize>(smooth + idx, &smooth_vec);
#pragma unroll
      for (int vi = 0; vi < VecSize; ++vi) {
        out_vals[c][vi] = (out_vals[c][vi] + shift_vec[vi]) * smooth_vec[vi];
        dst_vec[vi] =
            QuantHelperFunc<float, OutT>(static_cast<float>(out_vals[c][vi]),
                                         quant_scale,
                                         quant_round_type,
                                         quant_max_bound,
                                         quant_min_bound);
      }
      Store<OutT, VecSize>(dst_vec, out + idx);
    }
  }
}

template <int kNChunks,
          int VecSize,
          bool UseDiagonalBlockMatrix,
          typename T,
          typename OutT>
inline __device__ void quant_store_output(OutT *out,
                                          T out_vals[kNChunks][VecSize],
                                          const float quant_scale,
                                          const int quant_round_type,
                                          const float quant_max_bound,
                                          const float quant_min_bound,
                                          const int dim) {
  using DstVec = AlignedVector<OutT, VecSize>;
  using Vec = AlignedVector<T, VecSize>;
  DstVec dst_vec;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    int base_idx;
    if constexpr (UseDiagonalBlockMatrix) {
      base_idx = blockIdx.y * blockDim.x + threadIdx.x;
    } else {
      base_idx = c * blockDim.x + threadIdx.x;
    }
    const int idx = base_idx * VecSize;
    if (idx < dim) {
#pragma unroll
      for (int vi = 0; vi < VecSize; ++vi) {
        // out_vals[c][vi] = (out_vals[c][vi] + shift_vec[vi]) * smooth_vec[vi];
        dst_vec[vi] =
            QuantHelperFunc<float, OutT>(static_cast<float>(out_vals[c][vi]),
                                         quant_scale,
                                         quant_round_type,
                                         quant_max_bound,
                                         quant_min_bound);
      }
      Store<OutT, VecSize>(dst_vec, out + idx);
    }
  }
}

template <int kNChunks,
          int VecSize,
          bool UseDiagonalBlockMatrix,
          typename T,
          typename OutT>
inline __device__ void store_output(OutT *out,
                                    T out_vals[kNChunks][VecSize],
                                    int dim) {
  using vec_t = typename BytesToType<sizeof(T) * VecSize>::Type;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    int offset;
    if constexpr (UseDiagonalBlockMatrix) {
      offset = blockIdx.y * blockDim.x + threadIdx.x;
    } else {
      offset = c * blockDim.x + threadIdx.x;
    }
    if (offset * VecSize < dim) {
      reinterpret_cast<vec_t *>(out)[offset] =
          reinterpret_cast<const vec_t *>(out_vals)[c];
    }
  }
}

template <int kLogN, int kNChunks, typename T>
__device__ __forceinline__ void hadamard_mult_thread_transpose(
    T x[1 << kLogN][kNChunks]) {
  constexpr int N = 1 << kLogN;
#pragma unroll
  for (int i = 0; i < kLogN; ++i) {
    const int stride = 1 << i;
#pragma unroll
    for (int j = 0; j < N / 2; ++j) {
      const int lo = j & (stride - 1);
      const int idx = (j - lo) * 2 + lo;
#pragma unroll
      for (int c = 0; c < kNChunks; ++c) {
        const T a = x[idx][c];
        const T b = x[idx + stride][c];
        x[idx][c] = a + b;
        x[idx + stride][c] = a - b;
      }
    }
  }
}

template <int kLogN, int kNChunks, typename T>
__device__ __forceinline__ void hadamard_mult_thread(
    T x[kNChunks][1 << kLogN]) {
  constexpr int N = 1 << kLogN;
#pragma unroll
  for (int i = 0; i < kLogN; ++i) {
    const int stride = 1 << i;
#pragma unroll
    for (int j = 0; j < N / 2; ++j) {
      const int lo = j & (stride - 1);
      const int idx = (j - lo) * 2 + lo;
#pragma unroll
      for (int c = 0; c < kNChunks; ++c) {
        const T a = x[c][idx];
        const T b = x[c][idx + stride];
        x[c][idx] = a + b;
        x[c][idx + stride] = a - b;
      }
    }
  }
}

template <int kLogWarpSize,
          int kStepStart,
          int kNChunks,
          int kNItems,
          typename T>
__device__ __forceinline__ void hadamard_mult_warp(T x[kNChunks][kNItems]) {
  constexpr int N = 1 << kLogWarpSize;
  int lane_id = threadIdx.x % N;
#pragma unroll
  for (int step = kStepStart; step < kLogWarpSize; ++step) {
    const int lane_mask = 1 << step;
    const T sign = (lane_id & lane_mask) ? -1.f : 1.f;
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNItems; ++i) {
        T x_val_other = __shfl_xor_sync(FULL_MASK, x[c][i], lane_mask);
        x[c][i] = sign * x[c][i] + x_val_other;
      }
    }
  }
}

template <int kNChunks,
          int kChunksPerExchange,
          int kNElts,
          int kWarpSize,
          int kNWarps,
          bool Pre,
          typename vec_t,
          typename T>
inline __device__ void exchange_smem_pre(T x_vals[kNChunks][kNElts],
                                         vec_t *smem) {
  // kNChunks表示整体需要多少次循环才能处理完
  // kChunksPerExchange表示每次循环可以处理多少个chunk
  // kNExchanges表示多少次循环才能处理完所有数据
  constexpr int kNThreads = kWarpSize * kNWarps;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int row_t = threadIdx.x % kNWarps;
  const int col_t = threadIdx.x / kNWarps;
#pragma unroll
  for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
    // 搬多少次chunk算完所有数据
    __syncthreads();
#pragma unroll
    for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
      // 每次循环搬多少数据把smem塞满
      // smem[c1 * kNThreads + (Pre ? warp_id * kWarpSize + lane_id ^ warp_id :
      // row_t * kWarpSize + col_t ^ row_t)] =
      // *reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1]);
      smem[c1 * kNThreads +
           (Pre ? warp_id * kWarpSize + lane_id : row_t * kWarpSize + col_t)] =
          *reinterpret_cast<vec_t *>(x_vals[c0 * kChunksPerExchange + c1]);
    }
    __syncthreads();
#pragma unroll
    for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
      // *reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1]) =
      // smem[c1 * kNThreads + (Pre ? row_t * kWarpSize + col_t ^ row_t :
      // warp_id * kWarpSize + lane_id ^ warp_id)];
      *reinterpret_cast<vec_t *>(x_vals[c0 * kChunksPerExchange + c1]) =
          smem[c1 * kNThreads + (Pre ? row_t * kWarpSize + col_t
                                     : warp_id * kWarpSize + lane_id)];
    }
  }
}

constexpr int cilog2(int val) { return val > 0 ? 1 + cilog2(val >> 1) : -1; }

template <typename T,
          typename OutT,
          int kThreads,
          int kNBytes,
          int VecSize,
          int N,
          int kNChunks,
          int kSmeSize,
          int kRounds,
          int kChunksPerSmemSize,
          bool UseDiagonalBlockMatrix = false>
__global__ __launch_bounds__(kThreads) void moe_fast_hardamard_kernel(
    const T *x,
    const int64_t *expert_idx_per_token,
    const T *shift,
    const T *smooth,
    const float *quant_scales,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const int64_t token_num,
    const int64_t dim,
    OutT *out) {
  using vec_t = typename BytesToType<sizeof(T) * VecSize>::Type;
  constexpr int kLogVecSize = cilog2(VecSize);
  constexpr int kLogWarpSize = cilog2(32);
  constexpr int kWarpSize = 32;
  constexpr int kNWarps = kThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  constexpr int kLogNChunks = cilog2(kNChunks);

  extern __shared__ char smem_[];
  vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

  for (int token_id = blockIdx.x; token_id < token_num; token_id += gridDim.x) {
    const T *x_now = x + token_id * dim;
    OutT *out_now = out + token_id * dim;
    T init_value = static_cast<T>(0.f);
    T x_vals[kNChunks][VecSize] = {init_value};

    load_input<kNChunks, VecSize, UseDiagonalBlockMatrix, T>(
        x_now, x_vals, dim);
#ifdef DEBUG_HARDAMARD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < 1; ++i) {
        printf("chunk_id0: %d\n", i);
        for (int j = 0; j < VecSize; ++j) {
          printf("%f ", (float)x_vals[i][j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    hadamard_mult_thread<kLogVecSize, kNChunks>(x_vals);
#ifdef DEBUG_HARDAMARD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < 1; ++i) {
        printf("chunk_id1: %d, kLogVecSize: %d\n", i, kLogVecSize);
        for (int j = 0; j < VecSize; ++j) {
          printf("%f ", (float)x_vals[i][j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, VecSize>(x_vals);
#ifdef DEBUG_HARDAMARD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < 1; ++i) {
        printf("chunk_id2: %d\n", i);
        for (int j = 0; j < VecSize; ++j) {
          printf("%f ", (float)x_vals[i][j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    if constexpr (kNWarps > 1) {
      // 先让连续的NWARPS个线程拿到其余warps上的数据
      exchange_smem_pre<kNChunks,
                        kChunksPerSmemSize,
                        VecSize,
                        kWarpSize,
                        kNWarps,
                        true,
                        vec_t>(x_vals, smem_exchange);
      // 交叉计算
      hadamard_mult_warp<kLogNWarps, 0, kNChunks, VecSize>(x_vals);
      // 再换回来
      exchange_smem_pre<kNChunks,
                        kChunksPerSmemSize,
                        VecSize,
                        kWarpSize,
                        kNWarps,
                        false,
                        vec_t>(x_vals, smem_exchange);
    }
    if constexpr (kNChunks > 1) {
      if constexpr (kNChunks == 28) {
        hadamard_mult_thread_28_transpose<T, VecSize>(x_vals);
      } else if constexpr (kNChunks == 36) {
        hadamard_mult_thread_36_transpose<T, VecSize>(x_vals);
      } else {
        constexpr int kLogNChunks = cilog2(kNChunks);
        static_assert(1 << kLogNChunks == kNChunks,
                      "kNChunks must be a power of 2");
        hadamard_mult_thread_transpose<kLogNChunks, VecSize>(x_vals);
      }
    }
    if (quant_scales) {
      int64_t expert_id = expert_idx_per_token[token_id];
      float quant_scale = quant_scales[expert_id];
      if (shift) {
        smooth_quant_store_output<kNChunks,
                                  VecSize,
                                  UseDiagonalBlockMatrix,
                                  T,
                                  OutT>(out_now,
                                        shift,
                                        smooth,
                                        x_vals,
                                        quant_scale,
                                        quant_round_type,
                                        quant_max_bound,
                                        quant_min_bound,
                                        dim);
      } else {
        quant_store_output<kNChunks, VecSize, UseDiagonalBlockMatrix, T, OutT>(
            out_now,
            x_vals,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            dim);
      }
    } else {
      store_output<kNChunks, VecSize, UseDiagonalBlockMatrix, T>(
          out_now, x_vals, dim);
    }
  }
}

template <typename T,
          typename OutT,
          int kThreads,
          int kNBytes,
          int VecSize,
          int N,
          int kNChunks,
          int kSmeSize,
          int kRounds,
          int kChunksPerSmemSize,
          bool UseDiagonalBlockMatrix = false>
__global__ __launch_bounds__(kThreads) void masked_moe_fast_hardamard_kernel(
    const T *x,
    const int64_t *recv_expert_count,
    const T *shift,
    const T *smooth,
    const float *quant_scales,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const int64_t token_num,
    const int64_t dim,
    const int num_max_tokens_per_expert,
    OutT *out) {
  using vec_t = typename BytesToType<sizeof(T) * VecSize>::Type;
  constexpr int kLogVecSize = cilog2(VecSize);
  constexpr int kLogWarpSize = cilog2(32);
  constexpr int kWarpSize = 32;
  constexpr int kNWarps = kThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  constexpr int kLogNChunks = cilog2(kNChunks);

  extern __shared__ char smem_[];
  vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

  for (int token_id = blockIdx.x; token_id < token_num; token_id += gridDim.x) {
    const auto token_idx_in_expert = token_id % num_max_tokens_per_expert;
    const auto expert_id = token_id / num_max_tokens_per_expert;
    if (token_idx_in_expert >= recv_expert_count[expert_id]) {
      auto next_expert_start_idx = (expert_id + 1) * num_max_tokens_per_expert;
      auto num_iters_to_next_expert =
          (next_expert_start_idx - token_id - 1) / gridDim.x;
      token_id += num_iters_to_next_expert * gridDim.x;
      continue;
    }
    const T *x_now = x + token_id * dim;
    OutT *out_now = out + token_id * dim;
    T init_value = static_cast<T>(0.f);
    T x_vals[kNChunks][VecSize] = {init_value};

    load_input<kNChunks, VecSize, UseDiagonalBlockMatrix, T>(
        x_now, x_vals, dim);
#ifdef DEBUG_HARDAMARD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < 1; ++i) {
        printf("chunk_id0: %d\n", i);
        for (int j = 0; j < VecSize; ++j) {
          printf("%f ", (float)x_vals[i][j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    hadamard_mult_thread<kLogVecSize, kNChunks>(x_vals);
#ifdef DEBUG_HARDAMARD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < 1; ++i) {
        printf("chunk_id1: %d, kLogVecSize: %d\n", i, kLogVecSize);
        for (int j = 0; j < VecSize; ++j) {
          printf("%f ", (float)x_vals[i][j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, VecSize>(x_vals);
#ifdef DEBUG_HARDAMARD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < 1; ++i) {
        printf("chunk_id2: %d\n", i);
        for (int j = 0; j < VecSize; ++j) {
          printf("%f ", (float)x_vals[i][j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    if constexpr (kNWarps > 1) {
      // 先让连续的NWARPS个线程拿到其余warps上的数据
      exchange_smem_pre<kNChunks,
                        kChunksPerSmemSize,
                        VecSize,
                        kWarpSize,
                        kNWarps,
                        true,
                        vec_t>(x_vals, smem_exchange);
      // 交叉计算
      hadamard_mult_warp<kLogNWarps, 0, kNChunks, VecSize>(x_vals);
      // 再换回来
      exchange_smem_pre<kNChunks,
                        kChunksPerSmemSize,
                        VecSize,
                        kWarpSize,
                        kNWarps,
                        false,
                        vec_t>(x_vals, smem_exchange);
    }
    if constexpr (kNChunks > 1) {
      if constexpr (kNChunks == 28) {
        hadamard_mult_thread_28_transpose<T, VecSize>(x_vals);
      } else if constexpr (kNChunks == 36) {
        hadamard_mult_thread_36_transpose<T, VecSize>(x_vals);
      } else {
        constexpr int kLogNChunks = cilog2(kNChunks);
        static_assert(1 << kLogNChunks == kNChunks,
                      "kNChunks must be a power of 2");
        hadamard_mult_thread_transpose<kLogNChunks, VecSize>(x_vals);
      }
    }
    if (quant_scales) {
      float quant_scale = quant_scales[expert_id];
      if (shift) {
        smooth_quant_store_output<kNChunks,
                                  VecSize,
                                  UseDiagonalBlockMatrix,
                                  T,
                                  OutT>(out_now,
                                        shift,
                                        smooth,
                                        x_vals,
                                        quant_scale,
                                        quant_round_type,
                                        quant_max_bound,
                                        quant_min_bound,
                                        dim);
      } else {
        quant_store_output<kNChunks, VecSize, UseDiagonalBlockMatrix, T, OutT>(
            out_now,
            x_vals,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            dim);
      }
    } else {
      store_output<kNChunks, VecSize, UseDiagonalBlockMatrix, T>(
          out_now, x_vals, dim);
    }
  }
}

template <typename T,
          typename OutT,
          int kLogN,
          int VecSize,
          int kNChunks,
          int kThreads,
          bool UseDiagonalBlockMatrix>
void MoeFastHardamardImplWrapper(const T *x,
                                 const int64_t *expert_idx_per_token,
                                 const int64_t *recv_expert_count,
                                 const T *shift,
                                 const T *smooth,
                                 const float *quant_scales,
                                 const int quant_round_type,
                                 const float quant_max_bound,
                                 const float quant_min_bound,
                                 const int64_t token_num,
                                 const int64_t dim,
                                 const int num_max_tokens_per_expert,
                                 bool used_in_ep_low_latency,
                                 OutT *out,
                                 cudaStream_t stream) {
  using nv_type = typename nv_type_traits<T>::type;
  using out_type = typename nv_type_traits<OutT>::type;
  constexpr int kNBytes = sizeof(T);
  constexpr int N = 1 << kLogN;  // pad
  constexpr int kSmemSize = std::min(N * kNBytes, 32 * 1024);
  constexpr int kRounds = N * kNBytes / kSmemSize;
  constexpr int kChunksPerSmemSize = kSmemSize / (kThreads * VecSize * kNBytes);
  VLOG(1) << "real_dim: " << dim << ", N:  " << N;
  VLOG(1) << "kNChunks: " << kNChunks;
  VLOG(1) << "kNBytes: " << kNBytes;
  VLOG(1) << "kSmemSize: " << kSmemSize;
  VLOG(1) << "kRounds: " << kRounds;
  VLOG(1) << "kChunksPerSmemSize: " << kChunksPerSmemSize;
  const int dev_id = 0;
  int sm_count;
  int act_blocks_per_sm;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  if (used_in_ep_low_latency) {
    auto masked_kernel =
        masked_moe_fast_hardamard_kernel<nv_type,
                                         out_type,
                                         kThreads,
                                         kNBytes,
                                         VecSize,
                                         N,
                                         kNChunks,
                                         kSmemSize,
                                         kRounds,
                                         kChunksPerSmemSize,
                                         UseDiagonalBlockMatrix>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, masked_kernel, kThreads, kSmemSize);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    dim3 grid;
    grid.x = min(static_cast<int64_t>(num_blocks_per_wave), token_num);
    if constexpr (UseDiagonalBlockMatrix) {
      grid.y = ceil(dim / (kThreads * VecSize));
    }
    masked_kernel<<<grid, kThreads, kSmemSize, stream>>>(
        reinterpret_cast<const nv_type *>(x),
        recv_expert_count,
        reinterpret_cast<const nv_type *>(shift),
        reinterpret_cast<const nv_type *>(smooth),
        quant_scales,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        token_num,
        dim,
        num_max_tokens_per_expert,
        reinterpret_cast<out_type *>(out));
  } else {
    auto kernel = moe_fast_hardamard_kernel<nv_type,
                                            out_type,
                                            kThreads,
                                            kNBytes,
                                            VecSize,
                                            N,
                                            kNChunks,
                                            kSmemSize,
                                            kRounds,
                                            kChunksPerSmemSize,
                                            UseDiagonalBlockMatrix>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, kernel, kThreads, kSmemSize);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    dim3 grid;
    grid.x = min(static_cast<int64_t>(num_blocks_per_wave), token_num);
    if constexpr (UseDiagonalBlockMatrix) {
      grid.y = ceil(dim / (kThreads * VecSize));
    }
    kernel<<<grid, kThreads, kSmemSize, stream>>>(
        reinterpret_cast<const nv_type *>(x),
        expert_idx_per_token,
        reinterpret_cast<const nv_type *>(shift),
        reinterpret_cast<const nv_type *>(smooth),
        quant_scales,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        token_num,
        dim,
        reinterpret_cast<out_type *>(out));
  }
}
