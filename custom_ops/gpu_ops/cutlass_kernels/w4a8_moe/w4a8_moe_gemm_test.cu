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

// #include "paddle/phi/core/enforce.h"
#include "ctime"
#include "iostream"
#include "stdint.h"
#include "stdlib.h"
#include "w4a4_gemm_configs.h"
#include "w4a8_moe_gemm_kernel.h"
#include "weight_process_utils.h"
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <unistd.h>
// #include "paddle/phi/common/data_type.h"
#include "cutlass/numeric_types.h"
#include "cutlass/trace.h"
#define USE_NVTX

#ifdef USE_NVTX
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120900)
#include "nvtx3/nvToolsExt.h"
#else
#include "nvToolsExt.h"
#endif

const uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                           0xff00ffff, 0xffff0000, 0xffffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

using CutlassTileConfig = CutlassTileConfig;
using SplitKStyle = SplitKStyle;
using CutlassGemmConfig = CutlassGemmConfig;

#define PUSH_RANGE(name, cid)                                                  \
  {                                                                            \
    int color_id = cid;                                                        \
    color_id = color_id % num_colors;                                          \
    nvtxEventAttributes_t eventAttrib = {0};                                   \
    eventAttrib.version = NVTX_VERSION;                                        \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                          \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                   \
    eventAttrib.color = colors[color_id];                                      \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                         \
    eventAttrib.message.ascii = name;                                          \
    nvtxRangePushEx(&eventAttrib);                                             \
  }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

// namespace paddle {
// namespace operators{

// template class CutlassIntAIntBInterleavedGemmRunner<half, int8_t>;

// }
// }

template <typename T>
static void PrintMatrix(const T *mat_d, int num, std::string name,
                        int numOfCols) {
  std::vector<T> tmp(num);
  cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);

  std::ofstream outfile;
  outfile.open(name + ".dtxt", std::ios::out);
  std::stringstream ss;

  for (int i = 0; i < num; ++i) {
    if (std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value) {
      ss << static_cast<int>(tmp[i]) << " ";
    } else {
      ss << std::setprecision(8) << tmp[i] << " ";
    }
    if (i % numOfCols == numOfCols - 1) {
      ss << std::endl;
    }
  }
  outfile << ss.str();
  outfile.close();
}

uint as_uint(const float x) { return *(uint *)&x; }
uint16_t ConvertFloat2Half(const float x) {
  const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last
                                          // bit after truncated mantissa
  const uint e = (b & 0x7F800000) >> 23;  // exponent
  const uint m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 =
                                 // 0x00800000-0x00001000 = decimal indicator
                                 // flag - initial rounding
  return (b & 0x80000000) >> 16 |
         (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
         ((e < 113) & (e > 101)) *
             ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
         (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}

inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
#endif
}

inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
#endif
}

float CPUHalfConvert2Float(const uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the half-precision input was 0x1F (max possible value), the
   * biased exponent of the single-precision output must be 0xFF (max possible
   * value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
   * below) rather than by 0x70 suggested by the difference in the exponent bias
   * (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by
   * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
   * necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and
   * NaN would retain their value on at least partially IEEE754-compliant
   * implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
  constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
  // const float exp_scale = 0x1.0p-112f;
  constexpr uint32_t scale_bits = (uint32_t)15 << 23;
  float exp_scale_val;
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  const float exp_scale = exp_scale_val;
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructud single-precision
   * number by 2**(-24), i.e. the same amount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
  constexpr uint32_t magic_mask = UINT32_C(126) << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                          : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static void PrintHalfMatrix(const int16_t *mat_d, int num, std::string name,
                            int numOfCols) {
  std::vector<int16_t> tmp(num);
  cudaMemcpy(tmp.data(), mat_d, sizeof(int16_t) * num, cudaMemcpyDeviceToHost);

  std::ofstream outfile;
  outfile.open(name + ".dtxt", std::ios::out);
  std::stringstream ss;

  for (int i = 0; i < num; ++i) {
    // ss << std::setprecision(8) << static_cast<float>(tmp[i]) << " ";
    // ss << static_cast<int32_t>(tmp[i]) << " ";
    ss << CPUHalfConvert2Float(tmp[i]) << " ";
    if (i % numOfCols == numOfCols - 1) {
      ss << std::endl;
    }
  }
  outfile << ss.str();
  outfile.close();
}

template <typename T>
static void PrintMatrixCPU(const T *mat, int num, std::string name,
                           int numOfCols) {
  std::ofstream outfile;
  outfile.open(name + ".txt", std::ios::out);
  std::stringstream ss;
  for (int i = 0; i < num; ++i) {
    if (std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value) {
      ss << static_cast<int>(mat[i]) << " ";
    } else {
      ss << std::setprecision(8) << mat[i] << " ";
    }
    if (i % numOfCols == numOfCols - 1) {
      ss << std::endl;
    }
  }
  outfile << ss.str();
  outfile.close();
}

static void PrintMatrixCPU_int4(const int8_t *mat, int num, std::string name,
                                int numOfCols) {
  std::ofstream outfile;
  outfile.open(name + ".txt", std::ios::out);
  std::stringstream ss;
  for (int i = 0; i < num / 2; ++i) {
    int32_t output_value = mat[i] & 0x0F;
    ss << static_cast<int>(output_value) << " ";
    output_value = (mat[i] >> 4) & 0x0F;
    ss << static_cast<int>(output_value) << " ";
    if ((i * 2) % numOfCols == numOfCols - 2) {
      ss << std::endl;
    }
  }
  outfile << ss.str();
  outfile.close();
}
template <typename T>
static void PrintHalfMatrixCPU(const T *mat, int num, std::string name,
                               int numOfCols) {
  std::ofstream outfile;
  outfile.open(name + ".txt", std::ios::out);
  std::stringstream ss;
  for (int i = 0; i < num; ++i) {
    ss << static_cast<int>(mat[i]) << " ";
    if (i % numOfCols == numOfCols - 1) {
      ss << std::endl;
    }
  }
  outfile << ss.str();
  outfile.close();
}

template <typename T, typename outputT>
void naive_matmul(const T *a, const T *b, outputT *c, size_t m, size_t n,
                  size_t k) {
  for (int ik = 0; ik < k; ik++) {
    for (int im = 0; im < m; im++) {
      for (int in = 0; in < n; in++) {
        c[im * n + in] += a[im * k + ik] * b[ik * n + in];
      }
    }
  }
}

template <typename T, typename outputT, typename ScaleType = uint16_t>
void naive_matmul_fused_dequantize_nf4(const T *a, const T *b,
                                       const ScaleType *col_scale,
                                       const ScaleType *row_scale,
                                       const int32_t *nf4_look_up_table,
                                       outputT *c, size_t num_experts,
                                       int64_t *total_rows_before_experts,
                                       size_t total_rows, size_t n, size_t k) {
  // PrintMatrixCPU<T>(
  //     a, total_rows * k, "naive_matmul_a", k);
  // PrintMatrixCPU<T>(
  //     b, num_experts*k*n, "naive_matmul_b", n);
  // PrintMatrixCPU<outputT>(
  //     c, total_rows * n, "naive_matmul_c", n);
  // PrintMatrixCPU<ScaleType>(
  //     row_scale, total_rows, "naive_matmul_row_scale", 1);

  // PrintMatrixCPU<ScaleType>(
  //     col_scale, num_experts * n, "naive_matmul_col_scale", n);

  // PrintMatrixCPU<int32_t>(
  //     nf4_look_up_table, 16, "naive_matmul_nf4_lut", 1);
  // std::cout<<"####nf4_look_up_table"<<std::endl;
  // for(int i=0;i<16;++i){
  //   std::cout<<nf4_look_up_table[i]<<" ";
  // }
  // std::cout<<std::endl;
  // static constexpr uint32_t
  // loop_up_table[15]{0x03020100,0x07060504,0x0B0A0908,0x0F0E0D0C}; static
  // constexpr uint32_t
  // loop_up_table[16]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}; const int8_t*
  // loop_up_table_int8 = reinterpret_cast<const int8_t*>(&loop_up_table);
  for (int ie = 0; ie < num_experts; ie++) {
    int im_start, im_end;
    if (ie == 0) {
      im_start = 0;
      im_end = total_rows_before_experts[ie];
    } else {
      im_start = total_rows_before_experts[ie - 1];
      im_end = total_rows_before_experts[ie];
    }
    for (int im = im_start; im < im_end; im++) {
      for (int in = 0; in < n; in++) {
        int32_t accum_val = 0;
        // c[im*n+in]=0;
        for (int ik = 0; ik < k; ik++) {
          int8_t a_val = static_cast<int8_t>(a[im * k + ik]);
          int8_t b_val = static_cast<int8_t>(b[ie * n * k + ik * n + in]);
          // std::cout<<static_cast<int32_t>(a_val)<<", ";
          // std::cout<<static_cast<int32_t>(b[ie * n * k + ik * n + in])<<", ";
          // std::cout<<static_cast<int32_t>(b_val)<<", ";
          // std::cout<<static_cast<int32_t>(nf4_look_up_table[b_val])<<", " <<
          // std::endl; std::cout<<nf4_look_up_table[1]<<", ";
          int32_t b_val_int32 = nf4_look_up_table ? nf4_look_up_table[b_val]
                                                  : static_cast<int32_t>(b_val);
          int32_t matmul_res = static_cast<int32_t>(a_val) * b_val_int32;
          // std::cout<<matmul_res<<", ";
          accum_val += matmul_res;
          // std::cout<<accum_val<<", ";
          // std::cout<<"\n";
        }
        // std::cout<<"\n";
        uint16_t r_val = ConvertFloat2Half(row_scale ? row_scale[im] : 1.0);
        float row_scale_val =
            static_cast<float>(*reinterpret_cast<half *>(&r_val));
        // float row_scale_val = 1.0;
        uint16_t c_val =
            ConvertFloat2Half(col_scale ? col_scale[ie * n + in] * 112 : 1.0);
        float col_scale_val =
            static_cast<float>(*reinterpret_cast<half *>(&c_val));
        // printf("##### (%d,%d) accu_val = %d\n",im,in, accum_val);
        uint16_t res = ConvertFloat2Half(static_cast<float>(accum_val) *
                                         col_scale_val * row_scale_val);
        c[im * n + in] = static_cast<outputT>(*reinterpret_cast<half *>(&res));
      }
    }
  }
  // PrintMatrixCPU<outputT>(
  //   c, total_rows * n, "naive_matmul_c_computed", n);
}

// Author (zhengzekang): we use float to monitor half matmul in CPU.
void CheckHalfDiff(int16_t *device_res, float *host_result, size_t elem_cnt,
                   float atol, float rtol) {
  std::vector<int16_t> device_data(elem_cnt);
  cudaMemcpy(device_data.data(), device_res, sizeof(int16_t) * elem_cnt,
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < elem_cnt; i++) {
    float device_res_val = CPUHalfConvert2Float(device_data[i]);
    float host_res_val = static_cast<float>(host_result[i]);
    float absolute_diff = std::abs(device_res_val - host_res_val);
    bool check_flag = absolute_diff < (atol + rtol * std::abs(host_res_val));

    if (!check_flag) {
      printf("===== Error! ===== \n");
      printf(
          "Here in Idx: %d, CUDA result is: %f, Host result is: %f, absolute "
          "diff val is: %f \n",
          i, device_res_val, host_res_val, absolute_diff);
      return;
    }
  }
  printf("======= Check Success! =======\n");
}

// uint16_t float_to_half(const float x) { // IEEE-754 16-bit floating-point
// format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5,
// +-5.9604645E-8, 3.311 digits
//     const uint b = as_uint(x)+0x00001000; // round-to-nearest-even: add last
//     bit after truncated mantissa const uint e = (b&0x7F800000)>>23; //
//     exponent const uint m = b&0x007FFFFF; // mantissa; in line below:
//     0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial
//     rounding return (b&0x80000000)>>16 |
//     (e>112)*((((e-112)<<10)&0x7C00)|m>>13) |
//     ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; //
//     sign : normalized : denormalized : saturate
// }

template <typename T>
__global__ void CUDAPrintHalfMatrix(T *output, int m, int n) {
  for (int row_idx = 0; row_idx < m; row_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      // printf("%d ", static_cast<int32_t>(static_cast<float>(output[row_idx *
      // n + col_idx])));
      printf("%f ", static_cast<float>(output[row_idx * n + col_idx]));
    }
    printf("\n");
  }
}

CutlassGemmConfig GetGemmConfig(int token_nums,
                                std::vector<int> &gemm_config_tuple) {
  int len_of_gemm_config_tuple = gemm_config_tuple.size();
  if (len_of_gemm_config_tuple == 0) {
    CutlassGemmConfig gemm_config = CutlassGemmConfig{
        CutlassTileConfig::Undefined, SplitKStyle::NO_SPLIT_K, -1, -1};
    return gemm_config;
  }
  CutlassGemmConfig gemm_config = CutlassGemmConfig{
      CutlassTileConfig(gemm_config_tuple[len_of_gemm_config_tuple - 4]),
      SplitKStyle(gemm_config_tuple[len_of_gemm_config_tuple - 3]),
      gemm_config_tuple[len_of_gemm_config_tuple - 2],
      gemm_config_tuple[len_of_gemm_config_tuple - 1]};
  //                    0,1,2,3          ,4            ,5             ,6
  // gemm_config_tuple:[m,n,k,tile_config,split_k_style,split_k_factor,stages]
  for (int i = 0; i < len_of_gemm_config_tuple; i += 7) {
    gemm_config.tile_config =
        CutlassTileConfig(gemm_config_tuple[i + 3]); // tile_config
    gemm_config.split_k_style =
        SplitKStyle(gemm_config_tuple[i + 4]);             // split_k_style
    gemm_config.split_k_factor = gemm_config_tuple[i + 5]; // split_k_factor
    gemm_config.stages = gemm_config_tuple[i + 6];         // stages
    // make sure we have at least one tuned config
    if (token_nums <= gemm_config_tuple[i + 0]) {
      break;
    }
  }
  return gemm_config;
}

template <typename T, typename U = T>
void get_tensor_from_file(const std::string file_path, int64_t numel,
                          T *tensor_ptr) {
  std::fstream datafile;
  datafile.open(file_path, std::ios_base::in | std::ios_base::out);

  int index = 0;
  std::string line;
  while (std::getline(datafile, line)) {
    std::istringstream iss(line);
    if (index == 0) {
      std::cout << file_path << " line zero:" << line << std::endl;
    }
    U number;
    while (iss >> number) {
      tensor_ptr[index] = static_cast<T>(number);
      if (index == 0) {
        std::cout << file_path << ": " << number << "-"
                  << static_cast<U>(tensor_ptr[0]) << std::endl;
      }
      index++;
    }
  }
  std::cout << file_path << ": " << tensor_ptr[0] << std::endl;
  datafile.close();
}

int main(int argc, char *argv[]) {
  std::uniform_real_distribution<float> uniform(-0.02, 0.02);
  std::default_random_engine random_engine(0);

  // m n k
  // argv[1], argv[2], argv[3]
  size_t num_experts = strtol(argv[1], nullptr, 0);
  size_t n = strtol(argv[2], nullptr, 0);
  size_t k = strtol(argv[3], nullptr, 0);
  size_t tokens_per_expert = strtol(argv[4], nullptr, 0);
  size_t total_rows = num_experts * tokens_per_expert;
  std::vector<int64_t> total_rows_before_experts;
  std::cout << "total_rows_before_experts:  ";
  for (int i = 0; i < num_experts; ++i) {
    total_rows_before_experts.push_back(tokens_per_expert * (i + 1));
    std::cout << total_rows_before_experts[i] << " ";
  }
  std::cout << std::endl;

  bool do_check = false;
  if (argc >= 6) {
    do_check = strtol(argv[5], nullptr, 0);
  }
  if (do_check) {
    std::cout << "####do check#####" << std::endl;
  }
  std::cout << "num_experts: " << num_experts << " n: " << n << " k: " << k
            << std::endl;

  bool do_gemm_config_searching = false;
  if (argc >= 7) {
    do_gemm_config_searching = strtol(argv[6], nullptr, 0);
  }
  if (do_gemm_config_searching) {
    std::cout << "####do gemm config searching#####" << std::endl;
  }

  bool is_encryption = false;
  if (argc >= 6) {
    is_encryption = argv[7];
  }

  std::string gemm_config_search_log_file = "";
  if (argc >= 8) {
    gemm_config_search_log_file = argv[8];
  }

  std::ifstream gemm_config_file(gemm_config_search_log_file);

  std::string a_data_file = "";
  if (argc >= 9) {
    a_data_file = argv[9];
  }

  std::string b_data_file = "";
  if (argc >= 10) {
    b_data_file = argv[10];
  }

  std::string row_scale_data_file = "";
  if (argc >= 11) {
    row_scale_data_file = argv[11];
  }

  std::string col_scale_data_file = "";
  if (argc >= 12) {
    col_scale_data_file = argv[12];
  }
  std::vector<int> config_vec;

  if (gemm_config_file.is_open()) {
    std::string line;
    while (std::getline(gemm_config_file, line)) {
      // using printf() in all tests for consistency
      // printf("%s", line.c_str());
      if (line.find("#####best_gemm_config_tuple#####") != std::string::npos) {
        std::cout << line << std::endl;
        std::string config_str = line.substr(32, std::string::npos);
        std::istringstream in_str(config_str);
        int temp;
        while (in_str >> temp) {
          config_vec.push_back(temp);
        }
      }
    }
    gemm_config_file.close();
  }
  // auto best_gemm_config = GetGemmConfig(m, config_vec);
  const auto kWarmTime = 1;
  const auto kTestTime = 100;
  auto mixed_gemm_runner = W4A8MoeGemmRunner<half, int8_t, cutlass::uint4b_t>();

  // int mixgemm_max_size = std::max(m, k);
  int mixgemm_workspace_size_bytes = 1 * 1024 * 1024 * 1024; // 1G workspace
  std::cout << "mixgemm_workspace_size_bytes: " << mixgemm_workspace_size_bytes
            << std::endl;
  char *mixgemm_workspace_data;
  cudaMalloc(&mixgemm_workspace_data, mixgemm_workspace_size_bytes);
  cudaDeviceSynchronize();

  std::cout << "do init of a and b in cpu" << std::endl;
  std::vector<int8_t> a_int(total_rows * k);
  if (do_check) {
    if (a_data_file == "") {
      for (int i = 0; i < total_rows * k; i++) {
        // a_int[i] = 1;
        a_int[i] = rand() % 16;
        // a_int[i] = rand() % 128 - 64;
        // if(i>=k){
        //   // a_int[i] = a_int[i%k];
        //   a_int[i] = 1;
        // }
      }
    } else {
      std::cout << "get a data from: " << a_data_file << std::endl;
      get_tensor_from_file<int8_t, int32_t>(a_data_file, total_rows * k,
                                            a_int.data());
    }
    // PrintMatrixCPU<int8_t>(a_int.data(),total_rows*k,"a_int8_cpu",n);
  }

  std::vector<int8_t> b_int(num_experts * k * n);
  if (do_check) {
    for (int ii = 0; ii < num_experts; ++ii) {
      for (int i = ii * k * n; i < (ii + 1) * k * n; i++) {
        // author zhengzekang
        b_int[i] = rand() % 16;
        // b_int[i] = 1;
        // wangbojun
        // b_int[i] = 0x11;
        // int id_k = i / n;
        // int id_n = i % n;
        // b_int[i] = 0x00;
        // if(id_k < 8 && id_n < 1){
        //   b_int[i]=(i+id_k+1) % 16;
        // }
        // if(id_k >= 64 && id_k < 65 && id_n < 8){
        //   b_int[i]=(i+id_k+1) % 16;
        // }
        // if(id_k < 1 && id_n < 8){
        //   b_int[i]=(i+id_k+1) % 16;
        // }
      }
    }
    // PrintMatrixCPU<int8_t>(b_int.data(),num_experts * k *
    // n,"b_int8_cpu_init",n);
  }

  std::vector<int32_t> nf4_look_up_table(16);
  std::vector<int8_t> nf4_look_up_table_compress(16);
  if (do_check) {
    for (int i = 0; i < 4; ++i) {
      nf4_look_up_table_compress[i] = 0;
    }
    for (int i = 0; i < 16; ++i) {
      int32_t left4i = i << 4;
      int8_t tmp = *reinterpret_cast<int8_t *>(&(left4i));
      int32_t tmp_int32 = static_cast<int32_t>(tmp);
      nf4_look_up_table[i] = tmp_int32;
    }
    for (int i = 0; i < 16; ++i) {
      nf4_look_up_table_compress[i] =
          (static_cast<int8_t>(nf4_look_up_table[i]));
    }
    std::cout << "####nf4_look_up_table" << std::endl;
    for (int i = 0; i < 16; ++i) {
      std::cout << nf4_look_up_table[i] << " ";
    }
    std::cout << std::endl;
  }

  // printf("nf4 compress table:%08x,%08x,%08x,%08x
  // \n",nf4_look_up_table_compress[0],
  //                                                    nf4_look_up_table_compress[1],
  //                                                    nf4_look_up_table_compress[2],
  //                                                    nf4_look_up_table_compress[3]);

  std::cout << "finish init of a and b in cpu" << std::endl;
  cudaDeviceSynchronize();

  std::vector<int8_t> packed_b_int(num_experts * k * n / 2);
  if (do_check) {
    for (int ie = 0; ie < num_experts; ++ie) {
      int offset = ie * k * n / 2;
      for (int packed_i = 0; packed_i < k * n / 2; packed_i++) {
        packed_b_int[offset + packed_i] = 0;
        packed_b_int[offset + packed_i] |=
            b_int[(offset + packed_i) * 2] & 0x0f;
        packed_b_int[offset + packed_i] |=
            (b_int[(offset + packed_i) * 2 + 1] & 0x0f) << 4;
      }
    }
  }

  std::vector<int8_t> b_int_processed(num_experts * k * n / 2);
  std::vector<int8_t> b_int_processed_2(num_experts * k * n / 2);
  std::vector<int8_t> b_int_processed_3(num_experts * k * n / 2);
  if (do_check) {
    printf("do check\n");
    if (b_data_file == "") {
      for (int ie = 0; ie < num_experts; ie++) {
        // PrintMatrixCPU_int4(packed_b_int.data(),num_experts*k*n,"w4a8_packed_b_int4",n);
        permute_B_rows_for_mixed_gemm_int4<4>(
            b_int_processed.data() + ie * k * n / 2,
            packed_b_int.data() + ie * k * n / 2, std::vector<size_t>{k, n},
            (int64_t)80);

        // PrintMatrixCPU_int4(b_int_processed.data(),num_experts*k*n,"w4a8_permuted_int4",n);

        std::cout << "before subbyte_transpose_impl_int4" << std::endl;
        subbyte_transpose_impl_int4(b_int_processed_2.data() + ie * k * n / 2,
                                    b_int_processed.data() + ie * k * n / 2,
                                    std::vector<size_t>{k, n});
        // PrintMatrixCPU_int4(b_int_processed_2.data(),num_experts*k*n,"w4a8_subbyte_transpose_impl_int4",k);

        interleave_column_major_tensor_int4(
            b_int_processed_3.data() + ie * k * n / 2,
            b_int_processed_2.data() + ie * k * n / 2,
            std::vector<size_t>{k, n});
        // PrintMatrixCPU_int4(b_int_processed_3.data(),num_experts*k*n,"w4a8_interleave_column_major_tensor_int4",k);

        add_bias_and_interleave_int4s_inplace(
            b_int_processed_3.data() + ie * k * n / 2, k * n);
      }
    } else {
      get_tensor_from_file<int8_t, int32_t>(
          b_data_file, num_experts * k * n / 2, b_int_processed_3.data());
    }
    // PrintMatrixCPU_int4(b_int_processed_3.data(),
    //                     num_experts*k*n/2,
    //                     "w4a8_add_bias_and_interleave_int4s_inplace",
    //                     k);

    // PrintMatrixCPU<int8_t>(b_int_processed_3.data(),num_experts*k*n/2,"b_int8_cpu",n);

    // TODO(zhengzekang): temporary use uint16_t instead of half.
  }
  std::cout << "done weight interleaved;" << std::endl;
  std::cout << "begin init c: m: " << total_rows << " n: " << n << std::endl;

  std::cout << "#### 1" << std::endl;
  std::vector<float> c_float(total_rows * n);

  if (do_check) {
    for (int i = 0; i < c_float.size(); i++) {
      c_float[i] = 0.0f;
    }
  }

  std::cout << "#### 2" << std::endl;

  std::vector<uint16_t> c_half(total_rows * n);
  if (do_check) {
    for (int i = 0; i < c_half.size(); i++) {
      c_half[i] = ConvertFloat2Half(c_float[i]);
    }
  }

  std::cout << "#### 3" << std::endl;

  std::vector<float> row_scale_float(total_rows);
  if (do_check) {
    if (row_scale_data_file == "") {
      for (int32_t i = 0; i < row_scale_float.size(); i++) {
        // row_scale_float[i] = 0.1;
        // row_scale_float[i] = uniform(random_engine) * 0.1;
        row_scale_float[i] = 1.0;
        // if(i>0){
        //   row_scale_float[i] = row_scale_float[0];
        // }
      }
    } else {
      get_tensor_from_file<float>(row_scale_data_file, total_rows,
                                  row_scale_float.data());
    }
    // PrintMatrixCPU<float>(row_scale_float.data(),total_rows,"row_scale_float_cpu",total_rows);
  }

  std::vector<float> col_scale_float(num_experts * n);
  if (do_check) {
    if (col_scale_data_file == "") {
      for (int32_t i = 0; i < col_scale_float.size(); i++) {
        // col_scale_float[i] = 0.04;
        col_scale_float[i] =
            uniform(random_engine) * 0.06 * uniform(random_engine) * 0.1;
        // col_scale_float[i] = 0;
        // if(i<1){
        //   col_scale_float[i] = 1;
        // }
      }
    } else {
      get_tensor_from_file<float>(col_scale_data_file, num_experts * n,
                                  col_scale_float.data());
    }
    // PrintMatrixCPU<float>(col_scale_float.data(),num_experts*n,"col_scale_float_cpu",n);
  }

  std::vector<uint16_t> row_scale_half(total_rows);
  if (do_check) {
    for (int32_t i = 0; i < row_scale_half.size(); i++) {
      // row_scale_float[i] = 1;
      row_scale_half[i] = ConvertFloat2Half(row_scale_float[i]);
    }
  }

  std::vector<uint16_t> col_scale_half(num_experts * n);
  if (do_check) {
    for (int32_t i = 0; i < col_scale_float.size(); i++) {
      // col_scale_float[i] = 1;
      col_scale_half[i] = ConvertFloat2Half(col_scale_float[i]);
    }
  }

  std::cout << "done c init" << std::endl;

  void *d_a_int;
  void *d_b_int;
  void *d_c_int;
  void *d_col_scale_half;
  void *d_row_scale_half;
  void *d_nf4_look_up_table;
  void *d_total_rows_before_experts;
  cudaMalloc(&d_a_int, total_rows * k * sizeof(int8_t));
  cudaMalloc(&d_b_int, num_experts * k * n / 2 * sizeof(int8_t));
  cudaMalloc(&d_c_int, total_rows * n * sizeof(uint16_t));

  cudaMalloc(&d_row_scale_half, total_rows * sizeof(uint16_t));
  cudaMalloc(&d_col_scale_half, num_experts * n * sizeof(uint16_t));
  cudaMalloc(&d_nf4_look_up_table, 4 * sizeof(uint32_t));
  cudaMalloc(&d_total_rows_before_experts, num_experts * sizeof(int64_t));
  cudaMemcpy(d_a_int, a_int.data(), total_rows * k * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_int, b_int_processed_3.data(),
             num_experts * k * n / 2 * sizeof(int8_t), cudaMemcpyHostToDevice);

  cudaMemcpy(d_row_scale_half, row_scale_half.data(),
             total_rows * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_scale_half, col_scale_half.data(),
             num_experts * n * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nf4_look_up_table, nf4_look_up_table_compress.data(),
             4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c_int, c_half.data(), total_rows * n * sizeof(uint16_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_total_rows_before_experts, total_rows_before_experts.data(),
             num_experts * sizeof(int64_t), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  cudaError_t result = cudaGetLastError();

  if (result != cudaSuccess) {
    // Call cudaGetLastError() to clear the error bit
    CUTLASS_TRACE_HOST(" before kernel  grid launch failed with error "
                       << cudaGetErrorString(result));
  } else {
    CUTLASS_TRACE_HOST("after init");
  }

  std::cout << "=== do warm up for " << kWarmTime << " times" << std::endl;
  auto test_config =
      CutlassGemmConfig{CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                        SplitKStyle::NO_SPLIT_K, 1, 5};
  std::cout << "=== do warm up end" << std::endl;
  for (int i = 0; i < kWarmTime; i++) {
    printf("warm up %d\n", i);
    mixed_gemm_runner.moe_gemm(
        reinterpret_cast<const int8_t *>(d_a_int),
        reinterpret_cast<const cutlass::uint4b_t *>((void *)d_b_int),
        cutlass::epilogue::QuantMode::PerTokenChannelQuant,
        reinterpret_cast<const half *>(d_col_scale_half),
        reinterpret_cast<const half *>(d_row_scale_half),
        reinterpret_cast<const int32_t *>(d_nf4_look_up_table),
        reinterpret_cast<half *>(d_c_int),
        reinterpret_cast<int64_t *>(d_total_rows_before_experts), -1,
        total_rows, n, k, mixgemm_workspace_data, mixgemm_workspace_size_bytes,
        num_experts, 0, test_config);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "error: " << cudaGetErrorString(err) << std::endl;
    } else {
      std::cout << "cuda success" << std::endl;
    }
  }

  // if (gemm_config_search_log_file != ""){
  //   if(kTestTime > 0){
  //     cudaDeviceSynchronize();
  //     auto start = std::chrono::system_clock::now();
  //     for (int i = 0; i < 1; i++) {
  //       if(i == 0){
  //         std::string nvtx_name = "int4_gemm_" + std::to_string(m) + "-"
  //                                         + std::to_string(n) + "-"
  //                                         + std::to_string(k) + "-"
  //                                         +
  //                                         std::to_string(static_cast<std::underlying_type<CutlassTileConfig>::type>(best_gemm_config.tile_config))
  //                                         + "-"
  //                                         +
  //                                         std::to_string(static_cast<std::underlying_type<CutlassTileConfig>::type>(best_gemm_config.split_k_style))+"-"
  //                                                 +
  //                                                 std::to_string(best_gemm_config.split_k_factor)+"-"
  //                                                 +
  //                                                 std::to_string(best_gemm_config.stages);
  //         PUSH_RANGE(nvtx_name.c_str(), 1)
  //       }
  //   mixed_gemm_runner.moe_gemm(reinterpret_cast<const int8_t*>(d_a_int),
  //                          reinterpret_cast<const
  //                          cutlass::uint4b_t*>((void*)d_b_int),
  //                          QuantMode::PerTokenChannelQuant,
  //                          reinterpret_cast<const half*>(d_col_scale_half),
  //                          reinterpret_cast<const half*>(d_row_scale_half),
  //                          reinterpret_cast<const
  //                          int32_t*>(d_nf4_look_up_table),
  //                          reinterpret_cast<half*>(d_c_int),
  //                          total_rows_before_exports,
  //                          m,
  //                          n,
  //                          k,
  //                          mixgemm_workspace_data,
  //                          mixgemm_workspace_size_bytes,
  //                          num_experts,
  //                          0,
  //                          test_gemm_config);
  //       if(i == 0){
  //         POP_RANGE
  //       }
  //     }
  //     cudaDeviceSynchronize();
  //     auto stop = std::chrono::system_clock::now();
  //     auto duration =
  //     std::chrono::duration_cast<std::chrono::microseconds>((stop - start));
  //     // std::cout<<"avg time for "<<kTestTime<<"
  //     run:"<<duration.count()/(float)kTestTime<<" microseconds."<<std::endl;
  //   }
  //   cudaDeviceSynchronize();
  //   return 0;
  // }

  CutlassGemmConfig best_config = test_config;
  float best_time = 999999999;
  if (do_gemm_config_searching) {
    std::cout << "====== do_gemm_config_searching ====" << std::endl;
    std::vector<CutlassTileConfig> all_cutlass_tile_configs{
        CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
        CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
        CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
        CutlassTileConfig::CtaShape32x256x64_WarpShape32x64x64,
        CutlassTileConfig::CtaShape64x256x64_WarpShape64x64x64,
        CutlassTileConfig::CtaShape32x512x64_WarpShape32x128x64,
        CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64,
    };
    std::vector<SplitKStyle> all_split_k_style{SplitKStyle::NO_SPLIT_K};

    for (auto &tile_config : all_cutlass_tile_configs) {
      for (auto &split_k_style : all_split_k_style) {
        for (int stages = 3; stages <= 7; ++stages) {
          for (int split_k_factor = 1; split_k_factor <= 1;
               split_k_factor *= 2) {
            auto test_gemm_config = CutlassGemmConfig{
                tile_config, split_k_style, split_k_factor, stages};
            cudaEvent_t begin, end;
            cudaDeviceSynchronize();
            cudaEventCreate(&begin);
            cudaEventCreate(&end);
            cudaEventRecord(begin, 0);
            for (int i = 0; i < kTestTime; ++i) {

              mixed_gemm_runner.moe_gemm(
                  reinterpret_cast<const int8_t *>(d_a_int),
                  reinterpret_cast<const cutlass::uint4b_t *>((void *)d_b_int),
                  cutlass::epilogue::QuantMode::PerTokenChannelQuant,
                  reinterpret_cast<const half *>(d_col_scale_half),
                  reinterpret_cast<const half *>(d_row_scale_half),
                  reinterpret_cast<const int32_t *>(d_nf4_look_up_table),
                  reinterpret_cast<half *>(d_c_int),
                  reinterpret_cast<int64_t *>(d_total_rows_before_experts), -1,
                  total_rows, n, k, mixgemm_workspace_data,
                  mixgemm_workspace_size_bytes, num_experts, 0,
                  test_gemm_config);
            }
            cudaEventRecord(end, 0);
            auto cuda_error = cudaDeviceSynchronize();
            float cost_time;
            cudaEventElapsedTime(&cost_time, begin, end);
            float avg_time = cost_time / static_cast<float>(kTestTime) * 1000;
            if (cuda_error != cudaSuccess) {
              avg_time = 999999999;
              std::cout
                  << "#### test gemm_config, error "
                  << " with split-k factor: " << test_gemm_config.split_k_factor
                  << " tile_config: "
                  << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                         test_gemm_config.tile_config)
                  << " split_k_style: "
                  << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                         test_gemm_config.split_k_style)
                  << " stages: " << test_gemm_config.stages << std::endl;
            }
            std::cout
                << "#### test gemm_config, avg_time: " << avg_time
                << " with split-k factor: " << test_gemm_config.split_k_factor
                << " tile_config: "
                << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                       test_gemm_config.tile_config)
                << " split_k_style: "
                << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                       test_gemm_config.split_k_style)
                << " stages: " << test_gemm_config.stages << std::endl;

            if (avg_time < best_time) {
              best_time = avg_time;
              best_config = test_gemm_config;
            }
          }
        }
      }
    }
  }
  std::cout << "#### best gemm_config for total_rows: " << total_rows
            << "num_experts:" << num_experts << " n: " << n << " k: " << k
            << " avg_time: " << best_time
            << " with split-k factor: " << best_config.split_k_factor
            << " tile_config: "
            << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                   best_config.tile_config)
            << " split_k_style: "
            << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                   best_config.split_k_style)
            << " stages: " << best_config.stages << std::endl;
  std::cout << "#####best_gemm_config_tuple##### " << total_rows << " " << n
            << " " << k << " " << num_experts << " "
            << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                   best_config.tile_config)
            << " "
            << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                   best_config.split_k_style)
            << " " << best_config.split_k_factor << " " << best_config.stages
            << std::endl;

  // std::string output_config_path = is_encryption ?
  // "moe_w4a8_tuned_config.config" : "moe_w4a8_tuned_config.csv";
  std::string output_config_path = "moe_w4a8_tuned_config.csv";
  int fd =
      open(output_config_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd == -1) {
    perror("open error");
    return 1;
  }
  std::ofstream outfile;
  if (flock(fd, LOCK_EX) == -1) {
    perror("flock error");
    close(fd);
    return 1;
  }
  outfile.open(output_config_path, std::ios::app);
  outfile << total_rows << "," << n << "," << k << "," << num_experts << ","
          << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                 best_config.tile_config)
          << ","
          << static_cast<std::underlying_type<CutlassTileConfig>::type>(
                 best_config.split_k_style)
          << "," << best_config.split_k_factor << "," << best_config.stages
          << "\n";

  // if (!is_encryption) {
  //   outfile << tokens_per_expert << ","
  //           << n << ","
  //           << k << ","
  //           <<
  //           static_cast<std::underlying_type<CutlassTileConfig>::type>(best_config.tile_config)
  //           << ","
  //           <<
  //           static_cast<std::underlying_type<CutlassTileConfig>::type>(best_config.split_k_style)<<
  //           ","
  //           << best_config.split_k_factor << ","
  //           << best_config.stages <<"\n";
  // } else {
  //   std::stringstream ss;
  //   ss << tokens_per_expert << ","
  //     << n << ","
  //     << k << ","
  //     <<
  //     static_cast<std::underlying_type<CutlassTileConfig>::type>(best_config.tile_config)
  //     << ","
  //     <<
  //     static_cast<std::underlying_type<SplitKStyle>::type>(best_config.split_k_style)
  //     << ","
  //     << best_config.split_k_factor << ","
  //     << best_config.stages;
  //   std::string encrypted_str = paddle::operators::base64_encode(ss.str());
  //   outfile << encrypted_str << "\n";
  // }
  outfile.flush();
  if (flock(fd, LOCK_UN) == -1) {
    perror("flock error (unlock)");
    // 注意：即使解锁失败，也应尽量关闭文件描述符
  }
  outfile.close();
  close(fd);

  if (do_check) {
    std::cout << "=== do accuracy check " << std::endl;
    cudaMemset(d_c_int, 0, total_rows * n * sizeof(uint16_t));
    PrintHalfMatrix(static_cast<int16_t *>(d_c_int), total_rows * n,
                    "CUDA_c_dequantize_fp16_output_before_gemm", n);

    mixed_gemm_runner.moe_gemm(
        reinterpret_cast<const int8_t *>(d_a_int),
        reinterpret_cast<const cutlass::uint4b_t *>((void *)d_b_int),
        cutlass::epilogue::QuantMode::PerChannelQuant,
        reinterpret_cast<const half *>(d_col_scale_half),
        nullptr, // reinterpret_cast<const half*>(d_row_scale_half),
        nullptr, // reinterpret_cast<const int32_t*>(d_nf4_look_up_table),
        reinterpret_cast<half *>(d_c_int),
        reinterpret_cast<int64_t *>(d_total_rows_before_experts), -1,
        total_rows, n, k, mixgemm_workspace_data, mixgemm_workspace_size_bytes,
        num_experts, 0);
    cudaDeviceSynchronize();
    // PrintMatrix<int32_t>(reinterpret_cast<const
    // int32_t*>(d_nf4_look_up_table),4,"d_nf4_look_up_table",1);
    printf("##### d_nf4_look_up_table address: %p \n", d_nf4_look_up_table);

    naive_matmul_fused_dequantize_nf4<int8_t, float, float>(
        a_int.data(), b_int.data(), col_scale_float.data(),
        nullptr, // row_scale_float.data(),
        nullptr, // nf4_look_up_table.data(),
        c_float.data(), num_experts, total_rows_before_experts.data(),
        total_rows, n, k);
    PrintMatrixCPU<float>(c_float.data(), total_rows * n,
                          "CPU_c_fake_fp16_dequantize_output_base", n);
    PrintHalfMatrix(static_cast<int16_t *>(d_c_int), total_rows * n,
                    "CUDA_c_dequantize_fp16_output", n);
    CheckHalfDiff(static_cast<int16_t *>(d_c_int), c_float.data(),
                  total_rows * n, 1e-4, 1e-2);
  }

  // if(kTestTime > 0){
  //   cudaDeviceSynchronize();
  //   auto start = std::chrono::system_clock::now();
  //   for (int i = 0; i < kTestTime; i++) {
  //     if(i == 0){
  //       std::string nvtx_name = "int4_gemm_" +
  //       std::to_string(tokens_per_expert) + "-"
  //                                       + std::to_string(n) + "-"
  //                                       + std::to_string(k) + "-"
  //                                       +
  //                                       std::to_string(static_cast<std::underlying_type<CutlassTileConfig>::type>(best_config.tile_config))
  //                                       + "-"
  //                                       +
  //                                       std::to_string(static_cast<std::underlying_type<CutlassTileConfig>::type>(best_config.split_k_style))+"-"
  //                                               +
  //                                               std::to_string(best_config.split_k_factor)+"-"
  //                                               +
  //                                               std::to_string(best_config.stages);
  //       PUSH_RANGE(nvtx_name.c_str(), 1)
  //     }
  //   mixed_gemm_runner.moe_gemm(reinterpret_cast<const int8_t*>(d_a_int),
  //                         reinterpret_cast<const
  //                         cutlass::uint4b_t*>((void*)d_b_int),
  //                         cutlass::epilogue::QuantMode::PerTokenChannelQuant,
  //                         reinterpret_cast<const half*>(d_col_scale_half),
  //                         reinterpret_cast<const half*>(d_row_scale_half),
  //                         reinterpret_cast<const
  //                         int32_t*>(d_nf4_look_up_table),
  //                         reinterpret_cast<half*>(d_c_int),
  //                         reinterpret_cast<int64_t*>(d_total_rows_before_experts),
  //                         total_rows,
  //                         n,
  //                         k,
  //                         mixgemm_workspace_data,
  //                         mixgemm_workspace_size_bytes,
  //                         num_experts,
  //                         0,
  //                         best_config);
  //     if(i == 0){
  //       POP_RANGE
  //     }
  //   }
  //   cudaDeviceSynchronize();
  //   auto stop = std::chrono::system_clock::now();
  //   auto duration =
  //   std::chrono::duration_cast<std::chrono::microseconds>((stop - start));
  //   std::cout<<"avg time for "<<kTestTime<<"
  //   run:"<<duration.count()/(float)kTestTime<<" microseconds."<<std::endl;
  // }
  // cudaDeviceSynchronize();
  return 0;
}
