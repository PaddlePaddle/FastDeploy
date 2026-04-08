// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include "helper.h"
#include "cutlass_kernels/cutlass_preprocessors.h"

template <typename T>
inline T xabs(const T x) {
  return x < static_cast<T>(0.0) ? -x : x;
}

template <typename T, typename ScaleT>
void per_channel_scale(
    ScaleT* scale, const T* input, size_t m, size_t n, float bound) {
  for (size_t i = 0; i < n; ++i) {
    float max = static_cast<float>(input[i]);
    for (size_t j = 0; j < m; ++j) {
      max = static_cast<float>(xabs(input[j * n + i])) > max
                ? static_cast<float>(xabs(input[j * n + i]))
                : max;
    }
    scale[i] = static_cast<ScaleT>(max / bound);
  }
}

template <typename T, typename ScaleT>
void per_channel_quant(int8_t* output,
                       const T* input,
                       const ScaleT* scale,
                       size_t num_rows,
                       size_t num_cols) {
  size_t bytes_per_out_col = num_cols / 2;
  for (size_t ii = 0; ii < num_rows; ++ii) {
    int8_t* current_quantized_weight_row = output + ii * bytes_per_out_col;
    const T* current_weight_row = input + ii * num_cols;
    for (size_t jj = 0; jj < bytes_per_out_col; ++jj) {
        // We will pack two int4 elements per iteration of the inner loop.
        int8_t packed_int4s = 0;
        for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
            const size_t input_idx = 2 * jj + packed_idx;
            if (input_idx < num_cols) {
                const float col_scale = static_cast<float>(scale[input_idx]);
                const float weight_elt =
                    static_cast<float>(current_weight_row[input_idx]);
                const float scaled_weight = round(weight_elt / col_scale);
                int int_weight = static_cast<int>(scaled_weight);
                const int8_t clipped_weight = std::max(-7, std::min(7, int_weight));
                // Kill the sign extension bits (hence 0x0F mask) then shift to
                // upper bits if packing the second int4 and or the bits into the
                // final result.
                packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
            }
        }
        current_quantized_weight_row[jj] = packed_int4s;
    }
  }
}


template <typename T, typename ScaleT>
void group_wise_quant(int8_t* output,
                      const T* input,
                      const ScaleT* scale,
                      size_t num_rows,
                      size_t num_cols,
                      const int group_size) {
  size_t bytes_per_out_col = num_cols / 2;
  for (size_t ii = 0; ii < num_rows; ++ii) {
    int8_t* current_quantized_weight_row = output + ii * bytes_per_out_col;
    const T* current_weight_row = input + ii * num_cols;
    for (size_t jj = 0; jj < bytes_per_out_col; ++jj) {
        // We will pack two int4 elements per iteration of the inner loop.
        int8_t packed_int4s = 0;
        for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
          const size_t input_idx = 2 * jj + packed_idx;
          if (input_idx < num_cols) {
            size_t scale_cur_offset = input_idx + (ii / group_size) * num_cols;
            const float col_scale = static_cast<float>(scale[scale_cur_offset]);
            const float weight_elt =
                static_cast<float>(current_weight_row[input_idx]);
            const float scaled_weight = round(weight_elt / col_scale);
            int int_weight = static_cast<int>(scaled_weight);
            const int8_t clipped_weight = std::max(-7, std::min(7, int_weight));
            // Kill the sign extension bits (hence 0x0F mask) then shift to
            // upper bits if packing the second int4 and or the bits into the
            // final result.
            packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
          }
        }
        current_quantized_weight_row[jj] = packed_int4s;
    }
  }
}

template <typename T, typename ScaleT>
void group_wise_scale(ScaleT* scale,
                      const T* input,
                      size_t m,
                      size_t n,
                      float bound,
                      size_t group_size) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; j += group_size) {
      float max = static_cast<float>(0.f);
      for (size_t k = 0; k < group_size && j + k < m; ++k) {
        max = static_cast<float>(xabs(input[(j + k) * n + i])) > max
                  ? static_cast<float>(xabs(input[(j + k) * n + i]))
                  : max;
      }
      scale[static_cast<int>(j / group_size) * n + i] =
          static_cast<ScaleT>(max / bound);
    }
  }
}

std::vector<paddle::Tensor> Fp8Int4WeightQuantizeKernel(const paddle::Tensor &input,
    int groupsize,
    std::string scale_dtype) {
    auto input_cpu = input.copy_to(paddle::CPUPlace(), false);
    auto shape = input.shape();
    auto k = static_cast<size_t>(shape[0]);
    auto n = static_cast<size_t>(shape[1]);

    paddle::Tensor scale;
    auto packed_int4 = paddle::full({shape[0] * shape[1] / 2}, 0, paddle::DataType::INT8, paddle::CPUPlace());
    if (scale_dtype == "bfloat16") {
        if (groupsize > 0) {
            scale = paddle::full({shape[0] / groupsize * shape[1]}, 1.0, paddle::DataType::BFLOAT16, paddle::CPUPlace());
            group_wise_scale(scale.data<phi::dtype::bfloat16>(), input_cpu.data<float>(), k, n, 7.0f, groupsize);
            group_wise_quant(packed_int4.data<int8_t>(),
                            input_cpu.data<float>(),
                            scale.data<phi::dtype::bfloat16>(),
                            k,
                            n,
                            groupsize);
        } else {
            scale = paddle::full({shape[1]}, 1.0, paddle::DataType::BFLOAT16, paddle::CPUPlace());
            per_channel_scale(scale.data<phi::dtype::bfloat16>(), input_cpu.data<float>(), k, n, 7.0f);
            per_channel_quant(packed_int4.data<int8_t>(),
                              input_cpu.data<float>(),
                              scale.data<phi::dtype::bfloat16>(),
                              k,
                              n);
        }
    } else if (scale_dtype == "float16") {
        if (groupsize > 0) {
            scale = paddle::full({shape[0] / groupsize * shape[1]}, 1.0, paddle::DataType::FLOAT16, paddle::CPUPlace());
            group_wise_scale(scale.data<phi::dtype::float16>(), input_cpu.data<float>(), k, n, 7.0f, groupsize);
            group_wise_quant(packed_int4.data<int8_t>(),
                input_cpu.data<float>(),
                scale.data<phi::dtype::float16>(),
                k,
                n,
                groupsize);
        } else {
            scale = paddle::full({shape[1]}, 1.0, paddle::DataType::FLOAT16, paddle::CPUPlace());
            per_channel_scale(scale.data<phi::dtype::float16>(), input_cpu.data<float>(), k, n, 7.0f);
            per_channel_quant(packed_int4.data<int8_t>(),
                    input_cpu.data<float>(),
                    scale.data<phi::dtype::float16>(),
                    k,
                    n);
        }
    }

    auto out = paddle::full({shape[1] / 2, shape[0]}, 0, paddle::DataType::INT8, paddle::CPUPlace());
    preprocess_weights_for_mixed_gemm(
        out.data<int8_t>(),
        packed_int4.data<int8_t>(),
        {k, n},
        kernels::cutlass_kernels::QuantType::W4_AFP8,
        false);
    return {out, scale};
}

std::vector<std::vector<int64_t>> Fp8Int4WeightQuantizeInferShape(
    const std::vector<int64_t>& input_shape,
    int groupsize,
    std::string scale_dtype) {
    std::vector<int64_t> out_shape = {input_shape[1] / 2, input_shape[0]};
    std::vector<int64_t> scale_shape;
    if (groupsize > 0) {
        scale_shape = {input_shape[0] / groupsize, input_shape[1]};
    } else {
        scale_shape = {input_shape[1]};
    }
    return {out_shape, scale_shape};
}

std::vector<paddle::DataType> Fp8Int4WeightQuantizeInferDtype(
    const paddle::DataType& input_type,
    int groupsize,
    std::string scale_dtype) {
    paddle::DataType scale_data_type;
    if (scale_dtype == "bfloat16") {
        scale_data_type = paddle::DataType::BFLOAT16;
    } else if (scale_dtype == "float16") {
        scale_data_type = paddle::DataType::FLOAT16;
    } else {
        PD_THROW(
            "scaled_gemm_f8_i4_f16_half_gemm_fused only support bfloat16 and float16 output");
    }
    return {paddle::DataType::INT8, scale_data_type};
}


PD_BUILD_STATIC_OP(scaled_gemm_f8_i4_f16_weight_quantize)
    .Inputs({"input"})
    .Attrs({"groupsize: int",
            "scale_dtype: std::string"})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(Fp8Int4WeightQuantizeKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(Fp8Int4WeightQuantizeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Fp8Int4WeightQuantizeInferDtype));
