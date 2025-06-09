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
#include "cuda_bf16.h"
#include "common/configManager.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

using Pointer = void*;
using ConstPointer = void const*;

struct Params
{
    Pointer act;
    Pointer weight;
    Pointer scales;
    Pointer zeros;
    Pointer bias;
    Pointer out;
    float alpha;
    int m;
    int n;
    int k;
    int groupsize;
    bool apply_alpha_in_advance;

    Params(ConstPointer _act, ConstPointer _weight, ConstPointer _scales, ConstPointer _zeros,
        ConstPointer _bias, Pointer _out, float _alpha, int _m, int _n, int _k, int _groupsize,
        bool _apply_alpha_in_advance = false)
        : act(const_cast<Pointer>(_act))
        , weight(const_cast<Pointer>(_weight))
        , scales(const_cast<Pointer>(_scales))
        , zeros(const_cast<Pointer>(_zeros))
        , bias(const_cast<Pointer>(_bias))
        , out(_out)
        , alpha(_alpha)
        , m(_m)
        , n(_n)
        , k(_k)
        , groupsize(_groupsize)
        , apply_alpha_in_advance(_apply_alpha_in_advance)
    {
    }
};

template <typename T, cutlass::WeightOnlyQuantOp QuantOp>
void scaled_gemm_f8_i4_f16_launcher(Params& params,
                                    phi::Place place,
                                    cudaStream_t stream) {
    auto runner = std::make_shared<kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t, QuantOp, half, T, T>>();
    int ws_size = runner->getWorkspaceSize(params.m, params.n, params.k);
    phi::Allocator* allocator = paddle::GetAllocator(place);
    auto ws = allocator->Allocate(ws_size)->ptr();
    cutlass_extensions::CutlassGemmConfig best_config;
    if (getenv("FLAGS_fastdeploy_op_configs")) {
        std::string fastdeploy_op_configs = getenv("FLAGS_fastdeploy_op_configs");
        if (fastdeploy_op_configs == "tune") {
            auto& ConfigManager = ConfigManager::get_instance();
            auto configs = runner->getConfigs(params.k);
            best_config = configs[0];
            cudaEvent_t begin, end;
            cudaEventCreate(&begin);
            cudaEventCreate(&end);
            float fast_time = 1e8;
            printf("Begin tune cutlass kernel for [%d] confis...\n", static_cast<int>(configs.size()));
            for (int c_idx = 0; c_idx < static_cast<int>(configs.size()); ++c_idx) {
                auto config = configs[c_idx];
                for (int i = 0; i < 5; ++i) {
                    runner->gemm(params.act,
                                params.weight,
                                params.scales,
                                params.zeros,
                                params.bias,
                                params.alpha,
                                params.out,
                                params.m,
                                params.n,
                                params.k,
                                params.groupsize,
                                config,
                                ws,
                                ws_size,
                                stream);
                }
                cudaEventRecord(begin, stream);
                for (int i = 0; i < 15; ++i) {
                    runner->gemm(params.act,
                                params.weight,
                                params.scales,
                                params.zeros,
                                params.bias,
                                params.alpha,
                                params.out,
                                params.m,
                                params.n,
                                params.k,
                                params.groupsize,
                                config,
                                ws,
                                ws_size,
                                stream);
                }
                cudaEventRecord(end, stream);
                cudaEventSynchronize(end);
                float time;
                cudaEventElapsedTime(&time, begin, end);
                // std::cout << config.toString() << "time: " << time << std::endl;
                if (time < fast_time) {
                    fast_time = time;
                    best_config = config;
                }
            }
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
            ConfigManager.update("scaled_gemm_f8_i4_f16", params.m, params.n, params.k, best_config.toString());
        } else {
            if (!std::filesystem::exists(fastdeploy_op_configs)) {
                PADDLE_THROW(phi::errors::Fatal("Warning: The file \"" + fastdeploy_op_configs + "\" does not exist in the specified path." ));
            } else {
                auto& ConfigManager = ConfigManager::get_instance(fastdeploy_op_configs);
                auto best_config_string = ConfigManager.get_best_config("scaled_gemm_f8_i4_f16", params.m, params.n, params.k);
                if (!best_config_string.empty()) {
                    best_config.fromString(best_config_string);
                }
            }
        }
    }
    runner->gemm(params.act,
                params.weight,
                params.scales,
                params.zeros,
                params.bias,
                params.alpha,
                params.out,
                params.m,
                params.n,
                params.k,
                params.groupsize,
                best_config,
                ws,
                ws_size,
                stream);
}

std::vector<paddle::Tensor> scaled_gemm_f8_i4_f16(
                                const paddle::Tensor& x,
                                const paddle::Tensor& y,
                                const paddle::Tensor& scale,
                                const paddle::optional<paddle::Tensor>& zero_points,
                                const paddle::optional<paddle::Tensor>& bias,
                                int groupsize,
                                float out_scale,
                                std::string out_dtype) {
    // if (x.dims().size() != 2 ||
    //     y.dims().size() != 2 ||
    //     x.dims()[x.dims().size() - 1] != y.dims()[y.dims().size() - 1] ||
    //     x.dtype() != paddle::DataType::FLOAT8_E4M3FN ||
    //     y.dtype() != paddle::DataType::INT8) {
    //     PADDLE_THROW(phi::errors::Fatal(
    //         "Only support x[M, K](fp8_e4m3fn) and y[N, K](int8) as input"));
    // }

    int M = x.dims()[0];
    int K = x.dims()[1];
    int N = y.dims()[0] * 2;

    paddle::Tensor out;

    Pointer out_ptr = nullptr;
    ConstPointer x_ptr = nullptr;
    ConstPointer y_ptr = nullptr;
    ConstPointer scale_ptr = nullptr;
    ConstPointer zero_points_ptr = nullptr;
    ConstPointer bias_ptr = nullptr;

    x_ptr = reinterpret_cast<const void*>(x.data<phi::dtype::float8_e4m3fn>());
    y_ptr = reinterpret_cast<const void*>(y.data<int8_t>());
    std::vector<int64_t> out_shape = x.shape();
    out_shape[0] = M;
    out_shape[1] = N;

    auto place = x.place();
    cudaStream_t stream = x.stream();

    if (out_dtype == "float16") {
        scale_ptr = reinterpret_cast<const void*>(scale.data<phi::dtype::float16>());
        if (zero_points) {
            zero_points_ptr =
                reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
                    zero_points.get().data<phi::dtype::float16>()));
        }
        if (bias) {
            bias_ptr = reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
                bias.get().data<phi::dtype::float16>()));
        }
        out = paddle::empty(out_shape, paddle::DataType::FLOAT16, x.place());
        out_ptr = reinterpret_cast<void*>(out.data<phi::dtype::float16>());
        if (groupsize > 0) {
            Params params(x_ptr, y_ptr, scale_ptr, zero_points_ptr, bias_ptr, out_ptr, out_scale, M, N, K, groupsize);
            scaled_gemm_f8_i4_f16_launcher<half, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>(params, place, stream);
        } else {
            Params params(x_ptr, y_ptr, scale_ptr, nullptr, bias_ptr, out_ptr, out_scale, M, N, K, K);
            scaled_gemm_f8_i4_f16_launcher<half, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>(params, place, stream);
        }
    } else if (out_dtype == "bfloat16") {
        scale_ptr = reinterpret_cast<const void*>(scale.data<phi::dtype::bfloat16>());
        if (zero_points) {
            zero_points_ptr =
                reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
                    zero_points.get().data<phi::dtype::bfloat16>()));
        }
        if (bias) {
            bias_ptr = reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
                bias.get().data<phi::dtype::bfloat16>()));
        }
        out = paddle::empty(out_shape, paddle::DataType::BFLOAT16, x.place());
        out_ptr = reinterpret_cast<void*>(out.data<phi::dtype::bfloat16>());
        if (groupsize > 0) {
            Params params(x_ptr, y_ptr, scale_ptr, zero_points_ptr, bias_ptr, out_ptr, out_scale, M, N, K, groupsize);
            scaled_gemm_f8_i4_f16_launcher<__nv_bfloat16, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>(params, place, stream);
        } else {
            Params params(x_ptr, y_ptr, scale_ptr, nullptr, bias_ptr, out_ptr, out_scale, M, N, K, K);
            scaled_gemm_f8_i4_f16_launcher<__nv_bfloat16, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>(params, place, stream);
        }
    } else {
        PADDLE_THROW(
            phi::errors::Fatal("fp8_int4_gemm only support bfloat16 "
                             "and float16 output"));
    }
    return {out};
}

std::vector<std::vector<int64_t>> CutlassFp8Int4GemmInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& scale_shape,
    const paddle::optional<std::vector<int64_t>>& zero_points_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    int groupsize,
    float out_scale,
    std::string out_dtype) {
    // if (x_shape.size() != 2 ||
    //     y_shape.size() != 2 ||
    //     x_shape[x_shape.size()-1] != y_shape[y_shape.size()-1]) {
    //     PADDLE_THROW(phi::errors::Fatal(
    //         "Only support x[M, K] and y[N, K] as input"));
    // }

    int M = x_shape[0];
    int K = x_shape[1];
    int N = y_shape[0] * 2;

    std::vector<int64_t> out_shape = x_shape;
    out_shape[0] = M;
    out_shape[1] = N;
    return {out_shape};
}

std::vector<paddle::DataType> CutlassFp8Int4GemmInferDtype(
    const paddle::DataType& x_type,
    const paddle::DataType& y_type,
    const paddle::DataType& scale_type,
    const paddle::optional<paddle::DataType>& zero_points_type,
    const paddle::optional<paddle::DataType>& bias_type,
    int groupsize,
    float out_scale,
    std::string out_dtype) {
    paddle::DataType data_type;
    if (out_dtype == "bfloat16") {
        data_type = paddle::DataType::BFLOAT16;
    } else if (out_dtype == "float16") {
        data_type = paddle::DataType::FLOAT16;
    } else {
        PD_THROW(
            "fp8_int4_half_gemm_fused only support bfloat16 and float16 output");
    }
    return {data_type};
}

PD_BUILD_STATIC_OP(scaled_gemm_f8_i4_f16)
    .Inputs({"x", "y", "scale", paddle::Optional("zero_points"), paddle::Optional("bias")})
    .Attrs({"groupsize: int",
            "out_scale: float",
            "out_dtype: std::string"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(scaled_gemm_f8_i4_f16))
    .SetInferShapeFn(PD_INFER_SHAPE(CutlassFp8Int4GemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CutlassFp8Int4GemmInferDtype));
