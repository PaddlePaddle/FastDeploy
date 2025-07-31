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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include <xft/xdnn_plugin.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

std::vector<paddle::Tensor> ExtractTextTokenOutput(
            const paddle::Tensor& max_seq_len,
            const paddle::Tensor& max_seq_len_index,
            const paddle::Tensor& mm_token_num_len,
            const paddle::Tensor& seq_lens_this_time,
            const paddle::Tensor& cu_seqlens_q,
            const paddle::Tensor& score_text) {
    phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
    auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
    auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
    const int bsz = seq_lens_this_time.shape()[0];
    const int hidden_size = score_text.shape()[1];
    paddle::Tensor output = paddle::full({bsz, hidden_size}, 1, paddle::DataType::FLOAT32, score_text.place());
    
    switch (score_text.type()) {
        case paddle::DataType::FLOAT32: {
            using XPUType = typename XPUTypeTrait<float>::Type;
            //typedef paddle::float data_t;
            int r = baidu::xpu::api::plugin::extract_text_token_output<XPUType>(
                xpu_ctx->x_context(), 
                const_cast<int*>(max_seq_len.data<int>()),
                const_cast<int*>(max_seq_len_index.data<int>()),
                const_cast<int*>(mm_token_num_len.data<int>()),
                const_cast<int*>(seq_lens_this_time.data<int>()),
                const_cast<int*>(cu_seqlens_q.data<int>()),
                const_cast<XPUType*>(score_text.data<float>()),
                output.data<float>(),
                bsz,
                hidden_size
            );
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "text_image_gather_scatter");
            break;
        }
        default: {
            PD_THROW(
                "NOT supported data type. Only support FLOAT. ");
            break;
        }
    }
    return {output};
}

std::vector<std::vector<int64_t>> ExtractTextTokenOutputInferShape(const std::vector<int64_t>& max_seq_len_shape,
                                                             const std::vector<int64_t>& max_seq_len_index_shape,
                                                             const std::vector<int64_t>& mm_token_num_len_shape,
                                                             const std::vector<int64_t>& seq_lens_this_time_shape,
                                                             const std::vector<int64_t>& cu_seqlens_q_shape,
                                                             const std::vector<int64_t>& score_text_shape) {
    const int bsz = seq_lens_this_time_shape[0];
    const int hidden_size = score_text_shape[1];
    return {{bsz, hidden_size}};
}

std::vector<paddle::DataType> ExtractTextTokenOutputInferDtype(const paddle::DataType& max_seq_len_dtype,
                                                         const paddle::DataType& max_seq_len_index_dtype,
                                                         const paddle::DataType& mm_token_num_len_dtype,
                                                         const paddle::DataType& seq_lens_this_time_dtype,
                                                         const paddle::DataType& cu_seqlens_q_dtype,
                                                         const paddle::DataType& score_text_dtype) {
    return {score_text_dtype};
}

PD_BUILD_OP(extract_text_token_output)
    .Inputs({"max_seq_len",
             "max_seq_len_index",
             "mm_token_num_len",
             "seq_lens_this_time",
             "cu_seqlens_q",
             "score_text"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(ExtractTextTokenOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(ExtractTextTokenOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ExtractTextTokenOutputInferDtype));
