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
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "paddle/extension.h"
#include "../common_ngram_kernel.cuh"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void HybridMtpNgram(const paddle::Tensor &input_ids,
        const paddle::Tensor &input_ids_len,
        const paddle::Tensor &pre_ids,
        const paddle::Tensor &step_idx,
        const paddle::Tensor &draft_token_num,
        const paddle::Tensor &draft_tokens,
        const paddle::Tensor &seq_lens_this_time,
        const paddle::Tensor &seq_lens_decoder,
        const paddle::Tensor &max_dec_len,
        const int max_ngram_size,
        const int min_ngram_size,
        const int max_draft_tokens) {

    auto input_ids_shape = input_ids.shape();
    const int64_t input_ids_stride = input_ids_shape[1];

    auto pre_ids_shape = pre_ids.shape();
    const int64_t pre_ids_stride = pre_ids_shape[1];

    auto draft_tokens_shape = draft_tokens.shape();
    const int64_t draft_tokens_stride = draft_tokens_shape[1];

    const int64_t max_batch_size = seq_lens_this_time.shape()[0];

    /* 1. 阈值策略（与文件 1 差异①） */
    int threshold = 1024;
    if (char* e = getenv("SPEC_TOKENUM_THRESHOLD"))
        threshold = std::stoi(e);

    /* 2. 当前已用 token 数（device 侧 reduce） */
    int tokens_used = thrust::reduce(
        thrust::cuda::par.on(stream),
        thrust::device_ptr<const int32_t>(seq_lens_this_time.data<int32_t>()),
        thrust::device_ptr<const int32_t>(seq_lens_this_time.data<int32_t>() + max_batch_size),
        0,
        thrust::plus<int>());

    /* 3. device 输出 buffer */
    paddle::Tensor match_pos = paddle::empty({max_batch_size}, paddle::DataType::INT32, input_ids.place());
    paddle::Tensor draft_cnt = paddle::empty({max_batch_size}, paddle::DataType::INT32, input_ids.place());

    /* 4. 启动单线程 kernel（与文件 1 差异②：min_ngram_size 由参数传入） */
    ngram_match_kernel_single_thread<<<max_batch_size, 1, 0, stream>>>(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        max_ngram_size,
        min_ngram_size,                 // 文件 2 可配置
        max_draft_tokens,
        input_ids_stride,
        pre_ids_stride,
        match_pos.data<int>(),
        draft_cnt.data<int>());

    /* 5. 根据 kernel 结果写回草稿 token & 更新 seq_lens_this_time */
    thrust::for_each(
        thrust::cuda::par.on(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(max_batch_size)),
        [=] __device__ (int b) {
            int32_t& len = const_cast<int32_t*>(seq_lens_this_time.data<int32_t>())[b];
            if (seq_lens_decoder[b] == 0) { len = 0; return; }   // 文件 2 只判 decoder

            int ori_len = len;        // 与文件 1 差异③：需要累加
            int cnt     = draft_cnt.data<int>()[b];
            if (cnt <= 0) return;     // 无匹配，保持 ori_len 不变

            /* 阈值二次裁剪（与文件 1 相同逻辑） */
            int left = max_batch_size - b - 1;
            if (tokens_used + cnt + left > threshold)
                cnt = max(0, threshold - tokens_used - left);
            if (cnt == 0) return;

            /* 拷贝草稿 token（与文件 1 差异④：从 ori_len 位置开始写） */
            const int64_t* src = (match_pos.data<int>()[b] < input_ids_len.data<int64_t>()[b]) ?
                                 (input_ids.data<int64_t>() + b * input_ids_stride) :
                                 (pre_ids.data<int64_t>()   + b * pre_ids_stride);
            int64_t* dst = const_cast<int64_t*>(draft_tokens.data<int64_t>()) + b * draft_tokens_stride + ori_len;
            for (int i = 0; i < cnt; ++i) dst[i] = src[match_pos.data<int>()[b] + i];
            len = ori_len + cnt;      // 累加
        });
}

PD_BUILD_STATIC_OP(hybrid_mtp_ngram)
        .Inputs({"input_ids",
                "input_ids_len",
                "pre_ids",
                "step_idx",
                "draft_token_num",
                "draft_tokens",
                "seq_lens_this_time",
                "seq_lens_decoder",
                "max_dec_len"})
        .Attrs({"max_ngram_size: int", "min_ngram_size: int", "max_draft_tokens: int"})
        .Outputs({"draft_tokens_out", "seq_lens_this_time_out"})
        .SetKernelFn(PD_KERNEL(HybridMtpNgram))
        .SetInplaceMap({{"draft_tokens", "draft_tokens_out"}, {"seq_lens_this_time", "seq_lens_this_time_out"}});
