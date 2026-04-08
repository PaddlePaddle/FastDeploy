// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <assert.h>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <vector>

namespace paddle {
namespace inference {
namespace transfer {

#define MAX_CACHE_LENGTH 10000

using server_callback_fn = void(std::vector<int64_t>, void *);

struct BatchResult {
    BatchResult(int64_t cur_batch_size, std::vector<int64_t> &cur_tokens)
        : batch_size(cur_batch_size), tokens(cur_tokens) {}
    int64_t batch_size;
    std::vector<int64_t> tokens;
};

class TokenTransfer {
    public:
    TokenTransfer(const TokenTransfer &o) = delete;
    const TokenTransfer &operator=(const TokenTransfer &o) = delete;
    ~TokenTransfer() {}

    static TokenTransfer &Instance() {
        static TokenTransfer instance;
        return instance;
    }

    void RegisterCallback(server_callback_fn *cb_fn, void *cb_data) {
        stream_cb_fn_ = cb_fn;
        stream_cb_data_ = cb_data;
    }

    void UnRegisterCallback() {
        stream_cb_fn_ = nullptr;
        stream_cb_data_ = nullptr;
    }

    // once copy: cpu --> cpu
    // array length should be (1 + MAX_BATCH)
    bool GetBatchToken(int64_t *array) {
        if (Empty()) {
            return false;
        } else {
            assert(array != nullptr);
            std::lock_guard<std::mutex> mtx(mtx_);
            array[0] = q_.front().batch_size;
            if (array[0] != 0) {
                memmove(reinterpret_cast<void *>(array + 1),
                        reinterpret_cast<void *>(q_.front().tokens.data()),
                        sizeof(int64_t) * array[0]);
            }
            q_.pop();
            return true;
        }
    }

    void PushBatchToken(int64_t cur_batch_size, int64_t *cur_tokens) {
        std::lock_guard<std::mutex> mtx(mtx_);
        if (q_.size() > MAX_CACHE_LENGTH) {
            std::cout << "Warning: The queue that stores the results "
                      << "has exceeded MAX_CACHE_LENGTH and will be forcefully "
                         "cleared."
                      << std::endl;
            std::queue<BatchResult> empty;
            std::swap(q_, empty);
        }
        std::vector<int64_t> tmp(cur_tokens, cur_tokens + cur_batch_size);
        q_.emplace(cur_batch_size, tmp);
    }

    bool Empty() {
        std::lock_guard<std::mutex> mtx(mtx_);
        return q_.empty();
    }

    server_callback_fn *stream_cb_fn_ = nullptr;
    void *stream_cb_data_ = nullptr;

    private:
    TokenTransfer() {}

    std::mutex mtx_;
    std::queue<BatchResult> q_;
};

}  // namespace transfer
}  // namespace inference
}  // namespace paddle
