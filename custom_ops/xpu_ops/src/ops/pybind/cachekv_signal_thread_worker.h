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
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include "paddle/extension.h"
#include "xpu/runtime.h"

struct CacheKvSignalThreadWorker {
  CacheKvSignalThreadWorker();
  void push_signal_task(XPUEvent e1, void* meta_data);
  void push_signal_task_per_query(XPUEvent e1, void* meta_data);
  void sync_all_signals();
  std::thread worker_thread;
  std::queue<std::function<void()>> signal_task_queue;
  std::mutex write_mutex;
  XPUStream write_cache_kv_stream;
  bool stop;
};

paddle::Tensor create_cachekv_signal_thread();
void destroy_cachekv_signal_thread(const paddle::Tensor& t);
