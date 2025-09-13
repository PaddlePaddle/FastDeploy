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
#include "ops/pybind/cachekv_signal_thread_worker.h"
#include <cuda_runtime_api.h>
#include "ops/remote_cache_kv_ipc.h"
#include "ops/utility/env.h"
XPU_DECLARE_BOOL(fmt_write_cache_completed_signal, false);
CacheKvSignalThreadWorker::CacheKvSignalThreadWorker() : stop(false) {
  xpu_stream_create(&write_cache_kv_stream);
  int devid;
  auto ret = xpu_current_device(&devid);
  PD_CHECK(ret == 0, "xpu_current_device failed.");
  auto func = [this, devid]() {
    int old_dev;
    xpu_current_device(&old_dev);
    auto ret = xpu_set_device(devid);
    PD_CHECK(ret == 0, "xpu_set_device failed.");
    ret = cudaSetDevice(devid);
    PD_CHECK(ret == 0, "cudaSetDevice failed.");

    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(write_mutex);
        if (stop) return;
        if (!signal_task_queue.empty()) {
          task = std::move(signal_task_queue.front());
          signal_task_queue.pop();
        } else {
          lock.unlock();
          std::this_thread::sleep_for(std::chrono::microseconds(1));
          continue;
        }
      }
      task();  // 执行任务
    }
  };
  worker_thread = std::thread(func);
}

void CacheKvSignalThreadWorker::push_signal_task(XPUEvent e1, void* meta_data) {
  auto func = [this, e1, meta_data]() {
    xpu_stream_wait_event(write_cache_kv_stream, e1);
    xpu_wait(write_cache_kv_stream);
    RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise(meta_data);
    xpu_event_destroy(e1);
  };
  std::lock_guard<std::mutex> lock(write_mutex);
  signal_task_queue.push(func);
}

void CacheKvSignalThreadWorker::push_signal_task_per_query(XPUEvent e1,
                                                           void* meta_data) {
  auto func = [this, e1, meta_data]() {
    xpu_stream_wait_event(write_cache_kv_stream, e1);
    xpu_wait(write_cache_kv_stream);
    RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_per_query(
        meta_data);
    xpu_event_destroy(e1);
  };
  std::lock_guard<std::mutex> lock(write_mutex);
  signal_task_queue.push(func);
}

void CacheKvSignalThreadWorker::sync_all_signals() {
  {
    std::unique_lock<std::mutex> lock(write_mutex);
    while (!signal_task_queue.empty()) {
      // 1 微秒休眠
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      lock.lock();
    }
    stop = true;
  }
  worker_thread.join();
  xpu_stream_destroy(write_cache_kv_stream);
}

paddle::Tensor create_cachekv_signal_thread() {
  CacheKvSignalThreadWorker* worker = nullptr;
  if (FLAGS_fmt_write_cache_completed_signal) {
    worker = new CacheKvSignalThreadWorker();
  }
  auto t = paddle::full({1}, 0, paddle::DataType::INT64, paddle::CPUPlace());
  t.data<int64_t>()[0] = reinterpret_cast<int64_t>(worker);
  return t;
}
void destroy_cachekv_signal_thread(const paddle::Tensor& t) {
  auto worker =
      reinterpret_cast<CacheKvSignalThreadWorker*>(t.data<int64_t>()[0]);
  if (FLAGS_fmt_write_cache_completed_signal) {
    PD_CHECK(worker != nullptr, "cachekv_signal_thread should not be nullptr");
    worker->sync_all_signals();
    delete worker;
  } else {
    PD_CHECK(worker == nullptr,
             "cachekv_signal_thread should be nullptr if not pd split");
  }
}
