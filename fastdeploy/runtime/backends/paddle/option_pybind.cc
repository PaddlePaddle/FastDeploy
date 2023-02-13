// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/pybind/main.h"
#include "fastdeploy/runtime/backends/paddle/option.h"

namespace fastdeploy {

void BindIpuOption(pybind11::module& m) {
  pybind11::class_<IpuOption>(m, "IpuOption")
      .def(pybind11::init())
      .def_readwrite("ipu_device_num", &IpuOption::ipu_device_num)
      .def_readwrite("ipu_micro_batch_size", &IpuOption::ipu_micro_batch_size)
      .def_readwrite("ipu_enable_pipelining", &IpuOption::ipu_enable_pipelining)
      .def_readwrite("ipu_batches_per_step", &IpuOption::ipu_batches_per_step)
      .def_readwrite("ipu_enable_fp16", &IpuOption::ipu_enable_fp16)
      .def_readwrite("ipu_replica_num", &IpuOption::ipu_replica_num)
      .def_readwrite("ipu_available_memory_proportion",
                     &IpuOption::ipu_available_memory_proportion)
      .def_readwrite("ipu_enable_half_partial",
                     &IpuOption::ipu_enable_half_partial);
}

void BindPaddleOption(pybind11::module& m) {
  BindIpuOption(m);
  pybind11::class_<PaddleBackendOption>(m, "PaddleBackendOption")
      .def(pybind11::init())
      .def_readwrite("enable_log_info", &PaddleBackendOption::enable_log_info)
      .def_readwrite("enable_mkldnn", &PaddleBackendOption::enable_mkldnn)
      .def_readwrite("enable_trt", &PaddleBackendOption::enable_trt)
      .def_readwrite("ipu_option", &PaddleBackendOption::ipu_option)
      .def_readwrite("collect_trt_shape",
                     &PaddleBackendOption::collect_trt_shape)
      .def_readwrite("mkldnn_cache_size",
                     &PaddleBackendOption::mkldnn_cache_size)
      .def_readwrite("gpu_mem_init_size",
                     &PaddleBackendOption::gpu_mem_init_size)
      .def("disable_trt_ops", &PaddleBackendOption::DisableTrtOps)
      .def("delete_pass", &PaddleBackendOption::DeletePass)
      .def("set_ipu_config", &PaddleBackendOption::SetIpuConfig);
}

}  // namespace fastdeploy
