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

#include "pybind/main.h"
#include "fd_streamer.h"

namespace fastdeploy {
namespace streamer {
void BindFDStreamer(pybind11::module& m) {
  pybind11::class_<FDStreamer>(m, "FDStreamer")
      .def(pybind11::init<>(), "Default Constructor")
      .def("Init", [](FDStreamer& self, std::string config_file){
        return self.Init(config_file);
      })
      .def("Run", &FDStreamer::Run)
      .def("RunAsync", &FDStreamer::RunAsync);
}
}  // namespace streamer
}  // namespace fastdeploy
