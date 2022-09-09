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

namespace py = pybind11;

namespace fastdeploy {

void BindUIE(pybind11::module& m) {
  py::class_<text::UIEModel>(m, "UIEModel")
      .def(py::init<std::string, std::string, std::string, float, size_t,
                    std::vector<std::string>, RuntimeOption, Frontend>(),
           py::arg("model_file"), py::arg("params_file"), py::arg("vocab_file"),
           py::arg("position_prob"), py::arg("max_length"), py::arg("schema"),
           py::arg("custom_option") = fastdeploy::RuntimeOption(),
           py::arg("model_format") = fastdeploy::Frontend::PADDLE)
      .def("predict",
           [](text::UIEModel& self, const std::vector<std::string>& texts) {
             std::vector<
                 std::unordered_map<std::string, std::vector<text::UIEResult>>>
                 results;
             self.Predict(texts, &results);
             return results;
           },
           py::arg("text"));
}

}  // namespace fastdeploy
