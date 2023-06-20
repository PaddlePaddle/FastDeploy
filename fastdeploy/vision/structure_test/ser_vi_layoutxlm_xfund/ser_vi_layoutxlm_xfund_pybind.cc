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

namespace fastdeploy {
void BindSERVILayoutXLMXfund(pybind11::module& m) {
  pybind11::class_<vision::structure_test::SERViLayoutxlmModel,
                   FastDeployModel>(m, "SERViLayoutxlmModel")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("clone",
           [](vision::structure_test::SERViLayoutxlmModel& self) {
             return self.Clone();
           })
      .def("predict",
           [](vision::structure_test::SERViLayoutxlmModel& self,
              pybind11::array& data) {
             throw std::runtime_error(
                 "SERViLayoutxlmModel do not support predict.");
           })
      .def("batch_predict",
           [](vision::structure_test::SERViLayoutxlmModel& self,
              std::vector<pybind11::array>& data) {
             throw std::runtime_error(
                 "SERViLayoutxlmModel do not support batch_predict.");
           })
      .def("infer",
           [](vision::structure_test::SERViLayoutxlmModel& self,
              std::map<std::string, pybind11::array>& data) {
             std::vector<FDTensor> inputs(data.size());
             int index = 0;
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               std::vector<int64_t> data_shape;
               data_shape.insert(data_shape.begin(), iter->second.shape(),
                                 iter->second.shape() + iter->second.ndim());
               auto dtype = NumpyDataTypeToFDDataType(iter->second.dtype());

               inputs[index].Resize(data_shape, dtype);
               memcpy(inputs[index].MutableData(), iter->second.mutable_data(),
                      iter->second.nbytes());
               inputs[index].name = iter->first;
               index += 1;
             }

             std::vector<FDTensor> outputs(self.NumOutputsOfRuntime());
             self.Infer(inputs, &outputs);

             std::vector<pybind11::array> results;
             results.reserve(outputs.size());
             for (size_t i = 0; i < outputs.size(); ++i) {
               auto numpy_dtype = FDDataTypeToNumpyDataType(outputs[i].dtype);
               results.emplace_back(
                   pybind11::array(numpy_dtype, outputs[i].shape));
               memcpy(results[i].mutable_data(), outputs[i].Data(),
                      outputs[i].Numel() * FDDataTypeSize(outputs[i].dtype));
             }
             return results;
           })
      .def("get_input_info",
           [](vision::structure_test::SERViLayoutxlmModel& self, int& index) {
             return self.InputInfoOfRuntime(index);
           });
}
}  // namespace fastdeploy