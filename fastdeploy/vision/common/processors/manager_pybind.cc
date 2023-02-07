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
void BindProcessorManager(pybind11::module& m) {
  pybind11::class_<vision::ProcessorManager>(m, "ProcessorManager")
      .def("run",
           [](vision::ProcessorManager& self,
              std::vector<pybind11::array>& im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error("Failed to process the input data");
             }
             if (!self.CudaUsed()) {
               for (size_t i = 0; i < outputs.size(); ++i) {
                 outputs[i].StopSharing();
               }
             }
             return outputs;
           })
      .def("use_cuda",
           [](vision::ProcessorManager& self, bool enable_cv_cuda = false,
              int gpu_id = -1) { self.UseCuda(enable_cv_cuda, gpu_id); });
}
}  // namespace fastdeploy
