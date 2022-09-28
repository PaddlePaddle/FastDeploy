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

void BindArcFace(pybind11::module& m);
void BindInsightFaceRecognitionModel(pybind11::module& m);
void BindCosFace(pybind11::module& m);
void BindPartialFC(pybind11::module& m);
void BindVPL(pybind11::module& m);

void BindFaceId(pybind11::module& m) {
  auto faceid_module = m.def_submodule("faceid", "Face recognition models.");
  BindInsightFaceRecognitionModel(faceid_module);
  BindArcFace(faceid_module);
  BindCosFace(faceid_module);
  BindPartialFC(faceid_module);
  BindVPL(faceid_module);
}
}  // namespace fastdeploy
