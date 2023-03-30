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

namespace fastdeploy {
namespace streamer {

void BindFDStreamer(pybind11::module&);

PYBIND11_MODULE(fastdeploy_streamer_main, m) {
  m.doc() =
      "Make programer easier to deploy deeplearning model, save time to save "
      "the world!";

  BindFDStreamer(m);
}

}  // namespace streamer
}  // namespace fastdeploy
