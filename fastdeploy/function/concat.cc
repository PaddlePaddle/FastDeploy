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

#include "fastdeploy/function/reduce.h"

#include <limits>
#include <set>

#include "fastdeploy/function/eigen.h"
#include "fastdeploy/function/reduce_functor.h"
#include "fastdeploy/function/transpose.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  out << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != vec.size() - 1) {
      out << vec[i] << ", ";
    } else {
      out << vec[i] << "]";
    }
  }
  return out;
}


void Concat(const std::vector<FDTensor>& inputs, FDTensor* out, int axis) {
  FDASSERT(inputs.size() > 0, "The length of inputs is 0.");
  auto first_shape = inputs[0].Shape();
  std::vector<int64_t> out_shape(first_shape.begin(), first_shape.end());

  if (axis < 0) {
    axis += first_shape.size();
  }

  for (size_t i = 1; i < inputs.size(); ++i) {
    auto shape = inputs[i].Shape();
    FDASSERT(shape.size() == first_shape.size(), "Require all the rank of input tensors be same, but the first tensors' rank is %zu while the rank of tensor(index=%zu) is %zu.", first_shape.size(), i, shape.size());
    FDASSERT(inputs[i].Dtype() == inputs[0].Dtype(), "Require all the data type of input tensors be same.");
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      if (dim == axis) {
        out_shape[dim] += shape[axis];
        continue;
      }
      FDASSERT(shape[dim] == first_shape[dim], "Require all the input tensors' shape be same exclude the concatenate axis.");
    }
  }

  out->external_data_ptr = nullptr;
  out->device = Device::CPU;
  out->Resize(out_shape, inputs[0].Dtype());
  int index = 0;
  char* data_ptr = reinterpret_cast<char*>(out->MutableData());
  for (size_t i = 0; i < inputs.size(); ++i) {
    memcpy(data_ptr + index, inputs[i].CpuData(), inputs[i].Nbytes());
    index += inputs[i].Nbytes();
  }
}

}  // namespace fastdeploy
