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

#pragma once

#include <set>
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/function/eigen.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

template <typename T, size_t D, size_t R_D, typename Functor>
void ReduceFunctor(const FDTensor& input, FDTensor* output,
                   const std::vector<int64_t>& dims, bool keep_dim) {
  auto x = EigenTensor<T, D>::From(input);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto reduce_dim = Eigen::array<int, R_D>();
  std::vector<int64_t> dims_ref = dims;

  auto out_dims = input.shape;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) dims_ref[i] = x_rank + dims_ref[i];
    reduce_dim[i] = dims_ref[i];
    out_dims[dims_ref[i]] = 1;
  }
  // construct the squeezed output tensor
  if (keep_dim && x_rank > 1) {
    const int kDelFlag = -2;
    for (size_t i = 0; i < dims_ref.size(); ++i) {
      out_dims[dims_ref[i]] = kDelFlag;
    }
    out_dims.erase(remove(out_dims.begin(), out_dims.end(), kDelFlag),
                   out_dims.end());
  }
  auto& place = *EigenDeviceWrapper::GetInstance()->GetDevice();
  Functor functor;

  if (D == 1) {
    auto out = EigenScalar<T>::From(*output);
    functor(place, &x, &out, reduce_dim);
  } else {
    auto out = EigenTensor<T, (D - R_D)>::From(*output, out_dims);
    functor(place, &x, &out, reduce_dim);
  }
}

#define HANDLE_REDUCE_DIM(NDIM, RDIM)                                        \
  if (ndim == NDIM && rdim == RDIM) {                                        \
    ReduceFunctor<OutT, NDIM, RDIM, Functor>(input, output, dims, keep_dim); \
  }

inline void GetShuffledDim(const std::vector<int64_t>& src_dims,
                           std::vector<int64_t>* dst_dims,
                           const std::vector<int64_t>& reduced_dims,
                           std::vector<int>* perm_axis) {
  // check if it's a reduced dim
  std::vector<bool> src_dims_check(src_dims.size(), false);
  size_t src_size = src_dims.size();
  size_t reduce_size = reduced_dims.size();
  std::vector<int64_t> regular_reduced_dims = reduced_dims;
  for (size_t i = 0; i < regular_reduced_dims.size(); i++) {
    if (regular_reduced_dims[i] < 0) {
      regular_reduced_dims[i] = src_size + regular_reduced_dims[i];
    }
  }

  for (size_t i = 0; i < reduce_size; ++i) {
    dst_dims->at(src_size - reduce_size + i) =
        src_dims[regular_reduced_dims[i]];
    (*perm_axis)[src_size - reduce_size + i] = regular_reduced_dims[i];
    src_dims_check[regular_reduced_dims[i]] = true;
  }

  size_t offset = 0;
  for (size_t i = 0; i < src_dims_check.size(); ++i) {
    bool is_reduced = src_dims_check[i];
    if (!is_reduced) {
      (*perm_axis)[offset] = i;
      dst_dims->at(offset++) = src_dims[i];
    }
  }
}

template <typename OutT>
void GetShuffledInput(const FDTensor& input, FDTensor* shuffled_input,
                      const std::vector<int64_t>& dims) {
  auto shuffled_dims = input.shape;
  std::vector<int> perm_axis(input.shape.size());
  GetShuffledDim(input.shape, &shuffled_dims, dims, &perm_axis);

  shuffled_input->Allocate(shuffled_dims, input.dtype);
  // TODO(zhoushunjie) : Need to implement trans function
  // phi::funcs::TransposeNormal<DeviceContext, OutT> trans;
  // trans(dev_ctx, input, shuffled_input, perm_axis);
}

//////////////// HandleLargeDim
template <typename OutT, typename Functor>
void HandleLargeDim(const FDTensor& input, FDTensor* output,
                    const std::vector<int64_t>& dims, bool keep_dim) {
  //  shuffle the reduced dim to the end
  FDTensor shuffled_input;
  GetShuffledInput<OutT>(input, &shuffled_input, dims);

  // transpose to 2D tensor whose shape is {unreduced, reduced}.
  const int64_t unreduced = output->Numel();
  const int64_t reduced = shuffled_input.Numel() / unreduced;
  shuffled_input.Allocate({unreduced, reduced}, TypeToDataType<OutT>::dtype);

  auto output_dim = output->shape;
  output->Allocate({unreduced}, TypeToDataType<OutT>::dtype);

  ReduceFunctor<OutT, 2, 1, Functor>(shuffled_input, output, {1}, keep_dim);
  output->shape = output_dim;
}

////////////// ReduceKernel

template <typename T, typename OutT, typename Functor>
void ReduceKernelImpl(const FDTensor& input, FDTensor* output,
                      const std::vector<int64_t>& dims, bool keep_dim,
                      bool reduce_all) {
  output->Allocate({1}, TypeToDataType<OutT>::dtype);
  const auto& dev = *EigenDeviceWrapper::GetInstance()->GetDevice();
  if (reduce_all) {
    // Flatten and reduce 1-D tensor
    auto x = EigenVector<OutT>::Flatten(input);
    auto out = EigenScalar<OutT>::From(*output);
    auto reduce_dim = Eigen::array<int, 1>({{0}});

    Functor functor;
    functor(dev, &x, &out, reduce_dim);
  } else {
    int ndim = input.shape.size();
    int rdim = dims.size();
    if (ndim > 3) {
      HandleLargeDim<OutT, Functor>(input, output, dims, keep_dim);
    } else {
      HANDLE_REDUCE_DIM(4, 3);
      HANDLE_REDUCE_DIM(4, 2);
      HANDLE_REDUCE_DIM(4, 1);
      HANDLE_REDUCE_DIM(3, 2);
      HANDLE_REDUCE_DIM(3, 1);
      HANDLE_REDUCE_DIM(2, 1);
      HANDLE_REDUCE_DIM(1, 1);
    }
  }
}

template <typename OutT, typename Functor>
void BoolReduceKernel(const FDTensor& input, const std::vector<int64_t>& dims,
                      bool keep_dim, bool reduce_all, FDTensor* output) {
  output->Allocate({1}, TypeToDataType<OutT>::dtype);

  // The dims has full dim, set the reduce_all is True
  const auto& input_dim_size = input.shape.size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  ReduceKernelImpl<bool, OutT, Functor>(input, output, dims, keep_dim,
                                        reduce_all);
}

template <typename T, typename Functor>
void Reduce(const FDTensor& x, bool reduce_all,
            const std::vector<int64_t>& dims, bool keep_dim, FDTensor* out) {
  // If the dims has full dim, set the reduce_all is True
  const int& input_dim_size = x.shape.size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (int i = 0; i < input_dim_size; ++i) {
    if (dims_set.find(i) == dims_set.end() &&
        dims_set.find(i - input_dim_size) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  FD_VISIT_ALL_TYPES(x.dtype, "ReduceKernelImpl", ([&] {
                       ReduceKernelImpl<T, data_t, Functor>(
                           x, out, dims, keep_dim, reduce_all);
                     }));
}

void Max(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out);

void Min(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out);

void Sum(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out);

void All(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out);

void Any(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out);

void Mean(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
          bool keep_dim, FDTensor* out);

void Prod(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
          bool keep_dim, FDTensor* out);

}  // namespace fastdeploy
