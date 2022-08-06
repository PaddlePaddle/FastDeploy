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

#include <algorithm>
#include <vector>
#include "fastdeploy/core/fd_tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace fastdeploy {
// EigenDim converts shape into Eigen::DSizes.
template <int D>
struct EigenDim {
  using Type = Eigen::DSizes<Eigen::DenseIndex, D>;

  static Type From(const std::vector<int64_t>& dims) {
    Type ret;
    for (int64_t d = 0; d < dims.size(); d++) {
      ret[d] = dims[d];
    }
    return ret;
  }
};

// Interpret FDTensor as EigenTensor and EigenConstTensor.
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenTensor {
  using Type = Eigen::TensorMap<Eigen::Tensor<T, D, MajorType, IndexType>>;

  using ConstType =
      Eigen::TensorMap<Eigen::Tensor<const T, D, MajorType, IndexType>>;

  static Type From(FDTensor& tensor,
                   const std::vector<int64_t>& dims) {  // NOLINT
    return Type(reinterpret_cast<T*>(tensor.data.data()),
                EigenDim<D>::From(dims));
  }

  static Type From(FDTensor& tensor) {  // NOLINT
    return From(tensor, tensor.shape);
  }  // NOLINT

  static ConstType From(const FDTensor& tensor,
                        const std::vector<int64_t>& dims) {
    return ConstType(reinterpret_cast<const T*>(tensor.data.data()),
                     EigenDim<D>::From(dims));
  }

  static ConstType From(const FDTensor& tensor) {
    return From(tensor, tensor.shape);
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenScalar {
  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  using Type = Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, MajorType, IndexType>>;
  using ConstType = Eigen::TensorMap<
      Eigen::TensorFixedSize<const T, Eigen::Sizes<>, MajorType, IndexType>>;

  static Type From(FDTensor& tensor) {
    return Type(reinterpret_cast<T*>(tensor.data.data()));
  }  // NOLINT

  static ConstType From(const FDTensor& tensor) {
    return ConstType(reinterpret_cast<const T*>(tensor.data.data()));
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenVector : public EigenTensor<T, 1, MajorType, IndexType> {
  // Flatten reshapes a Tensor into an EigenVector.
  static typename EigenVector::Type Flatten(FDTensor& tensor) {  // NOLINT
    return EigenVector::From(tensor, {tensor.Numel()});
  }

  static typename EigenVector::ConstType Flatten(
      const FDTensor& tensor) {  // NOLINT
    return EigenVector::From(tensor, {tensor.Numel()});
  }
};

template <typename T, size_t D, size_t R_D, typename Functor>
void ReduceFunctor(const Eigen::DefaultDevice& dev, const FDTensor& input,
                   FDTensor* output, const std::vector<int64_t>& dims,
                   bool keep_dim = true) {
  auto x = EigenTensor<T, D>::From(input);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto reduce_dim = Eigen::array<int, R_D>();
  std::vector<int64_t> dims_ref = dims;
  std::vector<int> out_dims(input.shape.size());
  std::copy(input.shape.begin(), input.shape.end(), out_dims.begin());
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) dims_ref[i] = x_rank + dims_ref[i];
    out_dims[dims_ref[i]] = 1;
    reduce_dim[i] = dims_ref[i];
  }
  output->Allocate(out_dims, TypeToDataType<T>::dtype);
  if (keep_dim && x_rank > 1) {
    const int kDelFlag = -2;
    auto dims_vector = out_dims;
    for (size_t i = 0; i < dims_ref.size(); ++i) {
      dims_vector[dims_ref[i]] = kDelFlag;
    }
    dims_vector.erase(remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
                      dims_vector.end());
    out_dims = dims_vector;
  }
  Functor functor;

  if (D == 1) {
    auto out = EigenScalar<T>::From(*output);
    functor(dev, &x, &out, reduce_dim);
  } else {
    dims_ref.resize(out_dims.size());
    std::copy(out_dims.begin(), out_dims.end(), dims_ref.begin());
    for (int i = 0; i < dims_ref.size(); ++i) {
      std::cerr << dims_ref[i] << ", ";
    }
    std::cerr << std::endl;
    auto out = EigenTensor<T, (D - R_D)>::From(*output, dims_ref);
    functor(dev, &x, &out, reduce_dim);
  }
}

struct MaxFunctor {
  template <typename X, typename Y, typename Dim>
  void operator()(const Eigen::DefaultDevice& dev, X* x, Y* y, const Dim& dim) {
    y->device(dev) = x->maximum(dim);
  }
};

struct SumFunctor {
  template <typename X, typename Y, typename Dim>
  void operator()(const Eigen::DefaultDevice& dev, X* x, Y* y, const Dim& dim) {
    y->device(dev) = x->sum(dim);
  }
};

inline void GetBroadcastDimsArrays(const std::vector<int64_t>& x_dims,
                                   const std::vector<int64_t>& y_dims,
                                   int* x_dims_array, int* y_dims_array,
                                   int* out_dims_array, const int max_dim,
                                   const int axis) {
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.data(), x_dims.data() + x_dims.size(), x_dims_array);
    std::copy(y_dims.data(), y_dims.data() + y_dims.size(),
              y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.data(), x_dims.data() + x_dims.size(),
              x_dims_array + axis);
    std::copy(y_dims.data(), y_dims.data() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

inline int GetElementwiseIndex(const int* x_dims_array, const int max_dim,
                               const int* index_array) {
  int index_ = 0;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] > 1) {
      index_ = index_ * x_dims_array[i] + index_array[i];
    }
  }
  return index_;
}

inline void UpdateElementwiseIndexArray(const int* out_dims_array,
                                        const int max_dim, int* index_array) {
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_dims_array[i]) {
      index_array[i] -= out_dims_array[i];
    } else {
      break;
    }
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonElementwiseBroadcastForward(const FDTensor& x, const FDTensor& y,
                                       FDTensor* z, Functor func, int axis,
                                       const bool is_xsize_larger = true) {
  std::vector<int64_t> x_dims = x.shape;
  std::vector<int64_t> y_dims = y.shape;
  int max_dim = (std::max)(x_dims.size(), y_dims.size());
  int diff = x_dims.size() - y_dims.size();
  axis = (axis == -1 ? std::abs(diff) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                         y_dims_array.data(), out_dims_array.data(), max_dim,
                         axis);

  const T* x_data = reinterpret_cast<const T*>(x.Data());
  const T* y_data = reinterpret_cast<const T*>(y.Data());

  z->Allocate(out_dims_array, TypeToDataType<OutType>::dtype);
  OutType* out_data = reinterpret_cast<T*>(z->MutableData());

  const int out_size =
      std::accumulate(out_dims_array.data(), out_dims_array.data() + max_dim, 1,
                      std::multiplies<int>());
  int x_index, y_index;
  std::vector<int> index_array(max_dim, 0);
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index =
        GetElementwiseIndex(x_dims_array.data(), max_dim, index_array.data());
    y_index =
        GetElementwiseIndex(y_dims_array.data(), max_dim, index_array.data());
    if (is_xsize_larger) {
      out_data[out_index] = func(x_data[x_index], y_data[y_index]);
    } else {
      out_data[out_index] = func(y_data[y_index], x_data[x_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array.data(), max_dim,
                                index_array.data());
  }
}

template <typename T>
struct AddFunctor {
  T operator()(const T& lhs, const T& rhs) { return lhs + rhs; }
};

template <typename T>
struct SubFunctor {
  T operator()(const T& lhs, const T& rhs) { return lhs - rhs; }
};

template <typename T>
struct DivFunctor {
  T operator()(const T& lhs, const T& rhs) { return lhs / rhs; }
};

}  // namespace fastdeploy
