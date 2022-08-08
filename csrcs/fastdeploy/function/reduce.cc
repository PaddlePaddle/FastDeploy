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
#include "fastdeploy/function/reduce_functor.h"
namespace fastdeploy {

void Max(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "Max", ([&] {
                         Reduce<data_t, MaxFunctor>(x, reduce_all, dims,
                                                    keep_dim, out);
                       }));
  FD_VISIT_INT_TYPES(x.dtype, "Max", ([&] {
                       Reduce<data_t, MaxFunctor>(x, reduce_all, dims, keep_dim,
                                                  out);
                     }));
}

void Min(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "Min", ([&] {
                         Reduce<data_t, MinFunctor>(x, reduce_all, dims,
                                                    keep_dim, out);
                       }));
  FD_VISIT_INT_TYPES(x.dtype, "Min", ([&] {
                       Reduce<data_t, MinFunctor>(x, reduce_all, dims, keep_dim,
                                                  out);
                     }));
}

void Sum(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "Min", ([&] {
                         Reduce<data_t, SumFunctor>(x, reduce_all, dims,
                                                    keep_dim, out);
                       }));
  FD_VISIT_INT_TYPES(x.dtype, "Min", ([&] {
                       Reduce<data_t, SumFunctor>(x, reduce_all, dims, keep_dim,
                                                  out);
                     }));
}

void All(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out) {
  BoolReduceKernel<bool, AllFunctor>(x, dims, keep_dim, reduce_all, out);
}

void Any(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
         bool keep_dim, FDTensor* out) {
  BoolReduceKernel<bool, AnyFunctor>(x, dims, keep_dim, reduce_all, out);
}

void Mean(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
          bool keep_dim, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "Min", ([&] {
                         Reduce<data_t, MeanFunctor>(x, reduce_all, dims,
                                                     keep_dim, out);
                       }));
  FD_VISIT_INT_TYPES(x.dtype, "Min", ([&] {
                       Reduce<data_t, MeanFunctor>(x, reduce_all, dims,
                                                   keep_dim, out);
                     }));
}

void Prod(const FDTensor& x, bool reduce_all, const std::vector<int64_t>& dims,
          bool keep_dim, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "Min", ([&] {
                         Reduce<data_t, ProdFunctor>(x, reduce_all, dims,
                                                     keep_dim, out);
                       }));
  FD_VISIT_INT_TYPES(x.dtype, "Min", ([&] {
                       Reduce<data_t, ProdFunctor>(x, reduce_all, dims,
                                                   keep_dim, out);
                     }));
}

}  // namespace fastdeploy