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

void BindRuntime(pybind11::module&);
void BindFDModel(pybind11::module&);
void BindVision(pybind11::module&);

pybind11::dtype FDDataTypeToNumpyDataType(const FDDataType& fd_dtype) {
  pybind11::dtype dt;
  if (fd_dtype == FDDataType::INT32) {
    dt = pybind11::dtype::of<int32_t>();
  } else if (fd_dtype == FDDataType::INT64) {
    dt = pybind11::dtype::of<int64_t>();
  } else if (fd_dtype == FDDataType::FP32) {
    dt = pybind11::dtype::of<float>();
  } else if (fd_dtype == FDDataType::FP64) {
    dt = pybind11::dtype::of<double>();
  } else {
    FDASSERT(false, "The function doesn't support data type of " +
                        Str(fd_dtype) + ".");
  }
  return dt;
}

FDDataType NumpyDataTypeToFDDataType(const pybind11::dtype& np_dtype) {
  if (np_dtype.is(pybind11::dtype::of<int32_t>())) {
    return FDDataType::INT32;
  } else if (np_dtype.is(pybind11::dtype::of<int64_t>())) {
    return FDDataType::INT64;
  } else if (np_dtype.is(pybind11::dtype::of<float>())) {
    return FDDataType::FP32;
  } else if (np_dtype.is(pybind11::dtype::of<double>())) {
    return FDDataType::FP64;
  }
  FDASSERT(false,
           "NumpyDataTypeToFDDataType() only support "
           "int32/int64/float32/float64 now.");
  return FDDataType::FP32;
}

void PyArrayToTensor(pybind11::array& pyarray, FDTensor* tensor,
                     bool share_buffer) {
  tensor->dtype = NumpyDataTypeToFDDataType(pyarray.dtype());
  tensor->shape.insert(tensor->shape.begin(), pyarray.shape(),
                       pyarray.shape() + pyarray.ndim());
  if (share_buffer) {
    tensor->external_data_ptr = pyarray.mutable_data();
  } else {
    tensor->data.resize(pyarray.nbytes());
    memcpy(tensor->data.data(), pyarray.mutable_data(), pyarray.nbytes());
  }
}

#ifdef ENABLE_VISION
int NumpyDataTypeToOpenCvType(const pybind11::dtype& np_dtype) {
  if (np_dtype.is(pybind11::dtype::of<int32_t>())) {
    return CV_32S;
  } else if (np_dtype.is(pybind11::dtype::of<int8_t>())) {
    return CV_8U;
  } else if (np_dtype.is(pybind11::dtype::of<uint8_t>())) {
    return CV_8U;
  } else if (np_dtype.is(pybind11::dtype::of<float>())) {
    return CV_32F;
  } else {
    FDASSERT(
        false,
        "NumpyDataTypeToOpenCvType() only support int32/int8/uint8/float32 "
        "now.");
  }
  return CV_8U;
}

cv::Mat PyArrayToCvMat(pybind11::array& pyarray) {
  auto cv_type = NumpyDataTypeToOpenCvType(pyarray.dtype());
  FDASSERT(
      pyarray.ndim() == 3,
      "Require rank of array to be 3 with HWC format while converting it to "
      "cv::Mat.");
  int channel = *(pyarray.shape() + 2);
  int height = *(pyarray.shape());
  int width = *(pyarray.shape() + 1);
  return cv::Mat(height, width, CV_MAKETYPE(cv_type, channel),
                 pyarray.mutable_data());
}
#endif

PYBIND11_MODULE(fastdeploy_main, m) {
  m.doc() =
      "Make programer easier to deploy deeplearning model, save time to save "
      "the world!";

  BindRuntime(m);
  BindFDModel(m);
#ifdef ENABLE_VISION
  auto vision_module =
      m.def_submodule("vision", "Vision module of FastDeploy.");
  BindVision(vision_module);
#endif
}

}  // namespace fastdeploy
