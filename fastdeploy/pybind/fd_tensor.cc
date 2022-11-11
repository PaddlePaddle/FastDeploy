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

#include <dlpack/dlpack.h>

#include "fastdeploy/core/fd_type.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/pybind/main.h"

namespace fastdeploy {

DLDataType FDToDlpackType(FDDataType fd_dtype) {
  DLDataType dl_dtype;
  DLDataTypeCode dl_code;

  // Number of bits required for the data type.
  size_t dt_size = 0;

  dl_dtype.lanes = 1;
  switch (fd_dtype) {
    case FDDataType::BOOL:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 1;
      break;
    case FDDataType::UINT8:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 8;
      break;
    case FDDataType::INT8:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 8;
      break;
    case FDDataType::INT16:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 16;
      break;
    case FDDataType::INT32:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 32;
      break;
    case FDDataType::INT64:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 64;
      break;
    case FDDataType::FP16:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 16;
      break;
    case FDDataType::FP32:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 32;
      break;
    case FDDataType::FP64:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 64;
      break;

    default:
      FDASSERT(false,
              "Convert to DlPack, FDType \"%s\" is not supported.", Str(fd_dtype));
  }

  dl_dtype.code = dl_code;
  dl_dtype.bits = dt_size;
  return dl_dtype;
}

FDDataType
DlpackToFDType(const DLDataType& data_type) {
  FDASSERT(data_type.lanes == 1,
          "FDTensor does not support dlpack lanes != 1")

  if (data_type.code == DLDataTypeCode::kDLFloat) {
    if (data_type.bits == 16) {
      return FDDataType::FP16;
    } else if (data_type.bits == 32) {
      return FDDataType::FP32;
    } else if (data_type.bits == 64) {
      return FDDataType::FP64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLInt) {
    if (data_type.bits == 8) {
      return FDDataType::INT8;
    } else if (data_type.bits == 16) {
      return FDDataType::INT16;
    } else if (data_type.bits == 32) {
      return FDDataType::INT32;
    } else if (data_type.bits == 64) {
      return FDDataType::INT64;
    } else if (data_type.bits == 1) {
      return FDDataType::BOOL;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLUInt) {
    if (data_type.bits == 8) {
      return FDDataType::UINT8;
    }
  }

  return FDDataType::UNKNOWN1;
}

void DeleteUnusedDltensor(PyObject* dlp) {
  if (PyCapsule_IsValid(dlp, "dltensor")) {
    DLManagedTensor* dl_managed_tensor =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(dlp, "dltensor"));
    dl_managed_tensor->deleter(dl_managed_tensor);
  }
}

pybind11::capsule FDTensorToDLPack(FDTensor& fd_tensor) {
  DLManagedTensor* dlpack_tensor = new DLManagedTensor;
  dlpack_tensor->dl_tensor.ndim = fd_tensor.shape.size();
  dlpack_tensor->dl_tensor.byte_offset = 0;
  dlpack_tensor->dl_tensor.data = fd_tensor.MutableData();
  dlpack_tensor->dl_tensor.shape = &(fd_tensor.shape[0]);
  dlpack_tensor->dl_tensor.strides = nullptr;
  dlpack_tensor->manager_ctx = &fd_tensor;
  dlpack_tensor->deleter = [](DLManagedTensor* m) {
    if (m->manager_ctx == nullptr) {
      return;
    }

    FDTensor* tensor_ptr = reinterpret_cast<FDTensor*>(m->manager_ctx);
    pybind11::handle tensor_handle = pybind11::cast(tensor_ptr);
    tensor_handle.dec_ref();
    free(m);
  };

  pybind11::handle tensor_handle = pybind11::cast(&fd_tensor);

  // Increase the reference count by one to make sure that the DLPack
  // represenation doesn't become invalid when the tensor object goes out of
  // scope.
  tensor_handle.inc_ref();

  dlpack_tensor->dl_tensor.dtype = FDToDlpackType(fd_tensor.dtype);

  // TODO(liqi): FDTensor add device_id
  dlpack_tensor->dl_tensor.device.device_id = 0;
  if(fd_tensor.device == Device::GPU) {
    if (fd_tensor.is_pinned_memory) {
      dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCUDAHost;
    } else {
      dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCUDA;
    }
  } else {
    dlpack_tensor->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
  }

  return pybind11::capsule(
      static_cast<void*>(dlpack_tensor), "dltensor", &DeleteUnusedDltensor);
}


void BindFDTensor(pybind11::module& m) {
  pybind11::class_<FDTensor>(m, "FDTensor")
      .def(pybind11::init<>(), "Default Constructor")
      .def_readwrite("name", &FDTensor::name)
      .def_readonly("shape", &FDTensor::shape)
      .def_readonly("dtype", &FDTensor::dtype)
      .def_readonly("device", &FDTensor::device)
      .def("numpy", [](FDTensor& self) {
        return TensorToPyArray(self);
      })
      .def("data", &FDTensor::MutableData)
      .def("from_numpy", [](FDTensor& self, pybind11::array& pyarray, bool share_buffer = false) {
        PyArrayToTensor(pyarray, &self, share_buffer);
      })
      .def("to_dlpack", &FDTensorToDLPack);
}

}  // namespace fastdeploy
