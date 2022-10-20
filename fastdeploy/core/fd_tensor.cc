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
#include <cstring>

#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/core/float16.h"
#include "fastdeploy/utils/utils.h"
#ifdef WITH_GPU
#include <cuda_runtime_api.h>
#endif

namespace fastdeploy {

void* FDTensor::MutableData() {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

void* FDTensor::Data() {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

const void* FDTensor::Data() const {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

const void* FDTensor::CpuData() const {
  if (device == Device::GPU) {
#ifdef WITH_GPU
    auto* cpu_ptr = const_cast<std::vector<int8_t>*>(&temporary_cpu_buffer);
    cpu_ptr->resize(Nbytes());
    // need to copy cuda mem to cpu first
    if (external_data_ptr != nullptr) {
      FDASSERT(cudaMemcpy(cpu_ptr->data(), external_data_ptr, Nbytes(),
                          cudaMemcpyDeviceToHost) == 0,
               "[ERROR] Error occurs while copy memory from GPU to CPU");

    } else {
      FDASSERT(cudaMemcpy(cpu_ptr->data(), buffer_, Nbytes(),
                          cudaMemcpyDeviceToHost) == 0,
               "[ERROR] Error occurs while buffer copy memory from GPU to CPU");
    }
    return cpu_ptr->data();
#else
    FDASSERT(false,
             "The FastDeploy didn't compile under -DWITH_GPU=ON, so this is "
             "an unexpected problem happend.");
#endif
  }
  return Data();
}

void FDTensor::SetExternalData(const std::vector<int64_t>& new_shape,
                               const FDDataType& data_type, void* data_buffer,
                               const Device& new_device) {
  dtype = data_type;
  shape.assign(new_shape.begin(), new_shape.end());
  external_data_ptr = data_buffer;
  device = new_device;
}

void FDTensor::ExpandDim(int64_t axis) {
  size_t ndim = shape.size();
  FDASSERT(axis >= 0 && axis <= ndim,
           "The allowed 'axis' must be in range of (0, %lu)!", ndim);
  shape.insert(shape.begin() + axis, 1);
}

void FDTensor::Squeeze(int64_t axis) {
  size_t ndim = shape.size();
  FDASSERT(axis >= 0 && axis < ndim,
           "The allowed 'axis' must be in range of (0, %lu)!", ndim);
  FDASSERT(shape[axis]==1,
           "The No.%ld dimension of shape should be 1, but it is %ld!", (long)axis, (long)shape[axis]);
  shape.erase(shape.begin() + axis);
}

void FDTensor::Allocate(const std::vector<int64_t>& new_shape,
                        const FDDataType& data_type,
                        const std::string& tensor_name,
                        const Device& new_device) {
  dtype = data_type;
  name = tensor_name;
  shape.assign(new_shape.begin(), new_shape.end());
  device = new_device;
  size_t nbytes = Nbytes();
  FDASSERT(ReallocFn(nbytes),
           "The FastDeploy FDTensor allocate cpu memory error");
}

int FDTensor::Nbytes() const { return Numel() * FDDataTypeSize(dtype); }

int FDTensor::Numel() const {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

void FDTensor::Resize(size_t new_nbytes) { ReallocFn(new_nbytes); }

void FDTensor::Resize(const std::vector<int64_t>& new_shape) {
  int numel = Numel();
  int new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                  std::multiplies<int>());
  if (new_numel > numel) {
    size_t nbytes = new_numel * FDDataTypeSize(dtype);
    ReallocFn(nbytes);
  }
  shape.assign(new_shape.begin(), new_shape.end());
}

void FDTensor::Resize(const std::vector<int64_t>& new_shape,
                      const FDDataType& data_type,
                      const std::string& tensor_name,
                      const Device& new_device) {
  name = tensor_name;
  device = new_device;
  dtype = data_type;
  int new_nbytes = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>()) *
                   FDDataTypeSize(data_type);
  ReallocFn(new_nbytes);
  shape.assign(new_shape.begin(), new_shape.end());
}

template <typename T>
void CalculateStatisInfo(void* src_ptr, int size, double* mean, double* max,
                         double* min) {
  T* ptr = static_cast<T*>(src_ptr);
  *mean = 0;
  *max = -99999999;
  *min = 99999999;
  for (int i = 0; i < size; ++i) {
    if (*(ptr + i) > *max) {
      *max = *(ptr + i);
    }
    if (*(ptr + i) < *min) {
      *min = *(ptr + i);
    }
    *mean += *(ptr + i);
  }
  *mean = *mean / size;
}

void FDTensor::PrintInfo(const std::string& prefix) {
  double mean = 0;
  double max = -99999999;
  double min = 99999999;
  if (dtype == FDDataType::FP32) {
    CalculateStatisInfo<float>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::FP64) {
    CalculateStatisInfo<double>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::INT8) {
    CalculateStatisInfo<int8_t>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::UINT8) {
    CalculateStatisInfo<uint8_t>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::INT32) {
    CalculateStatisInfo<int32_t>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::INT64) {
    CalculateStatisInfo<int64_t>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == FDDataType::FP16) {
    CalculateStatisInfo<float16>(Data(), Numel(), &mean, &max, &min);
  } else {
    FDASSERT(false,
             "PrintInfo function doesn't support current situation, maybe you "
             "need enhance this function now.");
  }
  std::cout << prefix << ": name=" << name << ", shape=";
  for (int i = 0; i < shape.size(); ++i) {
    std::cout << shape[i] << " ";
  }
  std::cout << ", dtype=" << Str(dtype) << ", mean=" << mean << ", max=" << max
            << ", min=" << min << std::endl;
}

bool FDTensor::ReallocFn(size_t nbytes) {
  if (is_pinned_memory) {
#ifdef WITH_GPU
    size_t original_nbytes = Nbytes();
    if (nbytes > original_nbytes) {
      if (buffer_ != nullptr) {
        FDDeviceHostFree()(buffer_);
      }
      FDDeviceHostAllocator()(&buffer_, nbytes);
    }
    return buffer_ != nullptr;
#else
    FDASSERT(false,
             "The FastDeploy FDTensor allocator didn't compile under "
             "-DWITH_GPU=ON,"
             "so this is an unexpected problem happend.");
#endif
  }
  if (device == Device::GPU) {
#ifdef WITH_GPU
    size_t original_nbytes = Nbytes();
    if (nbytes > original_nbytes) {
      if (buffer_ != nullptr) {
        FDDeviceFree()(buffer_);
      }
      FDDeviceAllocator()(&buffer_, nbytes);
    }
    return buffer_ != nullptr;
#else
    FDASSERT(false,
             "The FastDeploy FDTensor allocator didn't compile under "
             "-DWITH_GPU=ON,"
             "so this is an unexpected problem happend.");
#endif
  }
  buffer_ = realloc(buffer_, nbytes);
  return buffer_ != nullptr;
}

void FDTensor::FreeFn() {
  if (external_data_ptr != nullptr) external_data_ptr = nullptr;
  if (buffer_ != nullptr) {
    if (is_pinned_memory) {
#ifdef WITH_GPU
      FDDeviceHostFree()(buffer_);
#endif
    } else if (device == Device::GPU) {
#ifdef WITH_GPU
      FDDeviceFree()(buffer_);
#endif
    } else {
      FDHostFree()(buffer_);
    }
    buffer_ = nullptr;
  }
}

void FDTensor::CopyBuffer(void* dst, const void* src, size_t nbytes) {
  if (is_pinned_memory) {
#ifdef WITH_GPU
    FDASSERT(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToHost) == 0,
             "[ERROR] Error occurs while copy memory from host to host");
#else
    FDASSERT(false,
             "The FastDeploy didn't compile under -DWITH_GPU=ON, so copying "
             "gpu buffer is "
             "an unexpected problem happend.");
#endif
  } else if (device == Device::GPU) {
#ifdef WITH_GPU
    FDASSERT(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice) == 0,
             "[ERROR] Error occurs while copy memory from GPU to GPU");
#else
    FDASSERT(false,
             "The FastDeploy didn't compile under -DWITH_GPU=ON, so copying "
             "gpu buffer is "
             "an unexpected problem happend.");
#endif
  } else {
    std::memcpy(dst, src, nbytes);
  }
}

FDTensor::FDTensor(const std::string& tensor_name) { name = tensor_name; }

FDTensor::FDTensor(const FDTensor& other)
    : shape(other.shape),
      name(other.name),
      dtype(other.dtype),
      device(other.device),
      external_data_ptr(other.external_data_ptr) {
  // Copy buffer
  if (other.buffer_ == nullptr) {
    buffer_ = nullptr;
  } else {
    size_t nbytes = Nbytes();
    FDASSERT(ReallocFn(nbytes),
             "The FastDeploy FDTensor allocate memory error");
    CopyBuffer(buffer_, other.buffer_, nbytes);
  }
}

FDTensor::FDTensor(FDTensor&& other)
    : buffer_(other.buffer_),
      shape(std::move(other.shape)),
      name(std::move(other.name)),
      dtype(other.dtype),
      external_data_ptr(other.external_data_ptr),
      device(other.device) {
  other.name = "";
  // Note(zhoushunjie): Avoid double free.
  other.buffer_ = nullptr;
  other.external_data_ptr = nullptr;
}

FDTensor& FDTensor::operator=(const FDTensor& other) {
  if (&other != this) {
    // Copy buffer
    if (other.buffer_ == nullptr) {
      FreeFn();
      buffer_ = nullptr;
    } else {
      Resize(other.shape);
      size_t nbytes = Nbytes();
      CopyBuffer(buffer_, other.buffer_, nbytes);
    }

    shape = other.shape;
    name = other.name;
    dtype = other.dtype;
    device = other.device;
    external_data_ptr = other.external_data_ptr;
  }
  return *this;
}

FDTensor& FDTensor::operator=(FDTensor&& other) {
  if (&other != this) {
    FreeFn();
    buffer_ = other.buffer_;
    external_data_ptr = other.external_data_ptr;

    shape = std::move(other.shape);
    name = std::move(other.name);
    dtype = other.dtype;
    device = other.device;

    other.name = "";
    // Note(zhoushunjie): Avoid double free.
    other.buffer_ = nullptr;
    other.external_data_ptr = nullptr;
  }
  return *this;
}

}  // namespace fastdeploy
