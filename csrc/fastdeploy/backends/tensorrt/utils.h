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

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

struct FDInferDeleter {
  template<typename T> void operator()(T* obj) const {
    delete obj;
  }
};

template<typename T> using FDUniquePtr = std::unique_ptr<T, FDInferDeleter>;

inline uint32_t GetElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kBOOL:
  case nvinfer1::DataType::kINT8:
    return 1;
  }
  return 0;
}

inline int64_t Volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline nvinfer1::Dims ToDims(const std::vector<int>& vec) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    FDWARNING << "Vector too long, only first 8 elements are used in dimension." << std::endl;
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

template <typename AllocFunc, typename FreeFunc> class FDGenericBuffer {
 public:
  //!
  //! \brief Construct an empty buffer.
  //!
  explicit FDGenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
      : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {}

  //!
  //! \brief Construct a buffer with the specified allocation size in bytes.
  //!
  FDGenericBuffer(size_t size, nvinfer1::DataType type)
      : mSize(size), mCapacity(size), mType(type) {
    if (!allocFn(&mBuffer, this->nbBytes())) {
      throw std::bad_alloc();
    }
  }

  FDGenericBuffer(FDGenericBuffer&& buf)
      : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType),
        mBuffer(buf.mBuffer) {
    buf.mSize = 0;
    buf.mCapacity = 0;
    buf.mType = nvinfer1::DataType::kFLOAT;
    buf.mBuffer = nullptr;
  }

  FDGenericBuffer& operator=(FDGenericBuffer&& buf) {
    if (this != &buf) {
      freeFn(mBuffer);
      mSize = buf.mSize;
      mCapacity = buf.mCapacity;
      mType = buf.mType;
      mBuffer = buf.mBuffer;
      // Reset buf.
      buf.mSize = 0;
      buf.mCapacity = 0;
      buf.mBuffer = nullptr;
    }
    return *this;
  }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  void* data() { return mBuffer; }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  const void* data() const { return mBuffer; }

  //!
  //! \brief Returns the size (in number of elements) of the buffer.
  //!
  size_t size() const { return mSize; }

  //!
  //! \brief Returns the size (in bytes) of the buffer.
  //!
  size_t nbBytes() const {
    return this->size() * GetElementSize(mType);
  }

  //!
  //! \brief Resizes the buffer. This is a no-op if the new size is smaller than
  //! or equal to the current capacity.
  //!
  void resize(size_t newSize) {
    mSize = newSize;
    if (mCapacity < newSize) {
      freeFn(mBuffer);
      if (!allocFn(&mBuffer, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      mCapacity = newSize;
    }
  }

  //!
  //! \brief Overload of resize that accepts Dims
  //!
  void resize(const nvinfer1::Dims& dims) {
    return this->resize(Volume(dims));
  }

  ~FDGenericBuffer() { freeFn(mBuffer); }

 private:
  size_t mSize{0}, mCapacity{0};
  nvinfer1::DataType mType;
  void* mBuffer;
  AllocFunc allocFn;
  FreeFunc freeFn;
};

class FDDeviceAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    return cudaMalloc(ptr, size) == cudaSuccess;
  }
};

class FDDeviceFree {
 public:
  void operator()(void* ptr) const { cudaFree(ptr); }
};

using FDDeviceBuffer = FDGenericBuffer<FDDeviceAllocator, FDDeviceFree>;

class FDTrtLogger : public nvinfer1::ILogger {
 public:
  static FDTrtLogger* logger;
  static FDTrtLogger* Get() {
    if (logger != nullptr) {
      return logger;
    }
    logger = new FDTrtLogger();
    return logger;
  }
  void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
    if (severity == nvinfer1::ILogger::Severity::kINFO) {
      FDINFO << msg << std::endl;
    } else if (severity == nvinfer1::ILogger::Severity::kWARNING) {
      FDWARNING << msg << std::endl;
    } else if (severity == nvinfer1::ILogger::Severity::kERROR) {
      FDERROR << msg << std::endl;
    } else if (severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR) {
      FDASSERT(false, "%s", msg);
    }
  }
};

}  // namespace fastdeploy
