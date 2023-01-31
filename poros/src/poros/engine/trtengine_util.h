// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
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

/**
* @file trtengine_util.h
* @author tianjinjin@baidu.com
* @date Wed Jul 21 11:45:49 CST 2021
* @brief 
**/

#pragma once

//from pytorch
#include "torch/script.h"
//from tensorrt
#include "NvInfer.h"

namespace baidu {
namespace mirana {
namespace poros {

//实现nvinfer::Dims的 == 运算符
inline bool operator==(const nvinfer1::Dims& in1, const nvinfer1::Dims& in2) {
    if (in1.nbDims != in2.nbDims) {
        return false;
    }
    // TODO maybe look to support broadcasting comparisons
    for (int64_t i = 0; i < in1.nbDims; i++) {
        if (in1.d[i] != in2.d[i]) {
            return false;
        }
    }
    return true;
}

//实现nvinfer::Dims的 != 运算符
inline bool operator!=(const nvinfer1::Dims& in1, const nvinfer1::Dims& in2) {
    return !(in1 == in2);
}

//实现nvinfer::Dims的<<运算符
template <typename T>
inline std::ostream& print_sequence(std::ostream& stream, const T* begin, int count) {
    stream << "[";
    if (count > 0) {
        std::copy_n(begin, count - 1, std::ostream_iterator<T>(stream, ", "));
        stream << begin[count - 1];
    }
    stream << "]";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::Dims& shape) {
    return print_sequence(stream, shape.d, shape.nbDims);
}

//实现nvinfer::DataType的<<运算符
inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::DataType& dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return stream << "Float32";
        case nvinfer1::DataType::kHALF:
            return stream << "Float16";
        case nvinfer1::DataType::kINT8:
            return stream << "Int8";
        case nvinfer1::DataType::kINT32:
            return stream << "Int32";
        case nvinfer1::DataType::kBOOL:
            return stream << "Bool";
        default:
            return stream << "Unknown Data Type";
    }
}

// 创建智能指针
template <class T>
std::shared_ptr<T> make_shared_ptr(T* p) {
    return std::shared_ptr<T>(p);
}

//int64_t volume(const nvinfer1::Dims& dim);  //move to nvdim_to_volume
bool broadcastable(nvinfer1::Dims a, nvinfer1::Dims b, bool multidirectional = true);

//以下四个函数，实现tensorrt的dims结构与vec形式的sizes的互换。
nvinfer1::Dims sizes_to_nvdim(const std::vector<int64_t>& sizes);
nvinfer1::Dims sizes_to_nvdim(c10::IntArrayRef sizes);
nvinfer1::Dims sizes_to_nvdim(c10::List<int64_t> sizes);
nvinfer1::Dims sizes_to_nvdim_with_pad(c10::IntArrayRef sizes, uint64_t pad_to);
nvinfer1::Dims sizes_to_nvdim_with_pad(c10::List<int64_t> sizes, uint64_t pad_to);
//以下三个函数，实现tensorrt的dim到其他形式的转换。
std::vector<int64_t> nvdim_to_sizes(const nvinfer1::Dims& dim);
std::string nvdim_to_str(const nvinfer1::Dims& dim);
int64_t nvdim_to_volume(const nvinfer1::Dims& dim);

//以下一个函数，实现dim的unpad
nvinfer1::Dims unpad_nvdim(const nvinfer1::Dims& dim);

//以下两个函数，实现tensorrt与aten的类型互转
//transform tensorrt-type to aten-type(which used in pytorch and torchscript)
at::ScalarType nvtype_to_attype(nvinfer1::DataType type);
//transform aten-type(which used in pytorch and torchscript) to tensorrt-type
nvinfer1::DataType attype_to_nvtype(at::ScalarType type);

//以下两个函数，实现tensorrt的dims的展开和压缩(???)。
nvinfer1::Dims unsqueeze_dims(const nvinfer1::Dims& d, int pos, int val = 1, bool use_zeros = true);
nvinfer1::Dims squeeze_dims(const nvinfer1::Dims& d, int pos, bool use_zeros = true);

/*
* @brief 通过node的输入信息，获取相应的tensor的类型。
**/
bool gen_tensor_type(const torch::jit::Node& node, const size_t index, nvinfer1::DataType & nv_type);
// 检查输入nvtensor是否为dynamic输入
bool check_nvtensor_is_dynamic(const nvinfer1::ITensor* nvtensor);
// 检查一个子图输入是否为dynamic
bool input_is_dynamic(torch::jit::Value* input);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
