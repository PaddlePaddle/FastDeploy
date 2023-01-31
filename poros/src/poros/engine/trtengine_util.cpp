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
* @file trtengine_util.cpp
* @author tianjinjin@baidu.com
* @date Wed Jul 21 11:45:49 CST 2021
* @brief 
**/
#include "poros/context/poros_global.h"
#include "poros/engine/trtengine_util.h"
#include "poros/util/poros_util.h"
#include "poros/util/macros.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
const std::unordered_map<at::ScalarType, nvinfer1::DataType>& get_at_trt_type_map() {
    static const std::unordered_map<at::ScalarType, nvinfer1::DataType> at_trt_type_map = {
        {at::kFloat, nvinfer1::DataType::kFLOAT},
        {at::kHalf, nvinfer1::DataType::kHALF},
        {at::kInt, nvinfer1::DataType::kINT32},
        {at::kChar, nvinfer1::DataType::kINT8},
        {at::kBool, nvinfer1::DataType::kBOOL},
        {at::kByte, nvinfer1::DataType::kINT8},
    };
    return at_trt_type_map;
}

const std::unordered_map<nvinfer1::DataType, at::ScalarType>& get_trt_at_type_map() {
    static const std::unordered_map<nvinfer1::DataType, at::ScalarType> trt_at_type_map = {
        {nvinfer1::DataType::kFLOAT, at::kFloat},
        {nvinfer1::DataType::kHALF, at::kHalf},
        {nvinfer1::DataType::kINT32, at::kInt},
        {nvinfer1::DataType::kINT8, at::kByte},  //TODO: should trans kChar or kByte???
        {nvinfer1::DataType::kBOOL, at::kBool},
    };
    return trt_at_type_map;
}
} // namespace

bool broadcastable(nvinfer1::Dims a, nvinfer1::Dims b, bool multidirectional) {
    if (a == b) {
        return true;
    }
    
    if (multidirectional) {
        nvinfer1::Dims a_dims_eq;
        nvinfer1::Dims b_dims_eq;
        if (a.nbDims > b.nbDims) {
            a_dims_eq = a;
            b_dims_eq = sizes_to_nvdim_with_pad(nvdim_to_sizes(b), a.nbDims);
        } else if (a.nbDims < b.nbDims) {
            a_dims_eq = sizes_to_nvdim_with_pad(nvdim_to_sizes(a), b.nbDims);
            b_dims_eq = b;
        } else {
            a_dims_eq = a;
            b_dims_eq = b;
        }

        bool broadcastable = true;
        for (int i = 0; i < a_dims_eq.nbDims; i++) {
            if (b_dims_eq.d[i] == a_dims_eq.d[i] || (b_dims_eq.d[i] == 1 || a_dims_eq.d[i] == 1)) {
                continue;
            } else {
                broadcastable = false;
                break;
            }
        }
        return broadcastable;
    } else {
        nvinfer1::Dims b_dims_eq;
        if (a.nbDims > b.nbDims) {
            b_dims_eq = sizes_to_nvdim_with_pad(nvdim_to_sizes(b), a.nbDims);
        } else if (a.nbDims < b.nbDims) {
            return false;
        } else {
            b_dims_eq = b;
        }

        bool broadcastable = true;
        for (int i = 0; i < a.nbDims; i++) {
            if (b_dims_eq.d[i] == a.d[i] || b_dims_eq.d[i] == 1) {
                continue;
            } else {
                broadcastable = false;
                break;
            }
        }
        return broadcastable;
    }
}

nvinfer1::Dims sizes_to_nvdim(const std::vector<int64_t>& sizes) {
    if (sizes.size() > nvinfer1::Dims::MAX_DIMS) {
        LOG(FATAL) << "given sizes is exceed of max dims of tensorrt";
        throw std::runtime_error("given sizes is exceed of max dims of tensorrt");
    }
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (size_t i = 0; i < sizes.size(); i++) {
        dims.d[i] = sizes[i];
    }
    return dims;
}

nvinfer1::Dims sizes_to_nvdim(c10::IntArrayRef sizes) {
    if (sizes.size() > nvinfer1::Dims::MAX_DIMS) {
        LOG(FATAL) << "given sizes is exceed of max dims of tensorrt";
        throw std::runtime_error("given sizes is exceed of max dims of tensorrt");
    }
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (size_t i = 0; i < sizes.size(); i++) {
        dims.d[i] = sizes[i];
    }
    return dims;
}

nvinfer1::Dims sizes_to_nvdim(c10::List<int64_t> sizes) {
    if (sizes.size() > nvinfer1::Dims::MAX_DIMS) {
        LOG(FATAL) << "given sizes is exceed of max dims of tensorrt";
        throw std::runtime_error("given sizes is exceed of max dims of tensorrt");
    }
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (size_t i = 0; i < sizes.size(); i++) {
        dims.d[i] = sizes[i];
    }
    return dims;
}

nvinfer1::Dims sizes_to_nvdim_with_pad(c10::IntArrayRef sizes, uint64_t pad_to) {
    if (pad_to > nvinfer1::Dims::MAX_DIMS || sizes.size() > nvinfer1::Dims::MAX_DIMS) {
        LOG(FATAL) << "given sizes is exceed of max dims of tensorrt";
        throw std::runtime_error("given sizes is exceed of max dims of tensorrt");
    }

    nvinfer1::Dims dims;
    //no need padding situation
    if (sizes.size() > pad_to) {
        dims.nbDims = sizes.size();
        for (size_t i = 0; i < sizes.size(); i++) {
            dims.d[i] = sizes[i];
        }
    //need padding situation
    } else {  
        dims.nbDims = pad_to;
        for (size_t i = 0; i < pad_to - sizes.size(); i++) {
            dims.d[i] = 1;
        }
        for (size_t i = pad_to - sizes.size(); i < pad_to; i++) {
            dims.d[i] = sizes[i - (pad_to - sizes.size())];
        }
    }
    return dims;
}

nvinfer1::Dims sizes_to_nvdim_with_pad(c10::List<int64_t> sizes, uint64_t pad_to) {
    if (pad_to > nvinfer1::Dims::MAX_DIMS || sizes.size() > nvinfer1::Dims::MAX_DIMS) {
        LOG(FATAL) << "given sizes is exceed of max dims of tensorrt";
        throw std::runtime_error("given sizes is exceed of max dims of tensorrt");
    }

    nvinfer1::Dims dims;
    //no need padding situation
    if (sizes.size() > pad_to) { 
        LOG(INFO) << "no need to pad, give sizes: " << sizes.size() 
                  << ", expected dims: " << pad_to;
        dims.nbDims = sizes.size();
        for (size_t i = 0; i < sizes.size(); i++) {
            dims.d[i] = sizes[i];
        }
    //need padding situation
    } else {  
        dims.nbDims = pad_to;
        for (size_t i = 0; i < pad_to - sizes.size(); i++) {
            dims.d[i] = 1;
        }
        for (size_t i = pad_to - sizes.size(); i < pad_to; i++) {
            dims.d[i] = sizes[i - (pad_to - sizes.size())];
        }
    }
    return dims;    
}

std::vector<int64_t> nvdim_to_sizes(const nvinfer1::Dims& dim) {
    std::vector<int64_t> sizes;
    for (int i = 0; i < dim.nbDims; i++) {
        sizes.push_back(dim.d[i]);
    }
    return std::move(sizes);
}

std::string nvdim_to_str(const nvinfer1::Dims& dim) {
    std::stringstream ss;
    ss << dim;
    return ss.str();
}

int64_t nvdim_to_volume(const nvinfer1::Dims& dim) {
    return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int64_t>());
}

nvinfer1::Dims unpad_nvdim(const nvinfer1::Dims& dim) {
    nvinfer1::Dims new_dim;
    int j = 0;
    bool pad_dims_done = false;
    
    for (int i = 0; i < dim.nbDims; i++) {
        if (dim.d[i] == 1 && !pad_dims_done) {
            // skip over unecessary dimension
            continue;
        } else {
            new_dim.d[j] = dim.d[i];
            j++;
            // keep all other dimensions (don't skip over them)
            pad_dims_done = true;
        }
    }
    new_dim.nbDims = j;
    return new_dim;
}

bool gen_tensor_type(const torch::jit::Node& node, const size_t index, nvinfer1::DataType& nv_type) {
    c10::optional<at::ScalarType> maybe_type;
    //at::ArrayRef<const torch::jit::Value*> inputs = node.inputs();
    std::shared_ptr<torch::jit::Graph> subgraph = node.g(torch::jit::attr::Subgraph);
    at::ArrayRef<torch::jit::Value*> inputs = subgraph->inputs();
    //for (size_t index = 0; index < inputs.size(); index++) {
    auto value = inputs[index];

    //extract scalar type from tensor.
    if (value->type()->isSubtypeOf(c10::TensorType::get())) {
        c10::TensorTypePtr op = value->type()->cast<c10::TensorType>();
        if (op->scalarType().has_value()) {
            maybe_type = op->scalarType().value();
        }

    //extract scalar type from tensorlist.
    } else if (value->type()->isSubtypeOf(c10::ListType::ofTensors())) {
        auto list_element = value->type()->cast<c10::ListType>()->getElementType();
        //TODO: ADD SOPPORT HERE
        LOG(WARNING) << "gen_tensor_type for tensorlist to add more";
        return false;
    }

    //this is added because tensorrt only support five kinds of date type 
    //(kFloat / kHalf / kINT8 / kINT32 / kBOOL) for now. (2021.08.01)
    if (maybe_type.has_value() && maybe_type.value() != at::kFloat &&
        maybe_type.value() != at::kHalf && maybe_type.value() != at::kChar &&
        maybe_type.value() != at::kInt && maybe_type.value() != at::kBool) {
        // when we meet at::KLong and globalContext allow us to down to at::KInt
        if (maybe_type.value() == at::kLong && PorosGlobalContext::instance().get_poros_options().long_to_int == true) {
            nv_type = attype_to_nvtype(at::kInt);
            LOG(WARNING) << "gen_tensor_type meets at::KLong tensor type, change this to at::KInt. "
                    << "Attention: this may leed to percision change";
            return true;       
        }
        LOG(WARNING) << "gen_tensor_type failed, reason: "
                 << "given scalartype is not supported by tensorrt";
        return false;
    }

    if (maybe_type.has_value()) {
        nv_type = attype_to_nvtype(maybe_type.value());
        return true;
    } else {
        LOG(WARNING) << "gen_tensor_type failed, reason: "
            << "cant't extract scalar type from all the input value";
        return false;
    }
}

at::ScalarType nvtype_to_attype(nvinfer1::DataType type) {
    auto trt_at_type_map = get_trt_at_type_map();
    if (trt_at_type_map.find(type) == trt_at_type_map.end()) {
        LOG(FATAL) << "unsupported tensorrt datatype";
        throw std::runtime_error("unsupported tensorrt datatype");
    }
    return trt_at_type_map.at(type);
}

nvinfer1::DataType attype_to_nvtype(at::ScalarType type) {
    auto at_trt_type_map = get_at_trt_type_map();
    if (at_trt_type_map.find(type) == at_trt_type_map.end()) {
        LOG(FATAL) << "unsupported aten datatype";
        throw std::runtime_error("unsupported aten datatype");
    }
    return at_trt_type_map.at(type);
}

nvinfer1::Dims unsqueeze_dims(const nvinfer1::Dims& d, int pos, int val, bool use_zeros) {
    // acceptable range for pos is [0, d.nbDims]
    POROS_CHECK(pos >= 0 && pos <= d.nbDims, "ERROR: Index to unsqueeze is out of bounds.");
    nvinfer1::Dims dims;
    for (int i = 0, j = 0; j <= d.nbDims; j++) {
        // add new dimension at pos
        if (j == pos) {
            dims.d[j] = val;
        } else {
            dims.d[j] = (use_zeros && d.d[i] == -1) ? 0 : d.d[i];
            ++i;
        }
    }
    dims.nbDims = d.nbDims + 1;
    return dims;
}

nvinfer1::Dims squeeze_dims(const nvinfer1::Dims& d, int pos, bool use_zeros) {
    // acceptable range for pos is [0, d.nbDims]
    POROS_CHECK(pos >= 0 && pos <= d.nbDims, "ERROR: Index to unsqueeze is out of bounds.");
    nvinfer1::Dims dims;
    int j = 0;
    for (int i = 0; i < d.nbDims; i++) {
        if (i != pos) {
            dims.d[j++] = (use_zeros && d.d[i] == -1) ? 0 : d.d[i];
        }
    }
    dims.nbDims = j;
    return dims;
}

bool check_nvtensor_is_dynamic(const nvinfer1::ITensor* nvtensor) {
    POROS_CHECK(nvtensor != nullptr, "input nvtensor is null");
    nvinfer1::Dims nvtensor_dims = nvtensor->getDimensions();
    for (int i = 0; i < nvtensor_dims.nbDims; i++) {
        if (nvtensor_dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

bool input_is_dynamic(torch::jit::Value* input) {
    auto _value_dynamic_shape_map = PorosGlobalContext::instance()._value_dynamic_shape_map;
    if (_value_dynamic_shape_map.find(input) != _value_dynamic_shape_map.end()) {
        auto min_shapes = _value_dynamic_shape_map[input].min_shapes;
        auto max_shapes = _value_dynamic_shape_map[input].max_shapes;
        for(size_t i = 0; i < min_shapes.size(); i++) {
            if (max_shapes[i] != min_shapes[i]) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
