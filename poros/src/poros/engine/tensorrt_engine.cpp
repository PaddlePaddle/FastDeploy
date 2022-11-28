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
* @file tensorrt_engine.cpp
* @author tianjinjin@baidu.com
* @author huangben@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/
#include "poros/engine/tensorrt_engine.h"

#include "poros/context/poros_global.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/iconverter.h"
// #include "poros/engine/trtengine_util.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {


TensorrtEngine::TensorrtEngine() : _logger(get_nvlogger().torch_level()), _builder(nullptr), 
    _network(nullptr), _cfg(nullptr), _runtime(nullptr), _cuda_engine(nullptr), 
    _exec_ctx(nullptr), _mutex(nullptr) {
    // init nvinfer plgins
    initLibNvInferPlugins(&_logger, "");
}

TensorrtEngine::TensorrtEngine(std::string engine_str) : _logger(get_nvlogger().torch_level()) {

    init();

    _cuda_engine = make_shared_ptr(_runtime->deserializeCudaEngine((void*)engine_str.c_str(), engine_str.length()));
    _exec_ctx = make_shared_ptr(_cuda_engine->createExecutionContext());
    binding_io();
}

//TensorrtEngine::~TensorrtEngine() {
//    //_exec_ctx->destroy();
//    //_cuda_engine->destroy();
//    //_runtime->destroy();
//
//}

void TensorrtEngine::binding_io() {
    uint64_t inputs = 0;
    uint64_t outputs = 0;

    for (int64_t idx = 0; idx < _cuda_engine->getNbBindings(); idx++) {
        std::string name = _cuda_engine->getBindingName(idx);
        //if (name.find("profile") != name.npos) {
        //    continue;
        //}
        std::string idx_s = name.substr(name.find("_") + 1);
        uint64_t idx_new = static_cast<uint64_t>(std::stoi(idx_s));
        if (_cuda_engine->bindingIsInput(idx)) {
            inputs++;
            _in_binding_map[idx] = idx_new;
        } else {
            outputs++;
            _out_binding_map[idx] = idx_new;
        }
    }
    _num_io = std::make_pair(inputs, outputs);
}

int TensorrtEngine::init() {
    // init nvinfer plgins
    initLibNvInferPlugins(&_logger, "");
    _mutex = std::make_shared<std::mutex>();
    _poros_options = PorosGlobalContext::instance().get_poros_options();

    _builder = make_shared_ptr(nvinfer1::createInferBuilder(_logger));
    _network = make_shared_ptr(_builder->createNetworkV2(1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    _cfg = make_shared_ptr(_builder->createBuilderConfig());
    _runtime = make_shared_ptr(nvinfer1::createInferRuntime(_logger));


    // Nvidia tf32 is enabled by default. 
    // if don't want to ues, the BuilderFlag::kTF32 should be clear.
    if (!_poros_options.use_nvidia_tf32) {
        _cfg->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }
    if (_poros_options.use_fp16) {
        _cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
#if NV_TENSORRT_MAJOR >=8 && NV_TENSORRT_MINOR >=3
    // trt version >= 8.3
    _cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, _poros_options.max_workspace_size);
#else
    _cfg->setMaxWorkspaceSize(_poros_options.max_workspace_size);
#endif
    return 0;
}


int TensorrtEngine::transform(const PorosGraph& sub_graph) {
    //step1. get the given graph
    torch::jit::Graph* to_trans_graph = sub_graph.graph;
    //PorosGraph to_trans_sub_graph = {to_trans_graph.get(), sub_graph.node};

    //step2. init the engine input
    if (init_engine_inputs(sub_graph) < 0) {
        LOG(ERROR) << " init engine inputs failed";
        return -1;
    }

    //step3. get op converter_map that tensortengine supports
    ConvertersMap* converter_map = PorosGlobalContext::instance().get_converter_map(who_am_i());
    if (converter_map == nullptr) {
        LOG(ERROR) << "could not find given engine [ " << who_am_i() << " ] in global context";
        return -1;
    }

    //step4. converter the nodes in the given graph one by one. this is the core function.
    const torch::jit::Block* block = to_trans_graph->block();
    for (const torch::jit::Node* node : block->nodes()) {
        IConverter* conv = converter_map->get_converter(node);
        if (nullptr ==  conv) {
            LOG(ERROR) << "pre judgment failed: " << node_info(node);
            return -1; 
        }
        
        LOG(INFO) << "start to converter for: " << node_info(node);
        if (!conv->converter(this, node)) {
            LOG(ERROR) << "converter for node failed [ " << *node->maybeSchema() <<  " ]";
            return -1;        
        }
    }

    //step5. mark the graph output.
    at::ArrayRef<const torch::jit::Value*> graph_outputs = block->outputs();
    if (mark_graph_outputs(graph_outputs) < 0) {
        LOG(ERROR) << " mark graph outputs failed";
        return -1;
    }

    //step6. build cuda engine.
    _cuda_engine = make_shared_ptr(_builder->buildEngineWithConfig(*_network, *_cfg));
    if (!_cuda_engine) {
        LOG(ERROR) << "build tensorrt engine failed";
        return -1;
    }

    //step7. create execution context and binding io
    // Easy way to get a unique name for each engine, maybe there is a more
    // descriptive way (using something associated with the graph maybe)
    _id = reinterpret_cast<EngineID>(_cuda_engine.get());
    _exec_ctx = make_shared_ptr(_cuda_engine->createExecutionContext());
    binding_io();

    return 0;
}

//DEPRECATED
inline void TensorrtEngine::gen_tensorrt_input_type(const torch::jit::Value* input, 
                                                nvinfer1::DataType& input_type) {
    if (input->type()->isSubtypeOf(c10::BoolType::get())) {
        input_type = nvinfer1::DataType::kBOOL;
    //NumberTypes below
    } else if (input->type()->isSubtypeOf(c10::IntType::get())) {
        input_type = nvinfer1::DataType::kINT32;
    } else if (input->type()->isSubtypeOf(c10::FloatType::get())) {
        input_type = nvinfer1::DataType::kFLOAT;
    } else {
        //TODO: TO ADD LOGGER
    }
}

nvinfer1::Dims TensorrtEngine::gen_dynamic_dims(torch::jit::Value* value) {
    if (PorosGlobalContext::instance()._value_dynamic_shape_map.count(value) <= 0) {
        LOG(ERROR) << "value is not in value_dynamic_shape_map"; 
        throw std::runtime_error("value is not in value_dynamic_shape_map");
    }
    std::vector<int64_t> sizes = PorosGlobalContext::instance()._value_dynamic_shape_map[value].sizes;
    return sizes_to_nvdim(sizes);
}

//try to extract input value from subgraph_node
int TensorrtEngine::init_engine_inputs(const PorosGraph& sub_graph) {
    torch::jit::Node* subgraph_node = sub_graph.node;
    torch::jit::Graph* subgraph = sub_graph.graph;
    AT_ASSERT(subgraph_node->kind() == torch::jit::prim::CudaFusionGroup);
    at::ArrayRef<torch::jit::Value*> graph_inputs = subgraph->inputs();
    at::ArrayRef<torch::jit::Value*> node_inputs = subgraph_node->inputs();

    nvinfer1::IOptimizationProfile* profile = _builder->createOptimizationProfile();

    bool total_is_dynamic = false;
    for (size_t i = 0; i < graph_inputs.size(); i++) {
        torch::jit::Value* in = graph_inputs[i];
        torch::jit::Value* node_in = node_inputs[i];
        std::string name = std::string("input_") + std::to_string(i);

        nvinfer1::DataType nv_type;
        if (!gen_tensor_type(*subgraph_node, i, nv_type)) {
            LOG(WARNING) << "init_engine_inputs failed:  reason: can't gen nv_type info from input";
            return -1;
        }

        //根据当前poros的设计，subgraph的输入在子图分割阶段，已经全部转换为tensor
        //此处如果出现了非tensor类型，则不在poros预期内，不做处理。
        if (in->type()->isSubtypeOf(c10::TensorType::get()) == false) {
            LOG(WARNING) << "not supported input type by tensorrt: " << node_info(in->node());
            return -1;
        }

        std::vector<int64_t> sizes;
        if (!gen_dims_for_tensor(in, sizes)) {
            LOG(WARNING) << "gen_dims_for_tensor failed for: " << in->debugName();
            return -1;
        };
        nvinfer1::Dims nv_dims = sizes_to_nvdim(sizes);
        bool current_dynamic = false;
        if (std::find(sizes.begin(), sizes.end(), -1) != sizes.end() || input_is_dynamic(node_in)) {
            total_is_dynamic = true;
            current_dynamic = true;
        }
        // mark: 从nv提供的api无法先验地去判断是否是shape tensor
        // 这里先这样判断输入的tensor scalar是否属于shape tensor范围
        // 可能会有误判
        bool is_shape_tensor = false;
        if (nv_type == nvinfer1::DataType::kINT32 && nv_dims.nbDims <= 1 && 
            node_in->node()->kind() == torch::jit::aten::tensor) {
            torch::jit::use_list in_use_list = in->uses();
            for (size_t u = 0; u < in_use_list.size(); u++) {
                if (in_use_list[u].user->kind() == torch::jit::aten::IntImplicit ||
                    in_use_list[u].user->kind() == torch::jit::prim::tolist) {
                    is_shape_tensor = true;
                    _in_shape_tensor_index.emplace(i);
                    break;
                }
            }
        }
        // 上面输入为tensor scalar（nv_type是nvinfer1::DataType::kINT32且nv_dims.nbDims <= 1）的状况
        // 是我们自己通过AdjustmentSalarInputs加的，有int_intlist_values_map预热数据支持，可获取到真实的max、min、opt，
        // 而不外乎有其他tensor scalar输入的情况，此时由value_dynamic_shape_map记录的max min opt全为0，
        // 输入到engine后面converter会报错，这里需要提前拦截。
        // todo: 预热时候给其他tensor scalar加上int_intlist_values_map预热数据支持
        if (nv_dims.nbDims < 1 && !is_shape_tensor) {
            LOG(WARNING) << "init_engine_inputs failed:  reason: Meet unknown tensor scalar with 0 dim.";
            return -1;
        }

        nvinfer1::ITensor* trt_in = nullptr;
        if (is_shape_tensor) {
            c10::List<int64_t> int_sizes = {1};
            nv_dims = nv_dims.nbDims == 0 ? sizes_to_nvdim(int_sizes) : nv_dims;
            trt_in = _network->addInput(name.c_str(), nv_type, nv_dims);
            int32_t nbvalues = nv_dims.d[0];


            std::unique_ptr<int32_t[]> max_values(new int32_t[nbvalues]);
            std::unique_ptr<int32_t[]> min_values(new int32_t[nbvalues]);
            std::unique_ptr<int32_t[]> opt_values(new int32_t[nbvalues]);

            if (PorosGlobalContext::instance()._value_dynamic_shape_map.count(node_in) == 0) {
                LOG(WARNING) << "can't find %" << node_in->debugName() << " in global _value_dynamic_shape_map!";
                return -1;
            } 
            ValueDynamicShape int_value_max_min_opt;
            int_value_max_min_opt = PorosGlobalContext::instance()._value_dynamic_shape_map[node_in];

            std::vector<int64_t> min_values_in_map = int_value_max_min_opt.min_shapes;
            std::vector<int64_t> max_values_in_map = int_value_max_min_opt.max_shapes;
            std::vector<int64_t> opt_values_in_map = int_value_max_min_opt.opt_shapes;

            if ((size_t)nbvalues != min_values_in_map.size() || 
                (size_t)nbvalues != max_values_in_map.size() || 
                (size_t)nbvalues != opt_values_in_map.size()) {
                LOG(WARNING) << "input %" << node_in->debugName() << " int or int[] length must match the size of max || min || opt vector!";
                return -1;
            }
            
            for (int i = 0; i < nbvalues; i++) {
                max_values[i] = max_values_in_map[i];
                min_values[i] = min_values_in_map[i];
                opt_values[i] = opt_values_in_map[i];
            }

            bool ret_min = profile->setShapeValues(trt_in->getName(), nvinfer1::OptProfileSelector::kMIN, min_values.get(), nbvalues);
            bool ret_max = profile->setShapeValues(trt_in->getName(), nvinfer1::OptProfileSelector::kMAX, max_values.get(), nbvalues);
            bool ret_opt = profile->setShapeValues(trt_in->getName(), nvinfer1::OptProfileSelector::kOPT, opt_values.get(), nbvalues);

            if (ret_min == false || ret_opt == false || ret_max == false) {
                LOG(WARNING) << "setDimensions for value: %" << node_in->debugName() << " failed"
                            << ", min_shape_info: " << sizes_to_nvdim(min_values_in_map)
                            << ", opt_shape_info: " << sizes_to_nvdim(opt_values_in_map)
                            << ", max_shape_info: " << sizes_to_nvdim(max_values_in_map);
                return -1;
            }

            LOG(INFO) << "Init shape tensor input ok: %" << node_in->debugName()
                            << ", min_shape_info: " << sizes_to_nvdim(min_values_in_map)
                            << ", opt_shape_info: " << sizes_to_nvdim(opt_values_in_map)
                            << ", max_shape_info: " << sizes_to_nvdim(max_values_in_map);

        } else {
            if (!current_dynamic) {
                trt_in = _network->addInput(name.c_str(), nv_type, nv_dims);
                LOG(INFO) << "init static tensor input ok : " << nv_dims;
            } else {
                if (PorosGlobalContext::instance()._value_dynamic_shape_map.count(node_in) <= 0) {
                    LOG(WARNING) << "can't generate max min opt input setting for value: %" << node_in->debugName();
                    return -1;
                }
                nvinfer1::Dims dynamic_nv_dims = gen_dynamic_dims(node_in);
                trt_in = _network->addInput(name.c_str(), nv_type, dynamic_nv_dims);
                std::vector<int64_t> min_shapes = PorosGlobalContext::instance()._value_dynamic_shape_map[node_in].min_shapes;
                bool ret_min = profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kMIN, sizes_to_nvdim(min_shapes));
                std::vector<int64_t> opt_shapes = PorosGlobalContext::instance()._value_dynamic_shape_map[node_in].opt_shapes;
                bool ret_opt = profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kOPT, sizes_to_nvdim(opt_shapes));
                std::vector<int64_t> max_shapes = PorosGlobalContext::instance()._value_dynamic_shape_map[node_in].max_shapes;
                bool ret_max = profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kMAX, sizes_to_nvdim(max_shapes));
                if (ret_min == false || ret_opt == false || ret_max == false) {
                    LOG(WARNING) << "setDimensions for value: %" << node_in->debugName() << " failed"
                                << ", min_shape_info: " << sizes_to_nvdim(min_shapes)
                                << ", opt_shape_info: " << sizes_to_nvdim(opt_shapes)
                                << ", max_shape_info: " << sizes_to_nvdim(max_shapes)
                                << ", dynamic tensor info: " << dynamic_nv_dims;
                    return -1;
                }
                LOG(INFO) << "Init dynamic tensor input ok: " << nv_dims
                            << ", min_shape_info: " << sizes_to_nvdim(min_shapes)
                            << ", opt_shape_info: " << sizes_to_nvdim(opt_shapes)
                            << ", max_shape_info: " << sizes_to_nvdim(max_shapes);
            }
        }            
        _context.set_tensor(in, trt_in);
    }

    if (total_is_dynamic) {
        POROS_CHECK(profile->isValid(), "Optimization profile is invalid, please check the input range provided");
        _cfg->addOptimizationProfile(profile);
    }
    return 0;
}

int TensorrtEngine::mark_graph_outputs(at::ArrayRef<const torch::jit::Value*> outputs) {
    int index = 0;
    for (const torch::jit::Value* out : outputs) {
        auto out_tensor = _context.get_tensor(out);
        if (out_tensor == nullptr) {
            LOG(WARNING) << "can't get output tensor from context. something is wrong";
            return -1;
        }
        //output should always be a tensor according to the segmentation setting.
        std::string name = std::string("output_") + std::to_string(index++);
        out_tensor->setName(name.c_str());
        _network->markOutput(*out_tensor);
        LOG(INFO) << "mark  " << out->debugName() << " named " << name << " as graph output";
    }
    return 0;
}

//DEPRECATED
std::string TensorrtEngine::convert_graph_to_engine(std::shared_ptr<torch::jit::Graph>& graph) {
    const torch::jit::Block* block = graph->block();
    ConvertersMap* converter_map = PorosGlobalContext::instance().get_converter_map(who_am_i());
    if (converter_map == nullptr) {
        LOG(ERROR) << "could not find given engine [ " << who_am_i() << " ] in global context";
        return "";
    }

    for (const torch::jit::Node* node :  block->nodes()) {
        IConverter* conv = converter_map->get_converter(node);
        LOG(INFO) << "start to converter for: " << node_info(node);
        if (!conv->converter(this, node)) {
            LOG(ERROR) << "converter for node failed [ " << *node->maybeSchema() <<  " ]";
            return "";        
        }
    }

    at::ArrayRef<const torch::jit::Value*> outputs = block->outputs();
    if (mark_graph_outputs(outputs) < 0) {
        LOG(ERROR) << " mark graph outputs failed";
        return "";
    }

    nvinfer1::ICudaEngine*  engine = _builder->buildEngineWithConfig(*_network, *_cfg);
    if (!engine) {
        LOG(FATAL) << "build tensorrt engine failed";
    }
    
    nvinfer1::IHostMemory* serialized_engine = engine->serialize();
    engine->destroy();
    std::string engine_str = std::string((const char*)serialized_engine->data(), serialized_engine->size());
    serialized_engine->destroy();
    return engine_str;
}

void TensorrtEngine::register_module_attribute(const std::string& name, torch::jit::Module& module) {
    //auto engine_ptr = c10::make_intrusive<TensorrtEngine>(*static_cast<TensorrtEngine*>(this));
    auto engine_ptr = c10::make_intrusive<TensorrtEngine>(*this);

    module.register_attribute(
            name,
            c10::getCustomClassType<c10::intrusive_ptr<TensorrtEngine>>(),
            c10::IValue(std::move(engine_ptr)),
            false);
}

std::vector<at::Tensor> TensorrtEngine::excute_engine(const std::vector<at::Tensor>& inputs) {
    std::vector<void*> gpu_handles;

    std::vector<at::Tensor> contig_inputs{};
    contig_inputs.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++) {
        uint64_t pyt_idx = _in_binding_map[i];
        //auto expected_type = nvtype_to_attype(_exec_ctx->getEngine().getBindingDataType(i));
        // POROS_CHECK(inputs[pyt_idx].dtype() == expected_type,
        //     "Expected input tensors to have type " << expected_type << ", found type " << inputs[pyt_idx].dtype());

        nvinfer1::Dims dims = sizes_to_nvdim_with_pad(inputs[pyt_idx].sizes(), 1);
        std::vector<int64_t> shape = nvdim_to_sizes(dims);
        // at::ScalarType::Long -> at::ScalarType::Int
        if (inputs[pyt_idx].scalar_type() == c10::ScalarType::Long) {
            LOG(WARNING) << "excute_engine input meets c10::ScalarType::Long tensor type, change this to c10::ScalarType::Int. "
                    << "Attention: this may leed to percision change";
            contig_inputs.push_back(inputs[pyt_idx].to(at::ScalarType::Int).view(shape).contiguous());
        } else {
            contig_inputs.push_back(inputs[pyt_idx].view(shape).contiguous());
        }

        // 输入可能不在cuda上面，要tocuda
        if (contig_inputs[i].device() != c10::DeviceType::CUDA) {
            contig_inputs[i] = contig_inputs[i].to(c10::kCUDA).contiguous();
        }
        // set input shape binding for nvidia shape tensor
        if (_in_shape_tensor_index.count(i) > 0) {
            size_t data_nb = inputs[pyt_idx].sizes()[0];
            if (data_nb == 0) {
                int32_t set_shape_int = c10::IValue(inputs[pyt_idx].item()).toInt();
                if (!_exec_ctx->setInputShapeBinding(i, &set_shape_int)) {
                    throw std::runtime_error("tensorrt setInputShapeBinding error");
                }
            } else {
                std::unique_ptr<int32_t[]> set_shape_ints(new int32_t[data_nb]);
                for (size_t s = 0; s < data_nb; s++) {
                    c10::IValue tmp_ivalue(inputs[pyt_idx][s].item());
                    if (tmp_ivalue.isInt()) {
                        set_shape_ints[s] = tmp_ivalue.toInt();
                    }
                }
                if (!_exec_ctx->setInputShapeBinding(i, set_shape_ints.get())) {
                    throw std::runtime_error("tensorrt setInputShapeBinding error");
                }
            }

        } else {
            if (_exec_ctx->setBindingDimensions(i, dims) == false) {
                throw std::runtime_error("tensorrt setBindingDimensions error");
            }
        }
        gpu_handles.push_back(contig_inputs.back().data_ptr());
    }

    std::vector<at::Tensor> outputs(_num_io.second);
    for (size_t o = inputs.size(); o < (_num_io.first + _num_io.second); o++) {
        uint64_t pyt_idx = _out_binding_map[o];
        nvinfer1::Dims out_shape = _exec_ctx->getBindingDimensions(o);
        std::vector<int64_t> dims = nvdim_to_sizes(out_shape);
        at::ScalarType type = nvtype_to_attype(_exec_ctx->getEngine().getBindingDataType(o));
        outputs[pyt_idx] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());
        gpu_handles.push_back(outputs[pyt_idx].data_ptr());
    }
    
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());
    {
        std::lock_guard<std::mutex> lock(*_mutex);
        _exec_ctx->enqueueV2(gpu_handles.data(), stream, nullptr);
    }
    return outputs;

}

std::vector<at::Tensor> execute_engine(const std::vector<at::Tensor>& inputs, 
                                    c10::intrusive_ptr<TensorrtEngine> compiled_engine) {
    return compiled_engine->excute_engine(inputs);
}

TORCH_LIBRARY(TensorrtEngine, m) {
    auto engine_class = m.class_<TensorrtEngine>("TensorrtEngine")
        .def(torch::init<>())
        .def_pickle(
            [](const c10::intrusive_ptr<TensorrtEngine>& self) -> std::string {
                auto serialized_engine = self->cuda_engine()->serialize();
                return std::string((const char*)serialized_engine->data(), serialized_engine->size());
            },
            [](std::string seralized_engine) -> c10::intrusive_ptr<TensorrtEngine> {
                return c10::make_intrusive<TensorrtEngine>(std::move(seralized_engine));
            });
    m.def("execute_engine", execute_engine);
}

POROS_REGISTER_ENGINE(TensorrtEngine);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
