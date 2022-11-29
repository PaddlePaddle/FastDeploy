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
* @file poros_global.cpp
* @author tianshaoqing@baidu.com
* @author huangben@baidu.com
* @date Fri Jul 23 11:21:10 CST 2021
* @brief 
**/

#include "poros/context/poros_global.h"
#include "poros/converter/iconverter.h"

namespace baidu {
namespace mirana {
namespace poros {

void ListSizeMap::update_value(torch::jit::Value* old_value, torch::jit::Value* new_value) {
    if (_list_size_map_input.count(old_value) != 0) {
        _list_size_map_input[new_value] = _list_size_map_input[old_value];
        _list_tensor_type_map_input[new_value] = _list_tensor_type_map_input[old_value];
    }
    if (_list_size_map_output.count(old_value) != 0) {
        _list_size_map_output[new_value] = _list_size_map_output[old_value];
        _list_tensor_type_map_output[new_value] = _list_tensor_type_map_output[old_value];
    }
}

void ListSizeMap::update_node(torch::jit::Node* old_node, torch::jit::Node* new_node) {
    for(size_t i = 0; i < new_node->inputs().size(); i++) {
        auto value = new_node->input(i);
        if (_list_size_map_input.count(value) != 0) {
            if (_list_size_map_input[value].count(old_node) != 0) {
                _list_size_map_input[value][new_node] = _list_size_map_input[value][old_node];
                _list_size_map_input[value].erase(old_node);

                _list_tensor_type_map_input[value][new_node] = _list_tensor_type_map_input[value][old_node];
                _list_tensor_type_map_input[value].erase(old_node);
            }    
        }
    }

    for(size_t i = 0; i < new_node->outputs().size(); i++) {
        auto value = new_node->output(i);
        if (_list_size_map_output.count(value) != 0) {
            if (_list_size_map_output[value].count(old_node) != 0) {
                _list_size_map_output[value][new_node] = _list_size_map_output[value][old_node];
                _list_size_map_output[value].erase(old_node);

                _list_tensor_type_map_output[value][new_node] = _list_tensor_type_map_output[value][old_node];
                _list_tensor_type_map_output[value].erase(old_node);
            }    
        }
    }
}

// 将PorosOptions放到全局类中，之后再初始化用户自定义不支持op列表
void PorosGlobalContext::set_poros_options(const PorosOptions& options) {
    _poros_options = options;
    for (auto i : _converters_map) {
        i.second->init_unsupport_op_set();
    }
} 

// 注册converter方法到全局的PorosGlobalContext。
void PorosGlobalContext::register_converter(const std::string& engine_name, IConverter* converter) {
    //根据engine_name 找到相应的 ConvertersMap 
    //(ps: 不同engine的ConvertersMap是相互独立的)
    auto search = _converters_map.find(engine_name);
    if (search == _converters_map.end()) {
        _converters_map[engine_name] = new ConvertersMap();
    }
    auto e_converter_map = _converters_map[engine_name];

    //根据converter的node_kind()和schema_string()信息，构造ConvRegistration。
    auto node_kind_list = converter->node_kind();
    auto schema_list = converter->schema_string();
    for (auto& node_kind : node_kind_list) {
        //converter that without schemas. such as aten::Constant
        ConvRegistration conv_reg;
        conv_reg.kind = node_kind;
        conv_reg.converter = converter;
        if (schema_list.size() == 0) {
            conv_reg.options = ConverterOptions();
        } else {
            conv_reg.options = ConverterOptions().set_valid_schemas(schema_list);
        }
        //调用ConvertersMap的add_converter方法, 完成注册。
        e_converter_map->add_converter(node_kind, conv_reg);
    }
    return;
};

ConvertersMap* PorosGlobalContext::get_converter_map(const std::string& engine_name) {
    auto search = _converters_map.find(engine_name);
    if (search == _converters_map.end()) {
        return nullptr;
    }
    return search->second;      
};

void PorosGlobalContext::destroy() {
    for (auto &e : _converters_map) {
        delete e.second;
    }
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
