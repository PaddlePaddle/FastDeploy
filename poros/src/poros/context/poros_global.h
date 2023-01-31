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
* @file poros_global.h
* @author tianjinjin@baidu.com
* @author huangben@baidu.com
* @date Fri Jul 23 11:21:10 CST 2021
* @brief 
**/

#pragma once

#include <memory>
#include <set>
#include <unordered_map>

#include <torch/csrc/jit/ir/ir.h>

#include "poros/compile/poros_module.h"
#include "poros/iplugin/plugin_create.h"
#include "poros/util/macros.h"

namespace baidu {
namespace mirana {
namespace poros {

// 设置双层map的原因是为了解决在同一个value不同node输入list时，因为append导致的引用问题
typedef std::map<torch::jit::Value*, std::map<torch::jit::Node*, std::set<int32_t>>> LIST_SIZE_MAP;
typedef std::map<torch::jit::Value*, std::map<torch::jit::Node*, std::map<int32_t, std::vector<c10::TensorTypePtr>>>> TENSOR_LIST_TYPE_MAP;

struct ListSizeMap {
    // 存储输入输出为list类型时的size信息
    LIST_SIZE_MAP _list_size_map_input;
    LIST_SIZE_MAP _list_size_map_output;

    // 存储输入输出类型为tensor list时的type信息
    TENSOR_LIST_TYPE_MAP _list_tensor_type_map_input;
    TENSOR_LIST_TYPE_MAP _list_tensor_type_map_output;

     /**
     * @brief 将old_value的信息更新到new_value上
     * @param [in] old_value : 原value
     * @param [in] new_value : 新的value
     * @return null
     **/
    void update_value(torch::jit::Value* old_value, torch::jit::Value* new_value);

    /**
     * @brief 将old_node的信息更新到new_node上
     * @param [in] old_node : 原node
     * @param [in] new_node : 新的node
     * @return null
     **/
    void update_node(torch::jit::Node* old_node, torch::jit::Node* new_node);
};

struct ValueDynamicShape {
    std::vector<int64_t> sizes;
    std::vector<int64_t> max_shapes;
    std::vector<int64_t> min_shapes;
    std::vector<int64_t> opt_shapes;
    bool is_dynamic = false;
};

// 前置声明
class ConvertersMap;
class IConverter;

class PorosGlobalContext {
public:
    static PorosGlobalContext& instance() {
        static PorosGlobalContext _instance;
        return _instance;
    }
 
    ~PorosGlobalContext() {
        destroy();
    }

    int init() {
        //to change
        return 0;
    }
    
    void set_poros_options(const PorosOptions& options);
   
    PorosOptions& get_poros_options() {
        return _poros_options;
    } 
   
    void destroy();

    // 注册converter方法到全局的PorosGlobalContext。
    void register_converter(const std::string& engine_name, IConverter* converter);

    ConvertersMap* get_converter_map(const std::string& engine_name);
public:
    plugin_creator_map_t _engine_creator_map;
    std::map<torch::jit::Value*, ValueDynamicShape> _value_dynamic_shape_map;
    ListSizeMap _list_size_map;
    std::map<torch::jit::Value*, ValueDynamicShape> _int_intlist_values_map;
    bool _disable_subblock_convert = false;

    const std::set<c10::Symbol> supported_mutable_ops_set = { 
        //aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)
        c10::Symbol::fromQualString("aten::append"),
        //"aten::_set_item.t(t [](a!) l, int idx, t(b -> *) el) -> t[](a!)"
        c10::Symbol::fromQualString("aten::_set_item"),
    };
private:
    PorosOptions _poros_options;
    std::unordered_map<std::string, ConvertersMap*> _converters_map;
};

/*-------------------------------------------------------------------------
                       converter自动注册相关宏
-------------------------------------------------------------------------*/
template <typename T>
class ConverterRegister {
public:
    public:
    inline ConverterRegister(std::string name = "",
        PorosGlobalContext& context = PorosGlobalContext::instance()) noexcept;
};

template <typename T>
inline ConverterRegister<T>::ConverterRegister(std::string name,
    PorosGlobalContext& context) noexcept {
    auto instance = new T();
    context.register_converter(name, instance);
}

#define POROS_REGISTER_CONVERTER(name, reg)                                   \
    static ConverterRegister<reg> POROS_CONVERTER_REGISTER_init_ ## reg (#name);

//engine自动注册
template <typename EngineType>
class EngineRegister {
public:
    EngineRegister(const std::string& name) {
        register_plugin_class<EngineType>(name, PorosGlobalContext::instance()._engine_creator_map);
    }
};

#define POROS_REGISTER_ENGINE(name)                                   \
    static EngineRegister<name> POROS_ENGINE_REGISTER_init_##name(#name);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
