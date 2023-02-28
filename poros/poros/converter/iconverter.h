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
* @file iconverter.h
* @author tianjinjin@baidu.com
* @author huangben@baidu.com
* @date Tue Jul 27 11:24:21 CST 2021
* @brief 
**/

#pragma once

#include <string>

#include "torch/script.h"
#include "ATen/core/function_schema.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/runtime/custom_operator.h"

#include "poros/context/poros_global.h"

namespace baidu {
namespace mirana {
namespace poros {

class IEngine;

// schema属性
// 默认支持dynamic shape，不支持tensor scalar输入
struct schema_attr {
    int is_support_dynamic_shape = 1;
    int is_support_tensor_scalar = 0;
};

class IConverter {
public:
    virtual ~IConverter() {}
    /**
     * @brief converter核心实现，注意要将输入tensor和engine的tensor对应起来，通过engine.context
     *        tensor->tensor,  constant->constant
     * @param [in] sub_graph  : 子图
     * @return [res]int
     * @retval 0 => success, <0 => fail
     **/
    virtual bool converter(IEngine* engine, const torch::jit::Node *node) = 0;
    virtual const std::vector<std::string> schema_string() = 0;
    virtual const std::vector<torch::jit::NodeKind> node_kind() = 0;
    const std::unordered_map<c10::OperatorName, schema_attr> get_schema_attr_map() {
        return _schema_attr_map;
    }
protected:
    /**
     * @brief help assign schema attr
     *
     * @param [in] schema_attr_vec : schema and schema_attr which want to assign
     * @return  true => succeed  false => failed
    **/
    virtual bool assign_schema_attr_helper(std::vector<std::pair<std::string, schema_attr>> schema_attr_vec) {
        if (_schema_attr_map.empty()) {
            LOG(INFO) << "the schema_attr_map may not have been initialized.";
            return false;
        }
        for (size_t i = 0; i < schema_attr_vec.size(); i++) {
            c10::OperatorName op_name = torch::jit::parseSchema(schema_attr_vec[i].first).operator_name();
            if (_schema_attr_map.count(op_name) == 0) {
                LOG(INFO) << "schema: [ " << schema_attr_vec[i].first << " ] was not found in schema_attr_map";
                return false;
            }
            _schema_attr_map[op_name] = schema_attr_vec[i].second;
        }
        return true;
    }
    // 给schema赋予attr，在子类中实现
    virtual bool assign_schema_attr() {
        return true;
    }
private:
    // 声明 ConvertersMap为友元，在其中调用init_schema_attr
    friend class ConvertersMap;
    // 初始化schema attr，需converter为子类时调用
    bool init_schema_attr() {
        _schema_attr_map.clear();
        std::vector<std::string> schema_strings = this->schema_string();
        schema_attr attr;
        for (const std::string& s : schema_strings) {
            _schema_attr_map.insert({torch::jit::parseSchema(s).operator_name(), attr});
        }
        return assign_schema_attr();
    }
    std::unordered_map<c10::OperatorName, schema_attr> _schema_attr_map;
};

struct ConverterOptions {
    std::vector<c10::OperatorName> valid_schemas;

    ConverterOptions() = default;

    ConverterOptions& set_valid_schemas(std::vector<std::string> schema_string) {
        use_options = true;
        for (auto s : schema_string) {
            valid_schemas.push_back(torch::jit::parseSchema(s).operator_name());
        }
        return *this;
    }

    bool use() {
        return use_options;
    }
private:
    bool use_options = false;
};

struct ConvRegistration {
    torch::jit::NodeKind kind;
    IConverter*   converter;
    ConverterOptions  options;
};

class ConvertersMap {
public:
    ConvertersMap() {}

    virtual ~ConvertersMap() {
    }
    
    //添加converter到当前的map。
    bool add_converter(torch::jit::NodeKind node_kind, ConvRegistration conv_reg) {
        auto iter = converters_map.find(node_kind);
        if (iter != converters_map.end()) {
            LOG(WARNING) << "override converter for [ " << node_kind.toQualString() <<  " ]";
        }
        converters_map[node_kind] = std::move(conv_reg);
        return true;
    }

    IConverter* get_converter(const torch::jit::Node* node) {
        if (!node_converterable(node)) {
            return nullptr;
        }
        auto node_kind = node->kind();
        auto iter = converters_map.find(node_kind);
        if (iter == converters_map.end()) {
            return nullptr;
        }
        auto conv_reg = iter->second;
        if (conv_reg.options.use()) {
            if (conv_reg.options.valid_schemas.size() != 0) {
                auto schema = node->maybeSchema();
                if (!schema) {
                    return nullptr;
                }
                for (auto reg_schema : conv_reg.options.valid_schemas) {
                    if (reg_schema == schema->operator_name()) {
                        return conv_reg.converter;
                    }
                }
                return nullptr;
            }
        }
        return conv_reg.converter;;
    }

    // 判断list类型的输入输出长度是否发生变化
    bool list_size_is_variable_length(const torch::jit::Node *node) {
        auto list_size_map_input = PorosGlobalContext::instance()._list_size_map._list_size_map_input;
        for (size_t i = 0; i < node->inputs().size(); i++) {
            auto value = node->input(i);
            // 如果是list类型
            if (value->type()->kind() == c10::TypeKind::ListType) {
                if (list_size_map_input.count(value) != 0 && list_size_map_input[value].count(const_cast<torch::jit::Node*>(node)) != 0) {
                    // 如果本node对应value（即list变量）记录的size有1个以上的，说明长度发生变化
                    // 返回true，外部no-converterable
                    if (list_size_map_input[value].at(const_cast<torch::jit::Node*>(node)).size() != 1){
                        return true;
                    }
                }
            }
        }
        // 输出list变量判断长度是否变化，原理同输入
        auto list_size_map_output = PorosGlobalContext::instance()._list_size_map._list_size_map_output;
        for (size_t i = 0; i < node->outputs().size(); i++) {
            auto value = node->output(i);
            if (value->type()->kind() == c10::TypeKind::ListType) {
                if (list_size_map_output.count(value) != 0 && list_size_map_output[value].count(const_cast<torch::jit::Node*>(node)) != 0) {
                    if (list_size_map_output[value].at(const_cast<torch::jit::Node*>(node)).size() != 1){
                        return true;
                    }
                }
            }
        }
        return false;
    }
    // 判断特殊输出类型节点例如list[list[]]
    bool special_node_check(const torch::jit::Node *node) {
        if (node->kind() == torch::jit::prim::ListConstruct) {
            const torch::jit::Value* output = node->outputs()[0];
            if (output->type()->str().find("[][]") != output->type()->str().npos) {
                return true;
            }
        }
        return false;
    }

    // 判断是否属于支持tensor scalar输入的op集合
    bool is_unsupport_tensor_scalar_inputs(const torch::jit::Node *node, 
                             std::unordered_map<c10::OperatorName, schema_attr> schema_attr_map) {
        if (node->kind() == torch::jit::prim::CudaFusionGroup) {
            return false;
        }
        for(size_t i = 0; i < node->inputs().size(); i++) {
            // 如果input是scalar或scalar list，且不来自于prim::Constant
            torch::jit::Value* current_input = node->input(i);
            if ((current_input->type()->isSubtypeOf(c10::NumberType::get()) || 
                current_input->type()->isSubtypeOf(c10::BoolType::get()) ||
                current_input->type()->isSubtypeOf(c10::StringType::get()) ||
                current_input->type()->isSubtypeOf(c10::ListType::ofFloats()) ||
                current_input->type()->isSubtypeOf(c10::ListType::ofInts()) || 
                current_input->type()->isSubtypeOf(c10::ListType::ofBools()) || 
                current_input->type()->isSubtypeOf(c10::ListType::ofStrings())
                ) &&
                current_input->node()->kind() != torch::jit::prim::Constant) {
                // 判断node是否属于: 1、prim::ListConstruct 2、prim::ListUnpack 3、支持scalar tensor输入的schema 
                // 都不属于则不支持，返回true，外部no-converterable
                if (!node->maybeSchema()) {
                    // 这两个op没有schema，需要单独判断
                    if (node->kind() == torch::jit::prim::ListConstruct || 
                        node->kind() == torch::jit::prim::ListUnpack) {
                        return false;
                    } else {
                        return true;
                    }
                } else {
                    if (schema_attr_map[node->maybeSchema()->operator_name()].is_support_tensor_scalar == 1) {
                        return false;
                    } else {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    bool node_converterable(const torch::jit::Node* node) {
        auto node_kind = node->kind();
        auto iter = converters_map.find(node_kind);
        if (iter == converters_map.end()) {
            LOG(WARNING) << "no converter find for [ " << node_kind.toQualString() <<  " ]";
            if (node->maybeSchema()) {
                LOG(WARNING) << "unsupported schema is [ " << *node->maybeSchema() <<  " ]";
            }
            return false;
        }
        auto conv_reg = iter->second;
        if (conv_reg.options.use()) {
            if (conv_reg.options.valid_schemas.size() != 0) {
                auto schema = node->maybeSchema();
                if (!schema) {
                    LOG(WARNING) << "no schema find for [ " << node_kind.toQualString() <<  " ]";
                    return false;
                }
                // 检查用户自定义不支持node schema
                if (_unsupport_schema_set.count(schema->operator_name())) {
                    LOG(WARNING) << "The user specifies that the unsupported node schema is [ " << *schema <<  " ]";
                    return false;
                } 
                // 检查用户自定义不支持node kind
                if (_unsupport_nodekind_set.count(node->kind())) {
                    LOG(WARNING) << "The user specifies that the unsupported node kind is [ " << node->kind().toQualString() <<  " ]";
                    return false;
                }
                // 由于tensorrt支持问题，aten::_convolution为反卷积时（transposed==true） 
                // output_padding参数必须为0，否则不支持
                if (node->kind() == torch::jit::aten::_convolution && node->inputs().size() >= 12) {
                    if (node->input(6)->node()->kind() == torch::jit::prim::Constant && 
                        node->input(6)->type()->kind() == c10::TypeKind::BoolType && 
                        node->input(7)->node()->kind() == torch::jit::prim::Constant && 
                        node->input(7)->type()->isSubtypeOf(c10::ListType::ofInts())) {

                        bool transposed = toIValue(node->input(6)).value().toBool();  
                        auto input_7_vec = toIValue(node->input(7)).value().toIntVector();  
                        if (transposed && (input_7_vec[0] > 0 || input_7_vec[1] > 0)) {
                            LOG(INFO) << "TensorRT does not have a notion of output_padding for deconvolution layers."
                            " output_padding has to be set as zeros.";
                            return false;
                        }
                    }
                }
                IConverter* current_converter = conv_reg.converter;
                if (!current_converter->init_schema_attr()) {
                    LOG(WARNING) << "converter [ " << node_kind.toQualString() << " ] failed to initialize schema attribute.";
                    return false;
                }
                auto conv_schema_attr = current_converter->get_schema_attr_map();
                if (conv_schema_attr.count(schema->operator_name()) == 0) {
                    LOG(WARNING) << "no supported schema find for [ " << node_kind.toQualString() <<  " ]";
                    LOG(WARNING) << "unsupported schema is [ " << *schema << " ]";
                    return false;
                }
                
                // 如果是dynamic，检查no_supported_dynamic_schema
                PorosOptions poros_options = PorosGlobalContext::instance().get_poros_options();
                if (poros_options.is_dynamic) {
                    if (conv_schema_attr[schema->operator_name()].is_support_dynamic_shape == 0) {
                        LOG(WARNING) << "no supported dynamic schema is [ " << *schema << " ]";
                        return false;
                    }
                }
                
                auto node_kind = node->kind();
                // 特殊op的特殊判断，比如list[list[]]在ListConstructConverter中的支持问题
                if (special_node_check(node)) {
                    LOG(WARNING) << "node input or output type is not support: " << node_kind.toQualString();
                    return false;
                }

                // list输入输出长度是否变化
                if (list_size_is_variable_length(node)) {
                    LOG(WARNING) << "input or output is variable length list [ " << node_kind.toQualString() <<  " ]";
                    return false;
                }

                // scalar的问题判断，判断一个op是否支持把scalar变成tensor来使用
                if (is_unsupport_tensor_scalar_inputs(node, conv_schema_attr)) {
                    LOG(WARNING) << "unsupport nvtensor scalar input node is [ " << node_kind.toQualString() <<  " ]";
                    return false;
                }

                // 如果是mutable的op，是否属于目前支持的范围
                if (node->kind().is_aten() && node->schema().is_mutable() && node->input(0)->type()->kind() == c10::TypeKind::ListType) {
                    if (PorosGlobalContext::instance().supported_mutable_ops_set.count(node->kind()) == 0) {
                        LOG(WARNING) << "Meet unsupport mutable node. The node is [ " << node_kind.toQualString() <<  " ]";
                        return false;
                    }
                }
            } else {
                // 用户指定不支持OP列表时，图中node只有kind没有schema的，只比较nodekind
                if (_unsupport_nodekind_set.count(node->kind())) {
                    LOG(WARNING) << "The user specifies that the unsupported node kind is [ " << node->kind().toQualString() <<  " ]";
                    return false;
                }
            }
        }
        return true;
    }
    // 由全局option初始化不支持op set
    void init_unsupport_op_set() {
        try {
            std::vector<std::string> unsupport_op_vec = PorosGlobalContext::instance().get_poros_options().unsupport_op_list;
            // 每次设置unsupport_op_list时刷新schema和nodekind set，避免用户下次编译想重新设置还保留之前不支持的op
            _unsupport_schema_set.clear();
            _unsupport_nodekind_set.clear();
            for (size_t i = 0; i < unsupport_op_vec.size(); i++) {
                std::string line = unsupport_op_vec[i];
                if (line.size() == 0) {
                    continue;
                }
                auto schema_or_opname = torch::jit::parseSchemaOrName(line);
                // operator name
                if (schema_or_opname.is_left()) {
                    torch::jit::NodeKind node_kind = c10::Symbol::fromQualString(line);
                    if (!converters_map.count(node_kind)) {
                        LOG(WARNING) << "WARNING: The user-defined unsupported nodekind [ " << node_kind.toQualString() << " ] cannot be found in the poros supported op set."
                        " Please check the PorosOptions.unsupport_op_list input.";
                    }
                    _unsupport_nodekind_set.insert(node_kind);
                    LOG(INFO) << "The user-defined unsupported node kind is [ " << node_kind.toQualString() << " ].";
                // schema
                } else {
                    c10::FunctionSchema fs = schema_or_opname.right();
                    std::string fs_name = fs.name();
                    auto end_index = fs_name.find_last_of('.') == std::string::npos ? fs_name.size() : fs_name.find_last_of('.');
                    std::string node_kind_name = fs_name.substr(0, end_index);
                    torch::jit::NodeKind node_kind = c10::Symbol::fromQualString(node_kind_name);
                    c10::OperatorName node_op_name = schema_or_opname.right().operator_name();
                    if (converters_map.count(node_kind)) {
                        std::vector<c10::OperatorName> node_valid_schema_vec = converters_map[node_kind].options.valid_schemas;
                        int e = 0;
                        for (auto i : node_valid_schema_vec) {
                            if (i == node_op_name) {
                                e++;
                            }
                        }
                        if (!e) {
                            LOG(WARNING) << "WARNING: The user-defined unsupported schema [ " << line << " ] cannot be found in the poros supported op set."
                            " Please check the PorosOptions.unsupport_op_list input.";
                        }
                    } else {
                        LOG(WARNING) << "WARNING: The user-defined unsupported schema nodekind [ " << node_kind.toQualString() << " ] cannot be found in the poros supported op set."
                        " Please check the PorosOptions.unsupport_op_list input.";
                    }
                    _unsupport_schema_set.insert(node_op_name);
                    LOG(INFO) << "The user-defined unsupported schema is [ " << fs << " ].";
                }
            }
        } catch (...) {
            LOG(WARNING) << "WARNING: Failed to initialize user-defined unsupport operator list. Please check the PorosOptions.unsupport_op_list parameter.";
        }
    }
    
private:
    std::set<std::string> converter_schemas;
    std::unordered_map<torch::jit::NodeKind, ConvRegistration> converters_map;
    std::unordered_set<torch::jit::NodeKind> _unsupport_nodekind_set;
    std::unordered_set<c10::OperatorName> _unsupport_schema_set;

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
