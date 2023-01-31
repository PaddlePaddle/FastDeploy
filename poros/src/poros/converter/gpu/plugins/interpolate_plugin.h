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
* @file interpolate_plugin.h
* @author tianjinjin@baidu.com
* @date Mon Sep 27 16:18:38 CST 2021
* @brief
**/

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//from tensorrt
#include "NvInferPlugin.h"

namespace baidu {
namespace mirana {
namespace poros {

/*
* @brief InterpolatePlugin 实现tensorrt的一个插件。
    该插件支持的pooling类算子包括：adaptive_avg_pool2d & adaptive_max_pool2d
    该插件支持的linear类算子包括：linear & bilinear & trilinear
    该插件会被上述算子对应的gpu-converter调用，注册到gpu-engine上，实现gpu-engine对该算子的支持；
    plugin的内部逻辑，主要是调用了pytorch-aten的实现(CUDA实现)。
**/
class InterpolatePlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    /*
     * @brief 多参数构造函数。
     * @param [in] in_shape : 输入tensors的shape信息
     * @param [in] out_shape : 输出tensors的shape信息
     * @param [in] size : 输出的 spatial 尺寸（当mode=linear, bilinear 和 trilinear）
     *                    输出的目标 size 信息（当mode=adaptive_avg_pool2d 和 adaptive_max_pool2d）
     * @param [in] scales : spatial 尺寸的缩放因子, 当use_scales=True时生效。
     * @param [in] mode : 算子标识，可选adaptive_avg_pool2d、adaptive_max_pool2d、linear、bilinear、trilinear。
     * @param [in] align_corners : 默认为false，如果 align_corners=True，则对齐input和output 的角点像素(corner pixels)，保持在角点像素的值. 
     *                             只会对 mode=linear, bilinear 和 trilinear 有作用。
     * @param [in] use_scales : 默认为false，如果use_scales=True，入参scales生效（且scales必须不为空）。
     *                          只会对 mode=linear, bilinear 和 trilinear 有作用。
     **/
    InterpolatePlugin(
        std::vector<int64_t> in_shape,
        std::vector<int64_t> out_shape,
        std::vector<int64_t> size,
        std::vector<double> scales,
        std::string mode,
        bool align_corners,
        bool use_scales);

    /*
     * @brief deserialize阶段使用的构造函数。
     * @param [in] data : 序列化好的data。
     * @param [in] length : 数据长度。
     **/
    InterpolatePlugin(const char* data, size_t length);

    /*
     * @brief InterpolatePlugin不应该存在无参数构造函数，将该默认构造函数删除。
     **/
    InterpolatePlugin() = delete;
   
    /*****************************************************************************
            以下部分是tensorrt定义的 IPluginV2DynamicExt API
    ******************************************************************************/
    /*
     * @brief clone函数，将这个插件对象克隆给tensorrt的builder/network/engine。
     **/   
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    /*
     * @brief 返回输出维度信息。
     **/
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    /*
     * @brief 插件的输入输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
     **/
    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept override;

    /*
     * @brief 配置这个插件，判断输入和输出类型数量是否正确等。
     **/
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    /*
     * @brief 返回需要中间显存变量的实际数据大小（bytesize）。
     **/
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    /*
     * @brief 该插件的的实际执行函数（重要！）。
     **/
    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;  

    /*****************************************************************************
            以下部分是tensorrt定义的 IPluginV2Ext API
    ******************************************************************************/
    /*
    * @brief 返回结果的数据类型（一般与输入类型一致）。
    **/
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
    
    /*****************************************************************************
            以下部分是tensorrt定义的 IPluginV2 API
    ******************************************************************************/
    /*
     * @brief 插件名称信息。
     **/  
    const char* getPluginType() const noexcept override;
    /*
     * @brief 插件版本信息。
     **/    
    const char* getPluginVersion() const noexcept override;
    /*
     * @brief 插件返回多少个Tensor。
     **/
    int getNbOutputs() const noexcept override;

    /*
     * @brief 初始化函数，在这个插件准备开始run之前执行。
     **/    
    int initialize() noexcept override;

    /*
     * @brief 资源释放函数，engine destory的时候调用该函数。
     **/ 
    void terminate() noexcept override {}

    /*
     * @brief 返回序列化时需要写多少字节到buffer中。
     **/      
    size_t getSerializationSize() const noexcept override;

    /*
     * @brief 把需要用的数据按照顺序序列化到buffer中。
     **/        
    void serialize(void* buffer) const noexcept override;

    /*
     * @brief 销毁插件对象，network/builder/engine destroy的时候调用该函数。
     **/
    void destroy() noexcept override {}

    /*
     * @brief 设置插件的namespace，默认为 ”“。
     **/    
    void setPluginNamespace(const char* pluginNamespace) noexcept override {};

    /*
     * @brief 返回插件的namespace。
     **/        
    const char* getPluginNamespace() const noexcept override;

    /*****************************************************************************
            以下部分为本插件自定义的API
    ******************************************************************************/
    /*
     * @brief 返回输入的shape信息。
     **/  
    std::vector<int64_t> getInputShape();

    /*
     * @brief 返回输出的shape信息。
     **/
    std::vector<int64_t> getOutputShape();

    /*
     * @brief 返回输出的spatial尺寸。
     **/
    std::vector<int64_t> getOutputSize();

    /*
     * @brief 插件序列化函数。
     **/
    std::string serializeToString() const;

private:
    nvinfer1::DataType dtype_;

    std::vector<int64_t> in_shape_;
    std::vector<int64_t> out_shape_;
    std::vector<int64_t> size_;
    std::vector<double> scales_;
    std::string mode_;
    bool align_corners_;
    bool use_scales_;
};

/*
* @brief InterpolatePluginCreator 实现插件的注册。
*        (配合全局宏REGISTER_TENSORRT_PLUGIN，将插件注册到tensorrt, 此后插件可以通过getPluginRegistry获取)
**/
class InterpolatePluginCreator : public nvinfer1::IPluginCreator {
public:
    /*
     * @brief 默认构造函数。
     **/
    InterpolatePluginCreator();

    /*
     * @brief 获取插件的名称信息。
     **/
    const char* getPluginName() const noexcept override;

    /*
     * @brief 获取插件的版本信息。
     **/
    const char* getPluginVersion() const noexcept override;

    /*
     * @brief 获取创建该插件需要的字段列表，该列表被 createPlugin 使用。
     **/
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    /*
     * @brief 根据插件名和字段列表信息，创建插件对象。
     **/
    nvinfer1::IPluginV2* createPlugin(
        const char* name, 
        const nvinfer1::PluginFieldCollection* fc) noexcept override;

    /*
     * @brief 反序列化插件。
     **/
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override;

    /*
     * @brief 设置插件的namespace信息。
     **/
    void setPluginNamespace(const char* libNamespace) noexcept override{};

    /*
     * @brief 获取插件的namespace信息。
     **/
    const char* getPluginNamespace() const noexcept override;

private:
    std::string name_;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC;
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
