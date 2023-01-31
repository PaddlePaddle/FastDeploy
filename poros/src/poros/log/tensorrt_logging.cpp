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
* @file tensorrt_logging.cpp
* @author tianjinjin@baidu.com
* @date Wed Jun  2 21:14:23 CST 2021
* @brief 
**/

#include <iostream>
#include "poros/context/poros_global.h"
#include "poros/log/tensorrt_logging.h"

namespace baidu {
namespace mirana {
namespace poros {

/*
TensorrtLogger::TensorrtLogger(logging::LogSeverity severity) {
    _severity = severity;
    switch (severity) {
        case logging::BLOG_INFO:
            _nv_level = nvinfer1::ILogger::Severity::kINFO;
            break;
        case logging::BLOG_NOTICE:
            _nv_level = nvinfer1::ILogger::Severity::kVERBOSE;
            break;
        case logging::BLOG_WARNING:
            _nv_level = nvinfer1::ILogger::Severity::kWARNING;
            break;
        case logging::BLOG_ERROR:
            _nv_level = nvinfer1::ILogger::Severity::kERROR;
            break;
        case logging::BLOG_FATAL:
            _nv_level = nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
            break;
        default:
            break;           
    }
} */

TensorrtLogger::TensorrtLogger() {
    auto debug = PorosGlobalContext::instance().get_poros_options().debug;
    _torch_level = debug ?  0 : 1;
    _nv_level = debug ? nvinfer1::ILogger::Severity::kVERBOSE : 
                        nvinfer1::ILogger::Severity::kWARNING;
}

TensorrtLogger::TensorrtLogger(uint32_t torch_level) {
    _torch_level = torch_level;
    switch (torch_level) {
    case 3: /*c10::GLOG_FATAL*/
        _nv_level = nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
        break;
    case 2: /*c10::GLOG_ERROR*/
        _nv_level = nvinfer1::ILogger::Severity::kERROR;
        break;
    case 1: /*c10::GLOG_WARNING*/
        _nv_level = nvinfer1::ILogger::Severity::kWARNING;
        break;
    case 0: /*c10::GLOG_INFO*/
        _nv_level = nvinfer1::ILogger::Severity::kVERBOSE;
        break;
    default: /*c10::GLOG_WARNING*/
        _nv_level = nvinfer1::ILogger::Severity::kWARNING;
        break;
    }
}

void TensorrtLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    // suppress unprintable messages
    if (severity > _nv_level) {
        return;
    }
    std::cout << msg << std::endl;  //TO MAKE THIS BETTER.
}

TensorrtLogger& get_nvlogger() {
  static TensorrtLogger nv_logger;
  return nv_logger;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
