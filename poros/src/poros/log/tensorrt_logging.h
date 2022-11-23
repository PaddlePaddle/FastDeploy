/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file tensorrt_logging.h
* @author tianjinjin@baidu.com
* @date Wed Jun  2 20:54:24 CST 2021
* @brief 
**/

#pragma once

#include <string>

//from pytorch
#include "c10/util/Logging.h"

//from tensorrt
#include "NvInfer.h"

#include "poros/log/poros_logging.h"

namespace baidu {
namespace mirana {
namespace poros {

/**
 * the required logger setting for tensorrt engine
 * **/
class TensorrtLogger : public nvinfer1::ILogger {
public:
    TensorrtLogger();
    TensorrtLogger(uint32_t torch_level);
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept;
    uint32_t torch_level() {
        return _torch_level;
    }

private:
    uint32_t _torch_level = 1;
    nvinfer1::ILogger::Severity _nv_level;
};

TensorrtLogger& get_nvlogger();

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
