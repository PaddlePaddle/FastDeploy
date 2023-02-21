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
* @file macros.h
* @author tianjinjin@baidu.com
* @date Fri Jun  4 16:16:38 CST 2021
* @brief 
**/

#pragma once

#include <sys/time.h>

#include <c10/util/Exception.h>

namespace baidu {
namespace mirana {
namespace poros {

#define POROS_TIME_COST_US(pre, now) ((now.tv_sec - pre.tv_sec) * 1000000 + (now.tv_usec - pre.tv_usec))
#define POROS_TIME_COST_MS(pre, now) ((now.tv_sec - pre.tv_sec) * 1000 + (now.tv_usec - pre.tv_usec) / 1000)

// ----------------------------------------------------------------------------
// Error reporting macros
// ----------------------------------------------------------------------------
#define POROS_CHECK_RET_EXIT(n, s) {  \
    if ((n) != 0) {                     \
        LOG(FATAL) << s;              \
        exit(1);                      \
    }                                 \
}
                        
#define POROS_CHECK_RET(n, s) {      \
    if ((n) != 0) {                    \
        LOG(WARNING) << s;           \
        return -1;                   \
    }                                \
}

#define POROS_CHECK_TRUE(n, s) {      \
    if ((n) != true) {                  \
        LOG(WARNING) << s;            \
        return false;                 \
    }                                 \
}

#define POROS_THROW_ERROR(msg)                                                             \
    throw ::c10::Error({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, #msg);

#define POROS_ASSERT(cond, ...)                                                            \
    if (!(cond)) {                                                                         \
    POROS_THROW_ERROR(                                                                     \
        #cond << " ASSERT FAILED at " << __FILE__ << ':' << __LINE__                       \
              << ", consider filing a bug to cudp@baidu.com \n"                            \
              << __VA_ARGS__);                                                             \
    }

#define POROS_CHECK(cond, ...)                                                               \
    if (!(cond)) {                                                                           \
    POROS_THROW_ERROR("Expected " << #cond << " to be true but got false\n" << __VA_ARGS__); \
    }

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu