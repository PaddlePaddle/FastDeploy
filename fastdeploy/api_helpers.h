// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// https://github.com/PaddlePaddle/Paddle-Lite/issues/8290
#if (defined(WITH_LITE_STATIC) && defined(WITH_STATIC_LIB))
// Whether to output some warning messages when using the 
// FastDepoy static library, default OFF. These messages
// are only reserve for debugging.
#if defined(WITH_STATIC_WARNING)
#warning You are using the FastDeploy static library. \
We will automatically add some registration codes for \
ops, kernels and passes for Paddle Lite.
#endif
#if !defined(WITH_STATIC_LIB_AT_COMPILING)
#include "paddle_use_ops.h"       // NOLINT
#include "paddle_use_kernels.h"   // NOLINT
#include "paddle_use_passes.h"    // NOLINT
#endif
#endif
