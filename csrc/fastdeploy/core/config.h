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
#pragma once

#ifndef FASTDEPLOY_DEBUG
/* #undef FASTDEPLOY_DEBUG */
#endif

#ifndef FASTDEPLOY_LIB
/* #undef FASTDEPLOY_LIB */
#endif

#ifndef ENABLE_PADDLE_FRONTEND
#define ENABLE_PADDLE_FRONTEND
#endif

#ifndef ENABLE_ORT_BACKEND
#define ENABLE_ORT_BACKEND
#endif

#ifndef ENABLE_PADDLE_BACKEND
#define ENABLE_PADDLE_BACKEND
#endif

#ifndef WITH_GPU
#define WITH_GPU
#endif

#ifndef ENABLE_TRT_BACKEND
/* #undef ENABLE_TRT_BACKEND */
#endif

#ifndef ENABLE_VISION
#define ENABLE_VISION
#endif

#ifndef ENABLE_OPENCV_CUDA
/* #undef ENABLE_OPENCV_CUDA */
#endif

#ifndef ENABLE_VISION_VISUALIZE
#define ENABLE_VISION_VISUALIZE
#endif
