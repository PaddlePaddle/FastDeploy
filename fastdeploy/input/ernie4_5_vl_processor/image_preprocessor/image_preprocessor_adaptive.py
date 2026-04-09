"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

# Backward compatibility: this module has been migrated to
# fastdeploy.input.image_processors.adaptive_processor
# This file will be removed in a future version.

from fastdeploy.input.image_processors.adaptive_processor import (  # noqa: F401
    AdaptiveImageProcessor,
    make_batched_images,
    make_batched_videos,
)
