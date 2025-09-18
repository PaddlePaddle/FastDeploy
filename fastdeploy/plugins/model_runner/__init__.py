"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from fastdeploy.plugins.utils import load_plugins_by_group

# use for model runner
PLUGINS_GROUP = "fastdeploy.model_runner_plugins"


def load_model_runner_plugins():
    """load_model_runner_plugins"""
    plugins = load_plugins_by_group(group=PLUGINS_GROUP)
    assert len(plugins) == 1, "Only one plugin is allowed to be loaded."
    return next(iter(plugins.values()))()
