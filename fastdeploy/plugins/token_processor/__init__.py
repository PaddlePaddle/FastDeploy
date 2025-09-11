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

# make sure one process only loads plugins once
PLUGINS_GROUP = "fastdeploy.token_processor_plugins"


def load_token_processor_plugins():
    """load_token_processor_plugins"""
    plugins = load_plugins_by_group(group=PLUGINS_GROUP)
    assert len(plugins) <= 1, "Most one plugin is allowed to be loaded."
    return next(iter(plugins.values()))()
