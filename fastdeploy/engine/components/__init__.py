# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
Engine components for the new modular architecture.
"""

from fastdeploy.engine.components.ipc_manager import IPCManager
from fastdeploy.engine.components.process_manager import ProcessManager
from fastdeploy.engine.components.resource_coordinator import ResourceCoordinator
from fastdeploy.engine.components.scheduler_coordinator import SchedulerCoordinator

__all__ = [
    "IPCManager",
    "ProcessManager",
    "ResourceCoordinator",
    "SchedulerCoordinator",
]
