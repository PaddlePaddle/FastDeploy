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

from dataclasses import dataclass

from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.entrypoints.openai.protocol import (
    ErrorResponse,
    ModelInfo,
    ModelList,
    ModelPermission,
)
from fastdeploy.utils import api_server_logger, get_host_ip


@dataclass
class ModelPath:
    name: str
    model_path: str


class OpenAIServingModels:
    """
    Shared instance to hold data about the loaded models
    """

    def __init__(
        self,
        engine_client: EngineClient,
        model_paths: list[ModelPath],
        max_model_len: int,
        pid,
        ips,
    ):
        self.engine_client = engine_client
        self.model_paths = model_paths
        self.max_model_len = max_model_len
        self.pid = pid
        self.master_ip = ips
        self.host_ip = get_host_ip()
        if self.master_ip is not None:
            if isinstance(self.master_ip, list):
                self.master_ip = self.master_ip[0]
            else:
                self.master_ip = self.master_ip.split(",")[0]

    def _check_master(self):
        if self.master_ip is None:
            return True
        if self.host_ip == self.master_ip:
            return True
        return False

    def is_supported_model(self, model_name) -> bool:
        """
        Check whether the specified model is supported.
        """
        return any(model.name == model_name for model in self.model_paths)

    def model_name(self) -> str:
        """
        Returns the current model name.
        """
        return self.model_paths[0].name

    async def list_models(self) -> ModelList:
        """
        Show available models.
        """
        if not self._check_master():
            err_msg = (
                f"Only master node can accept models request, please send request to master node: {self.pod_ips[0]}"
            )
            api_server_logger.error(err_msg)
            return ErrorResponse(message=err_msg, code=400)
        model_infos = [
            ModelInfo(
                id=model.name, max_model_len=self.max_model_len, root=model.model_path, permission=[ModelPermission()]
            )
            for model in self.model_paths
        ]
        return ModelList(data=model_infos)
