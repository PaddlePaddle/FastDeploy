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
from typing import List, Union

from fastdeploy.entrypoints.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    ModelInfo,
    ModelList,
    ModelPermission,
)
from fastdeploy.utils import ErrorType, api_server_logger, get_host_ip


@dataclass
class ModelPath:
    name: str
    model_path: str
    verification: bool = False


class OpenAIServingModels:
    """
    OpenAI-style models serving
    """

    def __init__(
        self,
        model_paths: list[ModelPath],
        max_model_len: int,
        ips: Union[List[str], str],
    ):
        self.model_paths = model_paths
        self.max_model_len = max_model_len
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

    def is_supported_model(self, model_name) -> tuple[bool, str]:
        """
        Check whether the specified model is supported.
        """
        if self.model_paths[0].verification is False:
            return True, self.model_name()
        if model_name == "default":
            return True, self.model_name()
        return any(model.name == model_name for model in self.model_paths), model_name

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
                f"Only master node can accept models request, please send request to master node: {self.master_ip}"
            )
            api_server_logger.error(err_msg)
            return ErrorResponse(error=ErrorInfo(message=err_msg, type=ErrorType.INTERNAL_ERROR))
        model_infos = [
            ModelInfo(
                id=model.name, max_model_len=self.max_model_len, root=model.model_path, permission=[ModelPermission()]
            )
            for model in self.model_paths
        ]
        return ModelList(data=model_infos)
