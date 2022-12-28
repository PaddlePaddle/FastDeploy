# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from fastapi import FastAPI
from .router import HttpRouterManager
from .model_manager import ModelManager


class SimpleServer(FastAPI):
    def __init__(self, **kwargs):
        """
        Initial function for the FastDeploy SimpleServer.
        """
        super().__init__(**kwargs)
        self._router_manager = HttpRouterManager(self)
        self._model_manager = None
        self._service_name = "FastDeploy SimpleServer"
        self._service_type = None

    def register(self, task_name, model_handler, predictor):
        """
        The register function for the SimpleServer, the main register argrument as follows:

        Args:
            task_name(str): API URL path.
            model_handler: To process request data, run predictor,
                and can also add your custom post processing on top of the predictor result
            predictor: To run model predict
        """
        self._server_type = "models"
        model_manager = ModelManager(model_handler, predictor)
        self._model_manager = model_manager
        # Register model server router
        self._router_manager.register_models_router(task_name)
