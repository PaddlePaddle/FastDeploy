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

from fastdeploy.hub.model_server import model_server
import fastdeploy.hub.env as hubenv
from fastdeploy.download import download


def download_model(name: str,
                   path: str=None,
                   format: str=None,
                   version: str=None):
    '''
    Download pre-trained model for FastDeploy inference engine.
    Args:
        name: model name
        path(str): local path for saving model. If not set, default is hubenv.MODEL_HOME
        format(str): FastDeploy model format
        version(str) : FastDeploy model version
    '''
    result = model_server.search_model(name, format, version)
    if path is None:
        path = hubenv.MODEL_HOME
    if result:
        url = result[0]['url']
        format = result[0]['format']
        version = result[0]['version']
        fullpath = download(url, path, show_progress=True)
        model_server.stat_model(name, format, version)
        print('Successfully download model at path: {}'.format(fullpath))
    else:
        print('ERROR: Could not find a model named {}'.format(name))
