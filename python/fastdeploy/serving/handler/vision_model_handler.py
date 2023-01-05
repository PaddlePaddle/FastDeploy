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
from .base_handler import BaseModelHandler
from ..utils import base64_to_cv2
from ...vision.utils import fd_result_to_json


class VisionModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, predictor, data, parameters):
        # TODO: support batch predict
        im = base64_to_cv2(data['image'])
        result = predictor.predict(im)
        r_str = fd_result_to_json(result)
        return r_str
