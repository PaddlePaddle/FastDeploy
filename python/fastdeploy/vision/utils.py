# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
import json
from .. import c_lib_wrap as C


def mask_to_json(result):
    r_json = {
        "data": result.data,
        "shape": result.shape,
    }
    return json.dumps(r_json)


def detection_to_json(result):
    masks = []
    for mask in result.masks:
        masks.append(mask_to_json(mask))
    r_json = {
        "boxes": result.boxes,
        "scores": result.scores,
        "label_ids": result.label_ids,
        "masks": masks,
        "contain_masks": result.contain_masks
    }
    return json.dumps(r_json)


def classify_to_json(result):
    r_json = {
        "label_ids": result.label_ids,
        "scores": result.scores,
    }
    return json.dumps(r_json)


def fd_result_to_json(result):
    if isinstance(result, list):
        r_list = []
        for r in result:
            r_list.append(fd_result_to_json(r))
        return r_list
    elif isinstance(result, C.vision.DetectionResult):
        return detection_to_json(result)
    elif isinstance(result, C.vision.Mask):
        return mask_to_json(result)
    elif isinstance(result, C.vision.ClassifyResult):
        return classify_to_json(result)
    else:
        assert False, "{} Conversion to JSON format is not supported".format(
            type(result))
    return {}
