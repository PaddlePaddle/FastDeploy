# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
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
 convert prewarm-data to c10:ivalues list that can handle in poros.
"""

import typing #  List & Dict & Any
import torch
import poros._C

from poros._module import DynamicOptions
from poros._module import PorosOptions
from poros._parse_util import _parse_device

def make_to_tuple(prewarm_input):
    """
    wrap a single torch.Tensor input to tuple
    """
    if isinstance(prewarm_input, torch.Tensor):
        return (prewarm_input,)
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(prewarm_input, tuple):
        return tuple(prewarm_input)
    return prewarm_input


def convert_prewarm_inputs(prewarm_inputs):
    # type: (Any) -> poros._C.PreWarmDatas
    """
    convert prewarm-data to c10:ivalues list that can handle in poros.
    we can accept 3 kinds of prewarm_inputs:
        one input that has a single tensor [torch.Tensor]
        one input that has multiple variables [tuple]
        more than one input, that each input has a single tensor [List of torch.Tensor]
        more that one input, that each input has multiple variables [List of tuple]
    """
    wraped_prewarm_inputs = []
    if isinstance(prewarm_inputs, torch.Tensor):
        wraped_prewarm_inputs.append(make_to_tuple(prewarm_inputs))
    elif isinstance(prewarm_inputs, tuple):
        wraped_prewarm_inputs.append(prewarm_inputs)
    elif isinstance(prewarm_inputs, list):
        for member in prewarm_inputs:
            if isinstance(member, torch.Tensor):
                wraped_prewarm_inputs.append(make_to_tuple(member))
            elif isinstance(member, tuple):
                wraped_prewarm_inputs.append(member)
            else:
                raise TypeError("prewarm_inputs for poros should be torch.Tensor or wraped as tuple, fix it")
    else:
        raise TypeError("prewarm_inputs for poros should be torch.Tensor or wraped as tuple or inputs-lists, fix it")
    return wraped_prewarm_inputs     
    # info = poros._C.PreWarmDatas()
    # info.set_data(prewarm_inputs)

def convert_poros_option(poros_option):
    # type: Dict[str, Any] -> poros._C.PorosOptions
    """
    converter key-value poros_option to PorosOptions that can handle in poros
    """
    option = poros._C.PorosOptions()
    if poros_option is None:
        #default situation. if user do not set the poros_option
        return option
    elif isinstance(poros_option, PorosOptions):
        return poros_option.to_internal()
    else:
        raise TypeError("poros_option for poros should be PorosOptions or a attribute dict fix it")
