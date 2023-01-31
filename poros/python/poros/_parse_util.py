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
 parse util for some python settings to poros c++ setting.
"""

import typing #  List & Dict & Any
import poros._C

def _parse_device(device):
    # type: Any -> poros._C.Device
    """
    converter device info to Device struct that can handle in poros
    """
    if isinstance(device, poros._C.Device):
        return device
    elif isinstance(device, str):
        if device == "GPU" or device == "gpu":
            return poros._C.Device.GPU
        elif device == "CPU" or device == "cpu":
            return poros._C.Device.CPU
        elif device == "XPU" or device == "xpu":
            return poros._C.Device.XPU
        else:
            ValueError("Got a device type unknown (type: " + str(device) + ")")
    else:
        raise TypeError("Device specification must be of type string or poros.Device, but got: " +
                str(type(device)))