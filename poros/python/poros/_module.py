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
poros module here
"""

import poros._C
from poros._parse_util import _parse_device

from torch.jit._script import RecursiveScriptModule
from torch.jit._script import ScriptModule
from torch.jit._script import script

class DynamicOptions(object):
    """
    dynamic settings for poros
    """
    def __init__(self):
        """set defalut dynamic options"""
        self.is_dynamic = False
        self.min_shapes = []
        self.opt_shapes = []
        self.max_shapes = []

    def set_dynamic_options(self, min, opt, max):
        """situation when give three inputs is given"""
        option_list = [min, opt, max]
        for item in option_list:
            if not (isinstance(item, list)):
                raise TypeError("dynamic_option for poros should be IntList, fix it")
        option_list.sort()
        self.min_shapes = option_list[0]
        self.opt_shapes = option_list[1]
        self.max_shapes = option_list[2]

    def set_dynamic_option(self, opt):
        """situation when only one input is given"""
        if not isinstance(opt, list):
            raise TypeError("dynamic_option for poros should be IntList, fix it")
        else:
            self.min_shapes = opt
            self.opt_shapes = opt
            self.max_shapes = opt
            self.is_dynamic = False
    
    def get_dynamic_options(self):
        """get dynamic options"""
        return [self.min_shapes, self.opt_shapes, self.max_shapes]

    def to_internal(self):
        """
        change DynamicOptions in python env to DynamicShapeOptions in c++ env
        """
        option = poros._C.DynamicShapeOptions()
        assert isinstance(self.is_dynamic, bool)
        option.is_dynamic = self.is_dynamic

        assert isinstance(self.min_shapes, list)
        option.min_shapes = self.min_shapes
        assert isinstance(self.opt_shapes, list)
        option.opt_shapes = self.opt_shapes
        assert isinstance(self.max_shapes, list)
        option.max_shapes = self.max_shapes
        return option

class PorosOptions(object):
    """
    options for poros
    """
    available_devices = ["GPU", "CPU", "XPU"]
    available_debug_mode = [True, False]
    def __init__(self):
        self.device = "GPU"
        self.debug = False
        self.use_fp16 = False
        self.max_workspace_size = 1 << 30
        self.is_dynamic = False
        self.long_to_int = True
        self.device_id = -1
        self.unconst_ops_thres = -1
        self.use_nvidia_tf32 = True
        self.preprocess_mode = 0
        self.unsupport_op_list = []

    def to_internal(self):
        """
        change PorosOptions in python env to PorosOptions in c++ env
        """
        option = poros._C.PorosOptions()
        option.device = _parse_device(self.device)
        assert isinstance(self.debug, bool)
        option.debug = self.debug
        assert isinstance(self.use_fp16, bool)
        option.use_fp16 = self.use_fp16
        assert type(self.max_workspace_size) is int
        option.max_workspace_size = self.max_workspace_size
        assert isinstance(self.is_dynamic, bool)
        option.is_dynamic = self.is_dynamic
        assert isinstance(self.long_to_int, bool)
        option.long_to_int = self.long_to_int
        assert type(self.device_id) is int
        option.device_id = self.device_id
        assert type(self.unconst_ops_thres) is int
        option.unconst_ops_thres = self.unconst_ops_thres
        assert type(self.use_nvidia_tf32) is bool
        option.use_nvidia_tf32 = self.use_nvidia_tf32
        assert type(self.preprocess_mode) is int
        option.preprocess_mode = self.preprocess_mode
        assert type(self.unsupport_op_list) is list
        option.unsupport_op_list = self.unsupport_op_list

        return option

    def set_device(self, device):
        """set device"""
        if device not in PorosOptions.available_devices:
            raise TypeError("device for poros invalid, only %s supported, fix it" % (PorosOptions.available_devices))
        self.device = device
    
    def set_debug(self, debug):
        """set debug"""
        if debug not in PorosOptions.available_debug_mode:
            raise TypeError("device for poros invalid, only %s supported, fix it" % (PorosOptions.available_debug_mode))
        self.debug = debug


class PorosModule(RecursiveScriptModule):
    """
    The core data structure of poros. 
    """
    def __init__(self, cpp_module):
        super(PorosModule, self).__init__(cpp_module)
        # self.options = PorosOptions()
        # if option is not None and isinstance(option, PorosOptions):
        #     self.options = option
            
    @staticmethod
    def _construct(cpp_module, init_fn):
        """
        Construct a PorosModule that's ready for use. 
        Args:
            cpp_module:  The C++ Module that will hold the actual state of
                            this PorosModule instance.
            init_fn:  Lambda that initializes the PorosModule passed to it.
        """
        script_module = PorosModule(cpp_module)
        init_fn(script_module)

        # Finalize the ScriptModule: replace the nn.Module state with our
        # custom implementations and flip the _initializing bit.
        PorosModule._finalize_scriptmodule(script_module)
        return script_module
    
    @property
    def supported_engine(self):
        """supported engine"""
        return ["tensorrt"]

    # @property
    # def options(self):
    #     """current options"""
    #     return self.options

