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
compile function for poros.
"""

from typing import List, Dict, Any
import torch
from torch import nn

import poros._C
from poros._input_convert import convert_prewarm_inputs
from poros._input_convert import convert_poros_option
from poros._module import PorosModule


def wrap_cpp_module(cpp_module):
    """
    Wrap torch._C.ScriptModule to porosModule, recursively for all submodules
    """
    def init_fn(script_module):
        """init_fn"""
        for name, cpp_module in torch._C.ModuleDict(script_module._c).items():
            setattr(script_module, name, wrap_cpp_module(cpp_module))
        script_module._concrete_type = torch._C.ConcreteModuleType.from_jit_type(script_module._c._type())

        for idx, fn in enumerate(script_module._c._get_forward_pre_hooks()):
            script_module._forward_pre_hooks[idx] = fn
        for idx, fn in enumerate(script_module._c._get_forward_hooks()):
            script_module._forward_hooks[idx] = fn

    return PorosModule._construct(cpp_module, init_fn)


def load(filename, poros_options):
    """
    Args:
        filename( str): poros model save path
        poros_options(PorosOptions / Dict of settings): compile settings for poros
    Returns:
        PorosModule: Compiled Module of poros, 
                    when run it will partially execute via inlined engine (which is TensorRT)
    """
    compiled_cpp_mod = poros._C.load(filename, convert_poros_option(poros_options))
    compiled_module = wrap_cpp_module(compiled_cpp_mod)
    return compiled_module

def save(module, filename):
    """
    Args:
        module（PorosModule）: poros module
        filename( str): poros model save path
    """
    assert type(module).__name__ == "PorosModule", "The type of module must be PorosModule"
    assert type(filename).__name__ == "str", "The type of filename must be str"
    module.save(filename)

def compile(module, prewarm_inputs, poros_options):
    """
    Compile a TorchScriptModule/nn.Module to porosModule
    Converts specifically the forward method of the original Module
    Args:
        module (torch.nn.Module / torch.jit.ScriptModule): Source module
        input (list of tensor input): prewarmed data.
        poros_options(PorosOptions): compile settings for poros
    Returns:
        PorosModule: Compiled Module of poros, 
                    when run it will partially execute via inlined engine (which is TensorRT)
    """  
    if poros_options.device == "GPU":  
        assert "cuda" in str(list(module.state_dict().values())[0].device), \
            "If the poros_options.device is GPU, the module.device should also is GPU"

    sp_model = None
    if isinstance(module, torch.jit.ScriptModule):
        sp_model = module
    else:
        if poros_options.preprocess_mode == 0:
            sp_model = torch.jit.script(module, optimize=None, _frames_up=0, _rcb=None)
        elif poros_options.preprocess_mode == 1:
            sp_model = torch.jit.trace(module, prewarm_inputs[0])
        else:
            raise ValueError(
                "preprocess_mode value err: The range of preprocess_mode is [0,1]")
    
    if sp_model is None:
        raise TypeError(
            "can't trans to poros module currently")

    wraped_inputs = convert_prewarm_inputs(prewarm_inputs)

    compiled_cpp_mod = poros._C.compile_graph(sp_model._c, wraped_inputs, convert_poros_option(poros_options))
    compiled_module = wrap_cpp_module(compiled_cpp_mod)
    return compiled_module
