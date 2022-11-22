"""
init file for poros
"""

import os
import sys

if sys.version_info < (3, 6):
    raise Exception("Poros can only work on Python 3.6+")

import ctypes
import torch

from poros._compile import *
from poros._module import PorosOptions

def _register_with_torch():
    poros_dir = os.path.dirname(__file__)
    torch.ops.load_library(poros_dir + '/lib/libporos.so')

_register_with_torch()