from __future__ import absolute_import
from ... import c_lib_wrap as C


class ResizeByShort():
    def __init__(self, target_size: int, interp=1, use_scale=True, max_hw=[]):
        self.processor = C.vision.processors.ResizeByShort(target_size, interp,
                                                           use_scale, max_hw)

    def __call__(self, mat):
        self.processor(mat)


class CenterCrop():
    def __init__(self, width, height):
        self.processor = C.vision.processors.CenterCrop(width, height)

    def __call__(self, mat):
        self.processor(mat)


class Pad():
    def __init__(self, top: int, bottom: int, left: int, right: int, value=[]):
        self.processor = C.vision.processors.Pad(top, bottom, left, right,
                                                 value)

    def __call__(self, mat):
        self.processor(mat)


class NormalizeAndPermute():
    def __init__(self,
                 mean=[],
                 std=[],
                 is_scale=True,
                 min=[],
                 max=[],
                 swap_rb=False):
        self.processor = C.vision.processors.NormalizeAndPermute(
            mean, std, is_scale, min, max, swap_rb)

    def __call__(self, mat):
        self.processor(mat)
