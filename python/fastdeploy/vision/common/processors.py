from __future__ import absolute_import
from ... import c_lib_wrap as C


class Processor():
    def __init__(self):
        self.processor

    def __call__(self, mat):
        self.processor(mat)


class ResizeByShort(Processor):
    def __init__(self, target_size: int, interp=1, use_scale=True, max_hw=[]):
        self.processor = C.vision.processors.ResizeByShort(target_size, interp,
                                                           use_scale, max_hw)
        """Create a ResizeByShort operation with the given parameters.

        :param target_size: the target short size to resize the image
        :param interp: optionally, the interpolation mode for resizing image
        :param use_scale: optionally, whether to scale image
        :param max_hw: max spatial size which is used by ResizeByShort
        """


class CenterCrop(Processor):
    def __init__(self, width, height):
        self.processor = C.vision.processors.CenterCrop(width, height)
        """Create a CenterCrop operation with the given parameters.

        :param width: desired width of the cropped image
        :param height: desired height of the cropped image
        """


class Pad(Processor):
    def __init__(self, top: int, bottom: int, left: int, right: int, value=[]):
        self.processor = C.vision.processors.Pad(top, bottom, left, right,
                                                 value)
        """Create a Pad operation with the given parameters.

        :param top: the top padding
        :param bottom: the bottom padding
        :param left: the left padding
        :param right: the right padding
        :param value: the value that is used to pad on the input image
        """


class NormalizeAndPermute(Processor):
    def __init__(self,
                 mean=[],
                 std=[],
                 is_scale=True,
                 min=[],
                 max=[],
                 swap_rb=False):
        self.processor = C.vision.processors.NormalizeAndPermute(
            mean, std, is_scale, min, max, swap_rb)
        """Creae a Normalize and a Permute operation with the given parameters.

        :param mean     A list containing the mean of each channel
        :param std      A list containing the standard deviation of each channel
        :param is_scale Specifies if the image are being scaled or not
        :param min      A list containing the minimum value of each channel
        :param max      A list containing the maximum value of each channel
        """
