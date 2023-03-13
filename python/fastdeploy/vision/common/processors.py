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


class Cast(Processor):
    def __init__(self, dtype="float"):
        self.processor = C.vision.processors.Cast(dtype)
        """Creat a new cast opereaton with given dtype

        :param dtype dtype of the output
        """


class HWC2CHW(Processor):
    def __init__(self):
        self.processor = C.vision.processors.HWC2CHW()
        """Creat a new hwc2chw processor with default dtype.

        :return An instance of processor `HWC2CHW`
        """


class Normalize(Processor):
    def __init__(self,
                 mean=[],
                 std=[],
                 is_scale=True,
                 min=[],
                 max=[],
                 swap_rb=False):
        self.processor = C.vision.processors.Normalize(mean, std, is_scale,
                                                       min, max, swap_rb)
        """Creat a new normalize opereator with given paremeters.

        :param mean     A list containing the mean of each channel
        :param std      A list containing the standard deviation of each channel
        :param is_scale Specifies if the image are being scaled or not
        :param min      A list containing the minimum value of each channel
        :param max      A list containing the maximum value of each channel
        """


class PadToSize(Processor):
    def __init__(self, width, height, value=[]):
        self.processor = C.vision.processors.PadToSize(width, height, value)
        """Create a new PadToSize opereator with given parameters.

        :param width     Desired width of the output image
        :param height    Desired height of the output image
        :param value     values to pad with
        """


class Resize(Processor):
    def __init__(self,
                 width,
                 height,
                 scale_w=-1.0,
                 scale_h=-1.0,
                 interp=1,
                 use_scale=False):
        self.processor = C.vision.processors.Resize(width, height, scale_w,
                                                    scale_h, interp, use_scale)
        """Create a Resize operation with the given parameters.

        :param width   Desired width of the output image
        :param height  Desired height of the output image
        :param scale_w Scales the width in x-direction
        :param scale_h Scales the height in y-direction
        :param interp: optionally, the interpolation mode for resizing image
        :param use_scale: optionally, whether to scale image
        """


class StridePad(Processor):
    def __init__(self, stride, value=[]):
        self.processor = C.vision.processors.StridePad(stride, value)
        """Create a StridePad processor with given parameters.

        :param stride Stride of the processor
        :param value  values to pad with
        """
