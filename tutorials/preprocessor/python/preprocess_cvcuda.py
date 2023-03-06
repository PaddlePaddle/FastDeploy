from re import T
from traceback import print_tb
from typing import List
from cv2 import resize
import fastdeploy as fd
import cv2
import numpy as np

# read jpg
im1 = cv2.imread('ILSVRC2012_val_00000010.jpeg')
im2 = cv2.imread('ILSVRC2012_val_00000010.jpeg')

mat1 = fd.C.vision.FDMat()
mat1.from_numpy(im1)
mat2 = fd.C.vision.FDMat()
mat2.from_numpy(im2)

# create op
hw = [500, 500]
resize_op = fd.C.vision.processors.ResizeByShort(100, 1, True, hw)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
is_scale = True
min = []
max = []
swap_rb = False
normalize_permute = fd.C.vision.processors.NormalizeAndPermute(
    mean, std, is_scale, min, max, swap_rb)


# create preprocessor
class Pyprocessor(fd.vision.common.manager.PyProcessorManager):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, image_batch, outputs):
        image_batch.mats[0].print_info('before')
        resize_op(image_batch.mats[0])
        image_batch.mats[0].print_info('after')
        normalize_permute(image_batch.mats[0])
        outputs.append(image_batch.mats[0])


preprocessor = Pyprocessor()

# use CVCUDA
preprocessor.use_cuda(True, -1)

# run the Processer with CVCUDA
outputs = []
images = [mat1, mat2]
preprocessor(images, outputs)

# show output
outputs[0].print_info('outputs[0]')
