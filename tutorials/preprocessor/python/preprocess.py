from re import T
from traceback import print_tb
from typing import List
from cv2 import resize
import fastdeploy as fd
import cv2
import numpy as np


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cvcuda",
        required=True,
        type=bool,
        help="Use CVCUDA in preprocess")
    return parser.parse_args()


# define CustomProcessor
class CustomProcessor(fd.vision.common.manager.PyProcessorManager):
    def __init__(self) -> None:
        super().__init__()
        # create op
        hw = [500, 500]
        self.resize_op = fd.C.vision.processors.ResizeByShort(100, 1, True, hw)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        is_scale = True
        min = []
        max = []
        swap_rb = False
        self.normalize_permute_op = fd.C.vision.processors.NormalizeAndPermute(
            mean, std, is_scale, min, max, swap_rb)

    def apply(self, image_batch, outputs):
        image_batch.mats[0].print_info('before')
        self.resize_op(image_batch.mats[0])
        image_batch.mats[0].print_info('after')
        self.normalize_permute_op(image_batch.mats[0])
        outputs.append(image_batch.mats[0])


if __name__ == "__main__":

    # read jpg
    im1 = cv2.imread('ILSVRC2012_val_00000010.jpeg')
    im2 = cv2.imread('ILSVRC2012_val_00000010.jpeg')

    mat1 = fd.C.vision.FDMat()
    mat1.from_numpy(im1)
    mat2 = fd.C.vision.FDMat()
    mat2.from_numpy(im2)

    args = parse_arguments()
    # creae processor
    preprocessor = CustomProcessor()

    # use CVCUDA
    if args.use_cvcuda:
        preprocessor.use_cuda(True, -1)

    # run the Processer with CVCUDA
    outputs = []
    images = [mat1, mat2]
    preprocessor(images, outputs)

    # show output
    outputs[0].print_info('outputs[0]')
