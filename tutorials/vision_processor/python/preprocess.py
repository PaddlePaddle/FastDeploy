import fastdeploy as fd
import cv2

from fastdeploy.vision.common.manager import PyProcessorManager


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cvcuda",
        required=False,
        type=bool,
        help="Use CV-CUDA in preprocess")
    return parser.parse_args()


# define CustomProcessor
class CustomProcessor(PyProcessorManager):
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

        width = 50
        height = 50
        self.centercrop_op = fd.C.vision.processors.CenterCrop(width, height)

        top = 5
        bottom = 5
        left = 5
        right = 5
        pad_value = [225, 225, 225]
        self.pad_op = fd.C.vision.processors.Pad(top, bottom, left, right,
                                                 pad_value)

    def apply(self, image_batch):
        outputs = []
        self.resize_op(image_batch)
        self.centercrop_op(image_batch)
        self.pad_op(image_batch)
        self.normalize_permute_op(image_batch)

        for i in range(len(image_batch.mats)):
            outputs.append(image_batch.mats[i])

        return outputs


if __name__ == "__main__":

    # read jpg
    im1 = cv2.imread('ILSVRC2012_val_00000010.jpeg')
    im2 = cv2.imread('ILSVRC2012_val_00000010.jpeg')

    mat1 = fd.C.vision.FDMat()
    mat1.from_numpy(im1)
    mat2 = fd.C.vision.FDMat()
    mat2.from_numpy(im2)
    images = [mat1, mat2]

    args = parse_arguments()
    # creae processor
    preprocessor = CustomProcessor()

    # use CV-CUDA
    if args.use_cvcuda:
        preprocessor.use_cuda(True, -1)

    # show input
    for i in range(len(images)):
        images[i].print_info('images' + str(i) + ': ')

    # run the Processer with CVCUDA
    outputs = preprocessor(images)

    # show output
    for i in range(len(outputs)):
        outputs[i].print_info('outputs' + str(i) + ': ')
