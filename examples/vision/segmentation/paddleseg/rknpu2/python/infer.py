import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", required=True, help="Path of PaddleSeg model.")
    parser.add_argument(
        "--config_file", required=True, help="Path of PaddleSeg config.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    option.use_rknpu2()
    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
model_file = args.model_file
params_file = ""
config_file = args.config_file
model = fd.vision.segmentation.PaddleSegModel(
    model_file, params_file, config_file, runtime_option=runtime_option,model_format=fd.ModelFormat.RKNN)

model.disable_normalize_and_permute()

# 预测图片分割结果
im = cv2.imread(args.image)
result = model.predict(im.copy())
print(result)

# 可视化结果
vis_im = fd.vision.vis_segmentation(im, result, weight=0.5)
cv2.imwrite("vis_img.png", vis_im)
