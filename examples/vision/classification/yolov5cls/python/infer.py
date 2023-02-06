import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of YOLOv5Cls model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--topk", type=int, default=1, help="Return topk results.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("images", [1, 3, 224, 224])
    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.classification.YOLOv5Cls(
    args.model, runtime_option=runtime_option)
model.postprocessor.topk = args.topk

# 预测图片分类结果
im = cv2.imread(args.image)
result = model.predict(im)
print(result)
