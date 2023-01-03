import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path of PaddleDetection model directory")
    parser.add_argument(
        "--image", required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.device.lower() == "kunlunxin":
        option.use_kunlunxin()

    if args.device.lower() == "ascend":
        option.use_ascend()

    return option


args = parse_arguments()

model_file = os.path.join(args.model_dir, "model.pdmodel")
params_file = os.path.join(args.model_dir, "model.pdiparams")
config_file = os.path.join(args.model_dir, "infer_cfg.yml")

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.detection.SSD(model_file,
                                params_file,
                                config_file,
                                runtime_option=runtime_option)

image_file_path = "../dataset/coco/val2017"
annotation_file_path = "../dataset/coco/annotations/instances_val2017.json"

res = fd.vision.evaluation.eval_detection(model, image_file_path,
                                          annotation_file_path)
print(res)
