import fastdeploy as fd
import cv2
import os
from fastdeploy import ModelFormat


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of smoke paddle model.")
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
        option.use_gpu(0)
    if args.device.lower() == "cpu":
        option.use_cpu()
    return option


args = parse_arguments()

model_file = os.path.join(args.model, "smoke.pdmodel")
params_file = os.path.join(args.model, "smoke.pdiparams")
config_file = os.path.join(args.model, "infer_cfg.yml")
# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.perception.Smoke(
    model_file, params_file, config_file, runtime_option=runtime_option)

# 预测图片检测结果
im = cv2.imread(args.image)
result = model.predict(im)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_perception(im, result, config_file)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
