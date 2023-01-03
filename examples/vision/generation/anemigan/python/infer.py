import cv2
import os
import fastdeploy as fd


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
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
    else:
        option.set_paddle_mkldnn(False)
    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
fd.download_model(name=args.model, path='./', format='paddle')
model_file = os.path.join(args.model, "model.pdmodel")
params_file = os.path.join(args.model, "model.pdiparams")
model = fd.vision.generation.AnimeGAN(
    model_file, params_file, runtime_option=runtime_option)

# 预测图片并保存结果
im = cv2.imread(args.image)
result = model.predict(im)
cv2.imwrite('style_transfer_result.png', result)
