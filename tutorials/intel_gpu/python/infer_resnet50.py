import fastdeploy as fd
import cv2
import os
import time


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleClas model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--topk", type=int, default=1, help="Return topk results.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'intel_gpu'.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    option.use_openvino_backend()

    assert args.device.lower(
    ) in ["cpu", "intel_gpu"], "--device only support ['cpu', 'intel_gpu']"

    if args.device.lower() == "intel_gpu":
        option.set_openvino_device("GPU")
        option.set_openvino_shape_info({"inputs": [1, 3, 224, 224]})

    return option


args = parse_arguments()

runtime_option = build_option(args)

model_file = os.path.join(args.model, "inference.pdmodel")
params_file = os.path.join(args.model, "inference.pdiparams")
config_file = os.path.join(args.model, "inference_cls.yaml")
model = fd.vision.classification.PaddleClasModel(
    model_file, params_file, config_file, runtime_option=runtime_option)

im = cv2.imread(args.image)

print("Warmup 20 times...")
for i in range(20):
    result = model.predict(im, args.topk)

print("Counting time...")
start = time.time()
for i in range(50):
    result = model.predict(im, args.topk)
end = time.time()
print("Elapsed time: {}ms".format((end - start) * 1000))
print(result)
