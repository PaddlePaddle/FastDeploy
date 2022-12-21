import fastdeploy as fd
import cv2
import os
from fastdeploy import ModelFormat


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of yolov5 onnx model.")
    parser.add_argument(
        "--image", required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    parser.add_argument(
        "--cpu_thread_num",
        type=int,
        default=9,
        help="Number of threads while inference on CPU.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    if args.device.lower() == "gpu":
        option.use_gpu(0)

    option.set_cpu_thread_num(args.cpu_thread_num)

    if args.backend.lower() == "trt":
        assert args.device.lower(
        ) == "gpu", "TensorRT backend require inference on device GPU."
        option.use_trt_backend()
    elif args.backend.lower() == "pptrt":
        assert args.device.lower(
        ) == "gpu", "TensorRT backend require inference on device GPU."
        option.use_trt_backend()
        option.enable_paddle_to_trt()
    elif args.backend.lower() == "ort":
        option.use_ort_backend()
    return option


args = parse_arguments()

model_file = os.path.join(args.model, "model.pdmodel")
params_file = os.path.join(args.model, "model.pdiparams")
# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.detection.YOLOv5(
    model_file,
    params_file,
    runtime_option=runtime_option,
    model_format=ModelFormat.PADDLE)

# 预测图片检测结果
im = cv2.imread(args.image)
result = model.predict(im)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_detection(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
