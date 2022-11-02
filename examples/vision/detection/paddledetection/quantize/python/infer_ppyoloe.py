import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PPYOLOE model.")
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
        ) == "gpu", "TensorRT backend require inferences on device GPU."
        option.use_trt_backend()
        option.set_trt_cache_file(os.path.join(args.model, "model.trt"))
        option.set_trt_input_shape("image", min_shape=[1, 3, 640, 640])
        option.set_trt_input_shape("scale_factor", min_shape=[1, 2])
    elif args.backend.lower() == "pptrt":
        assert args.device.lower(
        ) == "gpu", "TensorRT backend require inference on device GPU."
        option.use_trt_backend()
        option.enable_paddle_to_trt()
    elif args.backend.lower() == "ort":
        option.use_ort_backend()
    elif args.backend.lower() == "paddle":
        option.use_paddle_backend()
    elif args.backend.lower() == "openvino":
        assert args.device.lower(
        ) == "cpu", "OpenVINO backend require inference on device CPU."
        option.use_openvino_backend()
    return option


args = parse_arguments()

model_file = os.path.join(args.model, "model.pdmodel")
params_file = os.path.join(args.model, "model.pdiparams")
config_file = os.path.join(args.model, "infer_cfg.yml")

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.detection.PPYOLOE(
    model_file, params_file, config_file, runtime_option=runtime_option)

# 预测图片检测结果
im = cv2.imread(args.image)
result = model.predict(im.copy())
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
