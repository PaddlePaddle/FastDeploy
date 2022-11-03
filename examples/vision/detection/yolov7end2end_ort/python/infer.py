import fastdeploy as fd
import cv2


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None, help="Path of yolov7 end2end onnx model.")
    parser.add_argument(
        "--model_hub",
        type=str,
        default=None,
        help="Model name in model hub, the model will be downloaded automatically."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
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
        option.set_trt_input_shape("images", [1, 3, 640, 640])
    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)

assert args.model is None and args.model_hub is None, "Please set the model or model hub parameter."

if args.model is not None:
    model = args.model
else:
    model = fd.download_model(name=args.model_hub)

model = fd.vision.detection.YOLOv7End2EndORT(
    model, runtime_option=runtime_option)

# 预测图片检测结果
im = cv2.imread(args.image)
result = model.predict(im.copy())
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_detection(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
