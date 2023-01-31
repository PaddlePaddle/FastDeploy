import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path of PaddleDetection model directory")
    parser.add_argument(
        "--image", default=None, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'kunlunxin', 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "kunlunxin":
        option.use_kunlunxin()

    if args.device.lower() == "ascend":
        option.use_ascend()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
    return option


args = parse_arguments()

if args.model_dir is None:
    model_dir = fd.download_model(name='yolov3_darknet53_270e_coco')
else:
    model_dir = args.model_dir

model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.detection.YOLOv3(
    model_file, params_file, config_file, runtime_option=runtime_option)

# 预测图片检测结果
if args.image is None:
    image = fd.utils.get_detection_test_image()
else:
    image = args.image
im = cv2.imread(image)
result = model.predict(im)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
