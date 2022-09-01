import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tinypose_model_dir",
        required=True,
        help="path of paddletinypose model directory")
    parser.add_argument(
        "--det_model_dir", help="path of paddledetection model directory")
    parser.add_argument(
        "--image", required=True, help="path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="wether to use tensorrt.")
    return parser.parse_args()


def build_picodet_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("image", [1, 3, 320, 320])
        option.set_trt_input_shape("scale_factor", [1, 2])
    return option


def build_tinypose_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("image", [1, 3, 128, 96])
    return option


args = parse_arguments()
det_result = None
if args.det_model_dir:
    picodet_model_file = os.path.join(args.det_model_dir, "model.pdmodel")
    picodet_params_file = os.path.join(args.det_model_dir, "model.pdiparams")
    picodet_config_file = os.path.join(args.det_model_dir, "infer_cfg.yml")

    # 配置runtime，加载模型
    runtime_option = build_picodet_option(args)
    det_model = fd.vision.detection.PicoDet(
        picodet_model_file,
        picodet_params_file,
        picodet_config_file,
        runtime_option=runtime_option)

    # 预测图片检测结果
    im = cv2.imread(args.image)
    det_result = det_model.predict(im.copy())
    print("PicoDet Result:\n", det_result)

    # 预测结果可视化
    det_vis_im = fd.vision.vis_detection(im, det_result, score_threshold=0.5)
    cv2.imwrite("det_visualized_result.jpg", det_vis_im)
    print("Detection visualized result save in ./det_visualized_result.jpg")

tinypose_model_file = os.path.join(args.tinypose_model_dir, "model.pdmodel")
tinypose_params_file = os.path.join(args.tinypose_model_dir, "model.pdiparams")
tinypose_config_file = os.path.join(args.tinypose_model_dir, "infer_cfg.yml")
# 配置runtime，加载模型
runtime_option = build_tinypose_option(args)
tinypose_model = fd.vision.detection.PPTINYPOSE(
    tinypose_model_file,
    tinypose_params_file,
    tinypose_config_file,
    runtime_option=runtime_option)
# 预测图片检测结果
im = cv2.imread(args.image)
tinypose_result = tinypose_model.predict(im.copy(), det_result)
print("Paddle TinyPose Result:\n", tinypose_result)

# 预测结果可视化
vis_im = fd.vision.vis_keypoint_detection(
    im, tinypose_result, conf_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("TinyPose visualized result save in ./visualized_result.jpg")
