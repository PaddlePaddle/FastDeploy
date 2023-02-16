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
        "--image", required=True, help="path of test image file.")
    return parser.parse_args()


def build_tinypose_option(args):
    option = fd.RuntimeOption()
    option.use_rknpu()
    return option


args = parse_arguments()

tinypose_model_file = os.path.join(args.tinypose_model_dir, "PP_TinyPose_256x192_infer_rk3588_unquantized.rknn")
tinypose_params_file = os.path.join(args.tinypose_model_dir, "")
tinypose_config_file = os.path.join(args.tinypose_model_dir, "infer_cfg.yml")
# 配置runtime，加载模型
runtime_option = build_tinypose_option(args)
tinypose_model = fd.vision.keypointdetection.PPTinyPose(
    tinypose_model_file,
    tinypose_params_file,
    tinypose_config_file,
    runtime_option=runtime_option,
    model_format=fd.ModelFormat.RKNN)
tinypose_model.disable_normalize()
tinypose_model.disable_permute()

# 预测图片检测结果
im = cv2.imread(args.image)
tinypose_result = tinypose_model.predict(im)
print("Paddle TinyPose Result:\n", tinypose_result)

# 预测结果可视化
vis_im = fd.vision.vis_keypoint_detection(
    im, tinypose_result, conf_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("TinyPose visualized result save in ./visualized_result.jpg")
