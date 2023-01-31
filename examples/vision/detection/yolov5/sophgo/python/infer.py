import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path of model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")

    return parser.parse_args()


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

model_file = args.model
params_file = ""

model = fd.vision.detection.YOLOv5(
    model_file,
    params_file,
    runtime_option=runtime_option,
    model_format=fd.ModelFormat.SOPHGO)

# 预测图片分类结果
im = cv2.imread(args.image)
result = model.predict(im)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_detection(im, result)
cv2.imwrite("sophgo_result.jpg", vis_im)
print("Visualized result save in ./sophgo_result.jpg")
