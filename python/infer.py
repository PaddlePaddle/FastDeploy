import fastdeploy as fd
import cv2
import pickle

def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of yolov7face onnx model.")
    parser.add_argument(
        "--image", required=True, help="Path of test image file.")
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
model = fd.vision.facedet.YOLOv7Face(args.model, runtime_option=runtime_option)

# 预测图片检测结果
im = cv2.imread(args.image)
result = model.predict(im.copy())
print(result)

# 保存图片预测结果
# input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
# input_url2 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000570688.jpg"
# fd.download(input_url1, "resources")
# fd.download(input_url2, "resources")
# im1 = cv2.imread("./resources/000000014439.jpg")
# im2 = cv2.imread("./resources/000000570688.jpg")
# result1 = model.predict(im1.copy())
# result2 = model.predict(im2.copy())
# pickle_file1 = open('yolov7face_result1.pkl', 'wb')
# pickle_file2 = open('yolov7face_result2.pkl', 'wb')
# pickle.dump(result1, pickle_file1)
# pickle.dump(result2, pickle_file2)
# pickle_file1.close()
# pickle_file2.close()
# vis_im1 = fd.vision.vis_face_detection(im1, result1)
# vis_im2 = fd.vision.vis_face_detection(im2, result2)
# cv2.imwrite("visualized_result1.jpg", vis_im1)
# cv2.imwrite("visualized_result2.jpg", vis_im2)


# 预测结果可视化
vis_im = fd.vision.vis_face_detection(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
