import fastdeploy as fd
import cv2
import numpy as np


# 余弦相似度
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    mul_a = np.linalg.norm(a, ord=2)
    mul_b = np.linalg.norm(b, ord=2)
    mul_ab = np.dot(a, b)
    return mul_ab / (np.sqrt(mul_a) * np.sqrt(mul_b))


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of insightface onnx model.")
    parser.add_argument(
        "--face", required=True, help="Path of test face image file.")
    parser.add_argument(
        "--face_positive",
        required=True,
        help="Path of test face_positive image file.")
    parser.add_argument(
        "--face_negative",
        required=True,
        help="Path of test face_negative image file.")
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
        option.set_trt_input_shape("data", [1, 3, 112, 112])
    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.faceid.PartialFC(args.model, runtime_option=runtime_option)

# 加载图片
face0 = cv2.imread(args.face)  # 0,1 同一个人
face1 = cv2.imread(args.face_positive)
face2 = cv2.imread(args.face_negative)  # 0,2 不同的人

# 设置 l2 normalize
model.l2_normalize = True

# 预测图片检测结果
result0 = model.predict(face0)
result1 = model.predict(face1)
result2 = model.predict(face2)

# 计算余弦相似度
embedding0 = result0.embedding
embedding1 = result1.embedding
embedding2 = result2.embedding

cosine01 = cosine_similarity(embedding0, embedding1)
cosine02 = cosine_similarity(embedding0, embedding2)

# 打印结果
print(result0, end="")
print(result1, end="")
print(result2, end="")
print("Cosine 01: ", cosine01)
print("Cosine 02: ", cosine02)
print(model.runtime_option)
