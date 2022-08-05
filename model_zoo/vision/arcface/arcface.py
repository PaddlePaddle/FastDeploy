import fastdeploy as fd
import numpy as np
import cv2


# 余弦相似度
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    mul_a = np.linalg.norm(a, ord=2)
    mul_b = np.linalg.norm(b, ord=2)
    mul_ab = np.dot(a, b)
    return mul_ab / (np.sqrt(mul_a) * np.sqrt(mul_b))


# 加载模型
model = fd.vision.deepinsight.ArcFace("ms1mv3_arcface_r100.onnx")
print("Initialed model!")

# 加载图片
face0 = cv2.imread("face_recognition_0.png")  # 0,1 同一个人
face1 = cv2.imread("face_recognition_1.png")
face2 = cv2.imread("face_recognition_2.png")  # 0,2 不同的人

# 设置 l2 normalize
model.l2_normalize = True

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
