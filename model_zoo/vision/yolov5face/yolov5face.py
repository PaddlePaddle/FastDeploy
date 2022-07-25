import fastdeploy as fd
import cv2

# 加载模型
model = fd.vision.deepcam.YOLOv5Face("yolov5s-face.onnx")

# 预测图片
im = cv2.imread("test.jpg")
result = model.predict(im, conf_threshold=0.1, nms_iou_threshold=0.3)

# 可视化结果
fd.vision.visualize.vis_face_detection(im, result)
cv2.imwrite("vis_result.jpg", im)

# 输出预测结果
print(result)
print(model.runtime_option)
