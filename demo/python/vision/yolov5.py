import fastdeploy as fd
import cv2

# 获取模型 和 测试图片
# wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
# wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg
model = fd.vision.ultralytics.YOLOv5("yolov5s.onnx")
im = cv2.imread("bus.jpg")
result = model.predict(im, conf_threshold=0.25, nms_iou_threshold=0.5)
print(result)
