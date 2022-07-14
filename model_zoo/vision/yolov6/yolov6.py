import fastdeploy as fd
import cv2 

# 下载模型和测试图片
model_url = "https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx"
test_jpg_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
fd.download(model_url, ".", show_progress=True)
fd.download(test_jpg_url, ".", show_progress=True)

# 加载模型
model = fd.vision.meituan.YOLOv6("yolov6s.onnx")
print(model.is_dynamic_shape())

# 预测图片
im = cv2.imread("bus.jpg")
result = model.predict(im, conf_threshold=0.25, nms_iou_threshold=0.5)

# 可视化结果
fd.vision.visualize.vis_detection(im, result)
cv2.imwrite("vis_result.jpg", im)

# 输出预测结果
print(result)
print(model.runtime_option)
