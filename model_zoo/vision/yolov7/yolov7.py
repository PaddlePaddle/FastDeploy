import fastdeploy as fd
import cv2

# 下载模型和测试图片
model_url = "TODO "
test_jpg_url = "https://github.com/WongKinYiu/yolov7/blob/main/inference/images/horses.jpg"
fd.download(model_url, ".", show_progress=True)
fd.download(test_jpg_url, ".", show_progress=True)

# 加载模型
model = fd.vision.wongkinyiu.YOLOv7("yolov7.onnx")

# 预测图片
im = cv2.imread("horses.jpg")
result = model.predict(im, conf_threshold=0.25, nms_iou_threshold=0.5)

# 可视化结果
fd.vision.visualize.vis_detection(im, result)
cv2.imwrite("vis_result.jpg", im)

# 输出预测结果
print(result)
print(model.runtime_option)
