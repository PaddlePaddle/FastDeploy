import fastdeploy as fd
import cv2

# 下载模型和测试图片
test_jpg_url = "https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_2.jpg"
fd.download(test_jpg_url, ".", show_progress=True)

# 加载模型
# model = fd.vision.deepinsight.SCRFD("SCRFD.onnx")

# 加载无关键点预测模型
model = fd.vision.deepinsight.SCRFD("scrfd.onnx")

# 修改参数
model.use_kps = False
model.landmarks_per_face = 0

# 预测图片
im = cv2.imread("test_lite_face_detector.jpg")
result = model.predict(im, conf_threshold=0.5, nms_iou_threshold=0.5)

# 可视化结果
fd.vision.visualize.vis_face_detection(im, result)
cv2.imwrite("vis_result.jpg", im)

# 输出预测结果
print(result)
print(model.runtime_option)
