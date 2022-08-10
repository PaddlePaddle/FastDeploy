import fastdeploy as fd
import cv2

# 下载模型
model_url = "https://github.com/DefTruth/Pytorch_Retinaface/releases/download/v0.1/Pytorch_RetinaFace_mobile0.25-640-640.onnx"
test_img_url = "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/imgs/3.jpg"
fd.download(model_url, ".", show_progress=True)
fd.download(test_img_url, ".", show_progress=True)

# 加载模型
model = fd.vision.biubug6.RetinaFace(
    "Pytorch_RetinaFace_mobile0.25-640-640.onnx")

# 预测图片
im = cv2.imread("3.jpg")
result = model.predict(im, conf_threshold=0.7, nms_iou_threshold=0.3)

# 可视化结果
vis_im = fd.vision.visualize.vis_face_detection(im, result)
cv2.imwrite("vis_result.jpg", vis_im)

# 输出预测结果
print(result)
print(model.runtime_option)
