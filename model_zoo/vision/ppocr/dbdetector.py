import fastdeploy as fd
import cv2

# 下载模型和测试图
# 下载模型和测试图片
model_url = " https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar"
test_jpg_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/doc/imgs/12.jpg"
fd.download_and_decompress(model_url, ".")
fd.download(test_jpg_url, ".", show_progress=True)

model = fd.vision.ppocr.DBDetector(
    "ch_PP-OCRv3_det_infer/inference.pdmodel",
    "ch_PP-OCRv3_det_infer/inference.pdiparams",
    runtime_option=None,
    model_format=fd.fastdeploy_main.Frontend.PADDLE)
# 预测图片
im = cv2.imread("12.jpg")
result = model.predict(im)

# 可视化结果
fd.vision.visualize.vis_ppocr(im, result)
cv2.imwrite("vis_result.jpg", im)
# 输出预测结果
print(result)
print(model.runtime_option)
