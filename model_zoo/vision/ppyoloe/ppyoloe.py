import fastdeploy as fd
import cv2

# 下载模型和测试图片
model_url = "https://bj.bcebos.com/paddle2onnx/fastdeploy/models/ppdet/ppyoloe_crn_l_300e_coco.tgz"
test_jpg_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/2.4/demo/000000014439_640x640.jpg"
fd.download_and_decompress(model_url, ".")
fd.download(test_jpg_url, ".", show_progress=True)

# 加载模型
model = fd.vision.ppdet.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                "ppyoloe_crn_l_300e_coco/infer_cfg.yml")

# 预测图片
im = cv2.imread("000000014439_640x640.jpg")
result = model.predict(im, conf_threshold=0.5)

# 可视化结果
fd.vision.visualize.vis_detection(im, result)
cv2.imwrite("vis_result.jpg", im)

# 输出预测结果
print(result)
