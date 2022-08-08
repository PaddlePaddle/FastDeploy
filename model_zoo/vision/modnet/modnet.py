import fastdeploy as fd
import cv2

# 加载模型
model = fd.vision.zhkkke.MODNet("modnet_photographic_portrait_matting.onnx")

# 设置模型输入大小
model.size = (256, 256)

# 预测图片
im = cv2.imread("./matting_1.jpg")
im_old = im.copy()
vis_im = im.copy()

result = model.predict(im)
# 可视化结果
fd.vision.visualize.vis_matting_alpha(im_old, result, vis_im, False)
cv2.imwrite("vis_result.jpg", vis_im)

# 输出预测结果
print(result)
print(model.runtime_option)
