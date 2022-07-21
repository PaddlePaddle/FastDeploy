import fastdeploy as fd
import cv2
import tarfile

# 下载模型和测试图片
model_url = "https://github.com/felixhjh/Fastdeploy-Models/raw/main/unet_Cityscapes.tar.gz"
test_jpg_url = "https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png"
fd.download(model_url, ".", show_progress=True)
fd.download(test_jpg_url, ".", show_progress=True)

try:
    tar = tarfile.open("unet_Cityscapes.tar.gz", "r:gz")
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, ".")
    tar.close()
except Exception as e:
    raise Exception(e)

# 加载模型
model = fd.vision.ppseg.Model("./unet_Cityscapes/model.pdmodel",
                              "./unet_Cityscapes/model.pdiparams",
                              "./unet_Cityscapes/deploy.yaml")

# 预测图片
im = cv2.imread("./cityscapes_demo.png")
result = model.predict(im)

vis_im = im.copy()
# 可视化结果
fd.vision.visualize.vis_segmentation(im, result, vis_im)
cv2.imwrite("vis_img.png", vis_im)

# 输出预测结果
print(result)
print(model.runtime_option)
