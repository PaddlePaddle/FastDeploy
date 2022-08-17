import fastdeploy as fd
import cv2
import os

# 下载模型和测试图
# 下载模型和测试图片
test_jpg_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/doc/imgs/12.jpg"
test_label_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/ppocr/utils/ppocr_keys_v1.txt"

det_model_url = " https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar"
cls_model_url = " https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
rec_model_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar"

fd.download_and_decompress(det_model_url, ".")
fd.download_and_decompress(cls_model_url, ".")
fd.download_and_decompress(rec_model_url, ".")
fd.download(test_label_url, ".", show_progress=True)
fd.download(test_jpg_url, ".", show_progress=True)

#三个模型的启用与否
use_det = True
use_cls = True
use_rec = True

rec_op = fd.RuntimeOption()
rec_op.use_paddle_backend()
#PPinfer后端推理REC模型需要删除下面的Pass
rec_op.enable_paddle_delete_pass("matmul_transpose_reshape_fuse_pass")

#初始化sys
model = fd.vision.ppocr.PPocrsys(
    use_det,
    use_cls,
    use_rec,
    "ppocr_keys_v1.txt",
    "ch_PP-OCRv3_det_infer/inference.pdmodel",
    "ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel",
    "ch_PP-OCRv3_rec_infer/inference.pdmodel",
    "ch_PP-OCRv3_det_infer/inference.pdiparams",
    "ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams",
    "ch_PP-OCRv3_rec_infer/inference.pdiparams",
    rec_custom_option=rec_op)

#准备输入图片数据
img_dir = './'
imgs_file_lists = []
img_list = []
if os.path.isdir(img_dir):
    for single_file in os.listdir(img_dir):
        if 'jpg' in single_file:
            file_path = os.path.join(img_dir, single_file)
            if os.path.isfile(file_path):
                imgs_file_lists.append(file_path)

for img_file in imgs_file_lists:
    img = cv2.imread(img_file)
    img_list.append(img)

#开始预测
result = model.predict(img_list, use_det, use_cls, use_rec)

# 输出预测结果
print(result)
