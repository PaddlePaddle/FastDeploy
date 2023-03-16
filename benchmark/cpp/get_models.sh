#!/bin/bash
set -e
set +x

download_fd_model_zxvf() {
  local model="$1"
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget https://bj.bcebos.com/paddlehub/fastdeploy/$model && tar -zxvf $model
  else
     echo "[INFO] --- $model already exists!"
  fi
}
download_fd_model_xvf() {
  local model="$1"
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget https://bj.bcebos.com/paddlehub/fastdeploy/$model && tar -xvf $model
  else
     echo "[INFO] --- $model already exists!"
  fi
}
download_common_model_zxvf() {
  local model_url="$1"
  local model="$2"
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget ${model_url} && tar -zxvf $model
  else
     echo "[INFO] --- $model already exists!"
  fi
}
download_common_model_xvf() {
  local model_url="$1"
  local model="$2"
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget ${model_url} && tar -xvf $model
  else
     echo "[INFO] --- $model already exists!"
  fi
}
download_common_file() {
  local file_url="$1"
  local file="$2"
  if [ ! -f "${file}" ]; then
     echo "[INFO] --- downloading $file_url"
     wget ${file_url}
  else
     echo "[INFO] --- $file already exists!"
  fi
}

# PaddleDetection
download_fd_model_zxvf ppyoloe_crn_l_300e_coco_no_nms.tgz
download_fd_model_zxvf picodet_l_640_coco_lcnet_no_nms.tgz
download_fd_model_zxvf ppyoloe_plus_crn_m_80e_coco_no_nms.tgz
download_fd_model_zxvf yolox_s_300e_coco_no_nms.tgz
download_fd_model_zxvf yolov5_s_300e_coco_no_nms.tgz
download_fd_model_zxvf yolov6_s_300e_coco_no_nms.tgz
download_fd_model_zxvf yolov7_l_300e_coco_no_nms.tgz
download_fd_model_zxvf yolov8_s_500e_coco_no_nms.tgz
# PaddleClas
download_fd_model_zxvf PPLCNet_x1_0_infer.tgz
download_fd_model_zxvf PPLCNetV2_base_infer.tgz
download_fd_model_zxvf EfficientNetB7_infer.tgz
download_fd_model_zxvf GhostNet_x0_5_infer.tgz
download_fd_model_zxvf GhostNet_x1_3_infer.tgz
download_fd_model_zxvf GhostNet_x1_3_ssld_infer.tgz
download_fd_model_zxvf MobileNetV1_x0_25_infer.tgz
download_fd_model_zxvf MobileNetV1_ssld_infer.tgz
download_fd_model_zxvf MobileNetV2_x0_25_infer.tgz
download_fd_model_zxvf MobileNetV2_ssld_infer.tgz
download_fd_model_zxvf MobileNetV3_small_x0_35_ssld_infer.tgz
download_fd_model_zxvf MobileNetV3_large_x1_0_ssld_infer.tgz
download_fd_model_zxvf ShuffleNetV2_x0_25_infer.tgz
download_fd_model_zxvf ShuffleNetV2_x2_0_infer.tgz
download_fd_model_zxvf SqueezeNet1_1_infer.tgz
download_fd_model_zxvf InceptionV3_infer.tgz
download_fd_model_zxvf ResNet50_vd_infer.tgz
download_fd_model_zxvf ResNet50_infer.tgz
download_fd_model_zxvf PPHGNet_tiny_ssld_infer.tgz
download_fd_model_zxvf PPHGNet_base_ssld_infer.tgz
download_fd_model_zxvf EfficientNetB0_infer.tgz
download_fd_model_zxvf MobileNetV2_infer.tgz
download_fd_model_zxvf MobileNetV3_small_x1_0_infer.tgz
download_fd_model_zxvf ViT_large_patch16_224_infer.tgz
download_fd_model_zxvf ResNeXt50_32x4d_infer.tgz
download_fd_model_zxvf DenseNet121_infer.tgz
download_fd_model_zxvf PPHGNet_small_infer.tgz
download_fd_model_zxvf person_exists_infer.tgz
download_fd_model_zxvf EfficientNetB0_small_infer.tgz
# PaddleSeg
download_fd_model_zxvf PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
download_fd_model_zxvf PP_HumanSegV1_Lite_infer.tgz
download_fd_model_zxvf PP_HumanSegV2_Lite_192x192_with_argmax_infer.tgz
download_fd_model_zxvf Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer.tgz
download_fd_model_zxvf Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer.tgz
download_fd_model_zxvf SegFormer_B0-cityscapes-with-argmax.tgz
download_fd_model_xvf PP-Matting-512.tgz
download_fd_model_xvf PPHumanMatting.tgz
download_fd_model_xvf PPModnet_MobileNetV2.tgz
download_fd_model_xvf Unet_cityscapes_with_argmax_infer.tgz
download_fd_model_xvf PP_HumanSegV1_Server_with_argmax_infer.tgz
download_fd_model_xvf FCN_HRNet_W18_cityscapes_with_argmax_infer.tgz
download_fd_model_xvf PP_HumanSegV2_Mobile_192x192_with_argmax_infer.tgz


# PaddleOCR
download_common_model_xvf https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar ch_PP-OCRv3_det_infer.tar
download_common_model_xvf https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar ch_PP-OCRv3_rec_infer.tar
download_common_model_xvf https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar ch_ppocr_mobile_v2.0_cls_infer.tar
download_common_model_xvf https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar ch_PP-OCRv2_det_infer.tar
download_common_model_xvf https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar ch_PP-OCRv2_rec_infer.tar
# download images
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/rec_img.jpg rec_img.jpg
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/cityscapes_demo.png cityscapes_demo.png
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/portrait_heng.jpg portrait_heng.jpg
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg matting_input.jpg
download_common_file https://github.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg 12.jpg
download_common_file https://github.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg ILSVRC2012_val_00000010.jpeg
download_common_file https://github.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg 000000014439.jpg
download_common_file https://github.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt ppocr_keys_v1.txt
