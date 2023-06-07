#!/bin/bash
set +e
set +x

download_fd_model_zxvf() {
  local model="$1"  # xxx_model.tgz
  local len=${#model}
  local model_dir=${model:0:${#model}-4}  # xxx_model
  if [ -d "${model_dir}" ]; then
     echo "[INFO] --- $model_dir already exists!"
     return
  fi
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget https://bj.bcebos.com/paddlehub/fastdeploy/$model && tar -zxvf $model
     # remove tar crash
     rm $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -zxvf $model
        rm $(ls ./${model_dir}/._*)
     else
        echo "[INFO] --- $model_dir already exists!"
     fi
  fi
}
download_fd_model_xvf() {
  local model="$1"
  local model_dir=${model:0:${#model}-4}  # xxx_model
  if [ -d "${model_dir}" ]; then
     echo "[INFO] --- $model_dir already exists!"
     return
  fi
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget https://bj.bcebos.com/paddlehub/fastdeploy/$model && tar -xvf $model
     rm $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -xvf $model
        rm $(ls ./${model_dir}/._*)
     else
        echo "[INFO] --- $model_dir already exists!"
     fi
  fi
}
download_common_model_zxvf() {
  local model_url="$1"
  local model="$2"
  local model_dir=${model:0:${#model}-4}  # xxx_model
  if [ -d "${model_dir}" ]; then
     echo "[INFO] --- $model_dir already exists!"
     return
  fi
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget ${model_url} && tar -zxvf $model
     rm $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -zxvf $model
        rm $(ls ./${model_dir}/._*)
     else
        echo "[INFO] --- $model_dir already exists!"
     fi
  fi
}
download_common_model_xvf() {
  local model_url="$1"
  local model="$2"
  local model_dir=${model:0:${#model}-4}  # xxx_model
  if [ -d "${model_dir}" ]; then
     echo "[INFO] --- $model_dir already exists!"
     return
  fi
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget ${model_url} && tar -xvf $model
     rm $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -xvf $model
        rm $(ls ./${model_dir}/._*)
     else
        echo "[INFO] --- $model_dir already exists!"
     fi
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
download_fd_model_zxvf ppyoloe_crn_l_300e_coco.tgz
download_fd_model_zxvf picodet_l_640_coco_lcnet.tgz
download_fd_model_zxvf ppyoloe_plus_crn_m_80e_coco.tgz
download_fd_model_zxvf yolox_s_300e_coco.tgz
download_fd_model_zxvf yolov5_s_300e_coco.tgz
download_fd_model_zxvf yolov6_s_300e_coco.tgz
download_fd_model_zxvf yolov7_l_300e_coco.tgz
download_fd_model_zxvf yolov8_s_500e_coco.tgz

download_fd_model_zxvf ppyolo_r50vd_dcn_1x_coco.tgz
download_fd_model_zxvf ppyolov2_r101vd_dcn_365e_coco.tgz
download_fd_model_zxvf yolov3_darknet53_270e_coco.tgz
download_fd_model_zxvf faster_rcnn_r50_vd_fpn_2x_coco.tgz
download_fd_model_zxvf mask_rcnn_r50_1x_coco.tgz
download_fd_model_zxvf ssd_mobilenet_v1_300_120e_voc.tgz
download_fd_model_zxvf ssd_vgg16_300_240e_voc.tgz
download_fd_model_zxvf ssdlite_mobilenet_v1_300_coco.tgz
download_fd_model_zxvf rtmdet_l_300e_coco.tgz
download_fd_model_zxvf rtmdet_s_300e_coco.tgz
download_fd_model_zxvf yolov5_l_300e_coco.tgz
download_fd_model_zxvf yolov6_l_300e_coco.tgz
download_fd_model_zxvf yolov6_s_400e_coco.tgz
download_fd_model_zxvf cascade_rcnn_r50_fpn_1x_coco.tgz
download_fd_model_zxvf cascade_rcnn_r50_vd_fpn_ssld_2x_coco.tgz
download_fd_model_zxvf faster_rcnn_enhance_3x_coco.tgz
download_fd_model_zxvf fcos_r50_fpn_1x_coco.tgz
download_fd_model_zxvf gfl_r50_fpn_1x_coco.tgz
download_fd_model_zxvf ppyoloe_crn_l_80e_sliced_visdrone_640_025.tgz
download_fd_model_zxvf retinanet_r101_fpn_2x_coco.tgz
download_fd_model_zxvf retinanet_r50_fpn_1x_coco.tgz
download_fd_model_zxvf tood_r50_fpn_1x_coco.tgz
download_fd_model_zxvf ttfnet_darknet53_1x_coco.tgz
download_fd_model_zxvf yolov8_x_500e_coco.tgz
download_fd_model_zxvf yolov8_l_500e_coco.tgz
download_fd_model_zxvf yolov8_m_500e_coco.tgz
download_fd_model_zxvf yolov8_n_500e_coco.tgz
download_fd_model_zxvf picodet_l_320_coco_lcnet.tgz
download_fd_model_zxvf yolov7_x_300e_coco.tgz

download_fd_model_zxvf ppyoloe_crn_l_300e_coco_trt_nms.tgz
download_fd_model_zxvf picodet_l_640_coco_lcnet_trt_nms.tgz
download_fd_model_zxvf ppyoloe_plus_crn_m_80e_coco_trt_nms.tgz
download_fd_model_zxvf yolox_s_300e_coco_trt_nms.tgz
download_fd_model_zxvf yolov5_s_300e_coco_trt_nms.tgz
download_fd_model_zxvf yolov6_s_300e_coco_trt_nms.tgz
download_fd_model_zxvf yolov7_l_300e_coco_trt_nms.tgz
download_fd_model_zxvf yolov8_s_500e_coco_trt_nms.tgz

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
download_common_model_xvf https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar en_ppstructure_mobile_v2.0_SLANet_infer.tar

# download images
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/rec_img.jpg rec_img.jpg
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/cityscapes_demo.png cityscapes_demo.png
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/portrait_heng.jpg portrait_heng.jpg
download_common_file https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg matting_input.jpg
download_common_file https://github.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg 12.jpg
download_common_file https://github.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg ILSVRC2012_val_00000010.jpeg
download_common_file https://github.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg 000000014439.jpg
download_common_file https://github.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt ppocr_keys_v1.txt
