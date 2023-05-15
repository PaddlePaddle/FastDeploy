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
     rm -f $(ls ${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -zxvf $model
        rm -f $(ls ./${model_dir}/._*)
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
     rm -f $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -xvf $model
        rm -f $(ls ./${model_dir}/._*)
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
     rm -f $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -zxvf $model
        rm -f $(ls ./${model_dir}/._*)
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
     rm -f $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -xvf $model
        rm -f $(ls ./${model_dir}/._*)
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

# Convert model: paddle -> onnx -> mnn/tnn/ncnn
CONVERT_LOG=convert.$(date "+%Y.%m.%d.%H.%M.%S").log
CONVERT_FLAG=$1
CONVERT_MODE=$2

dump_convert_log() {
   local info="$1"
   echo "$info" >> ${CONVERT_LOG}
   echo "$info"
}
check_or_delete() {
   if [ "$CONVERT_MODE" = "delete" ]; then
      if [ -f "$1" ]; then
         rm -f $1
         echo "[WARN][DELETE] $1 DELETED!"
         dump_convert_log "[WARN][DELETE] $1 DELETED!"
      fi
   fi
}
paddle2onnx_cmd() {
   local model_dir=$1
   local model_file=$2
   local param_file=$3
   local onnx_file=${model_file:0:${#model_file}-8}
   if [ ! -d "${model_dir}" ]; then
      echo "[ERROR] Can not found model dir: ${model_dir}, skip!"
      return 0
   fi
   if [ -f "$model_dir/$onnx_file.onnx" ]; then
      echo "[INFO] --- $model_dir/$onnx_file.onnx already exists!"
      dump_convert_log "[INFO][Paddle2ONNX][$model_dir][$onnx_file.onnx] Found!"
      check_or_delete $model_dir/$onnx_file.onnx
   else
      if [ "$CONVERT_MODE" = "delete" ]; then
         return
      fi
      echo "[INFO][$model_dir] --- running paddle2onnx_cmd ... "
      local ret=$(paddle2onnx --model_dir $model_dir --model_filename $model_file --params_filename $param_file --save_file $model_dir/$onnx_file.onnx) && echo $ret
      local check=$(echo $(echo $ret | grep exported | grep -v ERROR | wc -l))
      if [ "$check" = "1" ]; then
         dump_convert_log "[INFO][Paddle2ONNX][$model_dir] Success!"
         onnxsim $model_dir/$onnx_file.onnx $model_dir/$onnx_file.onnx
      else
         dump_convert_log "[INFO][Paddle2ONNX][$model_dir] Failed!"
      fi
   fi
}
onnx2mnn_cmd() {
   local model_dir=$1
   local onnx_file=$2
   local mnn_file=${onnx_file:0:${#onnx_file}-5}
   if [ ! -d "${model_dir}" ]; then
      echo "[ERROR] Can not found model dir: ${model_dir}, skip!"
      return 0
   fi
   if [ -f "$model_dir/$mnn_file.mnn" ]; then
      echo "[INFO] --- $model_dir/$mnn_file.mnn already exists!"
      dump_convert_log "[INFO][ONNX2MNN][$model_dir][$mnn_file.mnn] Found!"
      check_or_delete $model_dir/$mnn_file.mnn
   else
      if [ "$CONVERT_MODE" = "delete" ]; then
         return
      fi
      echo "[INFO][$model_dir] --- running onnx2mnn_cmd ... "
      local ret=$(MNNConvert -f ONNX --modelFile $model_dir/$onnx_file --MNNModel $model_dir/$mnn_file.mnn --bizCode biz) && echo $ret
      local check=$(echo $(echo $ret | grep Success | wc -l))
      if [ "$check" = "1" ]; then
         dump_convert_log "[INFO][ONNX2MNN][$model_dir] Success!"
      else
         dump_convert_log "[INFO][ONNX2MNN][$model_dir] Failed!"
      fi
   fi
}
onnx2tnn_cmd() {
   local model_dir=$1
   local onnx_file=$2
   local tnn_file=${onnx_file:0:${#onnx_file}-5}
   if [ ! -d "${model_dir}" ]; then
      echo "[ERROR] Can not found model dir: ${model_dir}, skip!"
      return 0
   fi
   if [ -f "$model_dir/$tnn_file.opt.tnnmodel" ]; then
      echo "[INFO] --- $model_dir/$tnn_file.opt.tnnmodel already exists!"
      dump_convert_log "[INFO][ONNX2TNN][$model_dir][$tnn_file.opt.tnnmodel] Found!"
      check_or_delete $model_dir/$tnn_file.opt.tnnmodel
      check_or_delete $model_dir/$tnn_file.opt.tnnproto
      check_or_delete $model_dir/$tnn_file.opt.onnx
   else
      if [ "$CONVERT_MODE" = "delete" ]; then
         return
      fi
      echo "[INFO][$model_dir] --- running onnx2tnn_cmd ... "
      # ${@:3} may look like: -in image:1,3,640,640 scale_factor:1,2
      # TNNConvert onnx2tnn $model_dir/$onnx_file -v=v1.0 -o $model_dir
      TNNConvert onnx2tnn $model_dir/$onnx_file -optimize -v=v1.0 -o $model_dir ${@:3} > onnx2tnn.log 2>&1 && cat onnx2tnn.log
      local check=$(echo $(cat onnx2tnn.log | grep succeed | wc -l))
      rm onnx2tnn.log
      if [ "$check" = "1" ]; then
         dump_convert_log "[INFO][ONNX2TNN][$model_dir] Success!"
      else
         dump_convert_log "[INFO][ONNX2TNN][$model_dir] Failed!"
      fi
   fi
}
onnx2ncnn_cmd() {
   local model_dir=$1
   local onnx_file=$2
   local ncnn_file=${onnx_file:0:${#onnx_file}-5}
   if [ ! -d "${model_dir}" ]; then
      echo "[ERROR] Can not found model dir: ${model_dir}, skip!"
      return 0
   fi
   if [ -f "$model_dir/$ncnn_file.opt.param" ]; then
      echo "[INFO] --- $model_dir/$ncnn_file.opt.param already exists!"
      dump_convert_log "[INFO][ONNX2NCNN][$model_dir][$ncnn_file.opt.param] Found!"
      check_or_delete $model_dir/$ncnn_file.opt.param
      check_or_delete $model_dir/$ncnn_file.opt.bin
      check_or_delete $model_dir/$ncnn_file.param
      check_or_delete $model_dir/$ncnn_file.bin
   else
      if [ "$CONVERT_MODE" = "delete" ]; then
         return
      fi
      echo "[INFO][$model_dir] --- running onnx2ncnn_cmd ... "
      onnx2ncnn $model_dir/$onnx_file $model_dir/$ncnn_file.param $model_dir/$ncnn_file.bin > onnx2ncnn.log 2>&1 && cat onnx2ncnn.log
      local check=$(echo $(cat onnx2ncnn.log | wc -l))
      rm onnx2ncnn.log
      if [ "$check" = "0" ]; then
         dump_convert_log "[INFO][ONNX2NCNN][$model_dir] Success!"
         ncnnoptimize $model_dir/$ncnn_file.param $model_dir/$ncnn_file.bin $model_dir/$ncnn_file.opt.param $model_dir/$ncnn_file.opt.bin 0
      else
         dump_convert_log "[INFO][ONNX2NCNN][$model_dir] Failed!"
         # remove cache
         rm -f $model_dir/$ncnn_file.bin
         rm -f $model_dir/$ncnn_file.param
      fi
   fi
}
convert_fd_model() {
   local model_dir=$1
   local model_file=$(cd $model_dir && ls *.pdmodel && cd ..)
   local param_file=$(cd $model_dir && ls *.pdiparams && cd ..)
   local onnx_file=${model_file:0:${#model_file}-8}.onnx
   echo "[INFO] --- Processing $model_file, $param_file ..."
   paddle2onnx_cmd $model_dir $model_file $param_file
   onnx2mnn_cmd $model_dir $onnx_file
   onnx2tnn_cmd $model_dir $onnx_file ${@:2}
   onnx2ncnn_cmd $model_dir $onnx_file
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

# covert models -> onnx/mnn/tnn/ncnn
if [ "$CONVERT_FLAG" = "convert" ]; then
   convert_fd_model ppyoloe_crn_l_300e_coco_no_nms -in image:1,3,640,640 scale_factor:1,2
   convert_fd_model picodet_l_640_coco_lcnet_no_nms -in image:1,3,640,640
   convert_fd_model ppyoloe_plus_crn_m_80e_coco_no_nms -in image:1,3,640,640 scale_factor:1,2
   convert_fd_model yolox_s_300e_coco_no_nms -in image:1,3,640,640 scale_factor:1,2
   convert_fd_model yolov5_s_300e_coco_no_nms -in image:1,3,640,640 scale_factor:1,2
   convert_fd_model yolov6_s_300e_coco_no_nms -in image:1,3,640,640 scale_factor:1,2
   convert_fd_model yolov7_l_300e_coco_no_nms -in image:1,3,640,640 scale_factor:1,2
   convert_fd_model yolov8_s_500e_coco_no_nms -in image:1,3,640,640 scale_factor:1,2

   convert_fd_model PPLCNet_x1_0_infer -in 1,3,224,224
   convert_fd_model PPLCNetV2_base_infer -in 1,3,224,224
   convert_fd_model MobileNetV1_x0_25_infer -in 1,3,224,224
   convert_fd_model MobileNetV1_ssld_infer -in 1,3,224,224
   convert_fd_model MobileNetV3_large_x1_0_ssld_infer -in 1,3,224,224
   convert_fd_model ShuffleNetV2_x2_0_infer -in 1,3,224,224
   convert_fd_model ResNet50_vd_infer -in 1,3,224,224
   convert_fd_model EfficientNetB0_small_infer -in 1,3,224,224
   convert_fd_model PPHGNet_tiny_ssld_infer -in 1,3,224,224

   convert_fd_model PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer -in 1,3,512,512
   convert_fd_model PP_HumanSegV1_Lite_infer -in 1,3,192,192
   convert_fd_model PP_HumanSegV2_Lite_192x192_with_argmax_infer -in 1,3,192,192
   convert_fd_model Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer -in 1,3,144,256
   convert_fd_model Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer -in 1,3,512,512
   convert_fd_model SegFormer_B0-cityscapes-with-argmax -in 1,3,512,512
   convert_fd_model PPHumanMatting
   convert_fd_model PP-Matting-512
   convert_fd_model PPModnet_MobileNetV2 -in 1,3,512,512

   convert_fd_model ch_PP-OCRv3_det_infer -in x:1,3,960,608
   convert_fd_model ch_PP-OCRv3_rec_infer -in x:1,3,48,572
   convert_fd_model ch_ppocr_mobile_v2.0_cls_infer -in x:1,3,48,572
   convert_fd_model ch_PP-OCRv2_det_infer -in x:1,3,960,608
   convert_fd_model ch_PP-OCRv2_rec_infer -in x:1,3,48,572

   echo "-----------------------------------------Convert Status-----------------------------------------"
   cat ${CONVERT_LOG}
   echo "------------------------------------------------------------------------------------------------"
   echo "Saved -> ${CONVERT_LOG}"
fi

# ./get_models.sh
# ./get_models.sh convert # convert models -> onnx/mnn/tnn/ncnn
# ./get_models.sh convert delete # delete converted models
