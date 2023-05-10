# Run all models specify hardware and specify backend

CONFIG_PATH="config.gpu.paddle.fp32.txt"
if [ ! "$1" = "$CONFIG_PATH" ]; then
  if [ -f "$1" ]; then
    CONFIG_PATH="$1"
  fi
fi

# PaddleClas
./benchmark_ppcls --model PPLCNet_x1_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model PPLCNetV2_base_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model EfficientNetB7_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model EfficientNetB0_small_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model GhostNet_x0_5_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model GhostNet_x1_3_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model GhostNet_x1_3_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV1_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV2_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV2_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV3_small_x0_35_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV3_large_x1_0_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model ShuffleNetV2_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model ShuffleNetV2_x2_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model SqueezeNet1_1_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model InceptionV3_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model PPHGNet_tiny_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model PPHGNet_base_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model ResNet50_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model EfficientNetB0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV2_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model MobileNetV3_small_x1_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model ViT_large_patch16_224_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model ResNeXt50_32x4d_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model DenseNet121_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model PPHGNet_small_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH
./benchmark_ppcls --model person_exists_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH

# PaddleOCR
./benchmark_ppocr_det --model ch_PP-OCRv3_det_infer --image 12.jpg --config_path $CONFIG_PATH
./benchmark_ppocr_cls --model ch_ppocr_mobile_v2.0_cls_infer --image rec_img.jpg --config_path $CONFIG_PATH
./benchmark_ppocr_rec --model ch_PP-OCRv3_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path $CONFIG_PATH
./benchmark_ppocr_det --model ch_PP-OCRv2_det_infer --image 12.jpg --config_path $CONFIG_PATH
./benchmark_ppocr_rec --model ch_PP-OCRv2_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path $CONFIG_PATH
./benchmark_ppocr_table --model en_ppstructure_mobile_v2.0_SLANet_infer --image table.jpg --table_char_dict_path table_structure_dict.txt --config_path $CONFIG_PATH

# PaddleDetection
./benchmark_ppyolov5 --model yolov5_s_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov6 --model yolov6_s_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_s_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolox --model yolox_s_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyoloe --model ppyoloe_plus_crn_m_80e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_picodet --model picodet_l_640_coco_lcnet --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov7 --model yolov7_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyoloe --model ppyoloe_crn_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH

./benchmark_ppyolov5 --model yolov5_s_300e_coco_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms
./benchmark_ppyolov6 --model yolov6_s_300e_coco_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms
./benchmark_ppyolov7 --model yolov7_l_300e_coco_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms
./benchmark_ppyolov8 --model yolov8_s_500e_coco_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms
./benchmark_ppyolox --model yolox_s_300e_coco_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms
./benchmark_ppyoloe --model ppyoloe_crn_l_300e_coco_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms
./benchmark_ppyoloe --model ppyoloe_plus_crn_m_80e_coco_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms
./benchmark_picodet --model picodet_l_640_coco_lcnet_no_nms --image 000000014439.jpg --config_path $CONFIG_PATH --no_nms

./benchmark_ppyolo --model ppyolo_r50vd_dcn_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_yolov3 --model yolov3_darknet53_270e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolo --model ppyolov2_r101vd_dcn_365e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_picodet --model picodet_l_320_coco_lcnet --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_fasterrcnn --model faster_rcnn_r50_vd_fpn_2x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_maskrcnn --model mask_rcnn_r50_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_rtmdet --model rtmdet_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_rtmdet --model rtmdet_s_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_cascadercnn --model cascade_rcnn_r50_fpn_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_cascadercnn --model cascade_rcnn_r50_vd_fpn_ssld_2x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_fcos --model fcos_r50_fpn_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_gfl --model gfl_r50_fpn_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_retinanet --model retinanet_r101_fpn_2x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_retinanet --model retinanet_r50_fpn_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_tood --model tood_r50_fpn_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ttfnet --model ttfnet_darknet53_1x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov5 --model yolov5_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov6 --model yolov6_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov6 --model yolov6_s_400e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov7 --model yolov7_x_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_fasterrcnn --model faster_rcnn_enhance_3x_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyoloe --model ppyoloe_crn_l_80e_sliced_visdrone_640_025 --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ssd --model ssd_mobilenet_v1_300_120e_voc --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ssd --model ssd_vgg16_300_240e_voc --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ssd --model ssdlite_mobilenet_v1_300_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_x_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_l_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_m_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_n_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH

# PaddleSeg
./benchmark_ppseg --model Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH
./benchmark_ppseg --model PP_HumanSegV2_Lite_192x192_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH
./benchmark_ppseg --model PP_HumanSegV1_Lite_infer --image portrait_heng.jpg --config_path $CONFIG_PATH
./benchmark_ppseg --model PP_HumanSegV2_Mobile_192x192_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH
./benchmark_ppseg --model Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH
./benchmark_ppseg --model PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH
./benchmark_ppseg --model SegFormer_B0-cityscapes-with-argmax --image cityscapes_demo.png --config_path $CONFIG_PATH
./benchmark_ppmatting --model PP-Matting-512 --image matting_input.jpg --config_path $CONFIG_PATH
./benchmark_ppmatting --model PPHumanMatting --image matting_input.jpg --config_path $CONFIG_PATH
./benchmark_ppmatting --model PPModnet_MobileNetV2 --image matting_input.jpg --config_path $CONFIG_PATH
./benchmark_ppseg --model Unet_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH
./benchmark_ppseg --model PP_HumanSegV1_Server_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH
./benchmark_ppseg --model FCN_HRNet_W18_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH
