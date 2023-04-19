# Run all models specify hardware and specify backend

CONFIG_PATH="config.gpu.paddle_trt.fp32.txt"
if [ ! "$1" = "$CONFIG_PATH" ]; then
  if [ -f "$1" ]; then
    CONFIG_PATH="$1"
  fi
fi

# PaddleClas
./benchmark_ppcls --model PPLCNet_x1_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model PPLCNetV2_base_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model EfficientNetB7_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,600,600:1,3,600,600:1,3,600,600" --input_name "x"
./benchmark_ppcls --model EfficientNetB0_small_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model GhostNet_x0_5_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model GhostNet_x1_3_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model GhostNet_x1_3_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model MobileNetV1_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model MobileNetV2_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model MobileNetV2_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model MobileNetV3_small_x0_35_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model MobileNetV3_large_x1_0_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model ShuffleNetV2_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model ShuffleNetV2_x2_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model SqueezeNet1_1_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model InceptionV3_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,299,299:1,3,299,299:1,3,299,299" --input_name "x"
./benchmark_ppcls --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model PPHGNet_tiny_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model PPHGNet_base_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model ResNet50_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model EfficientNetB0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model MobileNetV2_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model MobileNetV3_small_x1_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model ViT_large_patch16_224_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model ResNeXt50_32x4d_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model DenseNet121_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "inputs"
./benchmark_ppcls --model PPHGNet_small_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"
./benchmark_ppcls --model person_exists_infer --image ILSVRC2012_val_00000010.jpeg --config_path $CONFIG_PATH --trt_shape "1,3,224,224:1,3,224,224:1,3,224,224" --input_name "x"

# PaddleOCR
./benchmark_ppocr_det --model ch_PP-OCRv3_det_infer --image 12.jpg --config_path $CONFIG_PATH --trt_shape 1,3,960,608:1,3,960,608:1,3,960,608
./benchmark_ppocr_cls --model ch_ppocr_mobile_v2.0_cls_infer --image rec_img.jpg --config_path $CONFIG_PATH --trt_shape 1,3,48,192:1,3,48,192:1,3,48,192
./benchmark_ppocr_rec --model ch_PP-OCRv3_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path $CONFIG_PATH --trt_shape 1,3,48,572:1,3,48,572:1,3,48,572
./benchmark_ppocr_det --model ch_PP-OCRv2_det_infer --image 12.jpg --config_path $CONFIG_PATH --trt_shape 1,3,960,608:1,3,960,608:1,3,960,608
./benchmark_ppocr_rec --model ch_PP-OCRv2_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path $CONFIG_PATH --trt_shape 1,3,32,10:1,3,32,320:1,3,32,2304


# PaddleDetection
./benchmark_ppyolov5 --model yolov5_s_300e_coco_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov6 --model yolov6_s_300e_coco_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_s_500e_coco_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolox --model yolox_s_300e_coco_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyoloe --model ppyoloe_plus_crn_m_80e_coco_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_picodet --model picodet_l_640_coco_lcnet_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov7 --model yolov7_l_300e_coco_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyoloe --model ppyoloe_crn_l_300e_coco_trt_nms --image 000000014439.jpg --config_path $CONFIG_PATH

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
./benchmark_yolov3 --model yolov3_darknet53_270e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_picodet --model picodet_l_320_coco_lcnet --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_rtmdet --model rtmdet_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_rtmdet --model rtmdet_s_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov5 --model yolov5_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov6 --model yolov6_l_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov6 --model yolov6_s_400e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov7 --model yolov7_x_300e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyoloe --model ppyoloe_crn_l_80e_sliced_visdrone_640_025 --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_x_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_l_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_m_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH
./benchmark_ppyolov8 --model yolov8_n_500e_coco --image 000000014439.jpg --config_path $CONFIG_PATH

# PaddleSeg
./benchmark_ppseg --model Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH --trt_shape 1,3,144,256:1,3,144,256:1,3,144,256
./benchmark_ppseg --model PP_HumanSegV2_Lite_192x192_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH --trt_shape 1,3,192,192:1,3,192,192:1,3,192,192
./benchmark_ppseg --model PP_HumanSegV1_Lite_infer --image portrait_heng.jpg --config_path $CONFIG_PATH --trt_shape 1,3,192,192:1,3,192,192:1,3,192,192
./benchmark_ppseg --model PP_HumanSegV2_Mobile_192x192_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH --trt_shape 1,3,192,192:1,3,192,192:1,3,192,192
./benchmark_ppseg --model Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppseg --model PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppseg --model SegFormer_B0-cityscapes-with-argmax --image cityscapes_demo.png --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppmatting --model PP-Matting-512 --image matting_input.jpg --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppmatting --model PPHumanMatting --image matting_input.jpg --config_path $CONFIG_PATH --trt_shape 1,3,2048,2048:1,3,2048,2048:1,3,2048,2048
./benchmark_ppmatting --model PPModnet_MobileNetV2 --image matting_input.jpg --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppseg --model Unet_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppseg --model PP_HumanSegV1_Server_with_argmax_infer --image portrait_heng.jpg --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppseg --model FCN_HRNet_W18_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path $CONFIG_PATH --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
