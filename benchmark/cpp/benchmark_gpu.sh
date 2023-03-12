# Run all models specify hardware and specify backend

# PaddleDetection
./benchmark_ppyolov5 --model yolov5_s_300e_coco_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms
./benchmark_ppyolov6 --model yolov6_s_300e_coco_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms
./benchmark_ppyolov7 --model yolov7_l_300e_coco_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms
./benchmark_ppyolov8 --model yolov8_s_500e_coco_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms
./benchmark_ppyolox --model yolox_s_300e_coco_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms
./benchmark_ppyoloe --model ppyoloe_crn_l_300e_coco_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms
./benchmark_ppyoloe --model ppyoloe_plus_crn_m_80e_coco_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms
./benchmark_picodet --model picodet_l_640_coco_lcnet_no_nms --image 000000014439.jpg --config_path config.gpu.txt --no_nms

# PaddleSeg
./benchmark_ppseg --model Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer --image portrait_heng.jpg --config_path config.gpu.txt
# use trt
# ./benchmark_ppseg --model Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer --image portrait_heng.jpg --config_path config.gpu.txt --trt_shape 1,3,144,256:1,3,144,256:1,3,144,256
./benchmark_ppseg --model PP_HumanSegV2_Lite_192x192_with_argmax_infer --image portrait_heng.jpg --config_path config.gpu.txt
# use trt
# ./benchmark_ppseg --model PP_HumanSegV2_Lite_192x192_with_argmax_infer --image portrait_heng.jpg --config_path config.gpu.txt --trt_shape 1,3,192,192:1,3,192,192:1,3,192,192
./benchmark_ppseg --model PP_HumanSegV1_Lite_infer --image portrait_heng.jpg --config_path config.gpu.txt
# use trt
# ./benchmark_ppseg --model PP_HumanSegV1_Lite_infer --image portrait_heng.jpg --config_path config.gpu.txt --trt_shape 1,3,192,192:1,3,192,192:1,3,192,192
./benchmark_ppseg --model Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path config.gpu.txt
# use trt
# ./benchmark_ppseg --model Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path config.gpu.txt --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppseg --model PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path config.gpu.txt
# use trt
# ./benchmark_ppseg --model PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer --image cityscapes_demo.png --config_path config.gpu.txt --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppseg --model SegFormer_B0-cityscapes-with-argmax --image cityscapes_demo.png --config_path config.gpu.txt
# use trt
# ./benchmark_ppseg --model SegFormer_B0-cityscapes-with-argmax --image cityscapes_demo.png --config_path config.gpu.txt --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppmatting --model PP-Matting-512 --image matting_input.jpg --config_path config.gpu.txt
# use trt
#./benchmark_ppmatting --model PP-Matting-512 --image matting_input.jpg --config_path config.gpu.txt --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppmatting --model PPHumanMatting --image matting_input.jpg --config_path config.gpu.txt
# use trt
#./benchmark_ppmatting --model PPHumanMatting --image matting_input.jpg --config_path config.gpu.txt --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512
./benchmark_ppmatting --model PPModnet_MobileNetV2 --image matting_input.jpg --config_path config.gpu.txt
# use trt
#./benchmark_ppmatting --model PPModnet_MobileNetV2 --image matting_input.jpg --config_path config.gpu.txt --trt_shape 1,3,512,512:1,3,512,512:1,3,512,512

# PaddleClas
./benchmark_ppcls --model PPLCNet_x1_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model PPLCNetV2_base_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model MobileNetV1_x0_25_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model MobileNetV1_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model MobileNetV3_large_x1_0_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model ShuffleNetV2_x2_0_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model EfficientNetB0_small_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt
./benchmark_ppcls --model PPHGNet_tiny_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.gpu.txt

# PaddleOCR
./benchmark_ppocr_det --model ch_PP-OCRv3_det_infer --image 12.jpg --config_path config.gpu.txt
# use trt
# ./benchmark_ppocr_det --model ch_PP-OCRv3_det_infer --image 12.jpg --config_path config.gpu.txt --trt_shape 1,3,64,64:1,3,640,640:1,3,960,960
./benchmark_ppocr_cls --model ch_ppocr_mobile_v2.0_cls_infer --image rec_img.jpg --config_path config.gpu.txt
# use trt
# ./benchmark_ppocr_cls --model ch_ppocr_mobile_v2.0_cls_infer --image rec_img.jpg --config_path config.gpu.txt --trt_shape 1,3,48,10:4,3,48,320:8,3,48,1024
./benchmark_ppocr_rec --model ch_PP-OCRv3_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path config.gpu.txt
# use trt
# ./benchmark_ppocr_rec --model ch_PP-OCRv3_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path config.gpu.txt --trt_shape 1,3,48,10:4,3,48,320:8,3,48,2304
./benchmark_ppocr_det --model ch_PP-OCRv2_det_infer --image 12.jpg --config_path config.gpu.txt
# use trt
# ./benchmark_ppocr_det --model ch_PP-OCRv2_det_infer --image 12.jpg --config_path config.gpu.txt --trt_shape 1,3,64,64:1,3,640,640:1,3,960,960
./benchmark_ppocr_rec --model ch_PP-OCRv2_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path config.gpu.txt
# use trt
# ./benchmark_ppocr_rec --model ch_PP-OCRv2_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path config.gpu.txt --trt_shape 1,3,32,10:4,3,32,320:8,3,32,2304
