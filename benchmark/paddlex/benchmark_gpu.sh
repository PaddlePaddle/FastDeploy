# Run all models specify hardware and specify backend

CONFIG_PATH="config.gpu.paddle.fp32.txt"
if [ ! "$1" = "$CONFIG_PATH" ]; then
  if [ -f "$1" ]; then
    CONFIG_PATH="$1"
  fi
fi

# PaddleClas
./benchmark_ppcls --model MobileNetV3_small_x1_0 --image ppcls_cls_demo.JPEG --config_path $CONFIG_PATH
./benchmark_ppcls --model PP-HGNet_small --image ppcls_cls_demo.JPEG --config_path $CONFIG_PATH
./benchmark_ppcls --model PP-LCNet_x1_0 --image ppcls_cls_demo.JPEG --config_path $CONFIG_PATH
./benchmark_ppcls --model SwinTransformer-Base --image ppcls_cls_demo.JPEG --config_path $CONFIG_PATH
./benchmark_ppcls --model ResNet50 --image ppcls_cls_demo.JPEG --config_path $CONFIG_PATH

# PaddleDetection
./benchmark_ppdet --model PP-YOLOE+_crn_l_80e --image ppdet_det_img.jpg --config_path $CONFIG_PATH
./benchmark_ppdet --model RT-DETR_R50vd_6x --image ppdet_det_img.jpg --config_path $CONFIG_PATH
./benchmark_ppdet --model PP-PicoDet_s_320_lcnet --image ppdet_det_img.jpg --config_path $CONFIG_PATH

# PaddleSeg
./benchmark_ppseg --model OCRNet_HRNetW48 --image ppseg_cityscapes_demo.png --config_path $CONFIG_PATH
./benchmark_ppseg --model PP-LiteSeg-STDC1 --image ppseg_cityscapes_demo.png --config_path $CONFIG_PATH
./benchmark_ppseg --model PP-MobileSeg-Base --image ppseg_ade_val_512x512.png --config_path $CONFIG_PATH
./benchmark_ppseg --model SegFormer-B0 --image ppseg_cityscapes_demo.png --config_path $CONFIG_PATH

# PaddleOCR
./benchmark_ppocr_rec --model PP-OCRv4-mobile-rec --image ppocrv4_word_1.jpg --rec_label_file ppocr_keys_v1.txt --config_path $CONFIG_PATH
./benchmark_ppocr_rec --model PP-OCRv4-server-rec --image ppocrv4_word_1.jpg --rec_label_file ppocr_keys_v1.txt --config_path $CONFIG_PATH

# PP-ShiTuV2
./benchmark_ppshituv2_rec --model PP-ShiTuv2-rec --image ppshituv2_wangzai.png --config_path $CONFIG_PATH

# PP-StructureV2
./benchmark_structurev2_layout --model PP-Structurev2-layout --image structurev2_layout_val_0002.jpg --config_path $CONFIG_PATH
./benchmark_structurev2_table --model PP-Structurev2-SLANet --image structurev2_table.jpg --table_char_dict_path table_structure_dict_ch.txt --config_path $CONFIG_PATH
