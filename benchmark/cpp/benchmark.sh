# Run all models specify hardware and specify backend
./benchmark_ppseg --model PP_HumanSegV2_Lite_192x192_with_argmax_infer --image human.jpg --config_path config.txt
./benchmark_ppseg --model PP_HumanSegV2_Mobile_192x192_with_argmax_infer --image human.jpg --config_path config.txt
./benchmark_ppcls --model MobileNetV2_ssld_infer --image ILSVRC2012_val_00000010.jpeg --config_path config.txt
./benchmark_ppocr_det --model ch_PP-OCRv3/ch_PP-OCRv3_det_infer --image 12.jpg --config_path config.txt
./benchmark_ppocr_cls --model ch_PP-OCRv3/ch_ppocr_mobile_v2.0_cls_infer --image rec_img.jpg --config_path config.txt
./benchmark_ppocr_rec --model ch_PP-OCRv3/ch_PP-OCRv3_rec_infer --image rec_img.jpg --rec_label_file ppocr_keys_v1.txt --config_path config.txt
./benchmark_precision_ppyolov8 --model yolov8_s_500e_coco --image 000000014439.jpg --config_path config.txt
./benchmark_yolov5 --model yolov5s.onnx --image 000000014439.jpg --config_path config.txt
