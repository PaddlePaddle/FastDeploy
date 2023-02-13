TARGET_DEVICE=ascend

python eval_picodet.py --model_dir ./models/picodet_l_320_coco_lcnet --image None --device $TARGET_DEVICE 2>&1 | tee ./log/picodet_l_320_coco_lcnet.log
python eval_ppyolo.py  --model_dir ./models/ppyolov2_r101vd_dcn_365e_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/ppyolov2_r101vd_dcn_365e_coco.log
python eval_ppyolo.py  --model_dir ./models/ppyolo_r50vd_dcn_1x_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/ppyolo_r50vd_dcn_1x_coco.log
python eval_ppyoloe.py  --model_dir ./models/ppyoloe_crn_l_300e_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/ppyoloe_crn_l_300e_coco.log
python eval_ppyoloe.py  --model_dir ./models/ppyoloe_plus_crn_m_80e_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/ppyoloe_plus_crn_m_80e_coco.log
python eval_ssd.py  --model_dir ./models/ssd_vgg16_300_240e_voc  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/ssd_vgg16_300_240e_voc.log
python eval_ssd.py  --model_dir ./models/ssdlite_mobilenet_v1_300_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/ssdlite_mobilenet_v1_300_coco.log
python eval_ssd.py  --model_dir ./models/ssd_mobilenet_v1_300_120e_voc  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/ssd_mobilenet_v1_300_120e_voc.log
python eval_yolov3.py  --model_dir ./models/yolov3_darknet53_270e_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/yolov3_darknet53_270e_coco.log
python eval_yolox.py --model_dir ./models/yolox_s_300e_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/yolox_s_300e_coco.log
python eval_faster_rcnn.py  --model_dir ./models/faster_rcnn_r50_vd_fpn_2x_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/faster_rcnn_r50_vd_fpn_2x_coco.log
python eval_mask_rcnn.py  --model_dir ./models/mask_rcnn_r50_1x_coco  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/mask_rcnn_r50_1x_coco.log
python eval_yolov5.py  --model ./models/yolov5s_infer --image None --device $TARGET_DEVICE 2>&1 | tee ./log/yolov5s_infer.log
python eval_yolov6.py  --model ./models/yolov6s_infer --image None --device $TARGET_DEVICE 2>&1 | tee ./log/yolov6s_infer.log
python eval_yolov7.py  --model ./models/yolov7_infer --image None --device $TARGET_DEVICE 2>&1 | tee ./log/yolov7_infer.log
