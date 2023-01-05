TARGET_DEVICE=ascend

python eval_ppocrv3.py.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt \
                 --image ../ICDAR2017_10 --device $TARGET_DEVICE 2>&1 | tee ./log/ppocrv3_diff.log

python eval_ppocrv2.py.py  --det_model ch_PP-OCRv2_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv2_rec_infer --rec_label_file ppocr_keys_v1.txt \
                 --image ../ICDAR2017_10 --device $TARGET_DEVICE 2>&1 | tee ./log/ppocrv2_diff.log
