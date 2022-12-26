mkdir models
cd models

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

wget https://bj.bcebos.com/paddlehub/fastdeploy/PPOCRv3_ICDAR10_BS116_1221.txt
wget https://bj.bcebos.com/paddlehub/fastdeploy/PPOCRv2_ICDAR10_BS116_1221.txt

ls *.tar | xargs -n1 tar xzvf
rm -rf *.tar

cd ..
