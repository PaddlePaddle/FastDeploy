mkdir models
cd models

wget https://bj.bcebos.com/paddlehub/fastdeploy/picodet_l_320_coco_lcnet.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://bj.bcebos.com/fastdeploy/models/ppyoloe_plus_crn_m_80e_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyolo_r50vd_dcn_1x_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyolov2_r101vd_dcn_365e_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov3_darknet53_270e_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s_300e_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ssd_mobilenet_v1_300_120e_voc.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ssd_vgg16_300_240e_voc.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ssdlite_mobilenet_v1_300_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_r50_vd_fpn_2x_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/mask_rcnn_r50_1x_coco.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_infer.tar
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s_infer.tar
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_infer.tar

ls *.tgz | xargs -n1 tar xzvf
ls *.tar | xargs -n1 tar xzvf
rm -rf *.tgz
rm -rf *.tar

cd ..
