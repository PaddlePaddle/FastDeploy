export LD_LIBRARY_PATH=${PWD}/lib

./test_clas models/mobilenetv1_ssld_ptq images/ILSVRC2012_val_00000010.jpeg results/mobilenetv1_cls.txt
./test_clas models/resnet50_vd_ptq/ images/ILSVRC2012_val_00000010.jpeg results/resnet50_cls.txt
./test_yolov5 models/yolov5s_ptq_model/ images/000000014439.jpg results/yolov5_result.txt
./test_ppyoloe models/ppyoloe_noshare_qat/ images/000000014439.jpg results/ppyoloe_result.txt
./test_ppliteseg models/ppliteseg images/cityscapes_demo.png results/ppliteseg_result.txt
