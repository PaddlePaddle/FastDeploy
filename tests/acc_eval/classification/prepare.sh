mkdir models
cd models

wget https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNet_x1_0_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNetV2_base_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB7_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB0_small_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x1_3_ssld_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x0_5_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_x0_25_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_ssld_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_x0_25_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_ssld_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_small_x0_35_ssld_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_large_x1_0_ssld_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x0_25_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x2_0_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/SqueezeNet1_1_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/InceptionV3_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_tiny_ssld_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_base_ssld_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz

ls *.tgz | xargs -n1 tar xzvf

rm -rf *.tgz

cd ..
