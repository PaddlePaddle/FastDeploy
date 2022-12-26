mkdir models
cd models

wget https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_with_argmax_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz

wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz

wget https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV1_Lite_with_argmax_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Lite_infer.tgz

wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Lite_192x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz

wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_with_argmax_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV1_Server_infer.tgz

wget https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_with_argmax_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/Portrait_PP_HumanSegV2_Lite_256x144_infer.tgz

wget https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_with_argmax_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz

wget https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_with_argmax_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz


ls *.tgz | xargs -n1 tar xzvf
rm -rf *.tgz
