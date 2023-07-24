#!/bin/bash
set +e
set +x

download() {
  local model="$1"  # xxx_model.tgz
  local len=${#model}
  local model_dir=${model:0:${#model}-4}  # xxx_model
  if [ -d "${model_dir}" ]; then
     echo "[INFO] --- $model_dir already exists!"
     return
  fi
  if [ ! -f "${model}" ]; then
     echo "[INFO] --- downloading $model"
     wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/$model && tar -zxvf $model
     # remove tar crash
     rm $(ls ./${model_dir}/._*)
  else
     echo "[INFO] --- $model already exists!"
     if [ ! -d "${model_dir}" ]; then
        tar -zxvf $model
        rm $(ls ./${model_dir}/._*)
     else
        echo "[INFO] --- $model_dir already exists!"
     fi
  fi
}


# PaddleClas
download MobileNetV3_small_x1_0.tgz
download PP-HGNet_small.tgz
download PP-LCNet_x1_0.tgz
download SwinTransformer-Base.tgz
download ResNet50.tgz
download CLIP_vit_base_patch16_224.tgz

# PaddleDetection
download rt_detr_hgnetv2_l.tgz
download dino_r50_4scale.tgz
download PP-PicoDet_s_320_lcnet.tgz # normal nms
download PP-YOLOE+_crn_l_80e.tgz
download PP-PicoDet_s_320_lcnet_with_nms.tgz
download PP-PicoDet_s_320_lcnet_without_nms.tgz
download PP-YOLOE+_crn_l_80e_with_nms.tgz
download PP-YOLOE+_crn_l_80e_without_nms.tgz
download PP-YOLOE+_crn_l_80e_with_trt_nms.tgz

# PaddleSeg
download OCRNet_HRNetW48.tgz
download PP-LiteSeg-STDC1.tgz
download PP-MobileSeg-Base.tgz
download SegFormer-B0.tgz

# PaddleOCR
download PP-OCRv4-mobile-rec.tgz
download PP-OCRv4-server-rec.tgz
download PP-OCRv4-mobile-det.tgz
download PP-OCRv4-server-det.tgz

# PP-ShiTuV2
download PP-ShiTuv2-rec.tgz
download PP-ShiTuv2-det.tgz

# PP-StructureV2
download PP-Structurev2-layout.tgz
download PP-Structurev2-SLANet.tgz
download PP-Structurev2-vi-layoutxlm.tgz

# Paddle3D
download CADDN_OCRNet-HRNetW18.tgz
download CenterPoint-Pillars-02Voxel.tgz
download PETRv1_v99.tgz
download PETRv2_v99.tgz

# Test resources
# PaddleClas
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppcls_cls_demo.JPEG

# PaddleDetection & ppshitu-det
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppdet_det_img.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppdet_det_img_800x800.jpg

# PaddleSeg
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppseg_cityscapes_demo.png
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppseg_cityscapes_demo_512x512.png
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppseg_ade_val_512x512.png

# PaddleOCR
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppocrv4_word_1.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppocrv4_det_1.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppocr_keys_v1.txt

# PP-ShiTuV2
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/ppshituv2_wangzai.png

# PP-StructureV2
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/structurev2_table.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/structurev2_layout_val_0002.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/structurev2_vi_layoutxml_zh_val_0.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/table_structure_dict_ch.txt
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/layout_cdla_dict.txt

# Paddle3D
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/paddle3d_cadnn_kitti_000780.png
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/paddle3d_centerpoint_n008_LIDAR_TOP__1533151603547590.pcd.bin
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/paddle3d_petrv1_v99_nuscenes_sample_6.tgz && tar -zxvf paddle3d_petrv1_v99_nuscenes_sample_6.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy_paddlex_2_0/paddle3d_petrv2_v99_nuscenes_sample_12.tgz && tar -zxvf paddle3d_petrv2_v99_nuscenes_sample_12.tgz
