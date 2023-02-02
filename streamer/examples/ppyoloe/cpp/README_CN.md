简体中文 | [English](README_EN.md)

# FastDeploy Streamer PP-YOLOE C++ Example

## 编译和运行

1. 该示例依赖DeepStream，需要准备DeepStream环境，并编译FastDeploy Streamer，请参考[README](../../../README_CN.md)

2. 编译Example
```
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-OF-FASTDEPLOY-INSTALL-DIR]
make -j
```

3. 下载模型
```
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_onnx_without_scale_factor.tgz
tar xvf ppyoloe_crn_l_300e_coco_onnx_without_scale_factor.tgz
mv ppyoloe_crn_l_300e_coco_onnx_without_scale_factor/ model/
```

4. 运行
```
cp ../nvinfer_config.txt .
cp ../streamer_cfg.yml .
./streamer_demo
```

## 导出ONNX模型，不包含NMS和scale factor
```
# 导出Paddle推理模型，exclude_nms=True and trt=True
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
python tools/export_model.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o  weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams exclude_nms=True trt=True --output_dir inference_model

# 转换为ONNX
paddle2onnx --model_dir inference_model/ppyoloe_crn_l_300e_coco/  --model_filename model.pdmodel  --params_filename model.pdiparams  --save_file ppyoloe.onnx  --deploy_backend tensorrt  --enable_dev_version True

# 裁剪ONNX，删除scale factor
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
python tools/onnx/prune_onnx_model.py --model ../PaddleDetection/ppyoloe.onnx --output_names concat_14.tmp_0 p2o.Mul.245 --save_file ppyoloe_without_scale_factor.onnx
```
