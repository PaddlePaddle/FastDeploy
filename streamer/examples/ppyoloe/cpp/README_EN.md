English | [简体中文](README_CN.md)

# FastDeploy Streamer PP-YOLOE C++ Example

## Build and Run

1. This example requires DeepStream, please prepare DeepStream environment and build FastDeploy Streamer, refer to [README](../../../README.md)

2. Build Example
```
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-OF-FASTDEPLOY-INSTALL-DIR]
make -j
```

3. Download model
```
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_onnx_without_scale_factor.tgz
tar xvf ppyoloe_crn_l_300e_coco_onnx_without_scale_factor.tgz
mv ppyoloe_crn_l_300e_coco_onnx_without_scale_factor/ model/
```

4. Run
```
cp ../nvinfer_config.txt .
cp ../streamer_cfg.yml .
./streamer_demo
```

## Export ONNX excluding scale_factor and NMS
```
# Export inference model with exclude_nms=True and trt=True
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
python tools/export_model.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o  weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams exclude_nms=True trt=True --output_dir inference_model

# Convert to ONNX
paddle2onnx --model_dir inference_model/ppyoloe_crn_l_300e_coco/  --model_filename model.pdmodel  --params_filename model.pdiparams  --save_file ppyoloe.onnx  --deploy_backend tensorrt  --enable_dev_version True

# Prune ONNX to delete scale factor
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
python tools/onnx/prune_onnx_model.py --model ../PaddleDetection/ppyoloe.onnx --output_names concat_14.tmp_0 p2o.Mul.245 --save_file ppyoloe_without_scale_factor.onnx
```
