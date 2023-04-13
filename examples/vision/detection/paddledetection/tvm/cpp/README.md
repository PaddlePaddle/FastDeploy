```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd tools/paddle
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov8_n_500e_coco.tgz
tar xvf yolov8_n_500e_coco.tgz
python prune_paddle_model.py --model_dir yolov8_n_500e_coco  \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --output_names  tmp_73 concat_15.tmp_0 \
                             --save_dir yolov8_n_500e_coco
# export model
```

```bash
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../compiled_fastdeploy_sdk
make -j
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
./infer_ppyoloe_demo ../tvm_save 000000014439.jpg
```
