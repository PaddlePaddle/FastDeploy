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
python3 -m tvm.driver.tvmc compile --target=llvm \
    --target-llvm-device=cpu \
    yolov8_n_500e_coco/model.pdmodel \
    --model-format=paddle \
    --module-name=yolov8_n_500e_coco \
    --input-shapes "image:[1,3,640,640],scale_factor:[1,2]" \
    --module-name=yolov8_n_500e_coco \
    --output=yolov8_n_500e_coco.tar
tar -xvf yolov8_n_500e_coco.tar
```

```bash
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../compiled_fastdeploy_sdk
make -j
./infer_ppyoloe_demo ../model 123
```