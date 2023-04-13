```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX/tools/paddle
wget https://bj.bcebos.com/fastdeploy/models/ppyoloe_plus_crn_m_80e_coco.tgz
tar xvf ppyoloe_plus_crn_m_80e_coco.tgz
python prune_paddle_model.py --model_dir ppyoloe_plus_crn_m_80e_coco  \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --output_names  tmp_17 concat_14.tmp_0 \
                             --save_dir ppyoloe_plus_crn_m_80e_coco
cp -r ppyoloe_plus_crn_m_80e_coco ../../../

# export model
cd ../../..
python ./convert_model.py --model_path=./ppyoloe_plus_crn_m_80e_coco/model \
                          --shape_dict="{'image': [1, 3, 640, 640], 'scale_factor': [1, 2]}"

# build example
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=/path/to/fastdeploy-sdk
make -j
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
./infer_ppyoloe_demo ../tvm_save 000000014439.jpg
```
