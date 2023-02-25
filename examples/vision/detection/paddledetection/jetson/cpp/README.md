```bash
mkdir build
cd build

cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../compiled_fastdeploy_sdk
make install

wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
./infer_solov2_demo ./solov2_r50_fpn_1x_coco 000000014439.jpg 0

```
