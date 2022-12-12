简体中文 | [English](README_EN.md)

# FastDeploy Streamer PP-YOLOE C++ Example

## 编译和运行

1. 需要先FastDeploy Streamer, 请参考[README](../../../README.md)

2. 编译Example
```
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-OF-FASTDEPLOY-INSTALL-DIR]
make -j
```

3. 下载模型
```
wget xxxxx/ppyoloe_crn_l_300e_coco_onnx_no_scale_factor.tgz
tar xvf ppyoloe_crn_l_300e_coco_onnx_no_scale_factor.tgz
mv ppyoloe_crn_l_300e_coco_onnx_no_scale_factor/ model/
```

4. 运行
```
cp ../nvinfer_config.txt .
cp ../streamer_cfg.yml .
./streamer_demo
```
