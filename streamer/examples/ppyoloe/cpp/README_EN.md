English | [简体中文](README_CN.md)

# FastDeploy Streamer PP-YOLOE C++ Example

## Build and Run

1. Build FastDeploy Streamer first, [README](../../../README.md)

2. Build Example
```
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-OF-FASTDEPLOY-INSTALL-DIR]
make -j
```

3. Download model
```
wget xxxxx/ppyoloe_crn_l_300e_coco_onnx_no_scale_factor.tgz
tar xvf ppyoloe_crn_l_300e_coco_onnx_no_scale_factor.tgz
mv ppyoloe_crn_l_300e_coco_onnx_no_scale_factor/ model/
```

4. Run
```

./streamer_demo
```
