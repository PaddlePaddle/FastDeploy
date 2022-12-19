简体中文 | [English](README_EN.md)

# FastDeploy Streamer Video Decoder Example

## 编译和运行

1. 需要先FastDeploy Streamer, 请参考[README](../../../README.md)

2. 编译Example
```
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-OF-FASTDEPLOY-INSTALL-DIR]
make -j
```

3. 运行
```
cp ../streamer_cfg.yml .
./video_decoder
```
