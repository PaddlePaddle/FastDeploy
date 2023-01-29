English | [简体中文](README_CN.md)

# FastDeploy Streamer Video Decoder Example

## Build and Run

1. Build FastDeploy Streamer first, [README](../../../)

2. Build Example
```
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=[PATH-OF-FASTDEPLOY-INSTALL-DIR]
make -j
```

3. Run
```
# GPU decoding(gpu.yml) or CPU decoding(cpu.yml)
cp ../gpu.yml streamer_cfg.yml
./video_decoder
```
