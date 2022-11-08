# PaddleSeg 量化模型 C++部署示例
本目录下提供的`infer.cc`,可以帮助用户快速完成PaddleSeg量化模型在CPU/GPU上的部署推理加速.

## 部署准备
### FastDeploy环境准备
- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

### 量化模型准备
- 1. 用户可以直接使用由FastDeploy提供的量化模型进行部署.
- 2. 用户可以使用FastDeploy提供的[一键模型自动化压缩工具](../../../../../../tools/auto_compression/),自行进行模型量化, 并使用产出的量化模型进行部署.(注意: 推理量化后的分类模型仍然需要FP32模型文件夹下的deploy.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可.)

## 以量化后的PP_LiteSeg_T_STDC1_cityscapes模型为例, 进行部署
在本目录执行如下命令即可完成编译,以及量化模型部署.
```bash
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.3.0.tgz
tar xvf fastdeploy-linux-x64-0.3.0.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.3.0
make -j

#下载FastDeloy提供的PP_LiteSeg_T_STDC1_cityscapes量化模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ.tar
tar -xvf PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ.tar
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 在CPU上使用Paddle-Inference推理量化模型
./infer_demo PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ cityscapes_demo.png 1
```
