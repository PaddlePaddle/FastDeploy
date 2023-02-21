English | [中文](README_CN.md)

# Example of PaddleClas models C++ Deployment

This directory provides example file `multi_thread.cc` to fast deploy PaddleClas models on CPU/GPU and GPU accelerated by TensorRT.

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install the FastDeploy Python whl package. Please refer to [FastDeploy Python Installation](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

Taking ResNet50_vd inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
mkdir build
cd build
# # Download FastDeploy precompiled library. Users can choose your appropriate version in the`FastDeploy Precompiled Library` mentioned above
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the ResNet50_vd model file and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU multi-thread inference
./multi_thread_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 0 1
# GPU multi-thread inference
./multi_thread_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 1 1
# TensorRT multi-inference inference on GPU
./multi_thread_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 2 1
```
>> **Notice**: the last number in above command is thread number

The above command works for Linux or MacOS. For SDK in Windows, refer to:  
- [How to use FastDeploy C++ SDK in Windows ](../../../../docs/cn/faq/use_sdk_on_windows.md)

The result returned after running is as follows
```
Thread Id: 0
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```
