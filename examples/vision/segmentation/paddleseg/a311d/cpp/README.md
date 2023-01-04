English | [简体中文](README_CN.md)
# PP-LiteSeg Quantized Model C++ Deployment Example

 `infer.cc` in this directory can help you quickly complete the inference acceleration of PP-LiteSeg quantization model deployment on A311D.

## Deployment Preparations
### FastDeploy Cross-compile Environment Preparations
- 1. For the software and hardware environment, and the cross-compile environment, please refer to [FastDeploy Cross-compile environment](../../../../../../docs/en/build_and_install/a311d.md#Cross-compilation-environment-construction)  

### Model Preparations
- 1. You can directly use the quantized model provided by FastDeploy for deployment.
- 2. You can use one-click automatical compression tool provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the deploy.yaml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)
- For more information, please refer to [Model Quantization](../../quantize/README.md)

## Deploying the Quantized PP-LiteSeg Segmentation model on A311D
Please follow these steps to complete the deployment of the PP-LiteSeg quantization model on A311D.
1. Cross-compile the FastDeploy library as described in [Cross-compile  FastDeploy](../../../../../../docs/en/build_and_install/a311d.md#FastDeploy-cross-compilation-library-compilation-based-on-Paddle-Lite)

2. Copy the compiled library to the current directory. You can run this line:
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/segmentation/paddleseg/a311d/cpp
```

3. Download the model and example images required for deployment in current path.
```bash
cd FastDeploy/examples/vision/segmentation/paddleseg/a311d/cpp
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz
tar -xvf ppliteseg.tar.gz
cp -r ppliteseg models
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
cp -r cityscapes_demo.png images
```

4. Compile the deployment example. You can run the following lines:
```bash
cd FastDeploy/examples/vision/segmentation/paddleseg/a311d/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=arm64 ..
make -j8
make install
# After success, an install folder will be created with a running demo and libraries required for deployment.
```

5. Deploy the PP-LiteSeg segmentation model to A311D based on adb. You can run the following lines:
```bash
# Go to the install directory.
cd FastDeploy/examples/vision/segmentation/paddleseg/a311d/cpp/build/install/
# The following line represents: bash run_with_adb.sh, demo needed to run, model path, image path, DEVICE ID.
bash run_with_adb.sh infer_demo ppliteseg cityscapes_demo.png $DEVICE_ID
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/205544166-9b2719ff-ed82-4908-b90a-095de47392e1.png">

需要特别注意的是，在 A311D 上部署的模型需要是量化后的模型，模型的量化请参考：[模型量化](../../../../../../docs/cn/quantize.md)
