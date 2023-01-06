[English](README.md) | 简体中文
# AnimeGAN C++部署示例

本目录下提供`infer.cc`快速完成AnimeGAN在CPU/GPU部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上AnimeGAN推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.2以上(x.x.x>=1.0.2)

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载准备好的模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/style_transfer_testimg.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/animegan_v1_hayao_60_v1.0.0.tgz
tar xvfz animegan_v1_hayao_60_v1.0.0.tgz

# CPU推理
./infer_demo --model animegan_v1_hayao_60 --image style_transfer_testimg.jpg  --device cpu
# GPU推理
./infer_demo --model animegan_v1_hayao_60 --image style_transfer_testimg.jpg  --device gpu
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## AnimeGAN C++接口

### AnimeGAN类

```c++
fastdeploy::vision::generation::AnimeGAN(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

AnimeGAN模型加载和初始化，其中model_file为导出的Paddle模型结构文件，params_file为模型参数文件。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

#### Predict函数

> ```c++
> bool AnimeGAN::Predict(cv::Mat& image, cv::Mat* result)
> ```
>
> 模型预测入口，输入图像输出风格迁移后的结果。
>
> **参数**
>
> > * **image**: 输入数据，注意需为HWC，BGR格式
> > * **result**: 风格转换后的图像，BGR格式

#### BatchPredict函数

> ```c++
> bool AnimeGAN::BatchPredict(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* results);
> ```
>
> 模型预测入口，输入一组图像并输出风格迁移后的结果。
>
> **参数**
>
> > * **images**: 输入数据，一组图像数据，注意需为HWC，BGR格式
> > * **results**: 风格转换后的一组图像，BGR格式

- [模型介绍](../../)
- [Python部署](../python)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
