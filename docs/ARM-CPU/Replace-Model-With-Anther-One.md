  <a name="0"></a>
# 简介

本文档介绍如何将FastDeploy的Demo模型，替换成开发者自己训练的AI模型。（**注意**：FastDeploy下载的SDK和Demo仅支持相同算法模型的替换）。本文档要求开发者已经将Demo和SDK运行跑通，如果要了解运行跑通Demo和SDK指导文档，可以参考[SDK使用文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/README.md#sdk使用)

* [简介](#0)<br>
* [模型替换](#1)<br>
  * [1.模型准备](#2)<br>
    * [1.1 Paddle模型](#3)<br>
    * [1.2 Paddle OCR模型增加一步特殊转换](#4)<br>
      * [1.2.1 下载模型转换工具](#5)<br>
      * [1.2.2 下载模型转换工具](#6)<br>
    * [1.3 其他框架模型](#7)<br>
  * [2.模型名修改和label文件准备](#8)<br>
    * [2.1 非OCR模型名修改](#9)<br>
    * [2.2 OCR模型名修改](#10)<br>
    * [2.3 模型label文件](#11)<br>
  * [3.修改配置文件](#12)<br>
* [测试效果](#13)<br>
* [完整配置文件说明](#14)<br>
  * [1.配置文件字段含义](#15)<br>
  * [2.预处理顺序](#16)<br>
* [FAQ](#17)<br>

**注意事项：** 

1. PP-PicoDet模型： 在FastDeploy中，支持PP-Picodet模型，是将后处理写到网络里面的方式（即后处理+NMS都在网络结构里面）。Paddle Detection导出静态模型时，有3种方法，选择将后处理和NMS导入到网络里面即可（参考[导出部分](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet#%E5%AF%BC%E5%87%BA%E5%8F%8A%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9E%8B)）。详细网络区别，可以通过netron工具对比。

2. PP-Picodet模型：在FastDeploy中，支持PP-Picodet模型，是将前处理写在网络外面的方式。Paddle Detection中的TinyPose算法中，会将PP-PicoDet模型的前处理写入网络中。如果要使用FastDeploy的SDK进行模型替换，需要将前处理写到网络外面。（参考[Detection中的导出命令](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint/tiny_pose#%E5%B0%86%E8%AE%AD%E7%BB%83%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%AE%9E%E7%8E%B0%E7%AB%AF%E4%BE%A7%E9%83%A8%E7%BD%B2)，将TestReader.fuse_normalize=False即可）。
   
   <a name="1"></a>

# 模型替换

开发者从PaddleDetection、PaddleClas、PaddleOCR、PaddleSeg等飞桨开发套件导出来的对应模型，完成 [1.模型准备](#)、[1.模型名修改和模型label](#)、[3.修改配置文件](#) 3步操作（需要相同算法才可替换），可完成自定义模型的模型文件，运行时指定新的模型文件，即可在自己训练的模型上实现相应的预测推理任务。

* Linux下模型资源文件夹路径：`EasyEdge-Linux-**/RES/` 。
* Windows下模型资源文件夹路径：`EasyEdge-Windows-**/data/model/`。
* Android下模型资源文件夹路径：`EasyEdge-Android-**/app/src/assets/infer/` 和 ` app/src/assets/demo/conf.json` 
* iOS下模型资源文件夹路径：`EasyEdge-iOS-**/RES/easyedge/` 

主要涉及到下面4个模型相关的文件（mode、params、label_list.txt、infer_cfg.json）和一个APP名相关的配置文件（仅Android、iOS、HTTP需要，APP名字，非必需。）

* ```
  ├── RES、model、infer  # 模型资源文件夹，一套模型适配不同硬件、OS和部署方式
  │   ├── conf.json        # Android、iOS系统APP名字需要
  │   ├── model            # 模型结构文件 
  │   ├── params           # 模型参数文件
  │   ├── label_list.txt   # 模型标签文件
  │   ├── infer_cfg.json   # 模型前后处理等配置文件
  ```
  
  > ❗注意：OCR模型在ARM CPU硬件上（包括Android、Linux、iOS 三款操作系统），因为任务的特殊性，替换在 [1.模型准备](#)、[1.模型名修改和模型label](#) 不同于其他任务模型，详细参考下面步骤。
  
  <a name="2"></a>

## 1.模型准备

 <a name="3"></a>

### 1.1 Paddle模型

* 通过PaddleDetection、PaddleClas、PaddleOCR、PaddleSeg等导出来飞桨模型文件，包括如下文件（可能存在导出时修改了名字的情况，后缀`.pdmodel`为模型网络结构文件，后缀`.pdiparams`为模型权重文件）：

```
model.pdmodel       # 模型网络结构
model.pdiparams   # 模型权重
model.yml           # 模型的配置文件（包括预处理参数、模型定义等）
```

 <a name="4"></a>

### 1.2 OCR模型特殊转换（仅在ARM CPU上需要）

因为推理引擎版本的问题，OCR模型需要在[1.1 Paddle模型](#3)导出`.pdmodel`和`.pdiparams`模型后，多增加一步模型转换的特殊处理，主要执行下面2步：

<a name="5"></a>

#### 1.2.1 下载模型转换工具

Linux 模型转换工具下载链接：[opt_linux](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11/opt_linux)</br>
M1 模型转换工具下载链接：[opt_m1](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11/opt_m1)</br>
mac 模型转换工具下载链接：[opt_mac](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.11/opt_mac)</br>

<a name="6"></a>

#### 1.2.2 模型转换

以下命令，以mac为例，完成模型转换。

```
* 转换 OCR 检测模型命名：
./opt_mac --model_dir=./ch_PP-OCRv3_det_infer/ --valid_targets=arm --optimize_out_type=naive_buffer --optimize_out=./ocr_det

* 转换 OCR 识别模型命名：
./opt_mac --model_dir=./ch_PP-OCRv3_rec_infer/ --valid_targets=arm --optimize_out_type=naive_buffer --optimize_out=./ocr_rec
```

产出：

<div align=center><img src="https://user-images.githubusercontent.com/54695910/175856746-501b05ad-6fba-482e-8e72-fdd68fe52101.png" width="400"></div>

 <a name="7"></a>

### 1.3 其他框架模型

* 如果开发着是PyTorch、TensorFLow、Caffe、ONNX等其他框架模型，可以参考[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)官网完成模型转换，即可得到对应的`model.pdmodel`和`model.pdiparams`模型文件。

<a name="8"></a>

## 2.模型名修改和label文件准备

<a name="9"></a>

### 2.1 非OCR模型名修改

按照下面的规则，修改套件导出来的模型名和标签文件，并替换到模型资源文件中。

```
1. model.pdmodel 修改成  model
2. model.pdiparams 修改成 params
```

<a name="10"></a>

### 2.2 OCR模型名修改

```
1. ocr_det.nb 修改成  model  # 将 检测模型 修改名称成 model
2. ocr_rec.nb 修改成 params  # 将 识别模型 修改名称成 model
```

<a name="11"></a>

### 2.3 模型label文件

同时需要准备模型文件对应的label文件`label_list.txt`。label文件可以参考原Demo中`label_list.txt`的格式准备。

<a name="12"></a>

## 3. 修改模型相关配置文件

（1）infer_cfg.json 文件修改

所有程序开发者都需要关注该配置文件。开发者在自己数据/任务中训练模型，可能会修改输入图像尺寸、修改阈值等操作，因此需要根据训练情况修改`Res文件夹下的infer_cfg.json`文件中的对应。CV任务涉及到的配置文件修改包括如下字段：

```
1. "best_threshold": 0.3,   #网络输出的阈值,根据开发者模型实际情况修改
2. "resize": [512, 512],    #[w, h]网络输入图像尺寸,用户根据实际情况修改。
```

（2）conf.json 文件修改
仅Android、iOS、HTTP服务应用开发者，需要关注该配置文件。开发者根据自己应用程序命名需要，参考已有`conf.json`即可。

通常，开发者修改FastDeploy项目中的模型，涉及到主要是这几个配置信息的修改。FastDeploy详细的配置文件介绍参考[完整配置文件说明](#8)。

<a name="13"></a>

# 测试效果

将自定义准备的`RES`文件，按照第2、3步完成修改后，参考可以参考[SDK使用文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/README.md#sdk%E4%BD%BF%E7%94%A8)完成自己模型上的不同预测体验。

<a name="14"></a>

# 完整配置文件说明

<a name="15"></a>

## 1. 配置文件字段含义

模型资源文件`infer_cfg.json`涉及到大量不同算法的前后处理等信息，下表是相关的字段介绍，通常开发者如果没有修改算法前出处理，不需要关心这些字段。非标记【必须】的可不填。

```json
{
    "version": 1,
    "model_info": { 
        "best_threshold": 0.3,   // 默认0.3
        "model_kind": 1, // 【必须】 1-分类，2-检测，6-实例分割，12-追踪，14-语义分割，401-人脸，402-姿态，10001-决策
    },
    "pre_process": { // 【必须】
        // 归一化， 预处理会把图像 (origin_img - mean) * scale 
        "skip_norm": false, // 默认为false, 如果设置为true，不做mean scale处理
        "mean": [123, 123, 123],  // 【必须，一般不需要动】图像均值，已经根据Paddle套件均值做了转换处理，开发者如果没有修改套件参数，可以不用关注。（X-mean）/ scale
        "scale": [0.017, 0.017, 0.017],  // 【必须，一般不需要动】
        "color_format": "RGB", // BGR 【必须，一般不需要动】
        "channel_order": "CHW", // HWC
        // 大小相关
        "resize": [300, 300],        // w, h 【必须】
        "rescale_mode": "keep_size", // 默认keep_size， keep_ratio, keep_ratio2, keep_raw_size, warp_affine
        "max_size": 1366, // keep_ratio 用。如果没有提供，则用 resize[0]
        "target_size": 800,  // keep_ratio 用。如果没有提供，则用 resize[1]
        "raw_size_range": [100, 10000], // keep_raw_size 用
        "warp_affine_keep_res": // warp_affine模式使用，默认为false
        "center_crop_size": [224, 224]， // w, h, 如果需要做center_crop，则提供，否则，无需提供该字段
        "padding": false,
        "padding_mode": "padding_align32",  // 【非必须】默认padding_align32, 其他可指定：padding_fill_size
        "padding_fill_size": [416, 416], // 【非必须】仅padding_fill_size模式下需要提供, [fill_size_w, fill_size_h], 这里padding fill对齐paddle detection实现，在bottom和right方向实现补齐
        "padding_fill_value": [114, 114, 114] // 【非必须】仅padding_fill_size模式下需要提供
        // 其他
        "letterbox": true,
     },
    "post_process": {
        "box_normed": true, // 默认为true, 如果为false 则表示该模型的box坐标输出不是归一化的
    }
}
```

<a name="16"></a>

## 2. 预处理顺序（没有的流程自动略过）

1. 灰度图 -> rgb图变换 
2. resize 尺寸变换 
3. center_crop
4. rgb/bgr变换
5. padding_fill_size
6. letterbox（画个厚边框，填上黑色）
7. chw/hwc变换
8. 归一化：mean, scale
9. padding_align32

rescale_mode说明：

* keep_size: 将图片缩放到resize指定的大小
* keep_ratio:将图片按比例缩放，长边不超过max_size，短边不超过target_size
* keep_raw_size:保持原图尺寸，但必须在raw_size_range之间
* warp_affine: 仿射变换，可以设置warp_affine_keep_res指定是否keep_res，在keep_res为false场景下，宽高通过resize字段指定

<a name="17"></a>

# FAQ

### 1. 如何处理一些 undefined reference / error while loading shared libraries?

> 如：./easyedge_demo: error while loading shared libraries: libeasyedge.so.1: cannot open shared object file: No such file or directory

遇到该问题时，请找到具体的库的位置，设置LD_LIBRARY_PATH；或者安装缺少的库。

> 示例一：libverify.so.1: cannot open shared object file: No such file or directory
> 链接找不到libveirfy.so文件，一般可通过 export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../lib 解决(实际冒号后面添加的路径以libverify.so文件所在的路径为准)

> 示例二：libopencv_videoio.so.4.5: cannot open shared object file: No such file or directory
>  链接找不到libopencv_videoio.so文件，一般可通过 export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../thirdparty/opencv/lib 解决(实际冒号后面添加的路径以libopencv_videoio.so所在路径为准)

> 示例三：GLIBCXX_X.X.X  not found
> 链接无法找到glibc版本，请确保系统gcc版本>=SDK的gcc版本。升级gcc/glibc可以百度搜索相关文献。

### 2. 使用libcurl请求http服务时，速度明显变慢

这是因为libcurl请求continue导致server等待数据的问题，添加空的header即可

```bash
headers = curl_slist_append(headers, "Expect:");
```

### 3. 运行二进制时，提示 libverify.so cannot open shared object file

可能cmake没有正确设置rpath, 可以设置LD_LIBRARY_PATH为sdk的lib文件夹后，再运行：

```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib ./easyedge_demo
```

### 4. 编译时报错：file format not recognized

可能是因为在复制SDK时文件信息丢失。请将整个压缩包复制到目标设备中，再解压缩、编译
