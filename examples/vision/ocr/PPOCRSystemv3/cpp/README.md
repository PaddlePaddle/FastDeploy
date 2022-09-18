# PPOCRSystemv3 C++部署示例

本目录下提供`infer.cc`快速完成PPOCRSystemv3在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```
mkdir build
cd build
wget https://https://bj.bcebos.com/paddlehub/fastdeploy/cpp/fastdeploy-linux-x64-gpu-0.2.1.tgz
tar xvf fastdeploy-linux-x64-0.2.1.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.2.1
make -j


# 下载模型,图片和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar

wget https://bj.bcebos.com/paddlehub/fastdeploy/ch_ppocr_mobile_v2.0_cls_infer.tar.gz
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar.gz

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# CPU推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# GPU推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
# GPU上TensorRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 2
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">


## PPOCRSystemv3 C++接口

### PPOCRSystemv3类

```
fastdeploy::application::ocrsystem::PPOCRSystemv3(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Classifier* cls_model,
                fastdeploy::vision::ocr::Recognizer* rec_model);
```

PPOCRSystemv3 的初始化，由检测，分类和识别模型串联构成

**参数**

> * **DBDetector**(model): OCR中的检测模型
> * **Classifier**(model): OCR中的分类模型
> * **Recognizer**(model): OCR中的识别模型

```
fastdeploy::application::ocrsystem::PPOCRSystemv3(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Recognizer* rec_model);
```
PPOCRSystemv3 的初始化，由检测，识别模型串联构成(无分类器)

**参数**

> * **DBDetector**(model): OCR中的检测模型
> * **Recognizer**(model): OCR中的识别模型

#### Predict函数

> ```  
> bool Predict(cv::Mat* img, fastdeploy::vision::OCRResult* result);
> ```
>
> 模型预测接口，输入一张图片，返回OCR预测结果
>
> **参数**
>
> > * **img**: 输入图像，注意需为HWC，BGR格式
> > * **result**: OCR预测结果,包括由检测模型输出的检测框位置,分类模型输出的方向分类,以及识别模型输出的识别结果, OCRResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


## DBDetector C++接口

### DBDetector类

```
fastdeploy::vision::ocr::DBDetector(const std::string& model_file, const std::string& params_file = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const Frontend& model_format = Frontend::PADDLE);
```

DBDetector模型加载和初始化，其中模型为paddle模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为Paddle格式

### Classifier类与DBDetector类相同

### Recognizer类
```
  Recognizer(const std::string& model_file,
             const std::string& params_file = "",
             const std::string& label_path = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const Frontend& model_format = Frontend::PADDLE);
```
Recognizer类初始化时,需要在label_path参数中,输入识别模型所需的label文件，其他参数均与DBDetector类相同

**参数**
> * **label_path**(str): 识别模型的label文件路径


### 类成员变量
#### DBDetector预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **max_side_len**(int): 检测算法前向时图片长边的最大尺寸，当长边超出这个值时会将长边resize到这个大小，短边等比例缩放,默认为960
> > * **det_db_thresh**(double): DB模型输出预测图的二值化阈值，默认为0.3
> > * **det_db_box_thresh**(double): DB模型输出框的阈值，低于此值的预测框会被丢弃，默认为0.6
> > * **det_db_unclip_ratio**(double): DB模型输出框扩大的比例，默认为1.5
> > * **det_db_score_mode**(string):DB后处理中计算文本框平均得分的方式,默认为slow，即求polygon区域的平均分数的方式
> > * **use_dilation**(bool):是否对检测输出的feature map做膨胀处理,默认为Fasle

#### Classifier预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **cls_thresh**(double): 当分类模型输出的得分超过此阈值，输入的图片将被翻转，默认为0.9

## 其它文档

- [PPOCR 系列模型介绍](../../)
- [PPOCRv3 Python部署](../python)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
