# PaddleOCR服务化部署示例

PaddleOCR 服务化部署示例是利用FastDeploy Serving搭建的服务化部署示例。FastDeploy Serving是基于Triton Inference Server框架封装的适用于高并发、高吞吐量请求的服务化部署框架，是一套可用于实际生产的完备且性能卓越的服务化部署框架。如没有高并发，高吞吐场景的需求，只想快速检验模型线上部署的可行性，请参考Simple Serving

## 1. PP-OCRv3服务化部署介绍
本文介绍了使用FastDeploy搭建PP-OCRv3模型服务的方法.
服务端必须在docker内启动，而客户端不是必须在docker容器内.

**本文所在路径($PWD)下的models里包含模型的配置和代码(服务端会加载模型和代码以启动服务), 需要将其映射到docker中使用.**

PP-OCRv3由det(检测)、cls(分类)和rec(识别)三个模型组成.

服务化部署串联的示意图如下图所示,其中`pp_ocr`串联了`det_preprocess`、`det_runtime`和`det_postprocess`,`cls_pp`串联了`cls_runtime`和`cls_postprocess`,`rec_pp`串联了`rec_runtime`和`rec_postprocess`.

特别的是,在`det_postprocess`中会多次调用`cls_pp`和`rec_pp`服务,来实现对检测结果(多个框)进行分类和识别,,最后返回给用户最终的识别结果。

<p align="center">
    <br>
<img src="https://user-images.githubusercontent.com/15235574/224879693-4bd7676e-9b49-4238-883c-b5762867fa8e.png">
    <br>
<p>


## 2. 服务端的使用

### 2.1 下载模型并使用服务化Docker
```bash
# 找到部署包内的模型，这里测试的模型包括检测模型、分类模型和识别模型，如果只导出了其中的某个模型，则其他模型可用预训练模型进行测试
# 例如只导出了检测模型，则分类和识别模型可用预训练模型ch_ppocr_mobile_v2.0_cls_infer.tar和识别模型ch_PP-OCRv3_rec_infer.tar

# 可用于测试的预训练模型，可替换为自己训练的模型
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar

# 模型重命名
tar xvf ch_PP-OCRv3_det_infer.tar
mv ch_PP-OCRv3_det_infer 1
mv 1/inference.pdiparams 1/model.pdiparams
mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/det_runtime/
rm -rf ch_PP-OCRv3_det_infer.tar

# 模型重命名
tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar
mv ch_ppocr_mobile_v2.0_cls_infer 1
mv 1/inference.pdiparams 1/model.pdiparams
mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/cls_runtime/
rm -rf ch_ppocr_mobile_v2.0_cls_infer.tar

# 模型重命名
tar xvf ch_PP-OCRv3_rec_infer.tar
mv ch_PP-OCRv3_rec_infer 1
mv 1/inference.pdiparams 1/model.pdiparams
mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/rec_runtime/
rm -rf ch_PP-OCRv3_rec_infer.tar

mkdir models/pp_ocr/1 && mkdir models/rec_pp/1 && mkdir models/cls_pp/1

# 准备字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt
mv ppocr_keys_v1.txt models/rec_postprocess/1/

# x.y.z为镜像版本号，需参照serving文档替换为数字
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
docker run -dit --net=host --name fastdeploy --shm-size="1g" -v $PWD:/ocr_serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash
docker exec -it -u root fastdeploy bash
```

### 2.2 安装(在docker内)
```bash
ldconfig
apt-get install libgl1
```

#### 2.3 启动服务端(在docker内)
```bash
fastdeployserver --model-repository=/ocr_serving/models
```

参数:
  - `model-repository`(required): 整套模型streaming_pp_tts存放的路径.
  - `http-port`(optional): HTTP服务的端口号. 默认: `8000`. 本示例中未使用该端口.
  - `grpc-port`(optional): GRPC服务的端口号. 默认: `8001`.
  - `metrics-port`(optional): 服务端指标的端口号. 默认: `8002`. 本示例中未使用该端口.


## 3. 客户端的使用
### 3.1 安装
```bash
pip3 install tritonclient[all]
```

### 3.2 发送请求
```bash
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
python3 client.py
```

## 4 .配置修改
当前默认配置在GPU上运行， 如果要在CPU或其他推理引擎上运行。 需要修改`models/runtime/config.pbtxt`中配置，详情请参考[配置文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_configuration.md)

## 5. 其他指南

- 使用PP-OCRv2进行服务化部署, 除了自行准备PP-OCRv2模型之外, 只需手动添加一行代码即可.
在[model.py](./models/det_postprocess/1/model.py#L109)文件**109行添加以下代码**：
```
self.rec_preprocessor.cls_image_shape[1] = 32
```

- [使用 VisualDL 进行 Serving 可视化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/vdl_management.md)
通过VisualDL的可视化界面对PP-OCRv3进行服务化部署只需要如下三步：
```text
1. 载入模型库：./vision/ocr/PP-OCRv3/serving
2. 下载模型资源文件：点击det_runtime模型，点击版本号1添加预训练模型，选择文字识别模型ch_PP-OCRv3_det进行下载。点击cls_runtime模型，点击版本号1添加预训练模型，选择文字识别模型ch_ppocr_mobile_v2.0_cls进行下载。点击rec_runtime模型，点击版本号1添加预训练模型，选择文字识别模型ch_PP-OCRv3_rec进行下载。点击rec_postprocess模型，点击版本号1添加预训练模型，选择文字识别模型ch_PP-OCRv3_rec进行下载。
3. 启动服务：点击启动服务按钮，输入启动参数。
```
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211709324-b07bb303-ced2-4137-9df7-0d2574ba84c8.gif" width="100%"/>
</p>

## 6. 常见问题
- [如何编写客户端 HTTP/GRPC 请求](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/client.md)
- [如何编译服务化部署镜像](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/compile.md)
- [服务化部署原理及动态Batch介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/demo.md)
- [模型仓库介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_repository.md)
