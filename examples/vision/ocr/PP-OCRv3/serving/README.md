[English](README_EN.md) | 简体中文
# PP-OCR服务化部署示例

在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考[FastDeploy服务化部署](../../../../../serving/README_CN.md)

## 介绍
本文介绍了使用FastDeploy搭建OCR文字识别服务的方法.

服务端必须在docker内启动,而客户端不是必须在docker容器内.

**本文所在路径($PWD)下的models里包含模型的配置和代码(服务端会加载模型和代码以启动服务), 需要将其映射到docker中使用.**

OCR由det(检测)、cls(分类)和rec(识别)三个模型组成.

服务化部署串联的示意图如下图所示,其中`pp_ocr`串联了`det_preprocess`、`det_runtime`和`det_postprocess`,`cls_pp`串联了`cls_runtime`和`cls_postprocess`,`rec_pp`串联了`rec_runtime`和`rec_postprocess`.

特别的是,在`det_postprocess`中会多次调用`cls_pp`和`rec_pp`服务,来实现对检测结果(多个框)进行分类和识别,,最后返回给用户最终的识别结果。

<p align="center">
    <br>
<img src='./ppocr.png'">
    <br>
<p>

## 使用
### 1. 服务端
#### 1.1 Docker
```bash
# 下载仓库代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/ocr/PP-OCRv3/serving/

# 下载模型,图片和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar && mv ch_PP-OCRv3_det_infer 1
mv 1/inference.pdiparams 1/model.pdiparams && mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/det_runtime/ && rm -rf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar && mv ch_ppocr_mobile_v2.0_cls_infer 1
mv 1/inference.pdiparams 1/model.pdiparams && mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/cls_runtime/ && rm -rf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xvf ch_PP-OCRv3_rec_infer.tar && mv ch_PP-OCRv3_rec_infer 1
mv 1/inference.pdiparams 1/model.pdiparams && mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/rec_runtime/ && rm -rf ch_PP-OCRv3_rec_infer.tar

mkdir models/pp_ocr/1 && mkdir models/rec_pp/1 && mkdir models/cls_pp/1

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt
mv ppocr_keys_v1.txt models/rec_postprocess/1/

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

# x.y.z为镜像版本号，需参照serving文档替换为数字
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
docker run -dit --net=host --name fastdeploy --shm-size="1g" -v $PWD:/ocr_serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash
docker exec -it -u root fastdeploy bash
```

#### 1.2 安装(在docker内)
```bash
ldconfig
apt-get install libgl1
```

#### 1.3 启动服务端(在docker内)
```bash
fastdeployserver --model-repository=/ocr_serving/models
```

参数:
  - `model-repository`(required): 整套模型streaming_pp_tts存放的路径.
  - `http-port`(optional): HTTP服务的端口号. 默认: `8000`. 本示例中未使用该端口.
  - `grpc-port`(optional): GRPC服务的端口号. 默认: `8001`.
  - `metrics-port`(optional): 服务端指标的端口号. 默认: `8002`. 本示例中未使用该端口.


### 2. 客户端
#### 2.1 安装
```bash
pip3 install tritonclient[all]
```

#### 2.2 发送请求
```bash
python3 client.py
```

## 配置修改

当前默认配置在GPU上运行， 如果要在CPU或其他推理引擎上运行。 需要修改`models/runtime/config.pbtxt`中配置，详情请参考[配置文档](../../../../../serving/docs/zh_CN/model_configuration.md)
