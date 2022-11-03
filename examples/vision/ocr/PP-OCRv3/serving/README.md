# PP-OCR服务化部署示例

## 介绍
本文介绍了使用FastDeploy搭建OCR文字识别服务的方法.

服务端必须在docker内启动,而客户端不是必须在docker容器内.

**本文所在路径($PWD)下的models里包含模型的配置和代码(服务端会加载模型和代码以启动服务), 需要将其映射到docker中使用.**

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
tar xvf ch_PP-OCRv3_rec_infer.tar && mv ch_PP-OCRv3_rec_infer.tar 1
mv 1/inference.pdiparams 1/model.pdiparams && mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/rec_runtime/ && rm -rf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt
mv ppocr_keys_v1.txt models/rec_postprocess/1/

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg


docker pull paddlepaddle/fastdeploy:0.3.0-gpu-cuda11.4-trt8.4-21.10
docker run -dit --net=host --name fastdeploy --shm-size="1g" -v $PWD:/ocr_serving paddlepaddle/fastdeploy:0.3.0-gpu-cuda11.4-trt8.4-21.10 bash
docker exec -it -u root fastdeploy bash
```

#### 1.2 安装(在docker内)
```bash
ldconfig
apt-get install libgl1
```

#### 1.3 启动服务端(在docker内)
```bash
fastdeployserver --model-repository=/models
```

参数:
  - `model-repository`(required): 整套模型streaming_pp_tts存放的路径.
  - `model-control-mode`(required): 模型加载的方式,现阶段, 使用'explicit'即可.
  - `load-model`(required): 需要加载的模型的名称.
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
