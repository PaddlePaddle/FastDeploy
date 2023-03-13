# PaddleOCR Python轻量服务化部署示例

PaddleOCR Python轻量服务化部署是FastDeploy基于Flask框架搭建的可快速验证线上模型部署可行性的服务化部署示例，基于http请求完成AI推理任务，适用于无并发推理的简单场景，如有高并发，高吞吐场景的需求请参考FastDeploy Serving

## 1. 启动服务
```bash
# 找到部署包内的模型，这里测试的模型包括检测模型、分类模型和识别模型，如果只导出了其中的某个模型，则其他模型可用预训练模型进行测试
# 例如只导出了检测模型，则分类和识别模型可用预训练模型ch_ppocr_mobile_v2.0_cls_infer.tar和识别模型ch_PP-OCRv3_rec_infer.tar

# 可用于测试的预训练模型，可替换为自己训练的模型
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar

# 下载字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 准备好模型后，按需修改server.py中的模型路径、字典路径等

# 启动服务，可修改server.py中的配置项来指定硬件、后端等
# 可通过--host、--port指定IP和端口号
fastdeploy simple_serving --app server:app
```

## 3. 客户端请求
```bash
# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

# 请求服务，获取推理结果（如有必要，请修改脚本中的IP和端口号）
python client.py
```
