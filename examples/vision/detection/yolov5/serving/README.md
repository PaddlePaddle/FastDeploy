# YOLOv5 Serving部署示例

```bash
#下载yolov5模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# 将模型放入 models/infer/1目录下, 并重命名为model.onnx
mv yolov5s.onnx models/infer/1/

# 拉取fastdeploy镜像
docker pull xxx

# 启动镜像和服务
docker run xx

# 客户端请求
python yolov5_grpc_client.py
```
