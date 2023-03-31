English | [简体中文](README_CN.md)
# YOLOv5 Python Deployment Example

Before deployment, the following step need to be confirmed:

- 1. Hardware and software environment meets the requirements. Please refer to [FastDeploy Environment Requirement](../../../../../../docs/en/build_and_install/sophgo.md)

`infer.py` in this directory provides a quick example of deployment of the YOLOv5 model on SOPHGO TPU. Please run the following script:

```bash
# Download the sample deployment code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/yolov5/sophgo/python

# Download images.
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# Set --auto True, automatic inference.
python3 infer.py --auto True

# Set --auto False, need to set the model path and image path manually.
python3 infer.py --model_file ./bmodel/yolov5s_1684x_f32.bmodel --image 000000014439.jpg

# The returned result.
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
268.480255,81.053055, 298.694794, 169.439026, 0.896569, 0
104.731163,45.661972, 127.583824, 93.449387, 0.869531, 0
378.909363,39.750137, 395.608643, 84.243454, 0.868430, 0
158.552979,80.361511, 199.185760, 168.181915, 0.842988, 0
414.375305,90.948090, 506.321899, 280.405182, 0.835842, 0
364.003448,56.608932, 381.978607, 115.968216, 0.815136, 0
351.725128,42.635330, 366.910309, 98.048386, 0.808936, 0
505.888306,114.366791, 593.124878, 275.995270, 0.801361, 0
327.708618,38.363693, 346.849915, 80.893021, 0.794725, 0
583.493408,114.532883, 612.354614, 175.873535, 0.760649, 0
186.470657,44.941360, 199.664505, 61.037643, 0.632591, 0
169.615891,48.014603, 178.141556, 60.888596, 0.613938, 0
25.810200,117.199692, 59.888783, 152.850128, 0.590614, 0
352.145294,46.712723, 381.946075, 106.752151, 0.505329, 0
1.875000,150.734375, 37.968750, 173.781250, 0.404573, 24
464.657288,15.901413, 472.512939, 34.116409, 0.346033, 0
64.625000,135.171875, 84.500000, 154.406250, 0.332831, 24
57.812500,151.234375, 103.000000, 174.156250, 0.332566, 24
165.906250,88.609375, 527.906250, 339.953125, 0.259424, 33
101.406250,152.562500, 118.890625, 169.140625, 0.253891, 24
```

## Other Documents
- [YOLOv5 C++ Deployment](../cpp)
- [Converting YOLOv5 SOPHGO model](../README.md)
