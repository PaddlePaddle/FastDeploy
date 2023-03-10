## Update notes

2020.11.4

1. Support model conversion from dynamic computational graph of PaddlePaddle to ONNX.
2. Reconfigure the code structure to support different versions of Paddlepaddle.
3. Improve the coverage rate of ONNX Opset, and Opset 9,10 and 11 will be stably supported.


2020.9.21

1. Support export to ONNX Opset 9,10 and 11.
2. Support new convertable ops: swish, floor, uniform_random, abs, instance_norm, clip, tanh, log, norm and pad2d.

2019.09.25

1. Add newly supported models: SE_ResNet50_vd,SqueezeNet1_0,SE_ResNext50_32x4d,Xception41,VGG16,InceptionV4 and YoloV3.
2. Solve the incompatibility of Paddle2ONNX v0.1 with ONNX.

2019.08.20

1. Solve the incompatibility of preview version with PaddlePaddle and ONNX.
2. Support mainstream models of image classification and object detection models.
3. Unify the interfaces and support PIP installation.