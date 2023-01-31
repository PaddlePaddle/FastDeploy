# Benchmark
## Environment
This benchmark is tested on CentOS7 with GPU `A10` and CPU `Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz`, and its environment is as follows:
| Package  |  Version |
|----------|----------|
| CUDA     | 11.3     |
| cuDNN    | 8.3.2.44 |
| TensorRT | 8.4.1.5  |
| Python   | 3.8.13   |
| PyTorch  | 1.12.1   |

## Performance
The following is the result of comparison between pytorch eager and poros, which measured by average latency time (ms) of model infering 1000 times.

### 1. ResNet50
Input shape: bx3x224x224  
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   6.17       |  1.70       |
| 2          |   6.02       |  2.41       |
| 4          |   6.33       |  3.23       |
| 8          |   8.55       |  4.75       |
| 16         |   16.22      |  7.82       |
| 32         |   32.09      |  14.00      |  

Model source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py  

### 2. VGG16
Input shape: bx3x224x224
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   3.20       |  2.71       |
| 2          |   4.97       |  3.78       |
| 4          |   8.20       |  6.09       |
| 8          |   14.64      |  10.20      |
| 16         |   27.47      |  19.17      |
| 32         |   53.09      |  36.47      |

Model source: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py  

### 3. MobileNetV2
Input shape: bx3x224x224
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   3.85       |  0.65       |
| 2          |   3.75       |  0.86       |
| 4          |   3.90       |  1.19       |
| 8          |   4.18       |  2.08       |
| 16         |   8.43       |  3.83       |
| 32         |   16.57      |  7.14       |

Model source: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py  

### 4. InceptionV3
Input shape: bx3x224x224
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   10.05      |  2.51       |
| 2          |   10.13      |  3.22       |
| 4          |   10.08      |  3.70       |
| 8          |   10.15      |  4.95       |
| 16         |   12.51      |  7.11       |
| 32         |   21.43      |  11.22      |

Model source: https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py  

### 5. Efficientnet_b0
Input shape: bx3x224x224
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   8.28       |  1.28       |
| 2          |   8.50       |  1.57       |
| 4          |   8.49       |  2.29       |
| 8          |   8.83       |  3.65       |
| 16         |   10.65      |  6.62       |
| 32         |   20.51      |  12.51      |

Model source: https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/efficientnet.py  

### 6. Bert-base-uncased
Input shape: bx128
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   6.40       |  2.02       |
| 2          |   7.14       |  2.59       |
| 4          |   11.58      |  4.39       |
| 8          |   21.64      |  8.41       |
| 16         |   44.20      |  16.90      |
| 32         |   92.69      |  32.21      |

Model source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py  

### 7. Vision Transformer (ViT)
Input shape: bx3x224x224
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   6.38       |  3.07       |
| 2          |   10.35      |  4.57       |
| 4          |   19.06      |  8.37       |
| 8          |   36.71      |  16.34      |
| 16         |   73.84      |  29.92      |
| 32         |   147.70     |  58.11      |

Model source: https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py 

### 8. YOLOv5s
Input shape: bx3x640x640
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   6.17       |  2.22       |
| 2          |   5.93       |  3.96       |
| 4          |   10.02      |  6.84       |
| 8          |   20.02      |  12.86      |
| 16         |   38.17      |  24.80      |
| 32         |   77.19      |  49.16      |

Model source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py  

### 9. Swin Transformer
Input shape: bx3x224x224
| Batch size | PyTorch (ms) |  Poros (ms) |
|------------|--------------|-------------|
| 1          |   14.11      |  7.68       |
| 2          |   22.73      |  11.99      |
| 4          |   42.21      |  21.74      |
| 8          |   83.07      |  42.18      |
| 16         |   162.34     |  78.34      |
| 32         |   317.43     |  149.72     |

Model source: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py 