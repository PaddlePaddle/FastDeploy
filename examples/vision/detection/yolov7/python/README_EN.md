English | [简体中文](README.md)

# YOLOv7 Python Deployment Demo

Two steps before deployment:

- 1. The hardware and software environment meets the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/docs_en/environment.md)
- 2. Install FastDeploy Python whl package. Please refer to [FastDeploy Python Installation](../../../../../docs/docs_en/quick_start)


This doc provides a quick `infer.py` demo of YOLOv7 deployment on CPU/GPU, and accelerated GPU deployment by TensorRT. Run the following command:

```bash
# Download sample deployment code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/yolov7/python/

# Download yolov7 model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU Inference
python infer.py --model yolov7.onnx --image 000000014439.jpg --device cpu
# GPU 
python infer.py --model yolov7.onnx --image 000000014439.jpg --device gpu
# GPU上使用TensorRT推理
python infer.py --model yolov7.onnx --image 000000014439.jpg --device gpu --use_trt True
```

The visualisation of the results is as follows.

<img width="640" src="https://user-images.githubusercontent.com/67993288/183847558-abcd9a57-9cd9-4891-b09a-710963c99b74.jpg">

## YOLOv7 Python Interface

```python
fastdeploy.vision.detection.YOLOv7(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOv7 model loading and initialisation, with model_file being the exported ONNX model format.

**Parameters**

> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path. If the model format is ONNX, the parameter can be filled with an empty string.
> * **runtime_option**(RuntimeOption): Back-end inference configuration. The default is None, i.e. the default is applied
> * **model_format**(ModelFormat): Model format. The default is ONNX format

### Predict Function

> ```python
> YOLOv7.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
> 
> Model prediction interface with direct output of detection results from the image input.
> 
> **Parameters**
> 
> > * **image_data**(np.ndarray): Input image. Images need to be in HWC or BGR format
> > * **conf_threshold**(float): Filter threshold for detection box confidence
> > * **nms_iou_threshold**(float): iou thresholds during NMS processing

> **Return**
> 
> > Return to`fastdeploy.vision.DetectionResult`Struct. For more details, please refer to [Vision Model Results](../../../../../docs/docs_en/api/vision_results/)

### Class Member Variables

#### Pre-processing parameters

Users can modify the following pre-processing parameters for their needs. This will affect the final reasoning and deployment results

> > * **size**(list[int]):  This parameter modifies the 'resize' during preprocessing and contains two integer elements representing [width, height]. The default value is [640, 640].
> > * **padding_value**(list[float]): This parameter modifies the value of the padding when resizing the image. It contains three floating-point elements, representing the values of the three channels. The default value is [114, 114, 114].
> > * **is_no_pad**(bool): This parameter determines whether the image is resized by padding, `is_no_pad=ture` means no padding is used. The default value is `is_no_pad=false`.
> > * **is_mini_pad**(bool): This parameter allows the width and height of the image after resize to be the closest value to the `size` member variable, which the pixel size of the padding can be divided by the `stride` member variable. The default value is `is_mini_pad=false`.
> > * **stride**(int): Used with`stris_mini_pad` member value. The default value is`stride=32`

## Related files

- [YOLOv7 Model Introduction](..)
- [YOLOv7 C++ Deployment](../cpp)
- [Vision Model Results](../../../../../docs/docs_en/api/vision_results/)
- [how to change inference backend](../../../../../docs/docs_en/runtime/how_to_change_inference_backend.md)
