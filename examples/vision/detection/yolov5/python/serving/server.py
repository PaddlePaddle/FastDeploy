import fastdeploy as fd
from fastdeploy.serving.server import SimpleServer
import os
import logging

logging.getLogger().setLevel(logging.INFO)

# Configurations
model_dir = 'yolov5s_infer'
device = 'cpu'
use_trt = False

# Prepare model
model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")

# Setup runtime option to select hardware, backend, etc.
option = fd.RuntimeOption()
if device.lower() == 'gpu':
    option.use_gpu()
if use_trt:
    option.use_trt_backend()
    option.set_trt_input_shape("images", [1, 3, 640, 640])
    option.set_trt_cache_file('yolov5s.trt')

# Create model instance
model_instance = fd.vision.detection.YOLOv5(
    model_file,
    params_file,
    runtime_option=option,
    model_format=fd.ModelFormat.PADDLE)

# Create server, setup REST API
app = SimpleServer()
app.register(
    task_name="fd/yolov5s",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=model_instance)
