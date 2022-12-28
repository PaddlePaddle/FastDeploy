import fastdeploy as fd
from fastdeploy.serving.server import SimpleServer
import os
import logging

logging.getLogger().setLevel(logging.INFO)

# Configurations
model_dir = 'ppyoloe_crn_l_300e_coco'
device = 'cpu'
use_trt = False

# Prepare model
model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

# Setup runtime option to select hardware, backend, etc.
option = fd.RuntimeOption()
if device.lower() == 'gpu':
    option.use_gpu()
if use_trt:
    option.use_trt_backend()
    option.set_trt_cache_file('ppyoloe.trt')

# Create model instance
model_instance = fd.vision.detection.PPYOLOE(
    model_file=model_file,
    params_file=params_file,
    config_file=config_file,
    runtime_option=option)

# Create server, setup REST API
app = SimpleServer()
app.register(
    task_name="fd/ppyoloe",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=model_instance)
