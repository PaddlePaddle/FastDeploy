import fastdeploy as fd
import os
import logging

logging.getLogger().setLevel(logging.INFO)

# Get arguments from envrionment variables
model_dir = os.environ.get('MODEL_DIR')
device = os.environ.get('DEVICE', 'cpu')
use_trt = os.environ.get('USE_TRT', False)

# Prepare model, download from hub or use local dir
if model_dir is None:
    model_dir = fd.download_model(name='ppyoloe_crn_l_300e_coco')

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
app = fd.serving.SimpleServer()
app.register(
    task_name="fd/ppyoloe",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=model_instance)
