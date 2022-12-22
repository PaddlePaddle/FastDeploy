import fastdeploy as fd
import os
import logging

logging.getLogger().setLevel(logging.INFO)

app = fd.serving.SimpleServer()

model_dir = fd.download_model(name='ppyoloe_crn_l_300e_coco')
model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

option = fd.RuntimeOption()
option.use_gpu()
option.use_trt_backend()
option.set_trt_cache_file('ppyoloe.trt')

app.register(
    task_name="fd/ppyoloe",
    model_handler=fd.serving.handler.VisionModelHandler,
    model_name="fd.vision.detection.PPYOLOE",
    model_file=model_file,
    params_file=params_file,
    config_file=config_file,
    runtime_option=option)
