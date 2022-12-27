import fastdeploy as fd
import os
import logging

logging.getLogger().setLevel(logging.INFO)

# Get arguments from envrionment variables
det_model_dir = os.environ.get('DET_MODEL_DIR')
cls_model_dir = os.environ.get('CLS_MODEL_DIR')
rec_model_dir = os.environ.get('REC_MODEL_DIR')
rec_label_file = os.environ.get('REC_LABEL_FILE')
device = os.environ.get('DEVICE', 'cpu')
use_trt = os.environ.get('USE_TRT', False)

# Prepare models
# Detection model
det_model_file = os.path.join(det_model_dir, "inference.pdmodel")
det_params_file = os.path.join(det_model_dir, "inference.pdiparams")
# Classification model
cls_model_file = os.path.join(cls_model_dir, "inference.pdmodel")
cls_params_file = os.path.join(cls_model_dir, "inference.pdiparams")
# Recognition model
rec_model_file = os.path.join(rec_model_dir, "inference.pdmodel")
rec_params_file = os.path.join(rec_model_dir, "inference.pdiparams")

# Setup runtime option to select hardware, backend, etc.
option = fd.RuntimeOption()
if device.lower() == 'gpu':
    option.use_gpu()
if use_trt:
    option.use_trt_backend()

det_option = option
det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                               [1, 3, 960, 960])

# det_option.set_trt_cache_file("det_trt_cache.trt")
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

cls_batch_size = 1
rec_batch_size = 6

cls_option = option
cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [cls_batch_size, 3, 48, 320],
                               [cls_batch_size, 3, 48, 1024])

# cls_option.set_trt_cache_file("cls_trt_cache.trt")
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

rec_option = option
rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [rec_batch_size, 3, 48, 320],
                               [rec_batch_size, 3, 48, 2304])

# rec_option.set_trt_cache_file("rec_trt_cache.trt")
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

# Create PPOCRv3 pipeline
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

ppocr_v3.cls_batch_size = cls_batch_size
ppocr_v3.rec_batch_size = rec_batch_size

# Create server, setup REST API
app = fd.serving.SimpleServer()
app.register(
    task_name="fd/ppocrv3",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=ppocr_v3)
