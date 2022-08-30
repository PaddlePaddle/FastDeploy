import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")

    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--det_use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    parser.add_argument(
        "--cls_use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    parser.add_argument(
        "--rec_use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_det_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.det_use_trt:
        option.use_trt_backend()
        #det_max_side_len 默认为960,当用户更改DET模型的max_side_len参数时，请将此参数同时更改
        det_max_side_len = 960
        option.set_trt_input_shape("x", [1, 3, 50, 50], [1, 3, 640, 640],
                                   [1, 3, det_max_side_len, det_max_side_len])

    return option


def build_cls_option(args):
    option = fd.RuntimeOption()
    option.use_paddle_backend()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.cls_use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("x", [1, 3, 32, 100])

    return option


def build_rec_option(args):
    option = fd.RuntimeOption()
    option.use_paddle_backend()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.rec_use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("x", [1, 3, 48, 10], [1, 3, 48, 320],
                                   [1, 3, 48, 2000])
    return option


args = parse_arguments()

#Det模型
det_model_file = os.path.join(args.det_model, "inference.pdmodel")
det_params_file = os.path.join(args.det_model, "inference.pdiparams")
#Cls模型
cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")
#Rec模型
rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
rec_label_file = args.rec_label_file

#默认
det_model = fd.vision.ocr.DBDetector()
cls_model = fd.vision.ocr.Classifier()
rec_model = fd.vision.ocr.Recognizer()

#模型初始化
if (len(args.det_model) != 0):
    det_runtime_option = build_det_option(args)
    det_model = fd.vision.ocr.DBDetector(
        det_model_file, det_params_file, runtime_option=det_runtime_option)

if (len(args.cls_model) != 0):
    cls_runtime_option = build_cls_option(args)
    cls_model = fd.vision.ocr.Classifier(
        cls_model_file, cls_params_file, runtime_option=cls_runtime_option)

if (len(args.rec_model) != 0):
    rec_runtime_option = build_rec_option(args)
    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file,
        rec_params_file,
        rec_label_file,
        runtime_option=rec_runtime_option)

ppocrsysv2 = fd.vision.ocr.PPOCRSystemv2(
    ocr_det=det_model._model,
    ocr_cls=cls_model._model,
    ocr_rec=rec_model._model)

# 预测图片准备
im = cv2.imread(args.image)

#预测并打印结果
result = ppocrsysv2.predict(im)
print(result)

# 可视化结果
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
