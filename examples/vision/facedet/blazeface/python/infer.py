import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of blazeface model dir.")
    parser.add_argument(
        "--image", required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("images", [1, 3, 640, 640])
    return option


args = parse_arguments()

model_dir = args.model

model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

# Configure runtime and load the model
runtime_option = build_option(args)
model = fd.vision.facedet.BlazeFace(model_file, params_file, config_file, runtime_option=runtime_option)

# Predict image detection results
im = cv2.imread(args.image)
result = model.predict(im)
print(result)
# Visualization of prediction Results
vis_im = fd.vision.vis_face_detection(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
