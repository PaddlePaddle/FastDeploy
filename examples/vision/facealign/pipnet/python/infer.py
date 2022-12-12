import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path of PIPNet model.")
    parser.add_argument("--image", type=str, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="ort",
        help="inference backend, ort, ov, trt, paddle, paddle_trt.")
    parser.add_argument(
        "--enable_trt_fp16",
        type=ast.literal_eval,
        default=False,
        help="whether enable fp16 in trt/paddle_trt backend")
    parser.add_argument(
        "--num_landmarks",
        type=int,
        default=19,
        help="whether enable fp16 in trt/paddle_trt backend")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    device = args.device
    backend = args.backend
    enable_trt_fp16 = args.enable_trt_fp16
    if device == "gpu":
        option.use_gpu()
        if backend == "ort":
            option.use_ort_backend()
        elif backend == "paddle":
            option.use_paddle_infer_backend()
        elif backend in ["trt", "paddle_trt"]:
            option.use_trt_backend()
            option.set_trt_input_shape("input", [1, 3, 112, 112])
            if backend == "paddle_trt":
                option.enable_paddle_to_trt()
            if enable_trt_fp16:
                option.enable_trt_fp16()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with GPU, only support default/ort/paddle/trt/paddle_trt now, {} is not supported.".
                format(backend))
    elif device == "cpu":
        if backend == "ort":
            option.use_ort_backend()
        elif backend == "ov":
            option.use_openvino_backend()
        elif backend == "paddle":
            option.use_paddle_infer_backend()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with CPU, only support default/ort/ov/paddle now, {} is not supported.".
                format(backend))
    else:
        raise Exception(
            "Only support device CPU/GPU now, {} is not supported.".format(
                device))

    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.facealign.PIPNet(args.model, runtime_option=runtime_option)
model.num_landmarks = args.num_landmarks
# for image
im = cv2.imread(args.image)
result = model.predict(im)
print(result)
# 可视化结果
vis_im = fd.vision.vis_face_alignment(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
