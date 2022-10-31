import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path of PFLD model.")
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
        help="inference backend, ort, ov, trt, paddle.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    device = args.device
    backend = args.backend
    if device == "gpu":
        option.use_gpu()

    if backend == "trt":
        assert device == "gpu", "the trt backend need device==gpu"
        option.use_trt_backend()
        option.set_trt_input_shape("input", [1, 3, 112, 112])
    elif backend == "ov":
        assert device == "cpu", "the openvino backend need device==cpu"
        option.use_openvino_backend()
    elif backend == "paddle":
        option.use_paddle_backend()
    elif backend == "ort":
        option.use_ort_backend()
    else:
        print("%s is an unsupported backend" % backend)

    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.facealign.PFLD(args.model, runtime_option=runtime_option)

# for image
im = cv2.imread(args.image)
result = model.predict(im.copy())
print(result)
# 可视化结果
vis_im = fd.vision.vis_face_alignment(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
