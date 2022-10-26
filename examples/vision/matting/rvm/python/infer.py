import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of RobustVideoMatting model.")
    parser.add_argument("--image", type=str, help="Path of test image file.")
    parser.add_argument("--video", type=str, help="Path of test video file.")
    parser.add_argument(
        "--bg",
        type=str,
        required=True,
        default=None,
        help="Path of test background image file.")
    parser.add_argument(
        '--output-composition',
        type=str,
        default="composition.mp4",
        help="Path of composition video file.")
    parser.add_argument(
        '--output-alpha',
        type=str,
        default="alpha.mp4",
        help="Path of alpha video file.")
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
        option.set_trt_input_shape("src", [1, 3, 1920, 1080])
        option.set_trt_input_shape("r1i", [1, 1, 1, 1], [1, 16, 240, 135],
                                   [1, 16, 240, 135])
        option.set_trt_input_shape("r2i", [1, 1, 1, 1], [1, 20, 120, 68],
                                   [1, 20, 120, 68])
        option.set_trt_input_shape("r3i", [1, 1, 1, 1], [1, 40, 60, 34],
                                   [1, 40, 60, 34])
        option.set_trt_input_shape("r4i", [1, 1, 1, 1], [1, 64, 30, 17],
                                   [1, 64, 30, 17])

    return option


args = parse_arguments()
output_composition = args.output_composition
output_alpha = args.output_alpha

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.matting.RobustVideoMatting(
    args.model, runtime_option=runtime_option)
bg = cv2.imread(args.bg)

if args.video is not None:
    # for video
    cap = cv2.VideoCapture(args.video)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    composition = cv2.VideoWriter(output_composition, fourcc, 20.0,
                                  (1080, 1920))
    alpha = cv2.VideoWriter(output_alpha, fourcc, 20.0, (1080, 1920))

    frame_id = 0
    while True:
        frame_id = frame_id + 1
        _, frame = cap.read()
        if frame is None:
            break
        result = model.predict(frame)
        vis_im = fd.vision.vis_matting(frame, result)
        vis_im_with_bg = fd.vision.swap_background_matting(frame, bg, result)
        alpha.write(vis_im)
        composition.write(vis_im_with_bg)
        cv2.waitKey(30)
    cap.release()
    composition.release()
    alpha.release()
    cv2.destroyAllWindows()
    print("Visualized result video save in {} and {}".format(
        output_composition, output_alpha))

if args.image is not None:
    # for image
    im = cv2.imread(args.image)
    result = model.predict(im.copy())
    print(result)
    # 可视化结果
    vis_im = fd.vision.vis_matting(im, result)
    vis_im_with_bg = fd.vision.swap_background_matting(im, bg, result)
    cv2.imwrite("visualized_result_fg.jpg", vis_im)
    cv2.imwrite("visualized_result_replaced_bg.jpg", vis_im_with_bg)
    print(
        "Visualized result save in ./visualized_result_replaced_bg.jpg and ./visualized_result_fg.jpg"
    )
