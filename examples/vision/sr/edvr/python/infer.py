import cv2
import os
import fastdeploy as fd


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path of model.")
    parser.add_argument(
        "--video", type=str, required=True, help="Path of test video file.")
    parser.add_argument("--frame_num", type=int, default=2, help="frame num")
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
        option.enable_paddle_trt_collect_shape()
        option.set_trt_input_shape("x", [1, 5, 3, 180, 320])
        option.enable_paddle_to_trt()
    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
model_file = os.path.join(args.model, "model.pdmodel")
params_file = os.path.join(args.model, "model.pdiparams")
model = fd.vision.sr.EDVR(
    model_file, params_file, runtime_option=runtime_option)

# 该处应该与你导出模型的第二个维度一致模型输入shape=[b,n,c,h,w]
capture = cv2.VideoCapture(args.video)
video_out_name = "output.mp4"
video_fps = capture.get(cv2.CAP_PROP_FPS)
video_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
# 注意导出模型时尺寸与原始输入的分辨一致比如：[1,2,3,180,320],经过4x超分后[1,3,720,1280](注意此处与PP-MSVSR不同)
# 所以导出模型相当重要
out_width = 1280
out_height = 720
print(f"fps: {video_fps}\tframe_count: {video_frame_count}")
# Create VideoWriter for output
video_out_dir = "./"
video_out_path = os.path.join(video_out_dir, video_out_name)
fucc = cv2.VideoWriter_fourcc(* "mp4v")
video_out = cv2.VideoWriter(video_out_path, fucc, video_fps,
                            (out_width, out_height), True)
if not video_out.isOpened():
    print("create video writer failed!")
# Capture all frames and do inference
frame_id = 0
imgs = []
while capture.isOpened():
    ret, frame = capture.read()
    if frame_id < args.frame_num and frame is not None:
        imgs.append(frame)
        frame_id += 1
        continue
    # 始终保持imgs队列中具有frame_num帧
    imgs.pop(0)
    imgs.append(frame)
    frame_id += 1
    # 视频读取完毕退出
    if not ret:
        break
    results = model.predict(imgs)
    for item in results:
        # cv2.imshow("13", item)
        # cv2.waitKey(30)
        video_out.write(item)
        print("Processing frame: ", frame_id)
        frame_id += 1
print("inference finished, output video saved at: ", video_out_path)
capture.release()
video_out.release()
