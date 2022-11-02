# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleSeg model.")
    parser.add_argument(
        "--video", type=str, required=True, help="Path of test video file.")
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
    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)
model_file = os.path.join(args.model, "model.pdmodel")
params_file = os.path.join(args.model, "model.pdiparams")
config_file = os.path.join(args.model, "infer_cfg.yml")
model = fd.vision.tracking.PPTracking(
    model_file, params_file, config_file, runtime_option=runtime_option)

# 初始化轨迹记录器
recorder = fd.vision.tracking.TrailRecorder()
# 绑定记录器 注意：每次预测时，往trail_recorder里面插入数据，随着预测次数的增加，内存会不断地增长，
# 可以通过unbind_recorder()方法来解除绑定
model.bind_recorder(recorder)
# 预测图片分割结果
cap = cv2.VideoCapture(args.video)
# count = 0
while True:
    _, frame = cap.read()
    if frame is None:
        break
    result = model.predict(frame)
    # count += 1
    # if count == 10:
    #     model.unbind_recorder()
    img = fd.vision.vis_mot(frame, result, 0.0, recorder)
    cv2.imshow("video", img)
    if cv2.waitKey(30) == ord("q"):
        break
model.unbind_recorder()
cap.release()
cv2.destroyAllWindows()
