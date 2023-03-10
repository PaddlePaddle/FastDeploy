# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        type=str,
        help="while use_paddle_predict, this means directory path of paddle model. Other wise, this means path of onnx model file."
    )
    parser.add_argument('--image_path', type=str, help="image filename")
    parser.add_argument(
        '--use_paddle_predict',
        type=bool,
        default=False,
        help="If use paddlepaddle to predict, otherwise use onnxruntime to predict."
    )
    parser.add_argument('--crop_size', default=224, help='crop_szie')
    parser.add_argument('--resize_size', default=256, help='resize_size')
    return parser.parse_args()


def preprocess(image_path):
    """ Preprocess input image file
    Args:
        image_path(str): Path of input image file


    Returns:
        preprocessed data(np.ndarray): Shape of [N, C, H, W]
    """
    import cv2

    def resize_by_short(im, resize_size):
        short_size = min(im.shape[0], im.shape[1])
        scale = FLAGS.resize_size / short_size
        new_w = int(round(im.shape[1] * scale))
        new_h = int(round(im.shape[0] * scale))
        return cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def center_crop(im, crop_size):
        h, w, c = im.shape
        w_start = (w - crop_size) // 2
        h_start = (h - crop_size) // 2
        w_end = w_start + crop_size
        h_end = h_start + crop_size
        return im[h_start:h_end, w_start:w_end, :]

    def normalize(im, mean, std):
        im = im.astype("float32") / 255.0
        # to rgb
        im = im[:, :, ::-1]
        mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
        std = np.array(std).reshape((1, 1, 3)).astype("float32")
        return (im - mean) / std

    # resize the short edge to `resize_size`
    im = cv2.imread(image_path)
    resized_im = resize_by_short(im, FLAGS.resize_size)

    # crop from center
    croped_im = center_crop(resized_im, FLAGS.crop_size)

    # normalize
    normalized_im = normalize(croped_im, [0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])

    # transpose to NCHW
    data = np.expand_dims(normalized_im, axis=0)
    data = np.transpose(data, (0, 3, 1, 2))

    return data


def postprocess(result, topk=5):
    # choose topk index and score
    scores = result.flatten()
    topk_indices = np.argsort(-1 * scores)[:topk]
    topk_scores = scores[topk_indices]
    print("TopK Indices: ", topk_indices)
    print("TopK Scores: ", topk_scores)


def paddle_predict(paddle_model_dir, image_path):
    import paddle
    import os
    model = paddle.jit.load(os.path.join(paddle_model_dir, "inference"))

    data = preprocess(image_path)
    result = model(paddle.to_tensor(data)).numpy()
    postprocess(result)


def onnx_predict(onnx_model_path, image_path):
    import onnxruntime
    sess = onnxruntime.InferenceSession(onnx_model_path)
    data = preprocess(image_path)
    result, = sess.run(None, {"inputs": data})
    postprocess(result)


if __name__ == '__main__':
    FLAGS = parse_args()

    if FLAGS.use_paddle_predict:
        paddle_predict(FLAGS.model, FLAGS.image_path)
    else:
        onnx_predict(FLAGS.model, FLAGS.image_path)
