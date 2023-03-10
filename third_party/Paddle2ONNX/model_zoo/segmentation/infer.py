# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import cv2
import numpy as np
import paddle
import onnxruntime as rt


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        help="while use_paddle_predict, this means directory path of paddle model. Other wise, this means path of "
        "onnx model file.")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        type=str,
        help='The directory or path or file list of the images to be predicted.')
    parser.add_argument(
        '--use_paddle_predict',
        type=bool,
        default=False,
        help="If use paddlepaddle to predict, otherwise use onnxruntime to predict."
    )
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./outputs')
    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    return parser.parse_args()


def paddle_predict(model_path, imgs_path):
    # run paddle inference
    model = paddle.jit.load(model_path)
    model.eval()
    data = preprocess(imgs_path)
    results = model(data).numpy()
    results = postprocess(results)
    return results


def onnx_predict(onnx_path, imgs_path):
    # run onnxruntime
    sess = rt.InferenceSession(onnx_path)
    data = preprocess(imgs_path)
    results = sess.run(None, {sess.get_inputs()[0].name: data})[0]
    results = postprocess(results)
    return results


def preprocess(image_path):
    def normalize(im, mean, std):
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        im = np.transpose(im, (2, 0, 1))
        return im

    im = cv2.imread(image_path).astype('float32')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    data = normalize(im, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    data = np.expand_dims(data, axis=0)
    return data


def postprocess(results):
    if FLAGS.with_argmax:
        results = np.argmax(results, axis=1)
    return results


def save_imgs(results, imgs_path, prefix=None):
    for i in range(results.shape[0]):
        result = get_pseudo_color_map(results[i])
        basename = os.path.basename(imgs_path)
        basename, _ = os.path.splitext(basename)
        if prefix is not None and isinstance(prefix, str):
            basename = prefix + "_" + basename
        basename = f'{basename}.png'
        result.save(os.path.join(FLAGS.save_dir, basename))


def get_pseudo_color_map(pred, color_map=None):
    from PIL import Image as PILImage
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


if __name__ == '__main__':
    FLAGS = parse_args()
    imgs_path = FLAGS.image_path
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    if FLAGS.use_paddle_predict:
        paddle_result = paddle_predict(FLAGS.model_path, imgs_path)
        save_imgs(paddle_result, imgs_path, "paddle")
    else:
        onnx_result = onnx_predict(FLAGS.model_path, imgs_path)
        save_imgs(onnx_result, imgs_path, "onnx")
    print("Finish")
