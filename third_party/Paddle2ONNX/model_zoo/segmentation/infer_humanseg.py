# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import os.path as osp

import os

import numpy as np
import cv2


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
        '--bg_image_path',
        dest='bg_image_path',
        help='Background image path for replacing. If not specified, a white background is used',
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')
    return parser.parse_args()


def get_bg_img(bg_img_path, img_shape):
    if bg_img_path is None:
        bg = 255 * np.ones(img_shape)
    elif not osp.exists(bg_img_path):
        raise Exception('The --bg_img_path is not existed: {}'.format(
            bg_img_path))
    else:
        bg = cv2.imread(bg_img_path)
    return bg


def preprocess(ori_img):
    def normalize(im, mean, std):
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        im = np.transpose(im, (2, 0, 1))
        return im

    ori_shapes = []
    processed_imgs = []
    im = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (398, 224), interpolation=cv2.INTER_LINEAR)
    data = normalize(im, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    processed_imgs.append(data)
    ori_shapes.append(ori_img.shape)

    processed_imgs = np.array(processed_imgs)
    return processed_imgs, ori_shapes


def postprocess(pred, img, ori_shape):
    score_map = pred[:, 1, :, :]
    score_map = np.transpose(score_map, [1, 2, 0])
    score_map = cv2.resize(
        score_map, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = score_map[..., np.newaxis]

    # background replace
    bg = get_bg_img(args.bg_image_path, img.shape)
    h, w, _ = img.shape
    bg = cv2.resize(bg, (w, h))
    if bg.ndim == 2:
        bg = bg[..., np.newaxis]

    comb = (alpha * img + (1 - alpha) * bg).astype(np.uint8)
    return comb


def save_imgs(results, imgs_path, prefix=None):
    basename = os.path.basename(imgs_path)
    basename, _ = os.path.splitext(basename)
    if prefix is not None and isinstance(prefix, str):
        basename = prefix + "_" + basename
    basename = f'{basename}.png'
    cv2.imwrite(os.path.join(args.save_dir, basename), results)


def paddle_predict(model_path, imgs_path):
    import paddle
    model = paddle.jit.load(model_path)
    model.eval()
    img = cv2.imread(imgs_path)
    data, ori_shapes = preprocess(img)
    output = model(data).numpy()
    results = postprocess(output, img, ori_shapes[0])
    return results


def onnx_predict(onnx_path, imgs_path):
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    img = cv2.imread(imgs_path)
    data, ori_shapes = preprocess(img)
    output = sess.run(None, {sess.get_inputs()[0].name: data})[0]
    results = postprocess(output, img, ori_shapes[0])
    return results


if __name__ == "__main__":
    args = parse_args()
    imgs_path = args.image_path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 图像背景替换
    if args.use_paddle_predict:
        paddle_result = paddle_predict(args.model_path, imgs_path)
        save_imgs(paddle_result, imgs_path, "paddle")
    else:
        onnx_result = onnx_predict(args.model_path, imgs_path)
        save_imgs(onnx_result, imgs_path, "onnx")
    print("Finish")
