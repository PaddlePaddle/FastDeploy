# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
from paddle2onnx.command import program2onnx
import onnxruntime as rt

paddle.enable_static()

import numpy as np
import paddle
import paddle.fluid as fluid
from onnxbase import randtool, compare


def anchor_generator_in_python(input_feat, anchor_sizes, aspect_ratios,
                               variances, stride, offset):
    num_anchors = len(aspect_ratios) * len(anchor_sizes)
    layer_h = input_feat.shape[2]
    layer_w = input_feat.shape[3]
    out_dim = (layer_h, layer_w, num_anchors, 4)
    out_anchors = np.zeros(out_dim).astype('float32')

    for h_idx in range(layer_h):
        for w_idx in range(layer_w):
            x_ctr = (w_idx * stride[0]) + offset * (stride[0] - 1)
            y_ctr = (h_idx * stride[1]) + offset * (stride[1] - 1)
            idx = 0
            for r in range(len(aspect_ratios)):
                ar = aspect_ratios[r]
                for s in range(len(anchor_sizes)):
                    anchor_size = anchor_sizes[s]
                    area = stride[0] * stride[1]
                    area_ratios = area / ar
                    base_w = np.round(np.sqrt(area_ratios))
                    base_h = np.round(base_w * ar)
                    scale_w = anchor_size / stride[0]
                    scale_h = anchor_size / stride[1]
                    w = scale_w * base_w
                    h = scale_h * base_h
                    out_anchors[h_idx, w_idx, idx, :] = [
                        (x_ctr - 0.5 * (w - 1)), (y_ctr - 0.5 * (h - 1)),
                        (x_ctr + 0.5 * (w - 1)), (y_ctr + 0.5 * (h - 1))
                    ]
                    idx += 1

    # set the variance.
    out_var = np.tile(variances, (layer_h, layer_w, num_anchors, 1))
    out_anchors = out_anchors.astype('float32')
    out_var = out_var.astype('float32')
    return out_anchors, out_var


def test_generate_proposals():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        scores = fluid.layers.data(
            name='scores',
            shape=[-1, 4, 16, 16],
            append_batch_size=False,
            dtype='float32')
        bbox_deltas = fluid.layers.data(
            name='bbox_deltas',
            shape=[-1, 16, 16, 16],
            append_batch_size=False,
            dtype='float32')
        im_info = fluid.layers.data(
            name='im_info',
            shape=[-1, 3],
            append_batch_size=False,
            dtype='float32')
        anchors = fluid.layers.data(
            name='anchors',
            shape=[16, 16, 4, 4],
            append_batch_size=False,
            dtype='float32')
        variances = fluid.layers.data(
            name='variances',
            shape=[16, 16, 4, 4],
            append_batch_size=False,
            dtype='float32')

        def init_test_input():
            batch_size = 1
            input_channels = 20
            layer_h = 16
            layer_w = 16
            input_feat = np.random.random((batch_size, input_channels, layer_h,
                                           layer_w)).astype('float32')
            # from test_anchor_generator_op import anchor_generator_in_python
            anchors, variances = anchor_generator_in_python(
                input_feat=input_feat,
                anchor_sizes=[16., 32.],
                aspect_ratios=[0.5, 1.0],
                variances=[1.0, 1.0, 1.0, 1.0],
                stride=[16.0, 16.0],
                offset=0.5)
            im_shape = np.array([[64, 64]]).astype('float32')
            num_anchors = anchors.shape[2]
            scores = np.random.random(
                (batch_size, num_anchors, layer_h, layer_w)).astype('float32')
            bbox_deltas = np.random.random((batch_size, num_anchors * 4,
                                            layer_h, layer_w)).astype('float32')
            im_info = np.array([[64., 64., 4]]).astype(
                'float32')  # im_height, im_width, scale
            return scores, bbox_deltas, im_shape, im_info, anchors, variances

        scores_data, bbox_deltas_data, im_shape_data, im_info_data, anchors_data, variances_data = init_test_input(
        )

        pre_nms_topN = 12000
        post_nms_topN = 5000
        nms_thresh = 0.7
        min_size = 3.0
        eta = 1.
        return_rois_num = False
        out = fluid.layers.generate_proposals(
            scores,
            bbox_deltas,
            im_info,
            anchors,
            variances,
            pre_nms_top_n=pre_nms_topN,
            post_nms_top_n=post_nms_topN,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta,
            return_rois_num=return_rois_num)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        result = exe.run(feed={
            "scores": scores_data,
            "bbox_deltas": bbox_deltas_data,
            "im_info": im_info_data,
            "anchors": anchors_data,
            "variances": variances_data
        },
                         fetch_list=[out],
                         return_numpy=False)

        path_prefix = "./generate_proposals"
        fluid.io.save_inference_model(
            path_prefix,
            ["scores", "bbox_deltas", "im_info", "anchors", "variances"], out,
            exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=12,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider'])
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        input_name3 = sess.get_inputs()[2].name
        input_name4 = sess.get_inputs()[3].name
        input_name5 = sess.get_inputs()[4].name
        pred_onnx = sess.run(None, {
            input_name1: scores_data,
            input_name2: bbox_deltas_data,
            input_name3: im_info_data,
            input_name4: anchors_data,
            input_name5: variances_data
        })
        compare(pred_onnx, result, 1e-5, 1e-5)


if __name__ == "__main__":
    test_generate_proposals()
