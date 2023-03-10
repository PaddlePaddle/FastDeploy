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

from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype


@paddle.jit.not_to_static
def generate_proposals_v2(scores,
                          bbox_deltas,
                          im_shape,
                          anchors,
                          variances,
                          pre_nms_top_n=6000,
                          post_nms_top_n=1000,
                          nms_thresh=0.5,
                          min_size=0.1,
                          eta=1.0,
                          pixel_offset=False,
                          return_rois_num=False,
                          name=None):
    """
    **Generate proposal Faster-RCNN**
    This operation proposes RoIs according to each box with their
    probability to be a foreground object and
    the box can be calculated by anchors. Bbox_deltais and scores
    to be an object are the output of RPN. Final proposals
    could be used to train detection net.
    For generating proposals, this operation performs following steps:
    1. Transposes and resizes scores and bbox_deltas in size of
       (H*W*A, 1) and (H*W*A, 4)
    2. Calculate box locations as proposals candidates.
    3. Clip boxes to image
    4. Remove predicted boxes with small area.
    5. Apply NMS to get final proposals as output.
    Args:
        scores(Tensor): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas(Tensor): A 4-D Tensor with shape [N, 4*A, H, W]
            represents the difference between predicted box location and
            anchor location. The data type must be float32.
        im_shape(Tensor): A 2-D Tensor with shape [N, 2] represents H, W, the
            origin image size or input size. The data type can be float32 or
            float64.
        anchors(Tensor):   A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 4]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (xmin, ymin, xmax, ymax) format an unnormalized. The data type must be float32.
        variances(Tensor): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 4]. Each variance is in
            (xcenter, ycenter, w, h) format. The data type must be float32.
        pre_nms_top_n(float): Number of total bboxes to be kept per
            image before NMS. The data type must be float32. `6000` by default.
        post_nms_top_n(float): Number of total bboxes to be kept per
            image after NMS. The data type must be float32. `1000` by default.
        nms_thresh(float): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size(float): Remove predicted boxes with either height or
            width < min_size. The data type must be float32. `0.1` by default.
        eta(float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
            `adaptive_threshold = adaptive_threshold * eta` in each iteration.
        return_rois_num(bool): When setting True, it will return a 1D Tensor with shape [N, ] that includes Rois's
            num of each image in one batch. The N is the image's num. For example, the tensor has values [4,5] that represents
            the first image has 4 Rois, the second image has 5 Rois. It only used in rcnn model.
            'False' by default.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        tuple:
        A tuple with format ``(rpn_rois, rpn_roi_probs)``.
        - **rpn_rois**: The generated RoIs. 2-D Tensor with shape ``[N, 4]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - **rpn_roi_probs**: The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.

    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            scores = paddle.static.data(name='scores', shape=[None, 4, 5, 5], dtype='float32')
            bbox_deltas = paddle.static.data(name='bbox_deltas', shape=[None, 16, 5, 5], dtype='float32')
            im_shape = paddle.static.data(name='im_shape', shape=[None, 2], dtype='float32')
            anchors = paddle.static.data(name='anchors', shape=[None, 5, 4, 4], dtype='float32')
            variances = paddle.static.data(name='variances', shape=[None, 5, 10, 4], dtype='float32')
            rois, roi_probs = ops.generate_proposals(scores, bbox_deltas,
                         im_shape, anchors, variances)
    """
    if in_dygraph_mode():
        assert return_rois_num, "return_rois_num should be True in dygraph mode."
        attrs = ('pre_nms_topN', pre_nms_top_n, 'post_nms_topN', post_nms_top_n,
                 'nms_thresh', nms_thresh, 'min_size', min_size, 'eta', eta,
                 'pixel_offset', pixel_offset)
        rpn_rois, rpn_roi_probs, rpn_rois_num = core.ops.generate_proposals_v2(
            scores, bbox_deltas, im_shape, anchors, variances, *attrs)
        return rpn_rois, rpn_roi_probs, rpn_rois_num

    else:
        helper = LayerHelper('generate_proposals_v2', **locals())

        check_variable_and_dtype(scores, 'scores', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(bbox_deltas, 'bbox_deltas', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(im_shape, 'im_shape', ['float32', 'float64'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(anchors, 'anchors', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(variances, 'variances', ['float32'],
                                 'generate_proposals_v2')

        rpn_rois = helper.create_variable_for_type_inference(
            dtype=bbox_deltas.dtype)
        rpn_roi_probs = helper.create_variable_for_type_inference(
            dtype=scores.dtype)
        outputs = {
            'RpnRois': rpn_rois,
            'RpnRoiProbs': rpn_roi_probs,
        }
        if return_rois_num:
            rpn_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            rpn_rois_num.stop_gradient = True
            outputs['RpnRoisNum'] = rpn_rois_num

        helper.append_op(
            type="generate_proposals_v2",
            inputs={
                'Scores': scores,
                'BboxDeltas': bbox_deltas,
                'ImShape': im_shape,
                'Anchors': anchors,
                'Variances': variances
            },
            attrs={
                'pre_nms_topN': pre_nms_top_n,
                'post_nms_topN': post_nms_top_n,
                'nms_thresh': nms_thresh,
                'min_size': min_size,
                'eta': eta,
                'pixel_offset': pixel_offset
            },
            outputs=outputs)
        rpn_rois.stop_gradient = True
        rpn_roi_probs.stop_gradient = True

        if return_rois_num:
            return rpn_rois, rpn_roi_probs, rpn_rois_num
        else:
            return rpn_rois, rpn_roi_probs


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


def test_generate_proposals_v2_1():
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
        im_shape = fluid.layers.data(
            name='im_shape',
            shape=[-1, 2],
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
        pixel_offset = False
        return_rois_num = True
        out = generate_proposals_v2(
            scores,
            bbox_deltas,
            im_shape,
            anchors,
            variances,
            pre_nms_top_n=pre_nms_topN,
            post_nms_top_n=post_nms_topN,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta,
            pixel_offset=pixel_offset,
            return_rois_num=return_rois_num)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        result = exe.run(feed={
            "scores": scores_data,
            "bbox_deltas": bbox_deltas_data,
            "im_shape": im_shape_data,
            "anchors": anchors_data,
            "variances": variances_data
        },
                         fetch_list=[out],
                         return_numpy=False)

        path_prefix = "./generate_proposals_v2"
        fluid.io.save_inference_model(path_prefix, [
            "scores", "bbox_deltas", "im_shape", "anchors", "variances"
        ], out, exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=12,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        input_name3 = sess.get_inputs()[2].name
        input_name4 = sess.get_inputs()[3].name
        input_name5 = sess.get_inputs()[4].name
        pred_onnx = sess.run(None, {
            input_name1: scores_data,
            input_name2: bbox_deltas_data,
            input_name3: im_shape_data,
            input_name4: anchors_data,
            input_name5: variances_data
        })
        compare(pred_onnx, result, 1e-5, 1e-5)


def anchor_generator_in_python_num_anchors_3(
        input_feat, anchor_sizes, aspect_ratios, variances, stride, offset):
    num_anchors = 3  #len(aspect_ratios) * len(anchor_sizes)
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
                for s in range(len(anchor_sizes) - 1):
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


def test_generate_proposals_v2_2():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    N = -1
    A = 3
    H = -1
    W = -1
    with paddle.static.program_guard(main_program, startup_program):
        scores = fluid.layers.data(
            name='scores',
            shape=[-1, A, H, W],
            append_batch_size=False,
            dtype='float32')
        bbox_deltas = fluid.layers.data(
            name='bbox_deltas',
            shape=[-1, 4 * A, H, W],
            append_batch_size=False,
            dtype='float32')
        im_shape = fluid.layers.data(
            name='im_shape',
            shape=[-1, 2],
            append_batch_size=False,
            dtype='float32')
        anchors = fluid.layers.data(
            name='anchors',
            shape=[-1, 4],
            append_batch_size=False,
            dtype='float32')
        variances = fluid.layers.data(
            name='variances',
            shape=[-1, 4],
            append_batch_size=False,
            dtype='float32')

        def init_test_input():
            batch_size = 1
            input_channels = 40
            layer_h = 25
            layer_w = 40
            input_feat = np.random.random((batch_size, input_channels, layer_h,
                                           layer_w)).astype('float32')
            anchors, variances = anchor_generator_in_python_num_anchors_3(
                input_feat=input_feat,
                anchor_sizes=[16., 32.],
                aspect_ratios=[0.5, 1.0],
                variances=[1.0, 1.0, 1.0, 1.0],
                stride=[16.0, 16.0],
                offset=0.5)
            im_shape = np.array([[800, 1267.32678223]]).astype('float32')
            num_anchors = anchors.shape[2]
            scores = np.random.random(
                (batch_size, num_anchors, layer_h, layer_w)).astype('float32')
            bbox_deltas = np.random.random((batch_size, num_anchors * 4,
                                            layer_h, layer_w)).astype('float32')
            im_info = np.array([[64., 64., 4]]).astype(
                'float32')  # im_height, im_width, scale
            anchors = anchors.reshape([-1, 4])
            variances = variances.reshape([-1, 4])
            return scores, bbox_deltas, im_shape, im_info, anchors, variances

        scores_data, bbox_deltas_data, im_shape_data, im_info_data, anchors_data, variances_data = init_test_input(
        )

        pre_nms_topN = 1000
        post_nms_topN = 1000
        nms_thresh = 0.7
        min_size = 0.0
        eta = 1.
        pixel_offset = False
        return_rois_num = True
        out = generate_proposals_v2(
            scores,
            bbox_deltas,
            im_shape,
            anchors,
            variances,
            pre_nms_top_n=pre_nms_topN,
            post_nms_top_n=post_nms_topN,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta,
            pixel_offset=pixel_offset,
            return_rois_num=return_rois_num)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        result = exe.run(feed={
            "scores": scores_data,
            "bbox_deltas": bbox_deltas_data,
            "im_shape": im_shape_data,
            "anchors": anchors_data,
            "variances": variances_data
        },
                         fetch_list=[out],
                         return_numpy=False)

        path_prefix = "./generate_proposals_v2"
        fluid.io.save_inference_model(path_prefix, [
            "scores", "bbox_deltas", "im_shape", "anchors", "variances"
        ], out, exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=12,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        input_name3 = sess.get_inputs()[2].name
        input_name4 = sess.get_inputs()[3].name
        input_name5 = sess.get_inputs()[4].name
        pred_onnx = sess.run(None, {
            input_name1: scores_data,
            input_name2: bbox_deltas_data,
            input_name3: im_shape_data,
            input_name4: anchors_data,
            input_name5: variances_data
        })
        compare(pred_onnx, result, 1e-5, 1e-5)


if __name__ == "__main__":
    test_generate_proposals_v2_1()
    test_generate_proposals_v2_2()
