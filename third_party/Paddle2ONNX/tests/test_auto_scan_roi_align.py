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

import paddle

from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype


@paddle.jit.not_to_static
def roi_align(input,
              rois,
              output_size,
              spatial_scale=1.0,
              sampling_ratio=-1,
              rois_num=None,
              aligned=True,
              name=None):
    """

    Region of interest align (also known as RoI align) is to perform
    bilinear interpolation on inputs of nonuniform sizes to obtain
    fixed-size feature maps (e.g. 7*7)

    Dividing each region proposal into equal-sized sections with
    the pooled_width and pooled_height. Location remains the origin
    result.

    In each ROI bin, the value of the four regularly sampled locations
    are computed directly through bilinear interpolation. The output is
    the mean of four locations.
    Thus avoid the misaligned problem.

    Args:
        input (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W],
            where N is the batch size, C is the input channel, H is Height, W is weight.
            The data type is float32 or float64.
        rois (Tensor): ROIs (Regions of Interest) to pool over.It should be
            a 2-D Tensor or 2-D LoDTensor of shape (num_rois, 4), the lod level is 1.
            The data type is float32 or float64. Given as [[x1, y1, x2, y2], ...],
            (x1, y1) is the top left coordinates, and (x2, y2) is the bottom right coordinates.
        output_size (int or tuple[int, int]): The pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float32, optional): Multiplicative spatial scale factor to translate ROI coords
            from their input scale to the scale used when pooling. Default: 1.0
        sampling_ratio(int32, optional): number of sampling points in the interpolation grid.
            If <=0, then grid points are adaptive to roi_width and pooled_w, likewise for height. Default: -1
        rois_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor:

        Output: The output of ROIAlignOp is a 4-D tensor with shape (num_rois, channels, pooled_h, pooled_w). The data type is float32 or float64.


    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()

            x = paddle.static.data(
                name='data', shape=[None, 256, 32, 32], dtype='float32')
            rois = paddle.static.data(
                name='rois', shape=[None, 4], dtype='float32')
            rois_num = paddle.static.data(name='rois_num', shape=[None], dtype='int32')
            align_out = ops.roi_align(input=x,
                                               rois=rois,
                                               ouput_size=(7, 7),
                                               spatial_scale=0.5,
                                               sampling_ratio=-1,
                                               rois_num=rois_num)
    """
    check_type(output_size, 'output_size', (int, tuple), 'roi_align')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        align_out = core.ops.roi_align(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale,
            "sampling_ratio", sampling_ratio, "aligned", aligned)
        return align_out

    else:
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'roi_align')
        check_variable_and_dtype(rois, 'rois', ['float32', 'float64'],
                                 'roi_align')
        helper = LayerHelper('roi_align', **locals())
        dtype = helper.input_dtype()
        align_out = helper.create_variable_for_type_inference(dtype)
        inputs = {
            "X": input,
            "ROIs": rois,
        }
        if rois_num is not None:
            inputs['RoisNum'] = rois_num
        helper.append_op(
            type="roi_align",
            inputs=inputs,
            outputs={"Out": align_out},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale,
                "sampling_ratio": sampling_ratio,
                "aligned": aligned,
            })
        return align_out


def test_generate_roi_align_lod_aligned_true():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        batch_size = 1
        channels = 256
        height = 200
        width = 320

        spatial_scale = 1.0 / 2.0
        pooled_height = 2
        pooled_width = 2
        sampling_ratio = 0
        aligned = False

        input = fluid.layers.data(
            name='input',
            shape=[-1, 256, -1, -1],
            append_batch_size=False,
            dtype='float32',
            lod_level=1)
        roi = fluid.layers.data(
            name='roi',
            shape=[-1, 4],
            append_batch_size=False,
            dtype='float32',
            lod_level=1)

        def init_test_input():
            # n, c, h, w
            x_dim = (batch_size, channels, height, width)
            x = np.random.random(x_dim).astype('float32')

            rois = []
            rois_lod = [[]]
            for bno in range(batch_size):
                rois_lod[0].append(bno + 1)
                for i in range(bno + 1):
                    x1 = np.random.random_integers(
                        0, width // spatial_scale - pooled_width)
                    y1 = np.random.random_integers(
                        0, height // spatial_scale - pooled_height)

                    x2 = np.random.random_integers(x1 + pooled_width,
                                                   width // spatial_scale)
                    y2 = np.random.random_integers(y1 + pooled_height,
                                                   height // spatial_scale)

                    roi = [bno, x1, y1, x2, y2]
                    rois.append(roi)
            rois_num = len(rois)
            rois = np.array(rois).astype("float32")
            return x, rois[:, 1:5]

        input_data, roi_data = init_test_input()

        out = roi_align(
            input,
            roi,
            output_size=2,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            rois_num=None,
            aligned=aligned)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        input_val = fluid.create_lod_tensor(input_data, [[1]], fluid.CPUPlace())
        roi_val = fluid.create_lod_tensor(roi_data, [[1]], fluid.CPUPlace())
        result = exe.run(feed={"input": input_val,
                               "roi": roi_val},
                         fetch_list=[out],
                         return_numpy=False)

        path_prefix = "./roi_align_v2"
        fluid.io.save_inference_model(path_prefix, ["input", "roi"], [out], exe)

        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=12,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        pred_onnx = sess.run(None, {
            input_name1: input_data,
            input_name2: roi_data,
        })
        compare(pred_onnx, result, 1e-5, 1e-5)
        print("Finish!!!")


def test_generate_roi_align_lod_aligned_false():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        batch_size = 1
        channels = 256
        height = 200
        width = 320

        spatial_scale = 1.0 / 2.0
        pooled_height = 2
        pooled_width = 2
        sampling_ratio = 0
        aligned = False

        input = fluid.layers.data(
            name='input',
            shape=[-1, 256, -1, -1],
            append_batch_size=False,
            dtype='float32',
            lod_level=1)
        roi = fluid.layers.data(
            name='roi',
            shape=[-1, 4],
            append_batch_size=False,
            dtype='float32',
            lod_level=1)

        def init_test_input():
            # n, c, h, w
            x_dim = (batch_size, channels, height, width)
            x = np.random.random(x_dim).astype('float32')

            rois = []
            rois_lod = [[]]
            for bno in range(batch_size):
                rois_lod[0].append(bno + 1)
                for i in range(bno + 1):
                    x1 = np.random.random_integers(
                        0, width // spatial_scale - pooled_width)
                    y1 = np.random.random_integers(
                        0, height // spatial_scale - pooled_height)

                    x2 = np.random.random_integers(x1 + pooled_width,
                                                   width // spatial_scale)
                    y2 = np.random.random_integers(y1 + pooled_height,
                                                   height // spatial_scale)

                    roi = [bno, x1, y1, x2, y2]
                    rois.append(roi)
            rois_num = len(rois)
            rois = np.array(rois).astype("float32")
            return x, rois[:, 1:5]

        input_data, roi_data = init_test_input()

        out = paddle.fluid.layers.roi_align(
            input,
            roi,
            pooled_height=pooled_height,
            pooled_width=pooled_width,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        input_val = fluid.create_lod_tensor(input_data, [[1]], fluid.CPUPlace())
        roi_val = fluid.create_lod_tensor(roi_data, [[1]], fluid.CPUPlace())
        result = exe.run(feed={"input": input_val,
                               "roi": roi_val},
                         fetch_list=[out],
                         return_numpy=False)

        path_prefix = "./roi_align"
        fluid.io.save_inference_model(path_prefix, ["input", "roi"], [out], exe)

        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=12,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        pred_onnx = sess.run(None, {
            input_name1: input_data,
            input_name2: roi_data,
        })
        compare(pred_onnx, result, 1e-5, 1e-5)
        print("Finish!!!")


if __name__ == "__main__":
    # test_generate_roi_align_lod_aligned_true()
    test_generate_roi_align_lod_aligned_false()
