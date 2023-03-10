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
def distribute_fpn_proposals(fpn_rois,
                             min_level,
                             max_level,
                             refer_level,
                             refer_scale,
                             pixel_offset=False,
                             rois_num=None,
                             name=None):
    r"""

    **This op only takes LoDTensor as input.** In Feature Pyramid Networks
    (FPN) models, it is needed to distribute all proposals into different FPN
    level, with respect to scale of the proposals, the referring scale and the
    referring level. Besides, to restore the order of proposals, we return an
    array which indicates the original index of rois in current proposals.
    To compute FPN level for each roi, the formula is given as follows:

    .. math::

        roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}

        level = floor(&\log(\\frac{roi\_scale}{refer\_scale}) + refer\_level)

    where BBoxArea is a function to compute the area of each roi.

    Args:

        fpn_rois(Variable): 2-D Tensor with shape [N, 4] and data type is
            float32 or float64. The input fpn_rois.
        min_level(int32): The lowest level of FPN layer where the proposals come
            from.
        max_level(int32): The highest level of FPN layer where the proposals
            come from.
        refer_level(int32): The referring level of FPN layer with specified scale.
        refer_scale(int32): The referring scale of FPN layer with specified level.
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image.
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tuple:

        multi_rois(List) : A list of 2-D LoDTensor with shape [M, 4]
        and data type of float32 and float64. The length is
        max_level-min_level+1. The proposals in each FPN level.

        restore_ind(Variable): A 2-D Tensor with shape [N, 1], N is
        the number of total rois. The data type is int32. It is
        used to restore the order of fpn_rois.

        rois_num_per_level(List): A list of 1-D Tensor and each Tensor is
        the RoIs' number in each image on the corresponding level. The shape
        is [B] and data type of int32. B is the number of images


    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            fpn_rois = paddle.static.data(
                name='data', shape=[None, 4], dtype='float32', lod_level=1)
            multi_rois, restore_ind = ops.distribute_fpn_proposals(
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224)
    """
    num_lvl = max_level - min_level + 1

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        attrs = ('min_level', min_level, 'max_level', max_level, 'refer_level',
                 refer_level, 'refer_scale', refer_scale, 'pixel_offset',
                 pixel_offset)
        multi_rois, restore_ind, rois_num_per_level = core.ops.distribute_fpn_proposals(
            fpn_rois, rois_num, num_lvl, num_lvl, *attrs)
        return multi_rois, restore_ind, rois_num_per_level

    else:
        check_variable_and_dtype(fpn_rois, 'fpn_rois', ['float32', 'float64'],
                                 'distribute_fpn_proposals')
        helper = LayerHelper('distribute_fpn_proposals', **locals())
        dtype = helper.input_dtype('fpn_rois')
        multi_rois = [
            helper.create_variable_for_type_inference(dtype)
            for i in range(num_lvl)
        ]

        restore_ind = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {'FpnRois': fpn_rois}
        outputs = {
            'MultiFpnRois': multi_rois,
            'RestoreIndex': restore_ind,
        }

        if rois_num is not None:
            inputs['RoisNum'] = rois_num
            rois_num_per_level = [
                helper.create_variable_for_type_inference(dtype='int32')
                for i in range(num_lvl)
            ]
            outputs['MultiLevelRoIsNum'] = rois_num_per_level

        helper.append_op(
            type='distribute_fpn_proposals',
            inputs=inputs,
            outputs=outputs,
            attrs={
                'min_level': min_level,
                'max_level': max_level,
                'refer_level': refer_level,
                'refer_scale': refer_scale,
                'pixel_offset': pixel_offset
            })

        if rois_num is not None:
            return multi_rois, restore_ind, rois_num_per_level
        return multi_rois, restore_ind


def test_generate_proposals():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        fpn_rois = fluid.data(
            name='fpn_rois', shape=[-1, 4], dtype='float32', lod_level=1)

        def init_test_input():
            images_shape = [512, 512]
            rois_lod = [[100, 200]]
            rois = []
            lod = rois_lod[0]
            bno = 0
            for roi_num in lod:
                for i in range(roi_num):
                    xywh = np.random.rand(4)
                    xy1 = xywh[0:2] * 20
                    wh = xywh[2:4] * (images_shape - xy1)
                    xy2 = xy1 + wh
                    roi = [bno, xy1[0], xy1[1], xy2[0], xy2[1]]
                    rois.append(roi)
                bno += 1
            rois = np.array(rois).astype("float32")
            rois = rois[:, 1:5]
            return rois, [rois.shape[0]]

        fpn_rois_data, rois_num_data = init_test_input()

        min_level = 2
        max_level = 5
        refer_level = 4
        refer_scale = 224

        out = distribute_fpn_proposals(fpn_rois, min_level, max_level,
                                       refer_level, refer_scale)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        fpn_rois_val = fluid.create_lod_tensor(fpn_rois_data,
                                               [[fpn_rois_data.shape[0]]],
                                               fluid.CPUPlace())
        result = exe.run(feed={"fpn_rois": fpn_rois_val},
                         fetch_list=list(out),
                         return_numpy=False)
        path_prefix = "./distribute_fpn_proposals"
        fluid.io.save_inference_model(
            path_prefix, ["fpn_rois"],
            [out[0][0], out[0][1], out[0][2], out[0][3], out[1]], exe)

        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        pred_onnx = sess.run(None, {input_name1: fpn_rois_data})

        compare(pred_onnx, result, 1e-5, 1e-5)


def test_generate_proposalsOpWithRoisNum():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        fpn_rois = fluid.layers.data(
            name='fpn_rois',
            shape=[-1, 4],
            append_batch_size=False,
            dtype='float32')
        rois_num = fluid.layers.data(
            name='rois_num', shape=[1], append_batch_size=False, dtype='int32')

        def init_test_input():
            images_shape = [512, 512]
            rois_lod = [[100, 200]]
            rois = []
            lod = rois_lod[0]
            bno = 0
            for roi_num in lod:
                for i in range(roi_num):
                    xywh = np.random.rand(4)
                    xy1 = xywh[0:2] * 20
                    wh = xywh[2:4] * (images_shape - xy1)
                    xy2 = xy1 + wh
                    roi = [bno, xy1[0], xy1[1], xy2[0], xy2[1]]
                    rois.append(roi)
                bno += 1
            rois = np.array(rois).astype("float32")
            rois = rois[:, 1:5]
            return rois, [rois.shape[0]]

        fpn_rois_data, rois_num_data = init_test_input()

        min_level = 2
        max_level = 5
        refer_scale = 224
        refer_level = 4
        pixel_offset = True

        out = distribute_fpn_proposals(fpn_rois, min_level, max_level,
                                       refer_level, refer_scale, pixel_offset,
                                       rois_num)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        result = exe.run(
            feed={"fpn_rois": fpn_rois_data,
                  "rois_num": rois_num_data},
            fetch_list=list(out),
            return_numpy=False)
        pass
        path_prefix = "./distribute_fpn_proposals2"
        fluid.io.save_inference_model(path_prefix, ["fpn_rois", "rois_num"], [
            out[0][0], out[0][1], out[0][2], out[0][3], out[1], out[2][0],
            out[2][1], out[2][2], out[2][3]
        ], exe)

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
            input_name1: fpn_rois_data,
            input_name2: rois_num_data,
        })
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_generate_proposalsOpNoOffset():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        fpn_rois = fluid.layers.data(
            name='fpn_rois',
            shape=[-1, 4],
            append_batch_size=False,
            dtype='float32')
        rois_num = fluid.layers.data(
            name='rois_num', shape=[1], append_batch_size=False, dtype='int32')

        def init_test_input():
            images_shape = [512, 512]
            rois_lod = [[100, 200]]
            rois = []
            lod = rois_lod[0]
            bno = 0
            for roi_num in lod:
                for i in range(roi_num):
                    xywh = np.random.rand(4)
                    xy1 = xywh[0:2] * 20
                    wh = xywh[2:4] * (images_shape - xy1)
                    xy2 = xy1 + wh
                    roi = [bno, xy1[0], xy1[1], xy2[0], xy2[1]]
                    rois.append(roi)
                bno += 1
            rois = np.array(rois).astype("float32")
            rois = rois[:, 1:5]
            return rois, [rois.shape[0]]

        fpn_rois_data, rois_num_data = init_test_input()

        min_level = 2
        max_level = 5
        refer_scale = 224
        refer_level = 4
        pixel_offset = False

        out = distribute_fpn_proposals(fpn_rois, min_level, max_level,
                                       refer_level, refer_scale, pixel_offset,
                                       rois_num)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        result = exe.run(
            feed={"fpn_rois": fpn_rois_data,
                  "rois_num": rois_num_data},
            fetch_list=list(out),
            return_numpy=False)
        pass
        path_prefix = "./distribute_fpn_proposals2"
        fluid.io.save_inference_model(path_prefix, ["fpn_rois", "rois_num"], [
            out[0][0], out[0][1], out[0][2], out[0][3], out[1], out[2][0],
            out[2][1], out[2][2], out[2][3]
        ], exe)

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
            input_name1: fpn_rois_data,
            input_name2: rois_num_data,
        })
        compare(pred_onnx, result, 1e-5, 1e-5)


if __name__ == "__main__":
    test_generate_proposals()
    test_generate_proposalsOpWithRoisNum()
    test_generate_proposalsOpNoOffset()
