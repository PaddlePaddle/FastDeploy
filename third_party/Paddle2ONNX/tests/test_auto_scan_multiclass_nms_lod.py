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


def test_generate_multiclass_nms_lod():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        M = 1000
        C = 80
        BOX_SIZE = 4
        background = 80
        nms_threshold = 0.5
        nms_top_k = -1
        keep_top_k = 100
        score_threshold = 0.01
        normalized = True

        boxes = fluid.layers.data(
            name='boxes',
            shape=[-1, C, 4],
            append_batch_size=False,
            dtype='float32',
            lod_level=1)
        scores = fluid.layers.data(
            name='scores',
            shape=[-1, C],
            append_batch_size=False,
            dtype='float32',
            lod_level=1)

        def init_test_input():
            def softmax(x):
                # clip to shiftx, otherwise, when calc loss with
                # log(exp(shiftx)), may get log(0)=INF
                shiftx = (x - np.max(x)).clip(-64.)
                exps = np.exp(shiftx)
                return exps / np.sum(exps)

            M = 1000
            C = 80
            BOX_SIZE = 4
            scores_lod = np.random.random((M, C)).astype('float32')
            scores_lod = np.apply_along_axis(softmax, 1, scores_lod)
            boxes_lod = np.random.random((M, C, BOX_SIZE)).astype('float32')
            boxes_lod[:, :, 0:2] = boxes_lod[:, :, 0:2] * 10
            boxes_lod[:, :, 2:4] = boxes_lod[:, :, 2:4] * 10 + 10

            return boxes_lod, scores_lod

        boxes_data, scores_data = init_test_input()
        out = paddle.fluid.layers.multiclass_nms(
            boxes,
            scores,
            score_threshold,
            nms_top_k,
            keep_top_k,
            nms_threshold=nms_threshold,
            normalized=normalized,
            nms_eta=1.0,
            background_label=background)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        boxes_val = fluid.create_lod_tensor(boxes_data, [[M]], fluid.CPUPlace())
        scores_val = fluid.create_lod_tensor(scores_data, [[M]],
                                             fluid.CPUPlace())
        result = exe.run(feed={"boxes": boxes_val,
                               "scores": scores_val},
                         fetch_list=[out],
                         return_numpy=False)

        path_prefix = "./multiclass_nms_lod"
        fluid.io.save_inference_model(path_prefix, ["boxes", "scores"], [out],
                                      exe)

        onnx_path = path_prefix + "/model.onnx"
        for opset in [10, 11, 12, 13, 14, 15]:
            program2onnx(
                model_dir=path_prefix,
                save_file=onnx_path,
                opset_version=opset,
                enable_onnx_checker=True)

            sess = rt.InferenceSession(onnx_path)
            input_name1 = sess.get_inputs()[0].name
            input_name2 = sess.get_inputs()[1].name
            pred_onnx = sess.run(None, {
                input_name1: boxes_data,
                input_name2: scores_data,
            })

            def all_sort(x):
                x1 = x.T
                y = np.split(x1, len(x1))
                z = list(reversed(y))
                index = np.lexsort(z)
                return x[index]

            pred_onnx = all_sort(pred_onnx[0])
            result = all_sort(np.array(result[0]))
            compare(pred_onnx, result, 1e-5, 1e-5)
            print("Finish!!!")


if __name__ == "__main__":
    test_generate_multiclass_nms_lod()
