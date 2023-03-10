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


def test_generate_multiclass_nms_tensor():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        N = 1
        M = 1200
        C = 21
        BOX_SIZE = 4
        background = 0
        nms_threshold = 0.3
        nms_top_k = 400
        keep_top_k = 200
        score_threshold = 0.01
        normalized = True

        boxes = fluid.layers.data(
            name='boxes',
            shape=[-1, 1200, 4],
            append_batch_size=False,
            dtype='float32')
        scores = fluid.layers.data(
            name='scores',
            shape=[-1, 21, 1200],
            append_batch_size=False,
            dtype='float32')

        def init_test_input():
            def softmax(x):
                # clip to shiftx, otherwise, when calc loss with
                # log(exp(shiftx)), may get log(0)=INF
                shiftx = (x - np.max(x)).clip(-64.)
                exps = np.exp(shiftx)
                return exps / np.sum(exps)

            scores = np.random.random((N * M, C)).astype('float32')
            scores = np.apply_along_axis(softmax, 1, scores)
            scores = np.reshape(scores, (N, M, C))
            scores = np.transpose(scores, (0, 2, 1))
            boxes = np.random.random((N, M, BOX_SIZE)).astype('float32')
            boxes[:, :, 0] = boxes[:, :, 0] * 10
            boxes[:, :, 1] = boxes[:, :, 1] * 10
            boxes[:, :, 2] = boxes[:, :, 2] * 10 + 10
            boxes[:, :, 3] = boxes[:, :, 3] * 10 + 10
            return boxes, scores

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
        result = exe.run(feed={"boxes": boxes_data,
                               "scores": scores_data},
                         fetch_list=[out],
                         return_numpy=False)

        path_prefix = "./multiclass_nms_tensor"
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
            compare(pred_onnx, result, 1e-5, 1e-5)
            print("Finish!!!")


if __name__ == "__main__":
    test_generate_multiclass_nms_tensor()
