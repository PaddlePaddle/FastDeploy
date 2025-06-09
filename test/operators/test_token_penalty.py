# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

""" UT for get_token_penalty """
import paddle
import numpy as np
from fastdeploy.model_executor.ops.gpu import get_token_penalty_once

paddle.seed(2023)

pre_ids = paddle.randint(0, 10000, (8, 1000))
pre_ids[:, -1] = pre_ids[:, -2]
print(pre_ids)
logits = paddle.rand(shape=[8, 10000], dtype="float16")
penalty_scores = np.array([1.2] * 8).astype(np.float16).reshape(-1, 1)
penalty_scores = paddle.to_tensor(penalty_scores)

print("logits[0][pre_ids[0]]: ", logits[0][pre_ids[0]])
res = get_token_penalty_once(pre_ids, logits, penalty_scores)
for i in range(8):
    print("res[{}]:{}".format(i, res[i][pre_ids[i]]))


input_ids = pre_ids
score = paddle.index_sample(logits, input_ids)
score = paddle.where(score < 0, score * penalty_scores, score / penalty_scores)

bsz = paddle.shape(logits)[
    0
]  # TODO: Bsz as input for inference with dynamic batch_size
bsz_range = paddle.arange(
    start=bsz * 0, end=bsz, step=bsz / bsz, name="bsz_range", dtype="int64"
).unsqueeze(-1)
input_ids = input_ids + bsz_range * logits.shape[-1]
res2 = paddle.scatter(logits.flatten(), input_ids.flatten(), score.flatten()).reshape(
    logits.shape
)
print("-------------------------------------------")
for i in range(8):
    print(res2[i][pre_ids[i]])

print("res_sub:")
for i in range(8):
    print(res2[i][pre_ids[i]] - res[i][pre_ids[i]])

print((res.numpy() - res2.numpy()).sum())
