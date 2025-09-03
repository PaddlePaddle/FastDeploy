# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from fastdeploy.model_executor.ops.xpu import get_token_penalty

paddle.seed(2023)

bs = 1
length = 12
length_id = 6
pre_ids = paddle.ones([bs, length_id], dtype="int64")
logits = paddle.randn([bs, length], dtype="float16")
penalty_scores = paddle.randn([bs], dtype="float16")
# pre_ids = np.array([[0, 1, 2, 3, 4, 5]]).astype('int64')
# logits = np.random.uniform(1, 10, size=(bs, length)).astype('float32')
# penalty_scores = np.random.uniform(1, 2, size=(bs)).astype('float32')
out = get_token_penalty(pre_ids, logits, penalty_scores)
print(pre_ids)
print(logits)
print(penalty_scores)
print(out)
pre_ids = paddle.ones([bs, length_id], dtype="int64")
logits = paddle.randn([bs, length], dtype="float32")
penalty_scores = paddle.randn([bs], dtype="float32")
# pre_ids = np.array([[0, 1, 2, 3, 4, 5]]).astype('int64')
# logits = np.random.uniform(1, 10, size=(bs, length)).astype('float32')
# penalty_scores = np.random.uniform(1, 2, size=(bs)).astype('float32')
out = get_token_penalty(pre_ids, logits, penalty_scores)
print(pre_ids)
print(logits)
print(penalty_scores)
print(out)
