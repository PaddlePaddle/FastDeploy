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

import unittest

from fastdeploy.worker.output import LogprobsTensors


class TestLogprobsOutput(unittest.TestCase):

    def test_logprobs_output(self):
        num_positions = 3
        num_tokens_per_position = 4
        shape = [num_positions, num_tokens_per_position]
        logprobs_tensors = LogprobsTensors.empty(num_positions, num_tokens_per_position)
        assert logprobs_tensors.logprob_token_ids.shape == shape
        assert logprobs_tensors.logprobs.shape == shape
        assert logprobs_tensors.selected_token_ranks.shape == [num_positions]

        sliced_logprobs_tensors = logprobs_tensors.slice_rows(1, 2)
        assert sliced_logprobs_tensors.logprob_token_ids.shape == [1, num_tokens_per_position]
        assert sliced_logprobs_tensors.logprobs.shape == [1, num_tokens_per_position]
        assert sliced_logprobs_tensors.selected_token_ranks.shape == [1]

        logprobs_tensors_cpu = LogprobsTensors.empty_cpu(num_positions, num_tokens_per_position)
        assert logprobs_tensors_cpu.logprob_token_ids.shape == shape
        assert logprobs_tensors_cpu.logprobs.shape == shape
        assert logprobs_tensors_cpu.selected_token_ranks.shape == [num_positions]

        logprobs_list = logprobs_tensors_cpu.tolists()
        assert isinstance(logprobs_list.logprobs, list)
        assert len(logprobs_list.logprobs) == num_positions

        row_sliced_logprobs_list = logprobs_list.slice_rows(1, 2)
        assert len(row_sliced_logprobs_list.logprobs) == 1

        col_sliced_logprobs_list = logprobs_list.slice_columns(1, 2)
        assert len(col_sliced_logprobs_list.logprobs) == num_positions


if __name__ == "__main__":
    unittest.main()
