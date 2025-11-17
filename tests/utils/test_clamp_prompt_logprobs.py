"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
"""

import unittest

from fastdeploy.utils import clamp_prompt_logprobs
from fastdeploy.worker.output import Logprob


class TestClampPromptLogprobs(unittest.TestCase):
    def test_none_input(self):
        """测试输入为None的情况"""
        result = clamp_prompt_logprobs(None)
        self.assertIsNone(result)

    def test_empty_list(self):
        """测试空列表输入"""
        result = clamp_prompt_logprobs([])
        self.assertEqual(result, [])

    def test_normal_logprobs(self):
        """测试正常的logprobs值（不包含-inf）"""
        logprob_dict = {
            1: Logprob(logprob=-2.5, rank=1, decoded_token="hello"),
            2: Logprob(logprob=-1.0, rank=2, decoded_token="world"),
        }
        prompt_logprobs = [logprob_dict]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # 原始值应该保持不变
        self.assertEqual(result[0][1].logprob, -2.5)
        self.assertEqual(result[0][2].logprob, -1.0)

    def test_negative_inf_logprobs_raises_error(self):
        """测试包含-inf的logprobs值会抛出AttributeError"""
        logprob_dict = {
            1: Logprob(logprob=float("-inf"), rank=1, decoded_token="hello"),
            2: Logprob(logprob=-1.0, rank=2, decoded_token="world"),
        }
        prompt_logprobs = [logprob_dict]

        # 由于Logprob是NamedTuple，无法修改其字段，应该抛出AttributeError
        with self.assertRaises(AttributeError) as context:
            clamp_prompt_logprobs(prompt_logprobs)

        self.assertIn("can't set attribute", str(context.exception))

    def test_multiple_negative_inf_raises_error(self):
        """测试多个-inf的logprobs值会抛出AttributeError"""
        logprob_dict = {
            1: Logprob(logprob=float("-inf"), rank=1, decoded_token="hello"),
            2: Logprob(logprob=float("-inf"), rank=2, decoded_token="world"),
            3: Logprob(logprob=-0.5, rank=3, decoded_token="test"),
        }
        prompt_logprobs = [logprob_dict]

        # 由于Logprob是NamedTuple，无法修改其字段，应该抛出AttributeError
        with self.assertRaises(AttributeError):
            clamp_prompt_logprobs(prompt_logprobs)

    def test_none_dict_in_list(self):
        """测试列表中包含None的情况"""
        prompt_logprobs = [None]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # None应该被跳过
        self.assertIsNone(result[0])

    def test_multiple_dicts_normal_values(self):
        """测试多个字典的情况（不包含-inf）"""
        logprob_dict1 = {
            1: Logprob(logprob=-2.0, rank=1, decoded_token="hello"),
        }
        logprob_dict2 = {
            2: Logprob(logprob=-2.0, rank=1, decoded_token="world"),
        }
        prompt_logprobs = [logprob_dict1, logprob_dict2]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # 应该正常返回，值保持不变
        self.assertEqual(result[0][1].logprob, -2.0)
        self.assertEqual(result[1][2].logprob, -2.0)

    def test_mixed_values_without_inf(self):
        """测试混合各种值的情况（不包含-inf）"""
        logprob_dict = {
            1: Logprob(logprob=-9999.0, rank=1, decoded_token="hello"),
            2: Logprob(logprob=-9999.0, rank=2, decoded_token="world"),
            3: Logprob(logprob=0.0, rank=3, decoded_token="test"),
            4: Logprob(logprob=-1.5, rank=4, decoded_token="again"),
        }
        prompt_logprobs = [logprob_dict]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # 所有值应该保持不变
        self.assertEqual(result[0][1].logprob, -9999.0)
        self.assertEqual(result[0][2].logprob, -9999.0)
        self.assertEqual(result[0][3].logprob, 0.0)
        self.assertEqual(result[0][4].logprob, -1.5)

    def test_return_same_object(self):
        """测试函数返回的是同一个对象（原地修改尝试）"""
        logprob_dict = {
            1: Logprob(logprob=-2.0, rank=1, decoded_token="hello"),
        }
        prompt_logprobs = [logprob_dict]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # 应该返回同一个对象（函数尝试原地修改）
        self.assertIs(result, prompt_logprobs)
        self.assertIs(result[0], prompt_logprobs[0])


if __name__ == "__main__":
    unittest.main()
