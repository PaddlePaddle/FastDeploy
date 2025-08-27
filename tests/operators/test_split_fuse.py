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

"""UT for set_stop_value"""
import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import get_mm_split_fuse


class TestSplitFuse(unittest.TestCase):
    def setUp(self):
        self.grid_thw = [[6, 20, 20], [6, 40, 20]]
        self.split_fuse_img_size = 16
        self.split_fuse_text_size = 384  # 1024
        self.max_seq_len = 2048
        self.image_token_id = 100295

    def split_grid(self, origin_grid_thw):
        # 划分grid_thw，该函数用于视频场景
        # origin_grid_thw = [6, 10, 12] ---> [2, 10, 12, 2, 10, 12, 2, 10, 12]
        grid_thw = []
        for t, h, w in origin_grid_thw:
            if t > 2:
                num_groups = t // 2
                remainder = t % 2
                for _ in range(num_groups):
                    grid_thw.extend([2, h, w])
                if remainder > 0:
                    grid_thw.extend([remainder, h, w])
            else:
                grid_thw.extend([t, h, w])
        return grid_thw

    def test_get_mm_split_fuse(self):
        grid_thw = self.split_grid(self.grid_thw)
        image_bs = len(grid_thw) // 3
        image_type_ids = [0] * image_bs

        # 随机拼接input_ids: [txt0+img1+tx1+img2]
        input_ids = [2] * 19
        img1 = [self.image_token_id] * 100 * 3
        txt1 = [3] * 19
        img2 = [self.image_token_id] * 200 * 3
        input_ids.extend(img1)
        input_ids.extend(txt1)
        input_ids.extend(img2)

        seq_len = len(input_ids)
        input_ids_tensor = paddle.to_tensor(input_ids, dtype="int64")
        image_type_ids_tensor = paddle.to_tensor(image_type_ids, dtype="int32")
        is_image_token = paddle.where(input_ids_tensor == self.image_token_id, 1, 0)
        image_token_sum = paddle.cumsum(is_image_token)  # 前缀和
        image_token_sum = paddle.concat([paddle.zeros([1], dtype="int64"), image_token_sum])

        grid_thw_tensor = paddle.to_tensor(grid_thw, dtype="int64")
        image_chunk_selections, split_fuse_cur_seq_lens = get_mm_split_fuse(
            input_ids_tensor.cpu(),
            image_type_ids_tensor.cast("int32").cpu(),
            image_token_sum.cast("int32").cpu(),
            grid_thw_tensor.cpu(),
            self.image_token_id,
            image_bs,
            0,
            seq_len,
            self.split_fuse_img_size,
            self.split_fuse_text_size,
            self.max_seq_len,
        )

        # Verify the outputs are not None
        self.assertIsNotNone(image_chunk_selections)
        self.assertIsNotNone(split_fuse_cur_seq_lens)

        # Verify the shapes are as expected
        self.assertEqual(len(image_chunk_selections.shape), 1)
        self.assertEqual(len(split_fuse_cur_seq_lens.shape), 1)


if __name__ == "__main__":
    unittest.main()
