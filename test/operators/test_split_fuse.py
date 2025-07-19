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
import paddle

from fastdeploy.model_executor.ops.gpu import get_mm_split_fuse

input_ids = []
image_type_ids = []
grid_thw = []


def split_grid(origin_grid_thw):
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


if __name__ == "__main__":
    grid_thw = [[6, 20, 20], [6, 40, 20]]
    grid_thw = split_grid(grid_thw)
    image_bs = len(grid_thw) // 3
    image_type_ids = [0] * image_bs
    # 随机拼接input_ids: [txt0+img1+tx1+img2]
    input_ids = [2] * 19
    img1 = [100295] * 100 * 3
    txt1 = [3] * 19
    img2 = [100295] * 200 * 3
    input_ids.extend(img1)
    input_ids.extend(txt1)
    input_ids.extend(img2)

    split_fuse_img_size = 16
    split_fuse_text_size = 384  # 1024

    seq_len = len(input_ids)
    input_ids_tensor = paddle.to_tensor(input_ids, dtype="int64")
    image_type_ids_tensor = paddle.to_tensor(image_type_ids, dtype="int32")
    is_image_token = paddle.where(input_ids_tensor == 100295, 1, 0)
    image_token_sum = paddle.cumsum(is_image_token)  # 前缀和
    image_token_sum = paddle.concat([paddle.zeros([1], dtype="int64"), image_token_sum])

    grid_thw_tensor = paddle.to_tensor(grid_thw, dtype="int64")
    image_chunk_selections, split_fuse_cur_seq_lens = get_mm_split_fuse(
        input_ids_tensor.cpu(),
        image_type_ids_tensor.cast("int32").cpu(),
        image_token_sum.cast("int32").cpu(),
        grid_thw_tensor.cpu(),
        100295,
        image_bs,
        0,
        seq_len,
        split_fuse_img_size,
        split_fuse_text_size,
        2048,
    )

    print("seq_len: ", seq_len)
    print("grid_thw", grid_thw_tensor)
    print("image_chunk_selections: ", image_chunk_selections)
    print("split_fuse_cur_seq_lens: ", split_fuse_cur_seq_lens)
