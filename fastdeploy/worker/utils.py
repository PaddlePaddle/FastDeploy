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
import os
from typing import List

import numpy as np
import paddle

from fastdeploy.input.mm_processor import DataProcessor
from fastdeploy.model_executor.models.ernie4_5_vl.modeling_resampler import \
    ScatterOp


def check_safetensors_model(model_dir: str):
    """
        model_dir : the directory of the model
        Check whther the model is safetensors format
    """
    model_files = list()
    all_files = os.listdir(model_dir)
    for x in all_files:
        if x.startswith("model") and x.endswith(".safetensors"):
            model_files.append(x)

    is_safetensors = len(model_files) > 0
    if not is_safetensors:
        return False

    if len(model_files) == 1 and model_files[0] == "model.safetensors":
        return True
    try:
        # check all the file exists
        safetensors_num = int(
            model_files[0].strip(".safetensors").split("-")[-1])
        flags = [0] * safetensors_num
        for x in model_files:
            current_index = int(x.strip(".safetensors").split("-")[1])
            flags[current_index - 1] = 1
        assert (
            sum(flags) == safetensors_num
        ), f"Number of safetensor files should be {len(model_files)}, but now it's {sum(flags)}"
    except Exception as e:
        raise Exception(f"Failed to check unified checkpoint, details: {e}.")
    return is_safetensors

def init_image_preprocess(tokenizer_path: str, image_preprocessor_path: str, patch_size: int) -> DataProcessor:
    processor = DataProcessor(
        tokenizer_name=tokenizer_path,
        image_preprocessor_name=str(image_preprocessor_path),
    )
    processor.eval()
    image_preprocess = processor.image_preprocessor
    image_preprocess.image_mean_tensor = paddle.to_tensor(
        image_preprocess.image_mean, dtype="float32").reshape([1, 3, 1, 1])
    image_preprocess.image_std_tensor = paddle.to_tensor(
        image_preprocess.image_std, dtype="float32").reshape([1, 3, 1, 1])
    image_preprocess.rescale_factor = paddle.to_tensor(
        image_preprocess.rescale_factor, dtype="float32")
    image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze(
        [-2, -1]).repeat_interleave(patch_size**2 * 1,
                                    -1)
    image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze(
        [-2, -1]).repeat_interleave(patch_size**2 * 1, -1)
    return image_preprocess

@paddle.no_grad()
def extract_vision_features(image_preprocess: DataProcessor ,inputs: list[paddle.Tensor], im_patch_id: int, amp_black: List[str], amp_white: List[str], dtype: paddle.dtype, model: paddle.nn.Layer,spatial_conv_size: int,tensor_parallel_size:int) -> paddle.Tensor:
    pass
    """extract_vision_features"""
    assert inputs["images"] is not None
    grid_thw = inputs["grid_thw"]

    images = inputs["images"].cast("float32")
    images = image_preprocess.rescale_factor * images - image_preprocess.image_mean_tensor
    images = images / image_preprocess.image_std_tensor
    images = images.cast("bfloat16")

    token_type_ids = inputs["token_type_ids"]
    token_type_ids_w_video = token_type_ids
    input_ids = inputs["input_ids"]
    # convert to img patch id
    image_mask = input_ids == im_patch_id
    image_type_ids = inputs["image_type_ids"]
    with paddle.amp.auto_cast(
            True,
            custom_black_list=amp_black,
            custom_white_list=amp_white,
            level="O2",
            dtype=dtype,
    ):
        image_features = model.vision_model.extract_feature(
            images, grid_thw)
        if tensor_parallel_size > 1:
            S, C = image_features.shape
            image_features = image_features.reshape(
                [-1, C * spatial_conv_size**2])
            image_features = ScatterOp.apply(image_features,
                                                axis=-1)  # mp 切 Fea
            image_features = image_features.reshape([S, -1])
        image_features = model.resampler_model(
            image_features,
            image_mask,
            token_type_ids_w_video,
            image_type_ids,
            grid_thw,
        )
    return image_features

def preprocess_mm_task(one: dict) -> dict:
    """process batch"""

    input_ids = one["input_ids"][np.newaxis, :]
    input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
    token_type_ids = one["token_type_ids"][np.newaxis, :]
    token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)

    if one["images"] is not None:
        image_type_ids = one["image_type_ids"][np.newaxis, :]
        images = one["images"]
        image_type_ids = paddle.to_tensor(image_type_ids,
                                            dtype=paddle.int64)
        images = paddle.to_tensor(images, dtype="uint8")
        grid_thw = paddle.to_tensor(one["grid_thw"], dtype="int64")
    else:
        image_type_ids = None
        images = None
        grid_thw = None

    if one["position_ids"] is not None:
        position_ids = paddle.to_tensor(one["position_ids"],
                                        dtype="int64").unsqueeze([0])
    else:
        position_ids = None

    result = dict(
        input_ids=input_ids,
        image_type_ids=image_type_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        grid_thw=grid_thw,
        images=images,
    )
    return result
