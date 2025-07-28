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

import copy
import itertools
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import paddle
from paddle.io import Dataset
from paddleformers.data import DataCollatorForSeq2Seq
from paddleformers.trainer import PdArgumentParser, TrainingArguments
from paddleformers.transformers.processing_utils import ProcessorMixin
from PIL import Image

from .mix_qwen2_5_tokenizer import MIXQwen2_5_Tokenizer
from .template import TEMPLATES
from .qwen2_5_vl_processing import (
    Qwen2_5_VLImageProcessor,
    Qwen2_5_VLProcessor,
)

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
VIDEO_PLACEHOLDER = "<video>"
IMAGE_PLACEHOLDER = "<image>"

dvqa_train_200k = {"annotation_path": "playground/opensource_json/dvqa_train_200k.json", "data_path": ""}
chartqa_train_18k = {"annotation_path": "playground/opensource_json/chartqa_train_18k.json", "data_path": ""}
ai2d_train_12k = {"annotation_path": "playground/opensource_json/ai2d_train_12k.json", "data_path": ""}
docvqa_train_10k = {"annotation_path": "playground/opensource_json/docvqa_train_10k.json", "data_path": ""}
geoqa = {"annotation_path": "playground/opensource_json/geoqa+.json", "data_path": ""}
synthdog_en = {"annotation_path": "playground/opensource_json/synthdog_en.json", "data_path": ""}

data_dict = {
    "dvqa_train_200k": dvqa_train_200k,
    "chartqa_train_18k": chartqa_train_18k,
    "ai2d_train_12k": ai2d_train_12k,
    "docvqa_train_10k": docvqa_train_10k,
    "geoqa": geoqa,
    "synthdog_en": synthdog_en,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


def get_rope_index(
    spatial_merge_size,
    input_ids: Optional[paddle.Tensor] = None,
    image_grid_thw: Optional[paddle.Tensor] = None,
    video_grid_thw: Optional[paddle.Tensor] = None,
    second_per_grid_ts: Optional[paddle.Tensor] = None,
    attention_mask: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`paddle.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`paddle.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`paddle.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`paddle.Tensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`paddle.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = paddle.ones_like(total_input_ids)
        position_ids = paddle.ones([3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype)
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            # TODO: CUDA error in some paddle version
            if attention_mask is not None:
                input_ids = paddle.to_tensor(input_ids[attention_mask[i] == 1])
            image_nums, video_nums = 0, 0
            vision_start_indices = paddle.nonzero(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum() if vision_tokens.numel() > 0 else 0
            video_nums = (vision_tokens == video_token_id).sum() if vision_tokens.numel() > 0 else 0
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + st_idx)
                range_tensor = paddle.arange(end=llm_grid_t).reshape([-1, 1])
                expanded_range = range_tensor.expand(shape=[-1, llm_grid_h * llm_grid_w])
                time_tensor = expanded_range * second_per_grid_t * 2
                time_tensor_long = time_tensor.astype(dtype="int64")
                t_index = time_tensor_long.flatten()
                h_index = (
                    paddle.arange(end=llm_grid_h)
                    .reshape([1, -1, 1])
                    .expand(shape=[llm_grid_t, -1, llm_grid_w])
                    .flatten()
                )
                w_index = (
                    paddle.arange(end=llm_grid_w)
                    .reshape([1, 1, -1])
                    .expand(shape=[llm_grid_t, llm_grid_h, -1])
                    .flatten()
                )
                llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + st_idx)
            llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape([3, -1])
            position_ids[..., i, attention_mask[i] == 1] = llm_positions

            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = paddle.cast(attention_mask, dtype="int64").cumsum(-1) - 1
            position_ids.masked_fill_(mask=attention_mask == 0, value=1)
            position_ids = position_ids.unsqueeze(0).expand([3, -1, -1])
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                paddle.arange(input_ids.shape[1]).reshape([1, 1, -1]).expand(shape=[3, input_ids.shape[0], -1])
            )
            mrope_position_deltas = paddle.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas


def rank0_print(*args):
    if paddle.distributed.get_rank() == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.copy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    tokenizer.init_chat_template(chat_template)
    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if source[0]["role"] != "user":
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        encode_output = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}], add_generation_prompt=False
        )  # 返回 dict
        input_id += encode_output["input_ids"]

        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + "<|image_pad|>" * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + "<|video_pad|>" * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_output = tokenizer.apply_chat_template(conv, add_generation_prompt=False)
            encode_id = encode_output["input_ids"]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = np.array(input_ids, dtype=np.int64)
    targets = np.array(targets, dtype=np.int64)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDatasetQwen2_5VL(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer, data_args, image_processor):  # NOTE: use paddlemix image_processor
        super(LazySupervisedDatasetQwen2_5VL, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_processor = image_processor
        self.image_processor.max_pixels = data_args.max_pixels
        self.image_processor.min_pixels = data_args.min_pixels
        self.image_processor.size["longest_edge"] = data_args.max_pixels
        self.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "images" in sample else 0
            length_list.append(sum(len(conv["content"].split()) for conv in sample["messages"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["content"].split()) for conv in sample["messages"])
            cur_len = cur_len if ("images" in sample) or ("video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="np")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def __getitem__(self, i):
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None

        if "images" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["images"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            raise NotImplementedError("Video is not supported yet.")

        chat_sources = copy.deepcopy([e["messages"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )

        position_ids = None
        if "images" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["messages"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged)
            position_ids = (
                np.arange(0, data_dict["input_ids"].shape[1]).reshape(1, -1)[np.newaxis, ...].expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].shape[0]]

        if "images" in self.list_data_dict[i]:
            data_dict["pixel_values"] = np.concatenate(image, axis=0)
            data_dict["image_grid_thw"] = np.concatenate([thw[np.newaxis, ...] for thw in grid_thw], axis=0)

        return data_dict


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    """
    image_resolution: int = field(
        default=768,
        metadata={"help": "Keeps the height or width of image below this resolution."},
    )


@dataclass
class ModelArguments(ProcessorArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the LLM decoder."},
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "Set the drop path rate for the ViT model. Default is 0."},
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use lora to train model."},
    )
    lora_path: Optional[str] = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: Optional[int] = field(
        default=128,
        metadata={"help": "Set the value of rank in lora. Default is 128."},
    )
    lora_alpha: Optional[int] = field(
        default=256,
        metadata={"help": "Set the value of alpha in lora. Default is 256."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Set the value of dropout in lora. Default is 0.0."},
    )
    lora_target_modules: Optional[str] = field(default=None, metadata={"help": "Lora target modules."})
    max_pixels: Optional[int] = field(default=None, metadata={"help": "max image input size"})
    min_pixels: Optional[int] = field(default=None, metadata={"help": "min image input size"})


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """

    max_seq_length: Optional[int] = field(
        default=8192,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_image_size: Optional[int] = field(
        default=768,
        metadata={"help": "Set the desired size for the image. Default is 224."},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={"help": "Pad the image to a square shape if set to True."},
    )
    conv_style: Optional[str] = field(default="qwen2_5_vl", metadata={"help": "Prompt style for a conversation."})
    meta_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the meta file of datasets."},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use data resampling."},
    )
    normalize_type: Optional[str] = field(
        default="imagenet",
        metadata={"help": "The normalize type for the image. Default is imagenet."},
    )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    group_by_length: bool = field(
        default=True,
        metadata={"help": ""},
    )
    save_safetensors: bool = field(
        default=True,
        metadata={"help": ""},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark (True/False)."},
    )


@dataclass
class MultiModalDataCollatorForSeq2SeqQwen2_5VLPaddle(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.
    Features should contain input_ids, attention_mask, labels, and optionally contain images.
    """

    template: Optional["TEMPLATES"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "paddle.Tensor"]:
        # process output from dataset
        new_feats = []
        for feat in features:
            new_feats_i = {}
            for k in feat:
                if k == "position_ids":
                    continue
                elif k == "attention_mask":
                    new_feats_i[k] = [1] * feat[k][0]
                elif k in ["input_ids", "labels"]:
                    new_feats_i[k] = feat[k].squeeze(0).tolist()
            new_feats.append(new_feats_i)

        new_feats: Dict[str, "paddle.Tensor"] = super().__call__(new_feats)

        pixel_values_list = []
        image_grid_thw_list = []
        for feat in features:
            pixel_values_list.append(feat["pixel_values"])
            image_grid_thw_list.append(feat["image_grid_thw"])

        new_pixel_values = paddle.to_tensor(np.concatenate(pixel_values_list, axis=0))
        image_grid_thw_list = paddle.to_tensor(np.concatenate(image_grid_thw_list, axis=0))

        new_feats["pixel_values"] = new_pixel_values
        new_feats["image_grid_thw"] = image_grid_thw_list
        features = new_feats
        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2_5_vl mrope
            rope_index_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": features.get("image_grid_thw"),
                "video_grid_thw": features.get("video_grid_thw"),
                "attention_mask": features["attention_mask"],
            }
            if "second_per_grid_ts" in features:
                rope_index_kwargs["second_per_grid_ts"] = features.get("second_per_grid_ts")

            features["position_ids"], features["rope_deltas"] = self.model.get_rope_index(**rope_index_kwargs)

        if "cross_attention_mask" in features:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = features.pop("cross_attention_mask")
            seq_len = features["input_ids"].shape[1]
            orig_len = cross_attention_mask.shape[1]
            features["cross_attention_mask"] = paddle.nn.functional.pad(
                cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len)
            )

        if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
            features = features.data  # use default_collate() instead of BatchEncoding.to()

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = paddle.arange(seq_length).long().tile([bsz, 1])
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}
        return features


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = paddle.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = paddle.concat(padded_tensors, axis=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(DataCollatorForSeq2Seq):
    """Collate examples for supervised fine-tuning."""

    tokenizer: MIXQwen2_5_Tokenizer

    def __call__(self, instances: Sequence[Dict], merge_size=2) -> Dict[str, paddle.Tensor]:
        # convert dataset output to paddle tensor
        # process output from dataset
        new_feats = []
        for feat in instances:
            new_feat = {}
            for k in feat:
                if k == "position_ids":
                    continue
                elif k == "attention_mask":
                    new_feat[k] = feat[k]
                else:
                    new_feat[k] = paddle.to_tensor(feat[k])

            new_feats.append(new_feat)

        instances = new_feats
        # compute rope index
        for inst in instances:
            rope_index_kwargs = {
                "input_ids": inst["input_ids"],
                "image_grid_thw": inst.get("image_grid_thw"),
                "video_grid_thw": inst.get("video_grid_thw"),
            }
            if "second_per_grid_ts" in instances:
                rope_index_kwargs["second_per_grid_ts"] = inst.get("second_per_grid_ts")
            inst["position_ids"], _ = get_rope_index(merge_size, **rope_index_kwargs)

        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = paddle.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = paddle.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = paddle.concat([image for image in images], axis=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = paddle.concat(grid_thw, aixs=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = paddle.concat([video for video in videos], axis=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = paddle.concat(video_grid_thw, axis=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: MIXQwen2_5_Tokenizer

    def __call__(self, instances: Sequence[Dict], merge_size=2) -> Dict[str, paddle.Tensor]:
        # convert dataset output to paddle tensor
        # process output from dataset
        new_feats = []
        for feat in instances:
            new_feat = {}
            for k in feat:
                if k == "position_ids":
                    continue
                elif k == "attention_mask":
                    new_feat[k] = feat[k]
                else:
                    new_feat[k] = paddle.to_tensor(feat[k])

            new_feats.append(new_feat)

        instances = new_feats
        # compute rope index
        for inst in instances:
            rope_index_kwargs = {
                "input_ids": inst["input_ids"],
                "image_grid_thw": inst.get("image_grid_thw"),
                "video_grid_thw": inst.get("video_grid_thw"),
            }
            if "second_per_grid_ts" in instances:
                rope_index_kwargs["second_per_grid_ts"] = inst.get("second_per_grid_ts")

            inst["position_ids"], _ = get_rope_index(merge_size, **rope_index_kwargs)

        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(*(instance["attention_mask"] for instance in instances if "attention_mask" in instance))
        )
        seq_lens = paddle.to_tensor([0] + attention_mask, dtype="int32")
        cumsum_seq_lens = paddle.cumsum(seq_lens, axis=0, dtype="int32")
        input_ids = paddle.concat(input_ids, axis=1)
        labels = paddle.concat(labels, axis=1)
        position_ids = paddle.concat(position_ids, axis=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = paddle.concat([image for image in images], axis=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = paddle.concat(grid_thw, axis=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = paddle.concat([video for video in videos], axis=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = paddle.concat(video_grid_thw, axis=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


if __name__ == "__main__":

    parser = PdArgumentParser((ModelArguments, DataTrainingArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(data_args)
    print("model_args: ", model_args)
    data_args.dataset_use = "dvqa_train_200k"
    data_args.max_pixels = 50176
    data_args.min_pixels = 784
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_processor = Qwen2_5_VLImageProcessor()
    tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(MODEL_NAME)
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer)
    dataset = LazySupervisedDatasetQwen2_5VL(
        tokenizer=processor.tokenizer, data_args=data_args, image_processor=image_processor
    )

    di = iter(dataset)
    d1 = next(di)
    d2 = next(di)
    data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)

    out = data_collator([d1, d2], merge_size=image_processor.merge_size)
    print(out)
