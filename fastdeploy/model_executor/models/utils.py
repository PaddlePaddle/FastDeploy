"""
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""

from __future__ import annotations

import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import struct
from functools import partial
from typing import Callable, Optional

import numpy as np
from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.transformers.utils import paddlenlp_load
from paddlenlp.transformers.model_utils import load_tp_checkpoint
from safetensors import safe_open

from paddlenlp.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)
from paddlenlp.utils.log import logger
from tqdm import tqdm

import paddle
import paddle.distributed as dist
from paddle.common_ops_import import convert_dtype
from paddle.distributed import fleet
from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.transformers.model_utils import _add_variant, load_tp_checkpoint
from paddlenlp.transformers.utils import paddlenlp_load
from paddlenlp.utils.env import (PADDLE_WEIGHTS_INDEX_NAME,
                                 SAFE_MASTER_WEIGHTS_INDEX_NAME,
                                 SAFE_PEFT_WEIGHTS_INDEX_NAME,
                                 SAFE_WEIGHTS_INDEX_NAME)
from paddlenlp.utils.log import logger
from safetensors import safe_open
from tqdm import tqdm

from fastdeploy.platforms import current_platform

from .tokenizer import ErnieBotTokenizer
import glob

MODEL_LIB_NAMES = [
    "ernie_bot.modeling",
    "ernie_bot.modeling_pp",
    "ernie_bot.modeling_moe",
    "ernie_bot.modeling_rm",
    "ernie_bot.proxy_distill",
]

MAX_BSZ = 512
MAX_DRAFT_TOKENS = 6


class UniqueIDGenerator:
    """
    The generator for the export model id
    """

    def __init__(self):
        pass

    def generate_unique_id(self, state_dict):
        """
        Generate the model id from the timestamp
        """
        keys = state_dict.keys()
        sorted_keys = sorted(keys)
        first_key = sorted_keys[0]
        first_parameter = state_dict[first_key].cast("float32")
        # 假设模型参数是唯一的，通过第一个key来获取md5sum
        model_md5 = hashlib.md5(str(
            first_parameter.sum()).encode("utf-8")).hexdigest()
        unique_id = f"{model_md5}-{random.randint(10000, 99999)}"
        return unique_id


def load_sharded_checkpoint(folder, variant=None, return_numpy=False):
    """

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.

    """
    # Load the index
    pdparams_file = os.path.join(folder,
                                 _add_variant("model_state.pdparams", variant))
    lora_pdparams_file = os.path.join(
        folder, _add_variant("lora_model_state.pdparams", variant))
    safetensors_file = os.path.join(folder,
                                    _add_variant("model.safetensors", variant))
    if os.path.isfile(pdparams_file):
        return paddle.load(pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(lora_pdparams_file):
        return paddle.load(lora_pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(safetensors_file):
        try:
            from paddlenlp.utils.safetensors import \
                fast_load_file as safe_load_file
        except ImportError:
            from safetensors.numpy import load_file as safe_load_file

        state_dict = safe_load_file(safetensors_file)
        if not return_numpy:
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = paddle.Tensor(state_dict.pop(key),
                                                    zero_copy=True)
        return state_dict

    index_file = os.path.join(folder,
                              _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(
        folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
    safe_master_file = os.path.join(
        folder, _add_variant(SAFE_MASTER_WEIGHTS_INDEX_NAME, variant))
    safe_peft_file = os.path.join(
        folder, _add_variant(SAFE_PEFT_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    safe_master_present = os.path.isfile(safe_master_file)
    safe_peft_present = os.path.isfile(safe_peft_file)

    load_safe = False
    load_index = None
    if safe_index_present:
        load_safe = True  # load safe due to preference
        load_index = safe_index_file
    elif safe_master_present:
        load_safe = True
        load_index = safe_master_file
    elif index_present:
        load_index = index_file
    elif safe_peft_present:
        load_safe = True
        load_index = safe_peft_file
    else:
        raise ValueError(
            f"Could not find {index_file} or {safe_index_file} or {safe_peft_file}"
        )

    if load_safe:
        try:
            from paddlenlp.utils.safetensors import \
                fast_load_file as safe_load_file
        except ImportError:
            from safetensors.numpy import load_file as safe_load_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loader = (safe_load_file if load_safe else partial(
        paddlenlp_load, map_location="np" if return_numpy else "cpu"))

    ret = {}
    for shard_file in tqdm(shard_files):
        state_dict = loader(os.path.join(folder, shard_file))
        ret.update(state_dict)

    if not return_numpy:
        for key in list(ret.keys()):
            if isinstance(ret[key], np.ndarray):
                ret[key] = paddle.Tensor(ret.pop(key), zero_copy=True)

    return ret


def convert_ndarray_dtype(np_array: np.ndarray,
                          target_dtype: str) -> np.ndarray:
    """convert ndarray

    Args:
        np_array (np.ndarray): numpy ndarray instance
        target_dtype (str): the target dtype

    Returns:
        np.ndarray: converted numpy ndarray instance
    """
    source_dtype = convert_dtype(np_array.dtype)
    if source_dtype == "uint16" or target_dtype == "bfloat16":
        if paddle.is_compiled_with_xpu():
            # xpu not support bf16.
            tensor = paddle.to_tensor(np_array, place=paddle.CPUPlace())
        else:
            tensor = paddle.to_tensor(np_array)
        tensor = paddle.cast(tensor, target_dtype)
        return tensor.numpy()

        # TODO(wj-Mcat): device_guard will slow the converting
        # with device_guard("cpu"):
        #     tensor = paddle.to_tensor(np_array)
        #     tensor = paddle.cast(tensor, target_dtype)
        # return tensor.numpy()

    if target_dtype == "bfloat16":
        target_dtype = "uint16"

    return np_array.astype(target_dtype)


def ernie_bot_postprocess_past_key_value(past_key_values):
    """
    ernie_bot_postprocess_past_key_values
    """
    Cache = collections.namedtuple("Cache", ["k", "v"])
    # (layer_num, bs, prefixlen, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 0, 1, 3,
                                                           4]).split(2)

    past_key_values = []
    for k, v in zip(keys, values):
        past_key_values.append(Cache(k, v))
    return past_key_values


def ernie_bot_pad_attention_mask(input_ids_shape, num_prefix_tokens,
                                 attention_mask):
    """
    ernie_bot_pad_attention_mask
    """
    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :]
        prefix_attention_mask = paddle.ones(
            [input_ids_shape[0], 1, 1, num_prefix_tokens],
            dtype=attention_mask.dtype,
        )
    else:
        prefix_attention_mask = paddle.ones(
            [input_ids_shape[0], 1, input_ids_shape[-1], num_prefix_tokens],
            dtype=attention_mask.dtype,
        )
    return paddle.concat((prefix_attention_mask, attention_mask), axis=3)


def set_seed(seed: int):
    """
    set random seed for all random modules
    """
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_infer_model_path(input_dir, model_prefix, is_export: bool = False):
    """when n_ranks = 1, infer_model_path is: `{input_dir}/{model_prefix}.pdiparams`
       when n_ranks > 1, infer_model_path is: `{input_dir}/rank_{idx}/{model_prefix}.pdiparams`

    Args:
        input_dir (str): the base input_dir
        model_prefix (str): the prefix name of model

    Returns:
        str: the path of infer model path
    """
    n_ranks = dist.get_world_size()
    try:
        local_rank = dist.ParallelEnv().dev_id
    except Exception:
        logger.info(
            "`dist.ParallelEnv().dev_id` is not supported on CPU devices,so set local_rank = 0."
        )
        local_rank = 0
    if n_ranks > 1:
        return os.path.join(input_dir, f"rank_{local_rank}", model_prefix)

    # if n_ranks director exist, return N-rank directory
    sub_rank_dir = os.path.join(input_dir, f"rank_{local_rank}")

    if is_export:
        return os.path.join(sub_rank_dir, model_prefix)
    else:
        # when inference, return sub_rank_dir when exists
        if os.path.exists(sub_rank_dir):
            return os.path.join(sub_rank_dir, model_prefix)
        else:
            return os.path.join(input_dir, model_prefix)


def pad_batch_data(insts, pad_id=0, return_seq_len=False, pad_style="right"):
    """Pad the instances to the max sequence length in batch."""
    # pad to max input len i bsz
    max_len = max(map(len, insts))
    # pad to max input len
    # max_len = args.max_len
    if pad_style == "left":
        inst_data = np.array([[pad_id] * (max_len - len(inst)) + list(inst)
                              for inst in insts])
    else:
        inst_data = np.array(
            [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])
    if return_seq_len:
        seq_len = np.array([len(inst) for inst in insts])
        return inst_data.astype("int64").reshape([-1, max_len]), seq_len
    else:
        return inst_data.astype("int64").reshape([-1, max_len])


def load_prefix_weights(
    prefix_path: str,
    inference: bool = False,
    batch_size: int = 1,
    dtype: str = "bfloat16",
) -> np.ndarray | list[paddle.Tensor]:
    """load prefix weight by path

    Args:
        prefix_path (str): the path of prefix weight
    """
    past_key_values = paddle.to_tensor(
        np.load(f"{prefix_path}/pre_caches.npy")).unsqueeze(2)

    if batch_size > 1:
        past_key_values = paddle.concat([past_key_values] * batch_size, axis=2)

    # .chatglm static model require one tensor, otherwise list of tensor
    past_key_values = past_key_values.astype(dtype)
    if inference:
        return past_key_values.numpy()
    return past_key_values


def build_for_generation(model, tokenizer: PretrainedTokenizer,
                         generation_kwargs: dict):
    """build `ErnieBotForGenerationFuse` to generate tokens

    Args:
        model (_type_): ErnieBotModel or ErnieBotFusedModel
        tokenizer (PretrainedTokenizer): pretrained tokenizer
        generation_kwargs (dict): generation_kwargs for model

    Returns:
        PretrainedModel: ErnieBotForGenerationFuse
    """
    from ernie_bot.single_model_fused import ErnieBotForGenerationFuse

    configs = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "initializer_range": 0.02,
        "fused_linear": False,
        "min_dec_len": 1,
        "max_dec_len": 1024,
        "top_k": 0,
        "top_p": 0.7,
        "temperature": 0.95,
        "use_topp_sampling": True,
        "inference": True,
    }
    configs.update(generation_kwargs)
    model = ErnieBotForGenerationFuse(model, configs=configs)
    model.eval()
    return model


def init_distributed_env() -> tuple[int, int]:
    """init distributed envs, and only support mp in ErnieBotModel

    Returns:
        tuple[int, int]: tensor_parallel_degree, tensor_parallel_rank
    """
    tensor_parallel_degree = dist.get_world_size()
    tensor_parallel_rank = 0

    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        tensor_parallel_rank = hcg.get_model_parallel_rank()

    return tensor_parallel_degree, tensor_parallel_rank


def generate_rank_mapping(output_dir: str):
    """generate current distributed rank mapping file

    Args:
        output_dir (str): the directory of rank_mapping file
    """
    os.makedirs(output_dir, exist_ok=True)

    # must in distributed env
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    ring_id = model_parallel_group.id

    world_size = dist.get_world_size()
    with open(os.path.join(output_dir, "rank_mapping.csv"), "w") as f:
        f.write("[ring_id -> ranks]\n")
        f.write(",".join(map(str, [0] + list(range(world_size)))) + "\n")
        f.write(",".join(map(str, [ring_id] + list(range(world_size)))) + "\n")

        f.write("[rank -> ring_ids]\n")
        for i in range(world_size):
            f.write(f"{i},0,{ring_id}\n")


def save_infer_result(trainer, dev_ds, k=100, src_length=256, tgt_length=512):
    """
    save infer result into jsonl format
    """
    from predict_generation import Predictor, batchfy_text

    all_instructions = []
    all_answers = []
    all_output = []

    # top k instruction from dev_ds
    for i, ds in enumerate(dev_ds.data):
        if i == k:
            break
        if "instruction" in ds:
            all_instructions.append(ds["instruction"])
            all_answers.append(ds["output"])
        elif "src" in ds:
            if isinstance(ds["src"], list):
                all_instructions.append(ds["src"][0])
                all_answers.append(ds["tgt"][0])
            else:
                all_instructions.append(ds["src"])
                all_answers.append(ds["tgt"])

    batch_texts = batchfy_text(all_instructions,
                               trainer.args.per_device_eval_batch_size)
    predictor = Predictor(
        tokenizer=trainer.tokenizer,
        model=trainer.model,
        src_length=src_length,
        tgt_length=tgt_length,
    )

    # infer results
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for i, (text, result) in enumerate(zip(texts, outputs["result"])):
            out = {
                "instruction":
                text,
                "answer":
                all_answers[bs * trainer.args.per_device_eval_batch_size + i],
                "output":
                result,
            }
            all_output.append(out)

    # save results
    if trainer.args.tensor_parallel_rank == 0:
        with open(os.path.join(trainer.args.output_dir, "infer_result.json"),
                  "w") as f:
            for out in all_output:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def w4a8_weight_convert(state_dict):
    """W4A8 权重转换函数
    Args:
        state_dict (dict): state_dict of model
    """

    def w4_weight_squash(value, name, w4a8_weight_bites_name_map):
        weight_dq = value
        # W8表象下的W4权重的absmax值为112，使用正负112进行权重类型判断
        if weight_dq.max() == 112 or weight_dq.min() == -112:
            weight_dq = weight_dq.cast("int8")
            np_weight_dq = np.array(weight_dq, dtype="int8").view("uint8")
            np_weight_dq_left_div_16 = (np_weight_dq / 16).astype("int8")
            # weight_q = (weight_dq/16).cast('int8')
            weight_q = paddle.to_tensor(np_weight_dq_left_div_16, dtype="int8")
            logger.debug(f"int4 weight:{name}")
            w4a8_weight_bites_name_map[name] = 4
            return weight_q.cast("int8")
        elif weight_dq.max() == 127 or weight_dq.min() == -128:
            logger.debug(f"int8 weight:{name}")
            w4a8_weight_bites_name_map[name] = 8
            return weight_dq.cast("int8")
        else:
            logger.debug(f"fp16/bf16/float weight:{name}")
            return weight_dq

    w4a8_weight_bites_name_map = {}
    for name, value in state_dict.items():
        if value.dtype == "uint16":
            weight_q = w4_weight_squash(
                paddle.to_tensor(value).cast("float32"),
                name,
                w4a8_weight_bites_name_map,
            )
            state_dict[name] = weight_q.numpy(
            ) if weight_q is not None else value
            del weight_q
    w4a8_weight_bites_layers_map = {}
    w4a8_weight_bites_layers_map["qkv_gemm_bits_map"] = []
    w4a8_weight_bites_layers_map["out_gemm_bits_map"] = []
    w4a8_weight_bites_layers_map["ffn1_gemm_bits_map"] = []
    w4a8_weight_bites_layers_map["ffn2_gemm_bits_map"] = []
    for name_keys, gemm_bits in w4a8_weight_bites_name_map.items():
        if "qkv_proj" in name_keys:
            w4a8_weight_bites_layers_map["qkv_gemm_bits_map"].append(gemm_bits)
        elif "out_proj" in name_keys:
            w4a8_weight_bites_layers_map["out_gemm_bits_map"].append(gemm_bits)
        elif "linear1" in name_keys:
            w4a8_weight_bites_layers_map["ffn1_gemm_bits_map"].append(
                gemm_bits)
        elif "linear2" in name_keys:
            w4a8_weight_bites_layers_map["ffn2_gemm_bits_map"].append(
                gemm_bits)
    logger.debug(
        f"w4a8_weight_bites_layers_map:{w4a8_weight_bites_layers_map}")
    return state_dict, w4a8_weight_bites_layers_map


def _vocab_size_with_padding(vocab_size, div_unit, mp_degree):
    padded_size = vocab_size
    multiple = div_unit * mp_degree
    while (padded_size % multiple) != 0:
        padded_size += 1
    # logger.warning(
    #     " > padded vocab (size: {}) with {} dummy tokens "
    #     "(new size: {})".format(vocab_size, padded_size - vocab_size, padded_size)
    # )
    return padded_size


def save_test_case(cases: list[list[dict]], file: str):
    """save test to result file

    Args:
        cases (list[list[dict]]): the content of case
        file (str): the path of saved file
    """
    with open(file, "w+", encoding="utf-8") as f:
        for case in cases:
            raw = json.dumps(case, ensure_ascii=False)
            f.write(raw + "\n")


def infer_save_test_case(cases: list[list[dict]], file: str):
    """save test to result file

    Args:
        cases (list[list[dict]]): the content of case
        file (str): the path of saved file
    """
    with open(file, "a+", encoding="utf-8") as f:
        for case in cases:
            raw = json.dumps(case, ensure_ascii=False)
            f.write(raw + "\n")


def deserialize_from_file(fp):
    """
    deserialize a binary file into an array
    """

    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b"0":
        data = fp.read(4)
        data_out = struct.unpack("f", data)[0]
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b"1":
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b"2":
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


def read_res(
    model_name_or_path,
    output_tensor_max_shape,
    result_queue: mp.Queue,
    msg_queue_id=None,
    use_ep=False,
    ep_just_for_test=False,
    tokenizer=None,
):
    """Read result from queue."""
    if msg_queue_id is None:
        if (current_platform.is_cuda() and
                current_platform.available()) or paddle.is_compiled_with_xpu():
            from fastdeploy.model_executor.ops.gpu import get_output
        elif paddle.is_compiled_with_custom_device("npu"):
            from paddle_custom_device.npu import get_output
        else:  # CPU
            from fastdeploy.model_executor.ops.cpu import get_output
    else:
        if (current_platform.is_cuda() and
                current_platform.available()) or paddle.is_compiled_with_xpu():
            from fastdeploy.model_executor.ops.gpu import get_output_dynamic
        elif paddle.is_compiled_with_custom_device("npu"):
            from paddle_custom_device.npu import get_output_dynamic
        else:  # CPU
            from fastdeploy.model_executor.ops.cpu import get_output_dynamic

    if tokenizer is None:
        tokenizer = ErnieBotTokenizer.from_pretrained(model_name_or_path)

    paddle.device.set_device("cpu")
    paddle.disable_static()
    output_tensor = paddle.full(output_tensor_max_shape,
                                fill_value=2,
                                dtype="int64")

    while True:
        outputs = []
        while True:
            if msg_queue_id is None:
                get_output(output_tensor, 0, True)
            else:
                get_output_dynamic(output_tensor, 0, True, msg_queue_id)
            if int(output_tensor[0, 0]) == -2:  # read none
                continue
            bsz = int(output_tensor[1, 0])
            output_numpy = output_tensor[2:bsz + 2].numpy()
            output_numpy[output_numpy == -1] = 2
            outputs.append(output_numpy)

            if int(output_tensor[0, 0]) < 0:
                break
        output = np.concatenate(outputs, axis=1)
        seqs = tokenizer.batch_decode(
            output.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if use_ep and (not ep_just_for_test):
            print("seqs: ", seqs)
        for i, seq in enumerate(seqs):
            result_queue.put([i, len(output.tolist()[i]), seq])


def speculate_read_res(
    model_name_or_path,
    output_tensor_max_shape,
    result_queue: mp.Queue,
    msg_queue_id=None,
):
    """Read result from queue."""
    if msg_queue_id is None:
        from fastdeploy.model_executor.ops.gpu import speculate_get_output
    else:
        from fastdeploy.model_executor.ops.gpu import \
            speculate_get_output_dynamic

    tokenizer = ErnieBotTokenizer.from_pretrained(model_name_or_path)
    paddle.device.set_device("cpu")
    paddle.disable_static()
    output_tensor = paddle.full(output_tensor_max_shape,
                                fill_value=2,
                                dtype="int64")
    while True:
        outputs = []
        for _ in range(MAX_BSZ):
            outputs.append([])

        while True:
            if msg_queue_id is None:
                speculate_get_output(output_tensor, 0, True)
            else:
                speculate_get_output_dynamic(output_tensor, 0, True,
                                             msg_queue_id)
            if int(output_tensor[0]) == -2:  # read none
                continue
            bsz = int(output_tensor[1])
            accept_num = output_tensor[2:bsz + 2].numpy()
            for bi in range(bsz):
                outputs[bi].extend(
                    output_tensor.numpy()[2 + MAX_BSZ +
                                          bi * MAX_DRAFT_TOKENS:2 + MAX_BSZ +
                                          bi * MAX_DRAFT_TOKENS +
                                          accept_num[bi]].tolist())
            if int(output_tensor[0]) == -1:
                break

        seqs = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for i in range(bsz):
            result_queue.put([i, len(outputs[i]), seqs[i]])


def calculate_effective_tokens(training_args, train_dataset, max_seq_len):
    """
    Calculate the effective tokens during training.
    """
    total_effective_tokens = 0
    try:
        data_parallel_degree = training_args.data_parallel_degree
    except Exception:
        data_parallel_degree = 1
    if training_args.sharding_parallel_degree > 1:
        sharding_parallel_degree = training_args.sharding_parallel_degree
    else:
        sharding_parallel_degree = 1

    total_batch = (training_args.max_steps *
                   training_args.per_device_train_batch_size *
                   training_args.gradient_accumulation_steps *
                   sharding_parallel_degree * data_parallel_degree)
    for i, data in enumerate(train_dataset):
        if i == total_batch:
            break
        for dd in data:
            total_effective_tokens += len(dd.token_ids)
    total_tokens = total_batch * max_seq_len

    return total_effective_tokens, total_tokens


def estimate_training(train_dataset, data_args, training_args, model_args):
    """
    根据训练数据估算训练所需的步数。

    Args:
        - None

    Returns:
        - dict: 返回一个字典，包含了训练所需的步骤数信息。

    """
    train_dataset.estimate = True
    logger.info("Start to estimate max training steps...")
    with open(data_args.train_task_config) as f:
        train_task_group = json.load(f)

    if len(train_task_group) > 1:
        logger.warning(
            "Suggest to use max_steps instead of num_train_epochs for multi source dataset."
        )
        logger.info(
            "Multi source dataset detected, number of samples will be estimated by following rule. "
            "num_samples = (source1_num_samples * prob1 + source2_num_samples * prob2 + ...) * epochs"
        )

    max_samples = train_dataset.max_estimate_samples

    if training_args.max_estimate_samples != -1:
        # Set estimate samples to max_estimate_samples
        logger.warning(
            "The results between sampling and non-sampling methods may differ."
        )
        train_dataset.max_estimate_samples = min(
            training_args.max_estimate_samples,
            train_dataset.max_estimate_samples)

    if train_dataset.max_estimate_samples > 0:
        train_batches = 0
        train_tokens = 0
        for sequences in train_dataset:
            if not train_dataset.estimate:
                break
            train_batches += 1
            for sequence in sequences:
                train_tokens += len(sequence.token_ids)

        train_tokens *= training_args.num_train_epochs
        train_batches *= training_args.num_train_epochs
        global_batch_size = (training_args.per_device_train_batch_size *
                             training_args.gradient_accumulation_steps *
                             max(training_args.data_parallel_degree, 1) *
                             max(training_args.sharding_parallel_degree, 1))
        max_steps = int(np.ceil(train_batches / global_batch_size))

        if max_samples != train_dataset.max_estimate_samples:
            max_steps *= max_samples / train_dataset.max_estimate_samples
            train_tokens *= max_samples / train_dataset.max_estimate_samples
            train_dataset.used_samples *= (max_samples /
                                           train_dataset.max_estimate_samples)
            train_dataset.unused_samples *= (
                max_samples / train_dataset.max_estimate_samples)

        res = {
            "num_train_epochs":
            int(training_args.num_train_epochs),
            "max_steps":
            int(np.ceil(max_steps)),
            "train_tokens":
            int(train_tokens),
            "global_batch_size":
            int(global_batch_size),
            "gradient_accumulation_steps":
            training_args.gradient_accumulation_steps,
            "warmup_steps":
            int(np.ceil(0.1 * max_steps)),
            "per_device_train_batch_size":
            int(training_args.per_device_train_batch_size),
            "tensor_parallel_degree":
            int(training_args.tensor_parallel_degree),
            "pipeline_parallel_degree":
            int(training_args.pipeline_parallel_degree),
            "sharding_parallel_degree":
            int(training_args.sharding_parallel_degree),
            "seed":
            training_args.seed,
            "num_samples_each_epoch":
            data_args.num_samples_each_epoch,
            "example_from_same_task_prob":
            data_args.example_from_same_task_prob,
            "pseudo_sampling_prob":
            data_args.pseudo_sampling_prob,
            "trigger_data_prob":
            data_args.trigger_data_prob,
            "max_seq_len":
            int(data_args.max_seq_len),
            "valid":
            True,
            "train_samples":
            int(max_samples * training_args.num_train_epochs),
            "estimate_samples":
            int(train_dataset.max_estimate_samples),
            "actual_train_samples":
            int(train_dataset.used_samples * training_args.num_train_epochs),
            "skip_samples":
            int(train_dataset.unused_samples * training_args.num_train_epochs),
        }
        if hasattr(training_args, "num_of_gpus"):
            res["num_of_gpus"] = training_args.num_of_gpus

        if train_batches / training_args.num_train_epochs / global_batch_size < 1:
            logger.warning(
                "This dataset is too small, you'd better enlarge your dataset."
            )
            res["valid"] = False

        if getattr(training_args, "estimation_output_file", None):
            with open(training_args.estimation_output_file,
                      "w",
                      encoding="utf-8") as f:
                json.dump(res, f)

        return max_steps
    else:
        res = {
            "num_train_epochs":
            int(training_args.num_train_epochs),
            "max_steps":
            0,
            "gradient_accumulation_steps":
            training_args.gradient_accumulation_steps,
            "train_tokens":
            0,
            "per_device_train_batch_size":
            int(training_args.per_device_train_batch_size),
            "tensor_parallel_degree":
            int(training_args.tensor_parallel_degree),
            "pipeline_parallel_degree":
            int(training_args.pipeline_parallel_degree),
            "sharding_parallel_degree":
            int(training_args.sharding_parallel_degree),
            "num_samples_each_epoch":
            data_args.num_samples_each_epoch,
            "example_from_same_task_prob":
            data_args.example_from_same_task_prob,
            "pseudo_sampling_prob":
            data_args.pseudo_sampling_prob,
            "trigger_data_prob":
            data_args.trigger_data_prob,
            "max_seq_len":
            int(data_args.max_seq_len),
            "seed":
            data_args.seed,
            "valid":
            False,
            "train_samples":
            0,
        }
        if hasattr(training_args, "num_of_gpus"):
            res["num_of_gpus"] = training_args.num_of_gpus

        if getattr(training_args, "estimation_output_file", None):
            with open(training_args.estimation_output_file,
                      "w",
                      encoding="utf-8") as f:
                json.dump(res, f)

        logger.error("No valid data found, please check your dataset format.")
        return 0


def get_w4a8_gemm_config_tuple(file_root_path):
    """读取预配置的gemm 配置表
    Args:
        file_root_path (str): the directory of w4a8_gemm_config files
    """

    def get_gemm_config_tuple_from_file(file):
        gemm_tuple_list = []
        for line in file:
            line_split = line.split(" ")
            gemm_tuple_list.append([
                int(line_split[1]),
                int(line_split[2]),
                int(line_split[3]),
                int(line_split[4]),
                int(line_split[5]),
                int(line_split[6]),
                int(line_split[7]),
            ])
        gemm_tuple_list.sort(key=lambda x: x[0])
        gemm_tuple_numpy = np.array(gemm_tuple_list, dtype="int32")
        gemm_tuple_numpy = gemm_tuple_numpy.flatten()
        return gemm_tuple_numpy

    qkv_gemm_config_tuple = []
    out_linear_gemm_config_tuple = []
    ffn1_gemm_config_tuple = []
    ffn2_gemm_config_tuple = []
    try:
        qkv_tuned_gemm_config_log_path = os.path.join(
            f"{file_root_path}", "qkv_tuned_gemm_config.log")
        with open(qkv_tuned_gemm_config_log_path) as file:
            qkv_gemm_config_tuple = get_gemm_config_tuple_from_file(file)
        out_linear_tuned_gemm_config_log_path = os.path.join(
            f"{file_root_path}", "out_linear_tuned_gemm_config.log")
        with open(out_linear_tuned_gemm_config_log_path) as file:
            out_linear_gemm_config_tuple = get_gemm_config_tuple_from_file(
                file)
        ffn1_tuned_gemm_config_log_path = os.path.join(
            f"{file_root_path}", "ffn1_tuned_gemm_config.log")
        with open(ffn1_tuned_gemm_config_log_path) as file:
            ffn1_gemm_config_tuple = get_gemm_config_tuple_from_file(file)
        ffn2_tuned_gemm_config_log_path = os.path.join(
            f"{file_root_path}", "ffn2_tuned_gemm_config.log")
        with open(ffn2_tuned_gemm_config_log_path) as file:
            ffn2_gemm_config_tuple = get_gemm_config_tuple_from_file(file)
    except Exception:
        logger.warning(
            "Found gemm config for W4A8 failed, using empty gemm tuple list for W4A8"
        )
    w4a8_gemm_config = {}
    w4a8_gemm_config["qkv_gemm_config_tuple"] = qkv_gemm_config_tuple
    w4a8_gemm_config[
        "out_linear_gemm_config_tuple"] = out_linear_gemm_config_tuple
    w4a8_gemm_config["ffn1_gemm_config_tuple"] = ffn1_gemm_config_tuple
    w4a8_gemm_config["ffn2_gemm_config_tuple"] = ffn2_gemm_config_tuple
    return w4a8_gemm_config


def update_refined_recompute(rr, sequence_parallel, lora=False):
    """update refined recompute dict."""
    # if rr is a dict, return it directly
    if isinstance(rr, dict):
        return rr
    if rr == "":
        return {}
    else:

        rr_res = {
            "mlp_row_ln": 0,
            "attention_row_ln": 0,
            "attention_column_ln": 0,
            "mlp_column_ln": 0,
            "flash_attn": 0,
        }
        ops = rr.split(",")
        for op in ops:
            if ":" not in op:
                raise ValueError(
                    "Illegal refined_recompute input, please check.")
            op_name, skip_num = op.split(":")[0], int(op.split(":")[1])
            if op_name not in rr_res:
                raise ValueError(
                    f"Refined recompute do not support {op_name}, please check."
                )

            if op_name in [
                    "mlp_row_ln",
                    "attention_row_ln",
                    "attention_column_ln",
                    "mlp_column_ln",
            ]:
                if not sequence_parallel:
                    logger.warning(
                        f"Currently, the `{op_name}` op is only supported "
                        "when `sequence_parallel=True`. This refined recompute op will be ignored."
                    )
                    continue
                if lora:
                    logger.warning(
                        "Currently, LoRA does not support refined recompute "
                        f"for the `{op_name}` op. This refined recompute op will be ignored."
                    )
                    continue
            rr_res[op_name] = skip_num

        return rr_res


def model_convert_fp8(model_path, device=None):
    """
    Convert a model checkpoint from bf16/fp16 to fp8 format.
    Args:
        model_path (str): The path to the directory containing the model checkpoint files
            (e.g., config.json and model_state.pdparams).
        device (str, optional): The device to set for paddle, such as 'cpu' or 'gpu'.
            If None, the default device is used.

    Note:
        This function requires non-smooth quantization 'act_scales' to be applied when using the converted model.
    """
    if device is not None:
        paddle.device.set_device(device)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as model_config_file:
        model_config = json.load(model_config_file)
        nums_layers = model_config["num_layers"]

    weight_scales_path = os.path.join(model_path, "weight_scales_0.json")
    with open(weight_scales_path, "r") as weight_scales_file:
        weight_scales = json.load(weight_scales_file)
        if "ernie.decoder.layers." + str(
                0) + ".gate.weight_quanter" in weight_scales:
            logger.info("FP8 model checkpoint already converted")
            return
        else:
            logger.info("Converting model checkpoint to fp8...")

    ffn1_weights_name = ".linear1.weight"
    ffn1_bias_name = ".linear1.bias"

    gate_weights_name = ".gate.weight"
    up_weights_name = ".up.weight"
    gate_bias_name = ".gate.bias"
    up_bias_name = ".up.bias"

    params_states = paddle.load(
        os.path.join(model_path, "model_state.pdparams"))
    new_path = os.path.join(model_path, "model_state.pdparams")

    for i in range(0, nums_layers):
        ffn1_weights = params_states["ernie.decoder.layers." + str(i) +
                                     ffn1_weights_name]
        ffn1_weights_0 = ffn1_weights[:, ::2]
        ffn1_weights_1 = ffn1_weights[:, 1::2]

        ffn1_weights_0_range = paddle.abs(ffn1_weights_0).max()
        ffn1_weights_1_range = paddle.abs(ffn1_weights_1).max()

        weight_scales["ernie.decoder.layers." + str(i) +
                      ".gate.weight_quanter"] = (paddle.cast(
                          ffn1_weights_0_range, "float").numpy().tolist())
        weight_scales["ernie.decoder.layers." + str(i) +
                      ".up.weight_quanter"] = (paddle.cast(
                          ffn1_weights_1_range, "float").numpy().tolist())
        params_states["ernie.decoder.layers." + str(i) +
                      gate_weights_name] = (ffn1_weights_0 * 448 /
                                            ffn1_weights_0_range)
        params_states["ernie.decoder.layers." + str(i) +
                      up_weights_name] = (ffn1_weights_1 * 448 /
                                          ffn1_weights_1_range)
        del params_states["ernie.decoder.layers." + str(i) + ffn1_weights_name]

        ffn1_bias = params_states["ernie.decoder.layers." + str(i) +
                                  ffn1_bias_name]
        params_states["ernie.decoder.layers." + str(i) +
                      gate_bias_name] = ffn1_bias[::2]
        params_states["ernie.decoder.layers." + str(i) +
                      up_bias_name] = ffn1_bias[1::2]
        del params_states["ernie.decoder.layers." + str(i) + ffn1_bias_name]

    with open(model_path + "/weight_scales_0.json", "w") as weight_scales_file:
        json.dump(weight_scales, weight_scales_file)

    paddle.save(params_states, new_path)



def load_ep_checkpoint(model_path, config, return_numpy=False, return_key_name=True):
    """
    load ep checkpoint
    """
    if return_key_name:
        merge_path = os.path.join(model_path, "merged_tp1_state_split")
        if os.path.isdir(merge_path):
            # load keyname

            state_dicts = []
            files = glob.glob(model_path + "/merged_tp1_state_split/*")
            for file_name in files:
                try:
                    state_dicts += [
                        {file_name.split("/")[-1]: file_name}
                    ]  # save {layer_name: weight_file_name}
                except Exception:
                    pass
            new_state_dict = {}
            for state_dict in state_dicts:
                for key, value in state_dict.items():
                    new_state_dict[key] = value
            state_dict = new_state_dict
        else:
            with open(
                os.path.join(model_path, "model.safetensors.index.json"), "r"
            ) as f:
                weight_map = json.load(f)["weight_map"]
                state_dict = {
                    k: "[" + k + "]" + os.path.join(model_path, v)
                    for k, v in weight_map.items()
                }
            return state_dict
    else:
        # return_numpy=True cpu
        # return_numpy=False gpu
        with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
            weight_list = json.load(f)["weight_map"]
        filtered_map = {k: v for k, v in weight_list.items() if "experts" not in k}
        num_local_ffn_keys = []
        quant_suffix = (
            "quant_weight"
            if config.use_offline_quant and config.moe_quant_type != "default"
            else ""
        )
        scale_suffix = (
            "quant_scale"
            if config.use_offline_quant and config.moe_quant_type != "default"
            else ""
        )

        for i in range(config.moe_layer_start_index, config.num_layers):
            for j in range(
                config.num_experts_start_offset,
                config.num_experts_start_offset + config.num_experts_per_rank,
            ):
                ffn1_quant_key = f"ernie.layers.{i}.mlp.experts.{j}.up_gate_proj.weight.{quant_suffix}"
                ffn2_quant_key = (
                    f"ernie.layers.{i}.mlp.experts.{j}.down_proj.weight.{quant_suffix}"
                )
                ffn1_scale_key = f"ernie.layers.{i}.mlp.experts.{j}.up_gate_proj.weight.{scale_suffix}"
                ffn2_scale_key = (
                    f"ernie.layers.{i}.mlp.experts.{j}.down_proj.weight.{scale_suffix}"
                )
                num_local_ffn_keys.append(ffn1_quant_key)
                num_local_ffn_keys.append(ffn2_quant_key)
                num_local_ffn_keys.append(ffn1_scale_key)
                num_local_ffn_keys.append(ffn2_scale_key)

        for k in num_local_ffn_keys:
            if k in weight_list:
                filtered_map[k] = weight_list[k]

        state_dict = {}
        for k, safetensor_path in filtered_map.items():
            with safe_open(
                os.path.join(model_path, safetensor_path), framework="np", device="cpu"
            ) as f:
                if k in f.keys():
                    weight = f.get_tensor(k)
                    if not return_numpy:
                        weight = paddle.Tensor(weight, zero_copy=True)
                        weight = weight._copy_to(
                            paddle.framework._current_expected_place(), False
                        )
                    state_dict[k] = weight
    return state_dict


def get_safe_tensor_file(model_path):
    """
    get_safe_tensor_file
    """
    with open(os.path.join(model_path, "model.safetensors.index.json"),
              "r") as f:
        weight_map = json.load(f)["weight_map"]
        safe_tensor_list = list(set(weight_map.values()))
        key_name_list = list(set(weight_map.keys()))
        safe_tensor_list = [os.path.join(model_path, v) for v in safe_tensor_list]

    return key_name_list, safe_tensor_list


def safetensors_weights_iterator(safe_tensor_list: list[str], ):
    """
    safetensors_weights_iterator
    """
    for st_file in tqdm(
            safe_tensor_list,
            desc="Loading safetensors checkpoint shards",
    ):
        with safe_open(st_file, framework="np") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def get_state_dict(model_path, config):
    """
    get_sate_dict
    """
    state_dict = {}
    _, safe_tensor_list = get_safe_tensor_file(
        os.path.join(model_path, f"rank{config.tensor_parallel_rank}"))
    weights_iterator = safetensors_weights_iterator(safe_tensor_list)
    for name, weight in weights_iterator:
        state_dict[name] = weight
    return state_dict


def load_checkpoint(model_path, cls, config, return_numpy=True):
    """
    load checkpoint
    """
    if config.use_ep:
        state_dict = load_ep_checkpoint(
            model_path, config, return_numpy=True, return_key_name=True
        )
    else:
        rank_dirs = [
            f
            for f in os.listdir(model_path)
            if f.startswith("rank") and os.path.isdir(os.path.join(model_path, f))
        ]
        if len(rank_dirs) > 1:
            if config.tensor_parallel_degree != len(rank_dirs):
                raise ValueError(
                    f"Your model only supports loading with tp{len(rank_dirs)}"
                )
            state_dict = get_state_dict(model_path, config)
        else:
            state_dict = load_tp_checkpoint(
                model_path, cls, config, return_numpy=return_numpy
            )
    return state_dict


def parser_quant_type(quant_type):
    """
    Parse the quantization type string and return the corresponding quantization types for weights,
    activations, and custom.

    Args:
        quant_type (str): The quantization type string. It can be one of the following formats:
            - "weight_only_int8" or "wint8": Only weights are quantized to int8.
            - "weight_only_int4" or "wint4": Only weights are quantized to int4.
            - A custom string in the format of "wxaybzcfp8", where 'x', 'y', 'z' are the quantization bitwidths
            for weights, activations, and custom respectively,
                and 'a', 'b', 'c' are the prefixes indicating the quantization types
                (e.g., 'fp8' for floating-point 8-bit).
                If a prefix is missing, the default quantization type will be used.

    Returns:
        tuple: A tuple of three strings representing the quantization types for weights, activations,
                and custom respectively.
                If the input is "weight_only_int8" or "wint8", returns ("int8", default_type, default_type).
                If the input is "weight_only_int4" or "wint4", returns ("int4", default_type, default_type).
                For custom strings, returns the parsed quantization types based on the input format.

    Raises:
        AssertionError: If the custom quantization type string format is incorrect.
    """
    default_type = paddle.get_default_dtype()
    conver_dict = {
        "8": "int8",
        "4": "int4",
        "16": paddle.get_default_dtype,
        "fp8": "float8_e4m3fn",
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32"
    }
    cache_type = default_type
    if "c8" in quant_type:
        cache_type = "int8"
    elif "cfp8" in quant_type:
        cache_type = "fp8"
    elif "c4" in quant_type:
        cache_type = "int4"
    if "weight_only_int8" in quant_type or "wint8" in quant_type:
        return "int8", default_type, cache_type
    elif "weight_only_int4" in quant_type or "wint4" in quant_type:
        return "int4", default_type, cache_type
    else:
        # split quant type, eg. w4afp8c8 -> ['w', '4', 'a', 'fp8', 'c', '8']
        pattern = f"({'|'.join(map(re.escape, ['w', 'a', 'c']))})"
        splited_type = re.split(pattern, quant_type)
        splited_type = [tmp_type for tmp_type in splited_type if tmp_type]
        assert (len(splited_type) % 2 == 0 and len(splited_type)
                <= 6), f"Quant type[{quant_type}] format error."

        quant_type_list = []
        if "w" in splited_type:
            w_idx = splited_type.index("w")
            quant_type_list.append(conver_dict[splited_type[w_idx + 1]])
        else:
            quant_type_list.append(default_type)
        if "a" in splited_type:
            a_idx = splited_type.index("a")
            quant_type_list.append(conver_dict[splited_type[a_idx + 1]])
        else:
            quant_type_list.append(default_type)
        if "c" in splited_type:
            c_idx = splited_type.index("c")
            quant_type_list.append(conver_dict[splited_type[c_idx + 1]])
        else:
            quant_type_list.append(default_type)

        return quant_type_list[0], quant_type_list[1], quant_type_list[2]
