"""
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
"""

import paddle
import os
from paddlenlp.trainer import strtobool
from efficientllm.models.utils import load_checkpoint
from efficientllm.inference_args import InferenceArgs
from paddlenlp.utils.log import logger
from efficientllm.models.configuration import ErnieBotConfig
from efficientllm.models.tokenizer import ErnieBotTokenizer
from safetensors.numpy import save_file as safe_save_file
from paddlenlp.utils.env import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
import shutil
import argparse
import importlib
import json
from paddlenlp.transformers.model_utils import shard_checkpoint

MODEL_LIB_NAMES = [
    "efficientllm.models.modeling_ernie_bot",
]


def parse_arguments():
    """
    parse_arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        required=True,
        help="The directory of model.",
    )
    parser.add_argument(
        "--output_dir",
        default="merged_output",
        required=True,
        help="The directory of merged model output.",
    )
    parser.add_argument(
        "--safe_serialization",
        type=strtobool,
        default="True",
        help="Whether merge the model into safetensors format.",
    )
    parser.add_argument(
        "--predict_model_type",
        type=str,
        default="",
        help="Quantization type for the model.",
    )

    parser.add_argument(
        "--draft_type",
        type=str,
        default=None,
        choices=["autoregressive", "inference_with_reference", "hydra", "mtp"],
        help="Quantization type for the model.",
    )

    parser.add_argument(
        "--moe_quant_type",
        default="default",
        type=str,
        choices=["weight_only_int4", "weight_only_int8", "w4a8", "fp8", "default"],
        help="quant type for moe part",
    )

    parser.add_argument(
        "--use_ep",
        type=strtobool,
        default="True",
        help="Whether merge the model into safetensors format.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    return parser.parse_args()


def get_model_cls(config):
    """
    Get model class from model configuration.
    """
    init_class = "ErnieBotFusedModel"
    for lib_name in MODEL_LIB_NAMES:
        eb_lib = importlib.import_module(lib_name)
        if hasattr(eb_lib, init_class):
            cls = getattr(eb_lib, init_class)
            return cls

    raise RuntimeError(f"Cannot find model architecture({init_class}) from eb_lib")


def save_safetensors(state_dict, args):
    """
    save_safetensors
    """
    logger.info("Move to numpy.")
    for k in list(state_dict.keys()):
        if isinstance(state_dict[k], paddle.Tensor):
            state_dict[k] = state_dict.pop(k).cpu().numpy()

    logger.info("Save safetensors files.")
    shards, index = shard_checkpoint(
        state_dict,
        max_shard_size="5GB",
        weights_name=SAFE_WEIGHTS_NAME,
        shard_format="naive",
    )
    for shard_file, shard in shards.items():
        save_file = os.path.join(args.output_dir, shard_file)
        logger.info(f"Saving {save_file}")
        safe_save_file(shard, save_file, metadata={"format": "np"})

    save_index_file = os.path.join(args.output_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(save_index_file, "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2) + "\n"
        f.write(content)


def quanted_tensor(cls, state_dict, config):
    """
    quanted_tensor
    """
    name_action_mappings = cls._get_tensor_quantization_mappings(config)
    state_keys_map = cls._resolve_prefix_keys(
        name_action_mappings.keys(), state_dict.keys()
    )
    for k, v in state_keys_map.items():
        name_action_mappings[v] = name_action_mappings.pop(k)
    state_dict_to_save = {}
    from efficientllm.layers.utils import get_tensor
    from tqdm import tqdm
    for key in tqdm(state_dict.keys(), desc="process quantized weights  "):
        tensor_path = state_dict[key]
        if key in name_action_mappings:
            ret = state_dict[key]
            action = name_action_mappings.pop(key)
            quanted_weight_tensor, weight_scale_tensor = action(get_tensor(ret))
            if quanted_weight_tensor._is_initialized():
                state_dict_to_save[key + ".quant_weight"] = quanted_weight_tensor.cpu()
            if weight_scale_tensor._is_initialized():
                state_dict_to_save[key + ".quant_scale"] = weight_scale_tensor.cpu()
            else:
                state_dict_to_save[key] = quanted_weight_tensor.cpu()
        else:
            state_dict_to_save[key] = get_tensor(tensor_path).cpu()

    if len(name_action_mappings) > 0:
        for x in name_action_mappings.keys():
            logger.debug(
                f"key <{x}> need to merge tensor parallel but we can't find in model state."
            )
    return state_dict_to_save


def get_quant_type(args):
    """
    get_quant_type
    """
    quant_type = args.predict_model_type.lower()
    if quant_type == "default":
        quant_type = ""
    moe_quant_type = args.moe_quant_type.lower()
    if moe_quant_type == "default":
        moe_quant_type = ""
    paddle.set_default_dtype(args.dtype)
    offline_args = InferenceArgs(
        quant_type=quant_type,
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        hidden_size=1,
        ffn_hidden_size=1,
        mp_rank=1,
        mp_size=1,
    )
    weight_dtype, act_dtype, cachekv_dtype = (
        offline_args.weight_dtype,
        offline_args.act_dtype,
        offline_args.cachekv_dtype,
    )
    return weight_dtype, act_dtype, cachekv_dtype, quant_type, moe_quant_type


def main():
    """
    main
    """
    args = parse_arguments()
    tokenizer = ErnieBotTokenizer.from_pretrained(args.model_name_or_path)
    config = ErnieBotConfig.from_pretrained(args.model_name_or_path)
    (
        config.weight_dtype,
        config.act_dtype,
        config.cachekv_dtype,
        config.quant_type,
        config.moe_quant_type,
    ) = get_quant_type(args)
    config.is_mtp = args.draft_type in ["eagle", "mtp"]
    config.use_ep = args.use_ep
    cls = get_model_cls(config)
    # load
    state_dict = load_checkpoint(
        args.model_name_or_path, cls, config, return_numpy=True
    )
    import time

    start = time.perf_counter()
    state_dict_to_save = quanted_tensor(cls=cls, state_dict=state_dict, config=config)
    end = time.perf_counter()
    logger.info("Finish Quantize.")
    logger.info(f"load和量化耗时: {end - start:.6f} 秒")

    logger.info("Begin to save model")
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.perf_counter()
    if not args.safe_serialization:
        paddle.save(
            state_dict_to_save,
            os.path.join(args.output_dir, "model_state.pdparams"),
        )
    else:
        save_safetensors(state_dict_to_save, args)

    config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if config.moe_quant_type == "w4a8":
        # cp act_scales.json
        shutil.copy(args.model_name_or_path + '/act_scales.json', args.output_dir)
        shutil.copy(args.model_name_or_path + '/weight_scales.json', args.output_dir)
    end = time.perf_counter()
    logger.info(f"save耗时: {end - start:.6f} 秒")
    logger.info("Finish.")


if __name__ == "__main__":
    main()